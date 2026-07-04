"""Weighted Sparse Logistic Regression with fixed L2.

Replaces the quadratic datafit in ``WeightedElasticNet`` with binary
cross-entropy (logistic loss), following the same bilevel structure:

    min_{β}  (1/n) Σ_i log(1 + exp(-y_i xᵢᵀβ))
            + (α_l2 / 2) ‖β‖²
            + Σ_j α_j |β_j|

where ``α = exp(log_alpha)`` is the feature-wise L1 hyperparameter vector
and ``α_l2`` is a fixed L2 coefficient.

Design notes
------------
* The inner BCD uses coordinate-wise adaptive Lipschitz constants from the
  logistic sigmoid Hessian, exactly as ``SparseLogreg``, with the L2 term
  added to each coordinate's step denominator.
* The Jacobian (forward-mode) propagation is *not* implemented: this model
  is designed for use with ``Implicit`` and ``ImplicitVariational``, both of
  which call ``compute_beta`` with ``compute_jac=False``.
* The ``__getattribute__`` binding trick (from ``WeightedElasticNet``) lets
  the Numba-compiled static methods capture the fixed ``alpha_l2`` from the
  instance via thin instance-method wrappers.
"""

import warnings

import numpy as np
from numba import njit
from numpy.linalg import norm
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator

from sparse_ho.models.base import BaseModel
from sparse_ho.utils import ST, sigma, dual_logreg, init_dbeta0_new_p


# ---------------------------------------------------------------------------
# Numba-compiled BCD kernels
# ---------------------------------------------------------------------------

@njit
def _bcd_dense_wlogreg(
    X, y, beta, dbeta, dual_var, ddual_var, alpha, L, alpha_l2, compute_jac
):
    """Dense BCD pass for logistic + L2 + weighted L1.

    dual_var convention: ``dual_var = y * (X @ beta)`` (same as SparseLogreg).
    ``compute_jac`` is accepted but ignored — forward Jacobian not implemented.
    """
    n_samples, n_features = X.shape
    for j in range(n_features):
        beta_old = beta[j]
        sigmar = sigma(dual_var)
        # Use the precomputed coordinate Lipschitz upper bound:
        #   ||X_j||^2 / (4 n) + alpha_l2
        # This avoids an extra dense reduction per coordinate.
        L_temp = L[j]
        if L_temp < 1e-14:
            continue
        # Full smooth gradient at β_j (logistic + L2)
        grad_j = X[:, j] @ (y * (sigmar - 1.0)) / n_samples + alpha_l2 * beta[j]
        z_j = beta[j] - grad_j / L_temp
        beta[j] = ST(z_j, alpha[j] / L_temp)
        # Maintain dual_var = y * X @ beta
        dual_var += y * X[:, j] * (beta[j] - beta_old)


@njit
def _bcd_dense_wlogreg_restricted(
    X, y, beta, dbeta, dual_var, ddual_var, alpha, L, alpha_l2, compute_jac,
    active_idx
):
    """Restricted dense BCD pass over a preselected working set."""
    n_samples = X.shape[0]
    for jj in range(active_idx.shape[0]):
        j = active_idx[jj]
        beta_old = beta[j]
        sigmar = sigma(dual_var)
        L_temp = L[j]
        if L_temp < 1e-14:
            continue
        grad_j = X[:, j] @ (y * (sigmar - 1.0)) / n_samples + alpha_l2 * beta[j]
        z_j = beta[j] - grad_j / L_temp
        beta[j] = ST(z_j, alpha[j] / L_temp)
        dual_var += y * X[:, j] * (beta[j] - beta_old)


@njit
def _bcd_sparse_wlogreg(
    data, indptr, indices, y, n_samples, n_features,
    beta, dbeta, dual_var, ddual_var, alphas, L, alpha_l2, compute_jac
):
    """Sparse BCD pass for logistic + L2 + weighted L1.

    ``compute_jac`` accepted but ignored.
    """
    for j in range(n_features):
        Xjs = data[indptr[j]:indptr[j + 1]]
        idx_nz = indices[indptr[j]:indptr[j + 1]]
        if Xjs.shape[0] == 0:
            continue
        beta_old = beta[j]
        sigmar = sigma(dual_var[idx_nz])
        L_temp_log = np.sum(Xjs ** 2 * sigmar * (1.0 - sigmar)) / n_samples
        L_temp = L_temp_log + alpha_l2
        if L_temp < 1e-14:
            continue
        grad_j = (Xjs @ (y[idx_nz] * (sigmar - 1.0))) / n_samples + alpha_l2 * beta[j]
        z_j = beta[j] - grad_j / L_temp
        beta[j:j + 1] = ST(z_j, alphas[j] / L_temp)
        dual_var[idx_nz] += y[idx_nz] * Xjs * (beta[j] - beta_old)


@njit
def _bcd_sparse_wlogreg_restricted(
    data, indptr, indices, y, n_samples, n_features,
    beta, dbeta, dual_var, ddual_var, alphas, L, alpha_l2, compute_jac,
    active_idx
):
    """Restricted sparse BCD pass over a preselected working set."""
    for jj in range(active_idx.shape[0]):
        j = active_idx[jj]
        Xjs = data[indptr[j]:indptr[j + 1]]
        idx_nz = indices[indptr[j]:indptr[j + 1]]
        if Xjs.shape[0] == 0:
            continue
        beta_old = beta[j]
        sigmar = sigma(dual_var[idx_nz])
        L_temp_log = np.sum(Xjs ** 2 * sigmar * (1.0 - sigmar)) / n_samples
        L_temp = L_temp_log + alpha_l2
        if L_temp < 1e-14:
            continue
        grad_j = (Xjs @ (y[idx_nz] * (sigmar - 1.0))) / n_samples + alpha_l2 * beta[j]
        z_j = beta[j] - grad_j / L_temp
        beta[j:j + 1] = ST(z_j, alphas[j] / L_temp)
        dual_var[idx_nz] += y[idx_nz] * Xjs * (beta[j] - beta_old)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class WeightedSparseLogReg(BaseModel):
    r"""Weighted L1 + fixed L2 Sparse Logistic Regression.

    Lower-level objective:

    .. math::

        \frac{1}{n} \sum_{i=1}^n \log\!\left(1 + e^{-y_i x_i^\top \beta}\right)
        + \frac{\alpha_{\mathrm{l2}}}{2} \|\beta\|_2^2
        + \sum_{j=1}^m \lambda_j |\beta_j|

    where :math:`\lambda = \exp(\log\_\alpha)` is the feature-wise L1
    hyperparameter (the optimized variable) and :math:`\alpha_{\mathrm{l2}}`
    is a fixed L2 coefficient.

    Parameters
    ----------
    alpha_l2 : float, default=0.0
        Fixed L2 regularization coefficient.
    """

    def __init__(self, alpha_l2: float = 0.0):
        self.alpha_l2 = float(alpha_l2)
        self.estimator = None  # use internal BCD via compute_beta
        self._cached_L_key = None
        self._cached_L_value = None
        self.use_dense_active_set = True
        self.use_sparse_active_set = True
        self.active_set_min_features = 256
        self.active_set_ratio = 4.0
        self.sparse_active_set_min_features_abs = 5000
        self.active_set_refresh = 5
        self.active_set_extra_factor = 5
        self.active_set_min_extra = 50
        self.active_set_violation_tol = 1e-12
        self.active_set_beta_tol = 1e-12

    # ------------------------------------------------------------------
    # Binding trick: lets instance methods capture self.alpha_l2
    # ------------------------------------------------------------------

    def __getattribute__(self, name):
        if name == "_update_beta_jac_bcd":
            return object.__getattribute__(self, "_update_beta_jac_bcd_impl")
        if name == "_update_beta_jac_bcd_sparse":
            return object.__getattribute__(self, "_update_beta_jac_bcd_sparse_impl")
        if name == "_get_pobj":
            return object.__getattribute__(self, "_get_pobj_impl")
        if name == "_get_dobj":
            return object.__getattribute__(self, "_get_dobj_impl")
        if name == "get_mat_vec":
            return object.__getattribute__(self, "get_mat_vec_impl")
        return object.__getattribute__(self, name)

    # ------------------------------------------------------------------
    # BCD implementations (bound via __getattribute__)
    # ------------------------------------------------------------------

    def _update_beta_jac_bcd_impl(
        self, X, y, beta, dbeta, dual_var, ddual_var, alpha, L, compute_jac=True
    ):
        if compute_jac:
            warnings.warn(
                "WeightedSparseLogReg: forward Jacobian not implemented; "
                "use Implicit or ImplicitVariational instead.",
                RuntimeWarning, stacklevel=2,
            )
        _bcd_dense_wlogreg(
            X, y, beta, dbeta, dual_var, ddual_var, alpha, L,
            self.alpha_l2, compute_jac,
        )

    def _update_beta_jac_bcd_restricted_impl(
        self, X, y, beta, dbeta, dual_var, ddual_var, alpha, L, active_idx,
        compute_jac=True
    ):
        if compute_jac:
            warnings.warn(
                "WeightedSparseLogReg: forward Jacobian not implemented; "
                "use Implicit or ImplicitVariational instead.",
                RuntimeWarning, stacklevel=2,
            )
        _bcd_dense_wlogreg_restricted(
            X, y, beta, dbeta, dual_var, ddual_var, alpha, L,
            self.alpha_l2, compute_jac, active_idx,
        )

    def _update_beta_jac_bcd_sparse_impl(
        self, data, indptr, indices, y, n_samples, n_features,
        beta, dbeta, dual_var, ddual_var, alphas, L, compute_jac=True
    ):
        if compute_jac:
            warnings.warn(
                "WeightedSparseLogReg: forward Jacobian not implemented; "
                "use Implicit or ImplicitVariational instead.",
                RuntimeWarning, stacklevel=2,
            )
        _bcd_sparse_wlogreg(
            data, indptr, indices, y, n_samples, n_features,
            beta, dbeta, dual_var, ddual_var, alphas, L,
            self.alpha_l2, compute_jac,
        )

    def _update_beta_jac_bcd_sparse_restricted_impl(
        self, data, indptr, indices, y, n_samples, n_features,
        beta, dbeta, dual_var, ddual_var, alphas, L, active_idx,
        compute_jac=True
    ):
        if compute_jac:
            warnings.warn(
                "WeightedSparseLogReg: forward Jacobian not implemented; "
                "use Implicit or ImplicitVariational instead.",
                RuntimeWarning, stacklevel=2,
            )
        _bcd_sparse_wlogreg_restricted(
            data, indptr, indices, y, n_samples, n_features,
            beta, dbeta, dual_var, ddual_var, alphas, L,
            self.alpha_l2, compute_jac, active_idx,
        )

    # ------------------------------------------------------------------
    # Primal / dual objectives (bound via __getattribute__)
    # ------------------------------------------------------------------

    def _get_pobj_impl(self, dual_var, X, beta, alphas, y):
        """Primal objective: logistic loss + L2/2 ‖β‖² + Σ α_j |β_j|."""
        return (
            np.logaddexp(0, -dual_var).mean()
            + 0.5 * self.alpha_l2 * np.dot(beta, beta)
            + np.abs(alphas * beta).sum()
        )

    def _get_dobj_impl(self, dual_var, X, beta, alpha, y):
        """Dual objective (vectorial-safe) for the BCD stopping criterion.

        For vectorial alpha the geometric mean is used as a scalar proxy.
        """
        alpha_arr = np.asarray(alpha, dtype=float)
        if alpha_arr.ndim > 0 and alpha_arr.size > 1:
            pos = alpha_arr[alpha_arr > 0]
            alpha_s = float(np.exp(np.mean(np.log(pos)))) if pos.size else 1.0
        else:
            alpha_s = float(alpha_arr)

        n_samples = len(y)
        theta = y * sigma(-dual_var) / (alpha_s * n_samples)
        if issparse(X):
            d_norm = float(np.max(np.abs(np.asarray(X.T @ theta).ravel())))
        else:
            d_norm = float(np.max(np.abs(X.T @ theta)))
        if d_norm > 1.0:
            theta /= d_norm
        return dual_logreg(y, theta, alpha_s)

    def _get_stop_crit(self, dual_var, X, beta, alphas, y):
        """Max KKT residual for logistic + fixed L2 + weighted L1.

        This is an exact optimality certificate for the current primal iterate
        and is preferable to the legacy dual-gap proxy when ``alpha_l2 > 0``.
        """
        n_samples = len(y)
        if issparse(X):
            grad = -np.asarray(X.T @ (y * sigma(-dual_var))).ravel() / n_samples
        else:
            grad = -(X.T @ (y * sigma(-dual_var))) / n_samples
        grad = grad + self.alpha_l2 * beta

        support = np.abs(beta) > self.active_set_beta_tol
        violation = np.maximum(0.0, np.abs(grad) - alphas)
        if np.any(support):
            violation[support] = np.abs(
                grad[support] + alphas[support] * np.sign(beta[support])
            )
        return float(np.max(violation))

    @staticmethod
    def _get_pobj0(dual_var, beta, alphas, y=None):
        """Reference objective at β=0: logistic loss = log(2) per sample."""
        return np.log(2)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_beta_dual_var(self, X, y, mask0=None, dense0=None):
        """Initialize β and dual_var = y * X @ β."""
        beta = np.zeros(X.shape[1])
        if dense0 is None or len(dense0) == 0:
            dual_var = np.zeros(X.shape[0])
        else:
            beta[mask0] = dense0.copy()
            if issparse(X):
                dual_var = y * np.asarray(X[:, mask0] @ dense0).ravel()
            else:
                dual_var = y * (X[:, mask0] @ dense0)
        return beta, dual_var

    def _init_dbeta_ddual_var(
        self, X, y, mask0=None, jac0=None, dense0=None, compute_jac=True
    ):
        """Initialize matrix Jacobian (n_features × n_features)."""
        n_samples, n_features = X.shape
        dbeta = np.zeros((n_features, n_features))
        ddual_var = np.zeros((n_samples, n_features))
        if jac0 is not None and compute_jac:
            dbeta[np.ix_(mask0, mask0)] = jac0.copy()
            ddual_var[:, mask0] = y[:, None] * (X[:, mask0] @ jac0)
        return dbeta, ddual_var

    @staticmethod
    def _init_dbeta0(mask, mask0, jac0):
        size_mat = mask.sum()
        if jac0 is None:
            return np.zeros((size_mat, size_mat))
        return init_dbeta0_new_p(jac0, mask, mask0)

    @staticmethod
    def _init_dbeta(n_features):
        return np.zeros((n_features, n_features))

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[np.ix_(mask, mask)]

    # ------------------------------------------------------------------
    # Lipschitz constant
    # ------------------------------------------------------------------

    def get_L(self, X):
        """Per-coordinate Lipschitz upper bound: ‖X[:,j]‖² / (4n) + α_l2."""
        key = (id(X), X.shape, getattr(X, "nnz", None))
        if self._cached_L_key == key and self._cached_L_value is not None:
            return self._cached_L_value
        n = X.shape[0]
        if issparse(X):
            col_norms_sq = np.asarray(X.power(2).sum(axis=0)).ravel()
        else:
            col_norms_sq = np.sum(X ** 2, axis=0)
        self._cached_L_key = key
        self._cached_L_value = col_norms_sq / (4.0 * n) + self.alpha_l2
        return self._cached_L_value

    def should_use_dense_active_set(self, X):
        return (
            self.use_dense_active_set
            and not issparse(X)
            and X.shape[1] >= self.active_set_min_features
            and X.shape[1] >= self.active_set_ratio * X.shape[0]
        )

    def should_use_sparse_active_set(self, X):
        return (
            self.use_sparse_active_set
            and issparse(X)
            and X.shape[1] >= self.active_set_min_features
            and (
                X.shape[1] >= self.active_set_ratio * X.shape[0]
                or X.shape[1] >= self.sparse_active_set_min_features_abs
            )
        )

    def get_dense_active_set(self, X, y, beta, dual_var, alphas):
        """Support plus top KKT violators for dense wide problems."""
        n_samples, n_features = X.shape
        support = np.flatnonzero(np.abs(beta) > self.active_set_beta_tol)
        residual = y * (sigma(dual_var) - 1.0)
        grad = (X.T @ residual) / n_samples + self.alpha_l2 * beta
        violation = np.maximum(0.0, np.abs(grad) - alphas)
        if support.size:
            violation[support] = 0.0

        active_mask = np.zeros(n_features, dtype=bool)
        active_mask[support] = True

        n_extra = min(
            n_features - support.size,
            max(
                self.active_set_min_extra,
                self.active_set_extra_factor * max(int(support.size), 1),
            ),
        )
        if n_extra > 0:
            violators = np.flatnonzero(violation > self.active_set_violation_tol)
            if violators.size > n_extra:
                top_pos = np.argpartition(violation[violators], -n_extra)[-n_extra:]
                violators = violators[top_pos]
            active_mask[violators] = True

        if not np.any(active_mask):
            active_mask[int(np.argmax(violation))] = True
        return np.flatnonzero(active_mask)

    def get_sparse_active_set(self, X, y, beta, dual_var, alphas):
        """Support plus top KKT violators for sparse wide problems."""
        n_samples, n_features = X.shape
        support = np.flatnonzero(np.abs(beta) > self.active_set_beta_tol)
        residual = y * (sigma(dual_var) - 1.0)
        grad = np.asarray(X.T @ residual).ravel() / n_samples + self.alpha_l2 * beta
        violation = np.maximum(0.0, np.abs(grad) - alphas)
        if support.size:
            violation[support] = 0.0

        active_mask = np.zeros(n_features, dtype=bool)
        active_mask[support] = True

        n_extra = min(
            n_features - support.size,
            max(
                self.active_set_min_extra,
                self.active_set_extra_factor * max(int(support.size), 1),
            ),
        )
        if n_extra > 0:
            violators = np.flatnonzero(violation > self.active_set_violation_tol)
            if violators.size > n_extra:
                top_pos = np.argpartition(violation[violators], -n_extra)[-n_extra:]
                violators = violators[top_pos]
            active_mask[violators] = True

        if not np.any(active_mask):
            active_mask[int(np.argmax(violation))] = True
        return np.flatnonzero(active_mask)

    # ------------------------------------------------------------------
    # Hessian operator for Implicit adjoint
    # ------------------------------------------------------------------

    @staticmethod
    def get_mat_vec(X, y, mask, dense, log_alpha):
        # Satisfied by the ABC; actual calls are redirected to get_mat_vec_impl
        # via __getattribute__.
        raise NotImplementedError(
            "WeightedSparseLogReg redirects get_mat_vec to get_mat_vec_impl."
        )

    def get_mat_vec_impl(self, X, y, mask, dense, log_alpha):
        """LinearOperator for the reduced Hessian: (1/n) X_S^T D X_S + α_l2 I_S.

        D = diag(σ(y * X_S dense)(1 - σ(y * X_S dense))).
        """
        X_m = X[:, mask]
        n_samples, size_supp = X_m.shape
        if issparse(X_m):
            a = y * np.asarray(X_m @ dense).ravel()
        else:
            a = y * (X_m @ dense)
        w = sigma(a) * (1.0 - sigma(a))
        alpha_l2 = self.alpha_l2

        def mv(v):
            if issparse(X_m):
                Xv = np.asarray(X_m @ v).ravel()
                XtDXv = np.asarray(X_m.T @ (w * Xv)).ravel()
            else:
                Xv = X_m @ v
                XtDXv = X_m.T @ (w * Xv)
            return XtDXv / n_samples + alpha_l2 * v

        return LinearOperator((size_supp, size_supp), matvec=mv)

    # ------------------------------------------------------------------
    # Hypergradient reconstruction for Implicit
    # ------------------------------------------------------------------

    @staticmethod
    def _get_grad(X, y, jac, mask, dense, alphas, v):
        """Element-wise hypergradient: α_S ⊙ sign(β_S) ⊙ p_S.

        For vectorial alpha each feature has its own gradient component.
        """
        return alphas[mask] * np.sign(dense) * jac

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        """Expand support-restricted gradient to full feature space."""
        res = np.zeros(n_features)
        res[mask] = jac_v
        return res

    @staticmethod
    def get_mask_jac_v(mask, jac_v):
        return jac_v[mask]

    def get_jac_v(self, X, y, mask, dense, jac, v):
        return jac.T @ v(mask, dense)

    # ------------------------------------------------------------------
    # Support helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reduce_alpha(alpha, mask):
        return alpha[mask]

    @staticmethod
    def reduce_X(X, mask):
        return X[:, mask]

    @staticmethod
    def reduce_y(y, mask):
        return y

    def sign(self, x, log_alpha):
        return np.sign(x)

    def get_beta(self, X, y, mask, dense):
        return mask, dense

    def generalized_supp(self, X, v, log_alpha):
        """Restrict to primal support (Sparse-HO adjoint behavior)."""
        return v

    # ------------------------------------------------------------------
    # Hyperparameter projection
    # ------------------------------------------------------------------

    def proj_hyperparam(self, X, y, log_alpha):
        """Clip each log_alpha_j near log(alpha_max) for logistic regression."""
        if not hasattr(self, "log_alpha_max"):
            n = X.shape[0]
            if issparse(X):
                corr = np.abs(np.asarray(X.T @ y).ravel())
            else:
                corr = np.abs(X.T @ y)
            alpha_max = float(corr.max()) / (2.0 * n)
            self.log_alpha_max = np.log(alpha_max)
        return np.clip(
            log_alpha,
            self.log_alpha_max - 8,
            self.log_alpha_max + np.log(0.9),
        )

    # ------------------------------------------------------------------
    # ImplicitVariational interface
    # ------------------------------------------------------------------

    def get_variational_lambdas(self, log_alpha, n_features):
        """Feature-wise L1 thresholds: Ψ(x̄) = exp(log_alpha)."""
        alpha = np.exp(log_alpha)
        if np.ndim(alpha) == 0:
            return np.full(n_features, float(alpha))
        return np.asarray(alpha, dtype=float)

    @staticmethod
    def get_variational_hypergrad(alpha, xi_diag, v_adj, beta, gamma):
        """Phase-4 hypergradient: α ⊙ ξ ⊙ (γ p)."""
        return np.asarray(alpha, dtype=float) * xi_diag * (float(gamma) * v_adj)

    # ------------------------------------------------------------------
    # Smooth-part derivatives (for ImplicitVariational Hessian builds)
    # ------------------------------------------------------------------

    def get_grad_smooth(self, X, y, beta):
        """Gradient of F(β) = logistic(β) + (α_l2/2)‖β‖²."""
        n = X.shape[0]
        if issparse(X):
            Xb = np.asarray(X @ beta).ravel()
        else:
            Xb = X @ beta
        yz = y * Xb
        s = sigma(-yz)          # σ(-y * Xβ) = P(mistake)
        if issparse(X):
            grad = -np.asarray(X.T @ (y * s)).ravel() / n
        else:
            grad = -(X.T @ (y * s)) / n
        return grad + self.alpha_l2 * beta

    def get_hess_smooth(self, X, y, beta, v):
        """Hessian-vector product of F: (1/n) X^T D X v + α_l2 v."""
        n = X.shape[0]
        if issparse(X):
            Xb = np.asarray(X @ beta).ravel()
        else:
            Xb = X @ beta
        yz = y * Xb
        p = sigma(yz)
        w = p * (1.0 - p)
        if issparse(X):
            Xv = np.asarray(X @ v).ravel()
            Hv = np.asarray(X.T @ (w * Xv)).ravel() / n
        else:
            Xv = X @ v
            Hv = (X.T @ (w * Xv)) / n
        return Hv + self.alpha_l2 * v

    def hessian_diag_weights(self, X_sub, y, beta_sub):
        """Diagonal IRLS weights ``w`` of the smooth loss on a support submatrix.

        Defined so that ``[∇²F(β)]_{S,S} = (1/n) X_S^T diag(w) X_S + α_l2 I``.
        For the logistic loss ``w = σ(yz)(1−σ(yz))`` with ``z = X_S β_S``.
        Consumed by the matrix-free reduced-adjoint solve in
        :mod:`sparse_ho.algo.implicit_variational`.
        """
        if issparse(X_sub):
            z = np.asarray(X_sub @ beta_sub).ravel()
        else:
            z = X_sub @ beta_sub
        p = sigma(y * z)
        return p * (1.0 - p)

    def get_hess_smooth_diag(self, X, y, beta):
        """Diagonal of the Hessian of F: (X² w)_j / n + α_l2."""
        n = X.shape[0]
        if issparse(X):
            Xb = np.asarray(X @ beta).ravel()
        else:
            Xb = X @ beta
        yz = y * Xb
        p = sigma(yz)
        w = p * (1.0 - p)
        if issparse(X):
            diag = np.asarray(X.power(2).T @ w).ravel() / n
        else:
            diag = ((X ** 2).T @ w) / n
        return diag + self.alpha_l2
