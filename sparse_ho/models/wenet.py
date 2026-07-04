import numpy as np
from numba import njit
from numpy.linalg import norm
from scipy.sparse import issparse
from scipy.sparse import eye as sparse_eye
from scipy.sparse import vstack as sparse_vstack
import scipy.sparse.linalg as slinalg
from scipy.sparse.linalg import LinearOperator

from sparse_ho.models.base import BaseModel
from sparse_ho.utils import init_dbeta0_new_p, prox_elasticnet


class CelerWeightedElasticNet:
    r"""`celer.Lasso` wrapper for weighted Elastic-Net with fixed L2.

    The ridge term is absorbed into an augmented weighted-Lasso problem:

    .. math::

        \frac{1}{2n} \|Xw - y\|_2^2 + \frac{\alpha_{\mathrm{l2}}}{2}\|w\|_2^2
        + \sum_j \lambda_j |w_j|
        =
        \frac{1}{2n}
        \left\|
            \begin{bmatrix}
                X \\
                \sqrt{n \alpha_{\mathrm{l2}}} I
            \end{bmatrix}
            w -
            \begin{bmatrix}
                y \\
                0
            \end{bmatrix}
        \right\|_2^2
        + \sum_j \lambda_j |w_j|.

    Parameters
    ----------
    alpha_l2 : float
        Fixed L2 regularization parameter.
    fit_intercept : bool, default=False
        Passed to the internal ``celer.Lasso`` estimator.
    warm_start : bool, default=True
        Passed to the internal ``celer.Lasso`` estimator.
    max_iter : int, default=100
        Passed to the internal ``celer.Lasso`` estimator.
    tol : float, default=1e-4
        Passed to the internal ``celer.Lasso`` estimator.
    kwargs : dict
        Extra keyword arguments forwarded to ``celer.Lasso``.
    """

    def __init__(
        self,
        alpha_l2,
        fit_intercept=False,
        warm_start=True,
        max_iter=100,
        tol=1e-4,
        **kwargs,
    ):
        self.alpha_l2 = float(alpha_l2)
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.kwargs = kwargs.copy()
        self._estimator = None
        self.coef_ = None
        self.intercept_ = 0.0

    def get_params(self, deep=True):
        params = {
            "alpha_l2": self.alpha_l2,
            "fit_intercept": self.fit_intercept,
            "warm_start": self.warm_start,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key == "alpha_l2":
                self.alpha_l2 = float(value)
            elif key == "fit_intercept":
                self.fit_intercept = value
            elif key == "warm_start":
                self.warm_start = value
            elif key == "max_iter":
                self.max_iter = value
            elif key == "tol":
                self.tol = value
            else:
                self.kwargs[key] = value
        return self

    def _get_celer_estimator(self):
        try:
            from celer import Lasso
        except ImportError as exc:
            raise ImportError(
                "CelerWeightedElasticNet requires `celer` to be installed."
            ) from exc

        if self._estimator is None:
            self._estimator = Lasso(
                fit_intercept=self.fit_intercept,
                warm_start=self.warm_start,
                max_iter=self.max_iter,
                tol=self.tol,
                **self.kwargs,
            )
        else:
            self._estimator.set_params(
                fit_intercept=self.fit_intercept,
                warm_start=self.warm_start,
                max_iter=self.max_iter,
                tol=self.tol,
                **self.kwargs,
            )
        return self._estimator

    def _augment(self, X, y):
        n_samples, n_features = X.shape
        n_aug = n_samples + n_features
        sample_scale = np.sqrt(n_aug / n_samples)
        ridge_scale = sample_scale * np.sqrt(n_samples * self.alpha_l2)
        if issparse(X):
            X_data = sample_scale * X
            X_ridge = ridge_scale * sparse_eye(
                n_features, format="csc", dtype=X.dtype
            )
            X_aug = sparse_vstack([X_data, X_ridge], format="csc")
        else:
            X_data = sample_scale * X
            X_ridge = ridge_scale * np.eye(n_features, dtype=X.dtype)
            X_aug = np.vstack([X_data, X_ridge])
        y_aug = np.concatenate(
            [sample_scale * y, np.zeros(n_features, dtype=y.dtype)]
        )
        return X_aug, y_aug

    def fit(self, X, y):
        if self.weights is None:
            raise ValueError("`weights` must be set before calling fit.")

        X_aug, y_aug = self._augment(X, y)
        est = self._get_celer_estimator()
        est.weights = np.asarray(self.weights, dtype=float)
        est.fit(X_aug, y_aug)
        self.coef_ = est.coef_.copy()
        self.intercept_ = getattr(est, "intercept_", 0.0)
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Estimator is not fitted.")
        return X @ self.coef_ + self.intercept_


class WeightedElasticNet(BaseModel):
    r"""Weighted Elastic-Net with fixed L2 regularization.

    The optimization objective is

    .. math::

        \|y - Xw\|_2^2 / (2 n_{\mathrm{samples}})
        + \frac{\alpha_{\mathrm{l2}}}{2} \|w\|_2^2
        + \sum_{j=1}^{n_{\mathrm{features}}} \lambda_j |w_j|

    where the tunable hyperparameter is the feature-wise vector
    ``lambda = exp(log_alpha)`` and ``alpha_l2`` is fixed.

    Parameters
    ----------
    alpha_l2 : float
        Fixed L2 regularization parameter.
    estimator : instance of ``sklearn.base.BaseEstimator``
        Optional estimator following the scikit-learn API.
    """

    def __init__(self, alpha_l2, estimator=None):
        self.alpha_l2 = float(alpha_l2)
        self.estimator = estimator

    def _init_dbeta_ddual_var(
        self, X, y, mask0=None, jac0=None, dense0=None, compute_jac=True
    ):
        n_samples, n_features = X.shape
        dbeta = np.zeros((n_features, n_features))
        ddual_var = np.zeros((n_samples, n_features))
        if jac0 is not None:
            dbeta[np.ix_(mask0, mask0)] = jac0.copy()
            ddual_var[:, mask0] = -X[:, mask0] @ jac0
        return dbeta, ddual_var

    def _init_beta_dual_var(self, X, y, mask0=None, dense0=None):
        beta = np.zeros(X.shape[1])
        if dense0 is None or len(dense0) == 0:
            dual_var = y.copy().astype(np.float64)
        else:
            beta[mask0] = dense0.copy()
            dual_var = y - X[:, mask0] @ dense0
        return beta, dual_var

    def _scale(self, L_j):
        return 1.0 / (1.0 + self.alpha_l2 / L_j)

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
        X, y, beta, dbeta, dual_var, ddual_var, alpha, L, compute_jac=True
    ):
        raise NotImplementedError(
            "WeightedElasticNet overrides this method at runtime with a bound implementation."
        )

    def _update_beta_jac_bcd_impl(
        self, X, y, beta, dbeta, dual_var, ddual_var, alpha, L, compute_jac=True
    ):
        return _update_beta_jac_bcd_wenet(
            X,
            y,
            beta,
            dbeta,
            dual_var,
            ddual_var,
            alpha,
            L,
            self.alpha_l2,
            compute_jac,
        )

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
        data,
        indptr,
        indices,
        y,
        n_samples,
        n_features,
        beta,
        dbeta,
        dual_var,
        ddual_var,
        alphas,
        L,
        compute_jac=True,
    ):
        raise NotImplementedError(
            "WeightedElasticNet overrides this method at runtime with a bound implementation."
        )

    def _update_beta_jac_bcd_sparse_impl(
        self,
        data,
        indptr,
        indices,
        y,
        n_samples,
        n_features,
        beta,
        dbeta,
        dual_var,
        ddual_var,
        alphas,
        L,
        compute_jac=True,
    ):
        return _update_beta_jac_bcd_sparse_wenet(
            data,
            indptr,
            indices,
            y,
            n_samples,
            n_features,
            beta,
            dbeta,
            dual_var,
            ddual_var,
            alphas,
            L,
            self.alpha_l2,
            compute_jac,
        )

    @staticmethod
    @njit
    def _update_bcd_jac_backward(X, alpha, jac_t_v, beta, v_, L):
        raise NotImplementedError(
            "WeightedElasticNet overrides this method at runtime with a bound implementation."
        )

    def _update_bcd_jac_backward_impl(self, X, alpha, jac_t_v, beta, v_, L):
        return _update_bcd_jac_backward_wenet(
            X, alpha, jac_t_v, beta, v_, L, self.alpha_l2
        )

    @staticmethod
    def _get_pobj(dual_var, X, beta, alphas, y=None):
        raise NotImplementedError(
            "WeightedElasticNet overrides this method at runtime with a bound implementation."
        )

    def _get_pobj_impl(self, dual_var, X, beta, alphas, y=None):
        n_samples = dual_var.shape[0]
        pobj = norm(dual_var) ** 2 / (2 * n_samples) + norm(alphas * beta, 1)
        pobj += 0.5 * self.alpha_l2 * norm(beta) ** 2
        return pobj

    def _get_stop_crit(self, dual_var, X, beta, alphas, y=None):
        """Max KKT residual for weighted elastic-net with fixed L2."""
        n_samples = dual_var.shape[0]
        if issparse(X):
            grad = -np.asarray(X.T @ dual_var).ravel() / n_samples
        else:
            grad = -(X.T @ dual_var) / n_samples
        grad = grad + self.alpha_l2 * beta

        support = np.abs(beta) > 1e-12
        violation = np.maximum(0.0, np.abs(grad) - alphas)
        if np.any(support):
            violation[support] = np.abs(
                grad[support] + alphas[support] * np.sign(beta[support])
            )
        return float(np.max(violation))

    @staticmethod
    def _get_pobj0(dual_var, beta, alphas, y=None):
        n_samples = dual_var.shape[0]
        return norm(y) ** 2 / (2 * n_samples)

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[np.ix_(mask, mask)]

    @staticmethod
    def _init_dbeta0(mask, mask0, jac0):
        size_mat = mask.sum()
        if jac0 is None:
            dbeta0_new = np.zeros((size_mat, size_mat))
        else:
            dbeta0_new = init_dbeta0_new_p(jac0, mask, mask0)
        return dbeta0_new

    @staticmethod
    def _init_dbeta(n_features):
        return np.zeros((n_features, n_features))

    @staticmethod
    def _init_ddual_var(dbeta, X, y, beta, alpha):
        return -X @ dbeta

    @staticmethod
    def _init_g_backward(jac_v0, n_features):
        if jac_v0 is None:
            return np.zeros(n_features)
        return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(
        Xs, y, dual_var, dbeta, ddual_var, L, alpha, beta
    ):
        raise NotImplementedError(
            "WeightedElasticNet overrides this method at runtime with a bound implementation."
        )

    def _update_only_jac_impl(self, Xs, y, dual_var, dbeta, ddual_var, L, alpha, beta):
        return _update_only_jac_wenet(
            Xs, y, dual_var, dbeta, ddual_var, L, alpha, beta, self.alpha_l2
        )

    @staticmethod
    @njit
    def _update_only_jac_sparse(
        data, indptr, indices, y, n_samples, n_features, dbeta, dual_var,
        ddual_var, L, alpha, beta
    ):
        raise NotImplementedError(
            "WeightedElasticNet overrides this method at runtime with a bound implementation."
        )

    def _update_only_jac_sparse_impl(
        self,
        data,
        indptr,
        indices,
        y,
        n_samples,
        n_features,
        dbeta,
        dual_var,
        ddual_var,
        L,
        alpha,
        beta,
    ):
        return _update_only_jac_sparse_wenet(
            data,
            indptr,
            indices,
            y,
            n_samples,
            n_features,
            dbeta,
            dual_var,
            ddual_var,
            L,
            alpha,
            beta,
            self.alpha_l2,
        )

    @staticmethod
    def _reduce_alpha(alpha, mask):
        return alpha[mask]

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        res = np.zeros(n_features)
        res[mask] = jac_v
        return res

    @staticmethod
    def get_mask_jac_v(mask, jac_v):
        return jac_v[mask]

    @staticmethod
    def _get_grad(X, y, jac, mask, dense, alphas, v):
        size_supp = mask.sum()
        jac_t_v = np.zeros(size_supp)
        jac_t_v = alphas[mask] * np.sign(dense) * jac
        return jac_t_v

    def proj_hyperparam(self, X, y, log_alpha):
        if not hasattr(self, "log_alpha_max"):
            alpha_max = np.max(np.abs(X.T @ y)) / X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        log_alpha = np.clip(
            log_alpha, self.log_alpha_max - 5, self.log_alpha_max + np.log(0.9)
        )
        return log_alpha

    @staticmethod
    def get_L(X):
        if issparse(X):
            return slinalg.norm(X, axis=0) ** 2 / X.shape[0]
        return norm(X, axis=0) ** 2 / X.shape[0]

    @staticmethod
    def get_mat_vec(X, y, mask, dense, log_alpha):
        raise NotImplementedError(
            "WeightedElasticNet overrides this method at runtime with a bound implementation."
        )

    def get_mat_vec_impl(self, X, y, mask, dense, log_alpha):
        X_m = X[:, mask]
        n_samples, size_supp = X_m.shape

        def mv(v):
            return X_m.T @ (X_m @ v) / n_samples + self.alpha_l2 * v

        return LinearOperator((size_supp, size_supp), matvec=mv)

    def _use_estimator(self, X, y, alpha, tol, max_iter=None):
        if self.estimator is None:
            raise ValueError("You did not pass a solver with sklearn API")
        params = dict(tol=tol)
        if max_iter is not None:
            params["max_iter"] = max_iter
        try:
            self.estimator.set_params(**params)
        except Exception:
            params.pop("max_iter", None)
            self.estimator.set_params(**params)
        self.estimator.weights = alpha
        self.estimator.alpha_l2 = self.alpha_l2
        self.estimator.fit(X, y)
        mask = self.estimator.coef_ != 0
        dense = self.estimator.coef_[mask]
        return mask, dense, None

    @staticmethod
    def reduce_X(X, mask):
        return X[:, mask]

    @staticmethod
    def reduce_y(y, mask):
        return y

    def sign(self, x, log_alpha):
        return x

    def get_beta(self, X, y, mask, dense):
        return mask, dense

    def get_jac_v(self, X, y, mask, dense, jac, v):
        return jac.T @ v(mask, dense)

    def generalized_supp(self, X, v, log_alpha):
        return v

    def get_jac_residual_norm(
        self, Xs, ys, n_samples, beta, dbeta, dual_var, ddual_var, alpha
    ):
        res = ddual_var.T @ ddual_var
        res += n_samples * self.alpha_l2 * np.sum(dbeta * dbeta, axis=0)
        res += n_samples * alpha * np.sign(beta) @ dbeta
        return norm(res)

    def set_variational_alpha(self, alpha):
        self._variational_alpha = np.asarray(alpha, dtype=float)

    def get_variational_lambdas(self, log_alpha, n_features):
        alpha = np.exp(log_alpha)
        return np.asarray(alpha, dtype=float)

    def get_variational_hypergrad(self, alpha, xi_diag, v_adj, beta, gamma):
        alpha = np.asarray(alpha, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        return alpha * xi_diag * (gamma * v_adj)

    def get_grad_smooth(self, X, y, beta):
        n_samples = X.shape[0]
        return -X.T @ (y - X @ beta) / n_samples + self.alpha_l2 * beta

    def get_hess_smooth(self, X, y, beta, v):
        n_samples = X.shape[0]
        return X.T @ (X @ v) / n_samples + self.alpha_l2 * v

    def hessian_diag_weights(self, X_sub, y, beta_sub):
        """Diagonal IRLS weights ``w`` of the smooth loss on a support submatrix.

        Defined so that ``[∇²F(β)]_{S,S} = (1/n) X_S^T diag(w) X_S + α_l2 I``.
        For the least-squares data-fidelity term the curvature is constant,
        hence ``w ≡ 1``. Consumed by the matrix-free reduced-adjoint solve in
        :mod:`sparse_ho.algo.implicit_variational` so that its ``H_S`` matches
        the model's own ``∇²F`` instead of assuming logistic curvature.
        """
        return np.ones(X_sub.shape[0], dtype=float)

    def get_hess_smooth_diag(self, X, y, beta):
        n_samples = X.shape[0]
        if issparse(X):
            hess_diag = np.array((X.power(2)).sum(axis=0)).ravel() / n_samples
        else:
            hess_diag = np.sum(X ** 2, axis=0) / n_samples
        return hess_diag + self.alpha_l2

    # Bind the methods that need access to the fixed alpha_l2 while preserving
    # the method names expected by the optimization routines.
    def __getattribute__(self, name):
        if name == "_update_beta_jac_bcd":
            return object.__getattribute__(self, "_update_beta_jac_bcd_impl")
        if name == "_update_beta_jac_bcd_sparse":
            return object.__getattribute__(self, "_update_beta_jac_bcd_sparse_impl")
        if name == "_update_bcd_jac_backward":
            return object.__getattribute__(self, "_update_bcd_jac_backward_impl")
        if name == "_update_only_jac":
            return object.__getattribute__(self, "_update_only_jac_impl")
        if name == "_update_only_jac_sparse":
            return object.__getattribute__(self, "_update_only_jac_sparse_impl")
        if name == "_get_pobj":
            return object.__getattribute__(self, "_get_pobj_impl")
        if name == "get_mat_vec":
            return object.__getattribute__(self, "get_mat_vec_impl")
        return object.__getattribute__(self, name)


@njit
def _update_beta_jac_bcd_wenet(
    X, y, beta, dbeta, dual_var, ddual_var, alpha, L, alpha_l2, compute_jac=True
):
    n_samples, n_features = X.shape
    non_zeros = np.where(L != 0)[0]
    for j in non_zeros:
        beta_old = beta[j]
        if compute_jac:
            dbeta_old = dbeta[j, :].copy()
        zj = beta[j] + dual_var @ X[:, j] / (L[j] * n_samples)
        beta[j] = prox_elasticnet(zj, alpha[j] / L[j], alpha_l2 / L[j])
        if compute_jac:
            dzj = dbeta[j, :] + X[:, j] @ ddual_var / (L[j] * n_samples)
            scale = 1.0 / (1.0 + alpha_l2 / L[j])
            dbeta[j : j + 1, :] = scale * np.abs(np.sign(beta[j])) * dzj
            dbeta[j : j + 1, j] -= scale * alpha[j] * np.sign(beta[j]) / L[j]
            ddual_var -= np.outer(X[:, j], (dbeta[j, :] - dbeta_old))
        dual_var -= X[:, j] * (beta[j] - beta_old)


@njit
def _update_beta_jac_bcd_sparse_wenet(
    data,
    indptr,
    indices,
    y,
    n_samples,
    n_features,
    beta,
    dbeta,
    dual_var,
    ddual_var,
    alphas,
    L,
    alpha_l2,
    compute_jac=True,
):
    non_zeros = np.where(L != 0)[0]
    for j in non_zeros:
        Xjs = data[indptr[j] : indptr[j + 1]]
        idx_nz = indices[indptr[j] : indptr[j + 1]]
        beta_old = beta[j]
        if compute_jac:
            dbeta_old = dbeta[j, :].copy()
        zj = beta[j] + dual_var[idx_nz] @ Xjs / (L[j] * n_samples)
        beta[j : j + 1] = prox_elasticnet(zj, alphas[j] / L[j], alpha_l2 / L[j])
        if compute_jac:
            dzj = dbeta[j, :] + Xjs @ ddual_var[idx_nz, :] / (L[j] * n_samples)
            scale = 1.0 / (1.0 + alpha_l2 / L[j])
            dbeta[j : j + 1, :] = scale * np.abs(np.sign(beta[j])) * dzj
            dbeta[j : j + 1, j] -= scale * alphas[j] * np.sign(beta[j]) / L[j]
            ddual_var[idx_nz, :] -= np.outer(Xjs, (dbeta[j, :] - dbeta_old))
        dual_var[idx_nz] -= Xjs * (beta[j] - beta_old)


@njit
def _update_bcd_jac_backward_wenet(X, alpha, jac_t_v, beta, v_, L, alpha_l2):
    n_samples, n_features = X.shape
    sign_beta = np.sign(beta)
    for j in np.arange(sign_beta.shape[0] - 1, -1, -1):
        scale = 1.0 / (1.0 + alpha_l2 / L[j])
        jac_t_v[j] -= scale * v_[j] * alpha[j] * sign_beta[j] / L[j]
        v_[j] *= scale * np.abs(sign_beta[j])
        v_ -= v_[j] / (L[j] * n_samples) * X[:, j] @ X
    return jac_t_v


@njit
def _update_only_jac_wenet(
    Xs, y, dual_var, dbeta, ddual_var, L, alpha, beta, alpha_l2
):
    n_samples, n_features = Xs.shape
    for j in range(n_features):
        dbeta_old = dbeta[j, :].copy()
        dzj = dbeta[j, :] + Xs[:, j] @ ddual_var / (L[j] * n_samples)
        scale = 1.0 / (1.0 + alpha_l2 / L[j])
        dbeta[j : j + 1, :] = scale * dzj
        dbeta[j : j + 1, j] -= scale * alpha[j] * np.sign(beta[j]) / L[j]
        ddual_var -= np.outer(Xs[:, j], (dbeta[j, :] - dbeta_old))


@njit
def _update_only_jac_sparse_wenet(
    data,
    indptr,
    indices,
    y,
    n_samples,
    n_features,
    dbeta,
    dual_var,
    ddual_var,
    L,
    alpha,
    beta,
    alpha_l2,
):
    for j in range(n_features):
        Xjs = data[indptr[j] : indptr[j + 1]]
        idx_nz = indices[indptr[j] : indptr[j + 1]]
        dbeta_old = dbeta[j, :].copy()
        dzj = dbeta[j, :] + Xjs @ ddual_var[idx_nz, :] / (L[j] * n_samples)
        scale = 1.0 / (1.0 + alpha_l2 / L[j])
        dbeta[j : j + 1, :] = scale * dzj
        dbeta[j : j + 1, j] -= scale * alpha[j] * np.sign(beta[j]) / L[j]
        ddual_var[idx_nz, :] -= np.outer(Xjs, (dbeta[j, :] - dbeta_old))
