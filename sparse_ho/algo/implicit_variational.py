"""Support-reduced implicit-variational hypergradient computation.

Implements Algorithm 1 (Support-Reduced Hypergradient Computation):
  Phase 1 — Lower-level resolution & support identification.
  Phase 2 — Selection policy Π, sign mask σ, and subsystem restriction to S.
  Phase 3 — Reduced adjoint system solve: H_S p_S = -[z*]_S.
  Phase 4 — Hypergradient reconstruction: h = ∇_x L + g_imp.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg

from sparse_ho.algo.forward import compute_beta

# Above this working-set size, the reduced adjoint H_S q = rhs is solved
# matrix-free (CG using Hessian-vector products) instead of forming the dense
# H_S explicitly. Forming H_S costs O(|S|) Hessian-vector products; CG costs
# O(#iterations) of them, which is far fewer once |S| is large (the regime of
# high-dimensional sparse designs). Small blocks stay on dense Cholesky.
_MATFREE_MIN_SUPPORT = 48


def _as_featurewise_alpha(log_alpha, n_features):
    """Return a feature-wise penalty vector from ``log_alpha``."""
    alpha = np.exp(log_alpha)
    if np.ndim(alpha) == 0:
        return np.ones(n_features) * float(alpha)
    return alpha


def _resolve_gamma(gamma, model, X):
    """Return a scalar step size γ ∈ (0, Λ_F^{-1})."""
    if gamma is None:
        L = np.asarray(model.get_L(X), dtype=float)
        if L.ndim == 0:
            L_max = float(L)
        else:
            positive_L = L[np.isfinite(L) & (L > 0.0)]
            L_max = float(np.max(positive_L)) if positive_L.size else 1.0
        if L_max <= 0.0:
            return 1.0
        return 1.0 / L_max

    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim == 0:
        gamma_scalar = float(gamma)
    else:
        if gamma.shape != (X.shape[1],):
            raise ValueError(
                "gamma must be scalar, None, or shape (n_features,), "
                f"got {gamma.shape}."
            )
        gamma_scalar = float(np.min(gamma))
        if not np.allclose(gamma, gamma_scalar):
            warnings.warn(
                "Support-reduced implicit differentiation requires a scalar "
                "gamma; using min(gamma) to preserve the SPD reduced system.",
                RuntimeWarning,
                stacklevel=2,
            )

    if gamma_scalar <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma_scalar}.")
    return gamma_scalar


def _resolve_lambdas(model, log_alpha, n_features):
    """Return feature-wise nonsmooth thresholds Ψ(x̄)."""
    if hasattr(model, "get_variational_lambdas"):
        lambdas = model.get_variational_lambdas(log_alpha, n_features)
    else:
        lambdas = _as_featurewise_alpha(log_alpha, n_features)
    lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.ndim == 0:
        return np.full(n_features, float(lambdas))
    if lambdas.shape != (n_features,):
        raise ValueError(
            "Variational lambdas must be scalar or shape (n_features,), "
            f"got {lambdas.shape} with n_features={n_features}."
        )
    return lambdas


def _make_full_beta(mask, dense, n_features):
    """Expand a support-restricted beta to full dimension."""
    beta = np.zeros(n_features, dtype=float)
    beta[mask] = dense
    return beta


def _partition_coordinates(
    v_bar,
    u_bar,
    *,
    biactive_tol_abs=0.0,
    biactive_tol_rel=1e-10,
    threshold_floor=1e-14,
    scale_floor=0.0,
):
    """Phase 1 step 3: Partition {0,...,n-1} into I+, I-, A, B+, B-.

    Uses the prox argument v̄ = ȳ - γ∇F(ȳ) and threshold ū = γΨ(x̄).

    Parameters
    ----------
    v_bar : ndarray, shape (n,)
    u_bar : ndarray, shape (n,)
    biactive_tol_abs : float, optional
        Absolute tolerance for detecting near-threshold coordinates.
    biactive_tol_rel : float, optional
        Relative tolerance for detecting near-threshold coordinates.
    threshold_floor : float, optional
        Minimal threshold below which coordinates are not considered biactive.
    scale_floor : float, optional
        Floor added to the relative scale max(|v̄_i|, ū_i). The default 0.0
        keeps the band scale-free: since (v̄, ū) are both γ-scaled while the
        exact kink condition |∇_iF(ȳ)| = Ψ(x̄)_i is γ-free, any positive
        floor silently turns the relative band into an absolute one whenever
        γΨ ≪ floor, classifying far-from-kink coordinates as biactive.
        Set to 1.0 to reproduce the legacy behaviour.

    Returns
    -------
    I_plus, I_minus, A, B_plus, B_minus : bool arrays of shape (n,)
    """
    v_bar = np.asarray(v_bar, dtype=float)
    u_bar = np.asarray(u_bar, dtype=float)
    abs_v = np.abs(v_bar)
    gap = np.abs(abs_v - u_bar)
    scale = np.maximum(abs_v, u_bar)
    if scale_floor > 0.0:
        scale = np.maximum(scale, float(scale_floor))
    tol = np.maximum(float(biactive_tol_abs), float(biactive_tol_rel) * scale)

    biactive = gap <= tol
    biactive &= u_bar > float(threshold_floor)
    non_biactive = ~biactive

    strict_active = non_biactive & (abs_v > u_bar)
    A = non_biactive & (abs_v <= u_bar)

    I_plus = strict_active & (v_bar >= 0)
    I_minus = strict_active & (v_bar < 0)
    B_plus = biactive & (v_bar >= 0)
    B_minus = biactive & (v_bar < 0)

    return I_plus, I_minus, A, B_plus, B_minus


def _build_reduced_hessian_block(selected_support, gamma, model, X, y, beta):
    """Form H_S = γ [∇²F(ȳ)]_{S,S}."""
    support = np.flatnonzero(selected_support)
    reduced_n = support.size
    if reduced_n == 0:
        return np.zeros((0, 0), dtype=float)

    n_features = X.shape[1]
    hessian_block = np.empty((reduced_n, reduced_n), dtype=float)
    for j, feat_idx in enumerate(support):
        basis = np.zeros(n_features, dtype=float)
        basis[feat_idx] = 1.0
        hess_col = model.get_hess_smooth(X, y, beta, basis)
        hessian_block[:, j] = np.asarray(hess_col, dtype=float)[selected_support]

    hessian_block = 0.5 * (hessian_block + hessian_block.T)
    return float(gamma) * hessian_block


def _hs_matvec_operator(S, gamma, model, X, y, beta, epsilon):
    """LinearOperator for H_S = γ[∇²F(ȳ)]_{S,S} (+εI) without forming it.

    Restricts the Hessian-vector product to the support submatrix X_S:
        [∇²F]_{S,S} p = (1/n) X_S^T (w ⊙ (X_S p)) + α_ℓ₂ p,
    where the diagonal IRLS weight ``w`` is supplied by the *model* via
    ``model.hessian_diag_weights`` (least-squares ⇒ w≡1; logistic ⇒
    w = σ(yXβ)(1−σ(yXβ))). Dispatching to the model keeps this matrix-free
    operator consistent with the model's own ∇²F — the dense path
    :func:`_build_reduced_hessian_block` already does this through
    ``get_hess_smooth`` — so the reduced adjoint is correct for every loss, not
    only the logistic one. Costs O(nnz(X_S)) per product, the same locality as
    Sparse-HO's support-restricted adjoint.
    """
    from scipy.sparse import issparse

    support = np.flatnonzero(S)
    nS = support.size
    n_features = X.shape[1]
    gamma = float(gamma)
    eps = float(epsilon)
    n_samples = X.shape[0]
    alpha_l2 = float(getattr(model, "alpha_l2", 0.0))

    X_S = X[:, support]
    if issparse(X_S):
        X_S = X_S.tocsr()
    beta_S = np.asarray(beta, dtype=float)[support]
    w = np.asarray(model.hessian_diag_weights(X_S, y, beta_S), dtype=float).ravel()

    def _matvec(p):
        Xp = np.asarray(X_S @ p).ravel()
        Hv = np.asarray(X_S.T @ (w * Xp)).ravel() / n_samples + alpha_l2 * p
        return gamma * Hv + eps * p

    return LinearOperator((nS, nS), matvec=_matvec, dtype=float), support


def _solve_HS(S, rhs, gamma, model, X, y, beta, *, epsilon, tol, maxiter, x0=None):
    """Solve H_S q = rhs on the working set S.

    Matrix-free CG for large |S| (avoids the O(|S|) Hessian-vector products
    needed to form H_S); dense Cholesky for small |S|. Returns (q_full, info,
    method) with q_full zero outside S.
    """
    n_features = X.shape[1]
    q = np.zeros(n_features, dtype=float)
    support = np.flatnonzero(S)
    nS = support.size
    if nS == 0:
        return q, 0, "none"

    if nS >= _MATFREE_MIN_SUPPORT and hasattr(model, "hessian_diag_weights"):
        A, support = _hs_matvec_operator(S, gamma, model, X, y, beta, epsilon)
        x0S = x0[support] if (x0 is not None and np.shape(x0) == (n_features,)) else None
        qS, info = cg(A, rhs, x0=x0S, rtol=tol, maxiter=maxiter)
        if info == 0:
            q[support] = qS
            return q, int(info), "cg_matfree"
        # fall through to the explicit path if CG did not converge

    H_S = _build_reduced_hessian_block(S, gamma, model, X, y, beta)
    if epsilon:
        H_S.flat[:: H_S.shape[0] + 1] += float(epsilon)
    try:
        qS = cho_solve(cho_factor(H_S, lower=True, check_finite=False), rhs,
                       check_finite=False)
        method, info = "cholesky", 0
    except LinAlgError:
        x0S = x0[support] if (x0 is not None and np.shape(x0) == (n_features,)) else None
        qS, info = cg(H_S, rhs, x0=x0S, rtol=tol, maxiter=maxiter)
        method = "cg"
    q[support] = qS
    return q, int(info), method


def _solve_reduced_adjoint(
    S,
    z_star,
    gamma,
    model,
    X,
    y,
    beta,
    *,
    sol_lin_sys=None,
    tol_lin_sys=1e-6,
    max_iter_lin_sys=100,
    epsilon=1e-8,
):
    """Phase 3: Solve H_S p_S = -[z*]_S.

    Parameters
    ----------
    S : bool array, shape (n,)
        Working set mask.
    z_star : ndarray, shape (n,)
        ∇_y L(x, ȳ).

    Returns
    -------
    p : ndarray, shape (n,)
        Full-dimension adjoint (zero outside S).
    info : int
    method : str
    """
    n_features = X.shape[1]
    if not np.any(S):
        return np.zeros(n_features, dtype=float), 0, "none"

    rhs = -z_star[S]
    p, info, method = _solve_HS(
        S, rhs, gamma, model, X, y, beta,
        epsilon=epsilon, tol=tol_lin_sys, maxiter=max_iter_lin_sys,
        x0=sol_lin_sys)
    if info != 0:
        warnings.warn(
            f"Reduced adjoint solve did not converge, info={info}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return p, int(info), method


def _compute_variational_hypergrad(model, alpha, sigma, p, beta, gamma):
    """Phase 4: Compute g_imp and return h = ∇_x L + g_imp.

    g_imp[S] = γ [JΨ(x)]_{S,S} (σ ⊙ p_S)

    For the default (scalar alpha, Lasso-type) case:
        h = α · γ · σᵀp     (∇_x L = 0 in standard held-out setup)

    For the vector-alpha case:
        h = α ⊙ γ · σ ⊙ p  (element-wise, summed by the outer optimizer)
    """
    if hasattr(model, "get_variational_hypergrad"):
        return model.get_variational_hypergrad(alpha, sigma, p, beta, gamma)
    if np.ndim(alpha) == 0:
        return alpha * float(np.dot(sigma, gamma * p))
    return alpha * (sigma * gamma * p)


def include_all_biactive(I_plus, I_minus, A, B_plus, B_minus):
    """Biactive selection policy: include every biactive coordinate in S.

    Assigns sign +1 to B+ coordinates and -1 to B- coordinates, selecting
    the prox-boundary branch of the subdifferential consistent with the
    proximal argument v_bar.  This is a valid element of the Mordukhovich
    subdifferential and requires no evaluation of the outer gradient (single-
    pass).

    This is the default policy for :class:`ImplicitVariational`.

    Parameters
    ----------
    I_plus, I_minus, A, B_plus, B_minus : bool arrays, shape (n_features,)
        Support partition from :func:`_partition_coordinates`.

    Returns
    -------
    M_B_plus, M_B_minus : bool arrays, shape (n_features,)
    """
    return B_plus.copy(), B_minus.copy()


def select_biactive_by_zstar_sign(
    I_plus,
    I_minus,
    A,
    B_plus,
    B_minus,
    *,
    z_star,
):
    """Select biactives using the sign of ``z_star``.

    This is the simplest descent-aligned heuristic consistent with the draft's
    sign convention:
      - select ``B+`` coordinates when ``z_star < 0``
      - select ``B-`` coordinates when ``z_star > 0``
    """
    if z_star is None:
        raise ValueError("`z_star` is required for z_star-sign selection.")
    z_star = np.asarray(z_star, dtype=float)
    return (B_plus & (z_star < 0.0), B_minus & (z_star > 0.0))


def make_select_biactive_topM(M):
    """Factory: descent-aligned biactive selection capped at M coordinates.

    Returns a policy function that:
      1. Filters to descent-aligned biactives (B+ where z*<0, B- where z*>0).
      2. Ranks by |z*_j| descending.
      3. Keeps at most M coordinates total.

    This bounds |S| ≤ |I| + M, keeping H_S well-conditioned when |B| >> n_tr.
    A natural choice is M = K (true sparsity level), giving |S| ≤ 2K << n_tr.

    Parameters
    ----------
    M : int
        Maximum number of biactive coordinates to include in S.

    Returns
    -------
    policy : callable
        A policy function compatible with :func:`compute_beta_grad_implicit_variational`.
    """
    def _policy(I_plus, I_minus, A, B_plus, B_minus, *, z_star):
        if z_star is None:
            raise ValueError("`z_star` is required for top-M descent-aligned policy.")
        z_star = np.asarray(z_star, dtype=float)

        # Descent-aligned candidates
        cand_plus = np.where(B_plus & (z_star < 0.0))[0]
        cand_minus = np.where(B_minus & (z_star > 0.0))[0]
        candidates = np.concatenate([cand_plus, cand_minus])

        n_features = len(z_star)
        M_B_plus = np.zeros(n_features, dtype=bool)
        M_B_minus = np.zeros(n_features, dtype=bool)

        if len(candidates) == 0:
            return M_B_plus, M_B_minus

        # Rank by |z*_j| descending, keep top M
        scores = np.abs(z_star[candidates])
        order = np.argsort(scores)[::-1]
        selected = candidates[order[:M]]

        M_B_plus[selected[np.isin(selected, cand_plus)]] = True
        M_B_minus[selected[np.isin(selected, cand_minus)]] = True
        return M_B_plus, M_B_minus

    _policy.__name__ = f"select_biactive_topM_{M}"
    return _policy


def select_biactive_self_consistent(
    I_plus, I_minus, A, B_plus, B_minus, *, z_star, gamma, model, X, y, beta
):
    """Self-consistent biactive selection policy (SC).

    Fixed-point iteration that retains only biactive coordinates whose
    selected sign agrees with the adjoint solution they produce.  Terminates
    in at most |B| steps since biactive indices are only removed, never added.

    Initialization uses the descent-aligned rule (B+ where z*<0, B- where
    z*>0).  When ∇²F(ȳ) is diagonal, D^(0) is already the fixed point and
    no refinement step is needed.

    Parameters
    ----------
    I_plus, I_minus, A, B_plus, B_minus : bool arrays, shape (n_features,)
        Coordinate partition from :func:`_partition_coordinates`.
    z_star : ndarray, shape (n_features,)
        ∇_y L(x̄, ȳ).
    gamma : float
        FBE step size.
    model, X, y, beta :
        Lower-level context for building the reduced Hessian H_S.

    Returns
    -------
    M_B_plus, M_B_minus : bool arrays, shape (n_features,)
        Selected biactive subsets (subsets of B_plus and B_minus).
    """
    # Initialization (step i): descent-aligned seed
    M_B_plus = (B_plus & (z_star < 0.0)).copy()
    M_B_minus = (B_minus & (z_star > 0.0)).copy()
    return _prune_sign_consistent(
        M_B_plus, M_B_minus, I_plus, I_minus,
        z_star=z_star, gamma=gamma, model=model, X=X, y=y, beta=beta)


def _prune_sign_consistent(
    M_B_plus, M_B_minus, I_plus, I_minus, *, z_star, gamma, model, X, y, beta
):
    """Definition-2 pruning loop from a given biactive seed.

    Repeatedly solves the reduced adjoint on S = I ∪ selected-biactive and
    removes every selected biactive index with σ_i q_i ≤ 0, until the
    sign-consistency fixed point is reached. Only removes, never adds, so it
    terminates in at most (seed size) iterations.
    """
    n_features = len(z_star)
    # No biactive coordinate is selected ⇒ nothing to prune, and the sign test
    # never looks at the strict-active block. Skip the adjoint solve entirely;
    # this is the common case on well-regularized sparse designs (B empty) and
    # avoids a redundant reduced-Hessian build per hypergradient.
    if not np.any(M_B_plus | M_B_minus):
        return M_B_plus, M_B_minus

    max_iters = int(np.sum(M_B_plus | M_B_minus)) + 1

    for _ in range(max_iters):
        S = I_plus | I_minus | M_B_plus | M_B_minus
        if not np.any(S):
            break

        # Build sign vector σ for current selection
        sigma = np.zeros(n_features, dtype=float)
        sigma[I_plus | M_B_plus] = +1.0
        sigma[I_minus | M_B_minus] = -1.0

        # Solve H_{S^(t)} q^(t) = -[z*]_{S^(t)}
        rhs = -z_star[S]
        q, _, _ = _solve_HS(S, rhs, gamma, model, X, y, beta,
                            epsilon=1e-8, tol=1e-6, maxiter=100)

        # Remove biactive indices where sign consistency fails: σ_i * q_i ≤ 0
        biactive_in_S = M_B_plus | M_B_minus
        inconsistent = biactive_in_S & (sigma * q <= 0.0)

        if not np.any(inconsistent):
            break  # Fixed point reached

        M_B_plus = M_B_plus & ~inconsistent
        M_B_minus = M_B_minus & ~inconsistent

    return M_B_plus, M_B_minus


def make_select_biactive_self_consistent_topM(M):
    """Factory: sign-consistent selection from a top-M capped seed.

    Same fixed-point pruning as :func:`select_biactive_self_consistent`, but
    the seed admits at most M descent-aligned biactive coordinates, ranked by
    |z*_j| descending. This bounds |S| ≤ |I| + M, keeping the reduced adjoint
    block well-conditioned in dense-degenerate regimes (|B| ~ p) and keeping
    the selection within the per-coordinate scope of the descent guarantee.

    Parameters
    ----------
    M : int
        Maximum number of biactive coordinates admitted to the seed.

    Returns
    -------
    policy : callable
        Policy compatible with :func:`compute_beta_grad_implicit_variational`.
    """
    _topM_seed = make_select_biactive_topM(M)

    def _policy(
        I_plus, I_minus, A, B_plus, B_minus,
        *, z_star, gamma, model, X, y, beta
    ):
        M_B_plus, M_B_minus = _topM_seed(
            I_plus, I_minus, A, B_plus, B_minus, z_star=z_star)
        return _prune_sign_consistent(
            M_B_plus, M_B_minus, I_plus, I_minus,
            z_star=z_star, gamma=gamma, model=model, X=X, y=y, beta=beta)

    _policy.__name__ = f"select_biactive_self_consistent_top{M}"
    return _policy


def _call_policy(
    policy, I_plus, I_minus, A, B_plus, B_minus, *, z_star, **extra_kwargs
):
    """Call a policy, forwarding only the keyword arguments it accepts.

    Supports both legacy policies (positional + optional z_star) and
    context-aware policies that additionally accept gamma, model, X, y, beta.
    """
    params = inspect.signature(policy).parameters
    accepts_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    all_kwargs = {"z_star": z_star, **extra_kwargs}
    if accepts_varkw:
        return policy(I_plus, I_minus, A, B_plus, B_minus, **all_kwargs)
    kwargs_to_pass = {k: v for k, v in all_kwargs.items() if k in params}
    return policy(I_plus, I_minus, A, B_plus, B_minus, **kwargs_to_pass)


class ImplicitVariational:
    """Support-reduced implicit differentiation via Algorithm 1.

    Parameters
    ----------
    max_iter : int (default=100)
        Maximum number of inner solver iterations.
    max_iter_lin_sys : int (default=100)
        Maximum CG iterations for the reduced adjoint solve (fallback).
    tol_lin_sys : float (default=1e-6)
        CG tolerance for the reduced adjoint solve.
    policy : callable or None
        Biactive selection policy Π.  Called as::

            M_B_plus, M_B_minus = policy(I_plus, I_minus, A, B_plus, B_minus)

        Policies may optionally accept ``z_star=...`` as a keyword argument
        for one-pass descent-aware selection.

        Must return two boolean arrays of shape ``(n_features,)`` that are
        subsets of ``B_plus`` and ``B_minus`` respectively.
        Defaults to : no biactive index selected.
        When ``None``, all biactive coordinates are excluded from S
        (reproduces the Sparse-HO baseline behaviour).
    biactive_tol_abs : float (default=0.0)
        Absolute tolerance used in biactive detection.
    biactive_tol_rel : float (default=1e-10)
        Relative tolerance used in biactive detection.
    biactive_scale_floor : float (default=0.0)
        Floor for the relative detection scale max(|v̄_i|, ū_i). Keep at 0.0
        for a scale-free (γ-independent) band; 1.0 reproduces the legacy
        max(|v̄_i|, ū_i, 1) behaviour.
    epsilon : float (default=1e-8)
        Tikhonov regularization added to the diagonal of the reduced Hessian
        H_S before the adjoint solve.  Increase this (e.g. to 1e-2) when
        |S| >> n_tr (rank-deficient H_S) to keep the solve well-conditioned.
    """

    def __init__(
        self,
        max_iter=100,
        max_iter_lin_sys=100,
        tol_lin_sys=1e-6,
        policy=None,
        biactive_tol_abs=0.0,
        biactive_tol_rel=1e-10,
        biactive_scale_floor=0.0,
        epsilon=1e-8,
    ):
        self.max_iter = max_iter
        self.max_iter_lin_sys = max_iter_lin_sys
        self.tol_lin_sys = tol_lin_sys
        self.policy = policy
        self.biactive_tol_abs = float(biactive_tol_abs)
        self.biactive_tol_rel = float(biactive_tol_rel)
        self.biactive_scale_floor = float(biactive_scale_floor)
        self.epsilon = float(epsilon)
        self.last_run_info = None

    def compute_beta_grad(
        self,
        X,
        y,
        log_alpha,
        model,
        get_grad_outer,
        mask0=None,
        dense0=None,
        quantity_to_warm_start=None,
        max_iter=1000,
        tol=1e-3,
        full_jac_v=False,
        gamma=None,
        policy=None,
        biactive_tol_abs=None,
        biactive_tol_rel=None,
        return_sets=False,
    ):
        """Compute β and the hypergradient via support-reduced adjoints.

        Parameters
        ----------
        policy : callable or None
            Per-call override for the instance policy.
        """
        if policy is None:
            policy = self.policy
        if biactive_tol_abs is None:
            biactive_tol_abs = self.biactive_tol_abs
        if biactive_tol_rel is None:
            biactive_tol_rel = self.biactive_tol_rel

        mask, dense, jac_v, sol_lin_sys, sets = compute_beta_grad_implicit_variational(
            X,
            y,
            log_alpha,
            get_grad_outer,
            mask0=mask0,
            dense0=dense0,
            max_iter=max_iter,
            tol=tol,
            sol_lin_sys=quantity_to_warm_start,
            tol_lin_sys=self.tol_lin_sys,
            max_iter_lin_sys=self.max_iter_lin_sys,
            model=model,
            gamma=gamma,
            policy=policy,
            biactive_tol_abs=biactive_tol_abs,
            biactive_tol_rel=biactive_tol_rel,
            biactive_scale_floor=self.biactive_scale_floor,
            epsilon=self.epsilon,
            return_sets=True,
        )
        self.last_run_info = {
            "strict_active_size": int(np.sum(sets["I_plus"] | sets["I_minus"])),
            "biactive_size": int(np.sum(sets["B_plus"] | sets["B_minus"])),
            "selected_biactive_size": int(
                np.sum(sets["M_B_plus"] | sets["M_B_minus"])
            ),
            "selected_support_size": int(np.sum(sets["S"])),
            "biactive_mask": (sets["B_plus"] | sets["B_minus"]).copy(),
            "selected_biactive_mask": (sets["M_B_plus"] | sets["M_B_minus"]).copy(),
            "selected_support_mask": sets["S"].copy(),
            "gmres_info": int(sets["gmres_info"]),
            "solve_method": sets["solve_method"],
        }

        if full_jac_v and np.ndim(jac_v) > 0:
            if jac_v.shape[0] != X.shape[1]:
                jac_v = model.get_full_jac_v(mask, jac_v, X.shape[1])

        if return_sets:
            return mask, dense, jac_v, sol_lin_sys, sets
        return mask, dense, jac_v, sol_lin_sys


def compute_beta_grad_implicit_variational(
    X,
    y,
    log_alpha,
    get_grad_outer,
    mask0=None,
    dense0=None,
    tol=1e-3,
    model=None,
    max_iter=1000,
    sol_lin_sys=None,
    tol_lin_sys=1e-6,
    max_iter_lin_sys=100,
    gamma=None,
    epsilon=1e-8,
    policy=None,
    biactive_tol_abs=0.0,
    biactive_tol_rel=1e-10,
    biactive_scale_floor=0.0,
    return_sets=False,
):
    """Algorithm 1: Support-Reduced Hypergradient Computation.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    log_alpha : float or ndarray
        Log hyperparameter(s): x̄.
    get_grad_outer : callable
        Returns z* = ∇_y L(x, ȳ) given (mask, dense_beta).
    policy : callable or None
        Biactive selection policy Π:
            M_B_plus, M_B_minus = policy(I_plus, I_minus, A, B_plus, B_minus)
        None means empty policy (biactives excluded from S).
    biactive_tol_abs : float, optional
        Absolute tolerance used in biactive detection.
    biactive_tol_rel : float, optional
        Relative tolerance used in biactive detection.

    Returns
    -------
    mask : bool array, shape (n_features,)
    dense : ndarray
    hypergrad : float or ndarray
    p : ndarray, shape (n_features,)
    sets : dict or None
    """
    # ---------------------------------------------------------------
    # Phase 1: Lower-Level Resolution & Support Identification
    # ---------------------------------------------------------------
    gamma = _resolve_gamma(gamma, model, X)
    alpha = np.exp(log_alpha)

    # Step 1: Compute ȳ ≈ S(x̄)
    mask, dense, _ = compute_beta(
        X,
        y,
        log_alpha,
        mask0=mask0,
        dense0=dense0,
        tol=tol,
        max_iter=max_iter,
        compute_jac=False,
        model=model,
    )
    n_features = X.shape[1]
    lambdas = _resolve_lambdas(model, log_alpha, n_features)
    beta = _make_full_beta(mask, dense, n_features)

    if hasattr(model, "set_variational_alpha"):
        model.set_variational_alpha(alpha)

    # Step 2: ū = γ Ψ(x̄),  v̄ = ȳ - γ ∇F(ȳ)
    grad_F = model.get_grad_smooth(X, y, beta)
    v_bar = beta - gamma * grad_F
    u_bar = gamma * lambdas

    # Step 3: Partition {1,...,n} into I+, I-, A, B+, B-
    I_plus, I_minus, A, B_plus, B_minus = _partition_coordinates(
        v_bar,
        u_bar,
        biactive_tol_abs=biactive_tol_abs,
        biactive_tol_rel=biactive_tol_rel,
        scale_floor=biactive_scale_floor,
    )

    # ---------------------------------------------------------------
    # Phase 2: Selection Policy, Sign Mask & Subsystem Restriction
    # ---------------------------------------------------------------

    # Steps 8-9: z* = ∇_y L(x, ȳ),  available before policy selection
    mask_full = np.ones(n_features, dtype=bool)
    z_star = np.asarray(get_grad_outer(mask_full, beta), dtype=float)

    # Step 4: Apply policy Π → binary selection masks M_B+, M_B-
    if policy is not None:
        M_B_plus, M_B_minus = _call_policy(
            policy, I_plus, I_minus, A, B_plus, B_minus,
            z_star=z_star, gamma=gamma, model=model, X=X, y=y, beta=beta,
        )
        M_B_plus = np.asarray(M_B_plus, dtype=bool) & B_plus
        M_B_minus = np.asarray(M_B_minus, dtype=bool) & B_minus
    else:
        M_B_plus = np.zeros(n_features, dtype=bool)
        M_B_minus = np.zeros(n_features, dtype=bool)

    # Step 5: S = I+ ∪ I- ∪ supp(M_B+) ∪ supp(M_B-)
    S = I_plus | I_minus | M_B_plus | M_B_minus

    # Step 6: Assert S ∩ A = ∅
    if np.any(S & A):
        raise AssertionError(
            "Working set S must be disjoint from the strictly inactive set A."
        )

    # Step 7: Build sign vector σ ∈ {±1}^n (zero outside S)
    sigma = np.zeros(n_features, dtype=float)
    sigma[I_plus | M_B_plus] = +1.0
    sigma[I_minus | M_B_minus] = -1.0

    # Steps 10-11: H_S = γ [∇²F(ȳ)]_{S,S},  solve H_S p_S = -[z*]_S
    p, info, solve_method = _solve_reduced_adjoint(
        S,
        z_star,
        gamma,
        model,
        X,
        y,
        beta,
        sol_lin_sys=sol_lin_sys,
        tol_lin_sys=tol_lin_sys,
        max_iter_lin_sys=max_iter_lin_sys,
        epsilon=epsilon,
    )

    # ---------------------------------------------------------------
    # Phase 4: Hypergradient Reconstruction
    # ---------------------------------------------------------------

    # Steps 12-14: g_imp[S] = γ [JΨ(x)]_{S,S} (σ ⊙ p_S),  h = ∇_x L + g_imp
    hypergrad = _compute_variational_hypergrad(model, alpha, sigma, p, beta, gamma)

    sets = None
    if return_sets:
        sets = {
            "I_plus": I_plus,
            "I_minus": I_minus,
            "A": A,
            "B_plus": B_plus,
            "B_minus": B_minus,
            "M_B_plus": M_B_plus,
            "M_B_minus": M_B_minus,
            "S": S,
            "sigma": sigma,
            "v_bar": v_bar,
            "u_bar": u_bar,
            "z_star": z_star,
            "p": p,
            "gmres_info": info,
            "solve_method": solve_method,
        }

    return mask, dense, hypergrad, p, sets
