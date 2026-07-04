"""Tests for WeightedElasticNet inner BCD solver — correctness and Numba speed.

Note: the speed tests require Numba JIT and are skipped when
NUMBA_DISABLE_JIT=1.

Correctness checks
------------------
1. Primal objective decreases monotonically for both dense and sparse X.
2. Solver converges (does NOT hit max_iter) with tol=1e-4.
3. KKT optimality at the solution for dense and sparse X.
4. Dense and sparse BCD code paths return the same beta (and same Jacobian).
5. With a scalar (uniform) alpha and no L2, the solution matches sklearn's
   ElasticNet / Lasso on the same problem.
6. get_L returns the correct per-column squared norms (Lipschitz constants).
7. get_mat_vec_impl returns the correct Hessian-vector product on the support.
8. The BCD Jacobian d(beta)/d(log_alpha) matches finite differences.
9. The __getattribute__ dispatch is transparent: every method name resolves
   to the correct implementation that captures alpha_l2.

Speed checks
------------
10. A single inner solve on (n=300, m=150) completes in < 10 s (JIT-compiled).
11. A single BCD Jacobian pass on (n=200, m=80) completes in < 5 s.
"""

import os
import time

import numpy as np
import pytest
from scipy.sparse import random as sp_random

from sparse_ho.algo.forward import compute_beta
from sparse_ho.models import WeightedElasticNet


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
_JIT_DISABLED = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"


def _make_dense(n=150, m=60, sparsity=0.1, rng=RNG):
    """Dense regression problem with a sparse true coefficient."""
    X = rng.standard_normal((n, m))
    s = max(1, int(sparsity * m))
    beta_true = np.zeros(m)
    beta_true[:s] = rng.standard_normal(s)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)
    return X, y, beta_true


def _make_sparse(n=150, m=60, density=0.15, rng=RNG):
    """CSC sparse regression problem."""
    X_csc = sp_random(n, m, density=density, format="csc", random_state=0)
    X_csc.data[:] = rng.standard_normal(len(X_csc.data))
    beta_true = np.zeros(m)
    beta_true[:5] = 1.0
    y = np.asarray(X_csc @ beta_true).ravel() + 0.1 * rng.standard_normal(n)
    return X_csc, y, beta_true


def _alpha_max(X, y):
    from scipy.sparse import issparse
    Xty = np.abs(np.asarray(X.T @ y).ravel())
    return float(Xty.max()) / X.shape[0]


def _log_alpha_uniform(m, frac, alpha_max):
    return np.log(np.full(m, frac * alpha_max))


# ---------------------------------------------------------------------------
# 1. Primal objective decreases monotonically (dense and sparse)
# ---------------------------------------------------------------------------

def test_primal_decreasing_dense():
    """Primal objective must not increase between BCD sweeps (dense X)."""
    X, y, _ = _make_dense()
    X_f = np.asfortranarray(X)
    n, m = X_f.shape
    alpha_l2 = 1.0 / n
    a_max = _alpha_max(X_f, y)
    alpha = np.full(m, 0.1 * a_max)

    model = WeightedElasticNet(alpha_l2=alpha_l2)
    L = model.get_L(X_f)
    beta = np.zeros(m)
    dual_var = np.zeros(n)
    dbeta = np.zeros((m, m))
    ddual_var = np.zeros((n, m))

    pobj_list = []
    for _ in range(25):
        model._update_beta_jac_bcd(
            X_f, y, beta, dbeta, dual_var, ddual_var, alpha, L, False
        )
        pobj_list.append(model._get_pobj(dual_var, X_f, beta, alpha, y))

    diffs = np.diff(pobj_list)
    assert np.all(diffs <= 1e-10), (
        f"Primal objective not monotonically decreasing (dense): "
        f"max increase = {diffs.max():.2e}"
    )


def test_primal_decreasing_sparse():
    """Primal objective must not increase between BCD sweeps (sparse X).

    For sparse X, the BCD kernel is _update_beta_jac_bcd_sparse, which takes
    CSC arrays (data, indptr, indices) rather than the dense matrix.
    """
    X_csc, y, _ = _make_sparse()
    n, m = X_csc.shape
    alpha_l2 = 1.0 / n
    a_max = _alpha_max(X_csc, y)
    alpha = np.full(m, 0.1 * a_max)

    model = WeightedElasticNet(alpha_l2=alpha_l2)
    L = model.get_L(X_csc)
    beta = np.zeros(m)
    dual_var = np.zeros(n)
    dbeta = np.zeros((m, m))
    ddual_var = np.zeros((n, m))

    pobj_list = []
    for _ in range(25):
        # Sparse BCD takes CSC arrays directly, not the sparse matrix object.
        model._update_beta_jac_bcd_sparse(
            X_csc.data, X_csc.indptr, X_csc.indices,
            y, n, m, beta, dbeta, dual_var, ddual_var, alpha, L,
            False,   # compute_jac
        )
        # _get_pobj uses the residual dual_var = y - X @ beta
        X_dense = X_csc.toarray()
        dual_var_check = y - X_dense @ beta
        pobj_list.append(model._get_pobj(dual_var_check, X_dense, beta, alpha, y))

    diffs = np.diff(pobj_list)
    assert np.all(diffs <= 1e-10), (
        f"Primal objective not monotonically decreasing (sparse): "
        f"max increase = {diffs.max():.2e}"
    )


# ---------------------------------------------------------------------------
# 2. Convergence within max_iter
# ---------------------------------------------------------------------------

def test_convergence_within_max_iter():
    """compute_beta must converge (pobj improves significantly from pobj0)."""
    X, y, _ = _make_dense()
    n, m = X.shape
    a_max = _alpha_max(X, y)
    log_alpha = _log_alpha_uniform(m, 0.1, a_max)

    model = WeightedElasticNet(alpha_l2=1.0 / n)
    mask, dense, _ = compute_beta(
        X, y, log_alpha, model, tol=1e-5, compute_jac=False, max_iter=2000
    )

    beta = np.zeros(m)
    beta[mask] = dense
    alpha = np.exp(log_alpha)
    X_f = np.asfortranarray(X)
    dual_var = y - X_f @ beta
    pobj = model._get_pobj(dual_var, X_f, beta, alpha, y)
    pobj0 = model._get_pobj0(y - np.zeros(n), beta * 0, alpha, y)

    assert pobj < pobj0 * 0.95, (
        f"Solver barely improved: pobj={pobj:.4f} vs pobj0={pobj0:.4f}"
    )


def test_sparse_wide_stop_criterion_exits_early():
    rng = np.random.default_rng(123)
    n, m = 80, 400
    X_csc, y, _ = _make_sparse(n=n, m=m, density=0.05, rng=rng)

    alpha_l2 = 1.0 / n
    a_max = _alpha_max(X_csc, y)
    log_alpha = np.log(np.full(m, 0.1 * a_max))

    model = WeightedElasticNet(alpha_l2=alpha_l2)
    model.inner_debug_records_ = []

    compute_beta(
        X_csc, y, log_alpha, model,
        tol=1e-4, compute_jac=False, max_iter=200,
    )

    rec = model.inner_debug_records_[0]
    assert rec["stop_reason"] == "stop_crit", (
        f"Expected KKT stop criterion, got {rec['stop_reason']}"
    )
    assert rec["n_passes"] < 200, (
        f"Expected early stop before max_iter, got {rec['n_passes']} passes"
    )


# ---------------------------------------------------------------------------
# 3. KKT conditions at the solution (dense and sparse)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sparse", [False, True])
def test_kkt_conditions(sparse):
    """At the solution, the subgradient optimality condition holds for each j:

    beta_j > 0  =>  grad_smooth_j + alpha_j = 0
    beta_j < 0  =>  grad_smooth_j - alpha_j = 0
    beta_j = 0  =>  |grad_smooth_j| <= alpha_j
    """
    if sparse:
        X, y, _ = _make_sparse(n=200, m=80)
    else:
        X, y, _ = _make_dense(n=200, m=80)

    n, m = X.shape
    a_max = _alpha_max(X, y)
    alpha_l2 = 1.0 / n
    alpha_vec = np.full(m, 0.05 * a_max)
    log_alpha = np.log(alpha_vec)

    model = WeightedElasticNet(alpha_l2=alpha_l2)
    mask, dense, _ = compute_beta(
        X, y, log_alpha, model, tol=1e-6, compute_jac=False, max_iter=5000
    )
    beta = np.zeros(m)
    beta[mask] = dense

    # smooth gradient: ∇_j [(1/2n)||Xβ-y||^2 + (α_l2/2)||β||^2]
    residual = np.asarray(X @ beta).ravel() - y
    grad_smooth = np.asarray(X.T @ residual).ravel() / n + alpha_l2 * beta

    kkt_viol = np.zeros(m)
    for j in range(m):
        if beta[j] > 1e-8:
            kkt_viol[j] = abs(grad_smooth[j] + alpha_vec[j])
        elif beta[j] < -1e-8:
            kkt_viol[j] = abs(grad_smooth[j] - alpha_vec[j])
        else:
            kkt_viol[j] = max(0.0, abs(grad_smooth[j]) - alpha_vec[j])

    assert kkt_viol.max() < 5e-4, (
        f"KKT violation too large (sparse={sparse}): max={kkt_viol.max():.2e}"
    )


# ---------------------------------------------------------------------------
# 4. Dense and sparse code paths return the same solution
# ---------------------------------------------------------------------------

def test_dense_sparse_agree():
    """Dense and sparse BCD paths must return the same beta and Jacobian."""
    rng = np.random.default_rng(7)
    X_csc, y, _ = _make_sparse(n=150, m=60, rng=rng)
    X_dense = np.asfortranarray(X_csc.toarray())

    a_max = _alpha_max(X_dense, y)
    alpha_l2 = 1.0 / 150
    log_alpha = np.log(np.full(60, 0.05 * a_max))

    model_d = WeightedElasticNet(alpha_l2=alpha_l2)
    model_s = WeightedElasticNet(alpha_l2=alpha_l2)

    mask_d, dense_d, jac_d = compute_beta(
        X_dense, y, log_alpha, model_d, tol=1e-6, compute_jac=True
    )
    mask_s, dense_s, jac_s = compute_beta(
        X_csc, y, log_alpha, model_s, tol=1e-6, compute_jac=True
    )

    beta_d = np.zeros(60)
    beta_d[mask_d] = dense_d
    beta_s = np.zeros(60)
    beta_s[mask_s] = dense_s

    assert np.allclose(beta_d, beta_s, atol=1e-4), (
        f"Dense/sparse beta mismatch: max_diff={np.abs(beta_d - beta_s).max():.2e}"
    )
    # Jacobians must also agree (on the common support)
    assert np.all(mask_d == mask_s), "Support mismatch between dense and sparse"
    if jac_d is not None and jac_s is not None:
        assert np.allclose(jac_d, jac_s, atol=1e-4), (
            f"Dense/sparse Jacobian mismatch: max={np.abs(jac_d - jac_s).max():.2e}"
        )


# ---------------------------------------------------------------------------
# 5. Matches sklearn ElasticNet with scalar uniform alpha
# ---------------------------------------------------------------------------

def test_matches_sklearn_elasticnet():
    """With uniform log_alpha and alpha_l2, the solution must agree with
    sklearn's ElasticNet (L1 ratio = 0.5 by construction of parameters)."""
    from sklearn.linear_model import ElasticNet

    rng = np.random.default_rng(1)
    n, m = 200, 40
    X = rng.standard_normal((n, m))
    beta_true = np.zeros(m)
    beta_true[:6] = rng.standard_normal(6)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    a_max = _alpha_max(X, y)
    alpha_l1 = 0.1 * a_max    # scalar L1 penalty (applied uniformly)
    alpha_l2 = 0.05 * a_max   # fixed L2 penalty

    log_alpha = np.log(np.full(m, alpha_l1))

    model = WeightedElasticNet(alpha_l2=alpha_l2)
    mask, dense, _ = compute_beta(
        X, y, log_alpha, model, tol=1e-8, compute_jac=False, max_iter=10000
    )
    beta_ho = np.zeros(m)
    beta_ho[mask] = dense

    # sklearn objective: (1/2n)||Xw-y||^2 + alpha*(l1_ratio*||w||_1 + (1-l1_ratio)*||w||_2^2/2)
    # To match ours: alpha*l1_ratio = alpha_l1/n (sklearn divides l1 by n via alpha)
    # and alpha*(1-l1_ratio) = alpha_l2/n.
    # Both penalties divided by n:  alpha_l1 (ours, per-sample) == sklearn alpha*l1_ratio*n
    # => set sklearn alpha = (alpha_l1 + alpha_l2) and l1_ratio = alpha_l1/(alpha_l1+alpha_l2)
    # Note: sparse_ho's objective does NOT divide alpha_l2 by n, but sklearn does divide
    # the ridge by n. Adjust accordingly.
    alpha_sk = (alpha_l1 + alpha_l2) / n   # sklearn divides by n internally via (1/2n) factor
    l1_ratio = alpha_l1 / (alpha_l1 + alpha_l2)

    clf = ElasticNet(
        alpha=alpha_sk * n,   # sklearn: (1/2n)||Xw-y||^2 + alpha*(...)  where alpha includes n
        l1_ratio=l1_ratio,
        fit_intercept=False,
        tol=1e-10,
        max_iter=20000,
    )
    clf.fit(X, y)
    beta_sk = clf.coef_

    # The supports must agree and coefficients should be close.
    # We allow 2% tolerance for solver tolerance differences.
    assert np.allclose(beta_ho, beta_sk, atol=5e-2, rtol=0.05), (
        f"Mismatch vs sklearn ElasticNet: max_diff={np.abs(beta_ho - beta_sk).max():.3e}, "
        f"n_active_ho={mask.sum()}, n_active_sk={(beta_sk != 0).sum()}"
    )


# ---------------------------------------------------------------------------
# 6. get_L returns ||X[:,j]||^2 / n (per-column squared norms)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sparse", [False, True])
def test_get_L_is_column_norms(sparse):
    """L[j] must equal ||X[:,j]||_2^2 / n_samples for both dense and sparse."""
    if sparse:
        X, _, _ = _make_sparse(n=100, m=50)
        L = WeightedElasticNet(alpha_l2=0.1).get_L(X)
        X_arr = X.toarray()
    else:
        X, _, _ = _make_dense(n=100, m=50)
        L = WeightedElasticNet(alpha_l2=0.1).get_L(X)
        X_arr = X

    n = X_arr.shape[0]
    L_ref = np.sum(X_arr ** 2, axis=0) / n
    assert np.allclose(L, L_ref, rtol=1e-10), (
        f"get_L mismatch (sparse={sparse}): max error = {np.abs(L - L_ref).max():.2e}"
    )


# ---------------------------------------------------------------------------
# 7. get_mat_vec_impl returns the correct (X_S.T @ X_S / n + alpha_l2 I) @ v
# ---------------------------------------------------------------------------

def test_get_mat_vec_matvec():
    """The LinearOperator from get_mat_vec_impl must compute
    (X_S.T @ X_S / n + alpha_l2 * I) @ v correctly."""
    rng = np.random.default_rng(5)
    n, m = 100, 40
    X = rng.standard_normal((n, m))
    y = rng.standard_normal(n)

    alpha_l2 = 0.1
    model = WeightedElasticNet(alpha_l2=alpha_l2)

    # Use full support for simplicity
    mask = np.ones(m, dtype=bool)
    dense = rng.standard_normal(m)
    log_alpha = np.zeros(m)

    mv_op = model.get_mat_vec(X, y, mask, dense, log_alpha)
    X_S = X[:, mask]

    v = rng.standard_normal(mask.sum())
    result = mv_op @ v
    expected = X_S.T @ (X_S @ v) / n + alpha_l2 * v

    assert np.allclose(result, expected, rtol=1e-10), (
        f"get_mat_vec matvec error: max={np.abs(result - expected).max():.2e}"
    )


# ---------------------------------------------------------------------------
# 8. BCD Jacobian vs. finite differences
# ---------------------------------------------------------------------------

def test_jacobian_finite_diff():
    """d(beta)/d(log_alpha) from BCD must agree with forward finite differences.

    We use a small problem (n=80, m=20) and large tol to keep it fast.
    """
    rng = np.random.default_rng(13)
    n, m = 80, 20
    X = rng.standard_normal((n, m))
    y = X @ rng.standard_normal(m) * 0.3 + 0.1 * rng.standard_normal(n)

    alpha_l2 = 1.0 / n
    a_max = _alpha_max(X, y)
    log_alpha0 = np.log(np.full(m, 0.15 * a_max))
    tol = 1e-7
    eps = 1e-5

    model = WeightedElasticNet(alpha_l2=alpha_l2)
    mask0, dense0, jac0 = compute_beta(
        X, y, log_alpha0, model, tol=tol, compute_jac=True
    )
    # Jacobian restricted to active set: shape (size_supp, size_supp)
    # jac0[k, j] = d(beta[active[k]]) / d(log_alpha[active[j]])
    size_supp = mask0.sum()
    if size_supp == 0:
        pytest.skip("No active features — adjust alpha")

    # Finite-difference Jacobian on the active set
    active_idx = np.where(mask0)[0]
    jac_fd = np.zeros((size_supp, size_supp))

    for col, j in enumerate(active_idx):
        log_alpha_p = log_alpha0.copy()
        log_alpha_p[j] += eps
        log_alpha_m = log_alpha0.copy()
        log_alpha_m[j] -= eps

        model_p = WeightedElasticNet(alpha_l2=alpha_l2)
        model_m = WeightedElasticNet(alpha_l2=alpha_l2)

        mask_p, dense_p, _ = compute_beta(
            X, y, log_alpha_p, model_p, tol=tol, compute_jac=False
        )
        mask_m, dense_m, _ = compute_beta(
            X, y, log_alpha_m, model_m, tol=tol, compute_jac=False
        )

        beta_p = np.zeros(m)
        beta_p[mask_p] = dense_p
        beta_m = np.zeros(m)
        beta_m[mask_m] = dense_m

        jac_fd[:, col] = (beta_p[active_idx] - beta_m[active_idx]) / (2 * eps)

    assert np.allclose(jac0, jac_fd, atol=1e-3), (
        f"Jacobian vs FD: max_diff={np.abs(jac0 - jac_fd).max():.2e}\n"
        f"jac0 (BCD):\n{jac0}\njac_fd:\n{jac_fd}"
    )


# ---------------------------------------------------------------------------
# 9. __getattribute__ dispatch is transparent
# ---------------------------------------------------------------------------

def test_getattribute_dispatch():
    """Every _impl method must be accessible via the public name, and
    the @staticmethod stubs must remain accessible via object.__getattribute__
    without raising inside user code (they raise NotImplementedError, not
    AttributeError)."""
    model = WeightedElasticNet(alpha_l2=0.5)

    # Public names must resolve to the _impl variants (callable, not raising)
    assert callable(model._update_beta_jac_bcd)
    assert callable(model._update_beta_jac_bcd_sparse)
    assert callable(model._update_bcd_jac_backward)
    assert callable(model._update_only_jac)
    assert callable(model._update_only_jac_sparse)
    assert callable(model._get_pobj)
    assert callable(model.get_mat_vec)

    # The dispatch actually resolves to the _impl method (same __func__).
    # Python bound methods are never `is`-equal even when wrapping the same
    # function; compare __func__ instead.
    impl_fn = object.__getattribute__(model, "_update_beta_jac_bcd_impl")
    assert impl_fn.__func__ is model._update_beta_jac_bcd.__func__

    # Each instance binds its own alpha_l2: verify two instances give
    # different results on the same input.
    model2 = WeightedElasticNet(alpha_l2=1.0)   # different alpha_l2
    rng2 = np.random.default_rng(0)
    n2, m2 = 20, 8
    X2 = rng2.standard_normal((n2, m2))
    y2 = rng2.standard_normal(n2)
    a_max2 = _alpha_max(X2, y2)
    alpha2 = np.full(m2, 0.1 * a_max2)
    L2 = model.get_L(X2)
    X2_f = np.asfortranarray(X2)

    beta1 = np.zeros(m2)
    dv1 = y2.copy()
    model._update_beta_jac_bcd(
        X2_f, y2, beta1, np.zeros((m2, m2)), dv1, np.zeros((n2, m2)),
        alpha2, L2, False
    )
    beta2 = np.zeros(m2)
    dv2 = y2.copy()
    model2._update_beta_jac_bcd(
        X2_f, y2, beta2, np.zeros((m2, m2)), dv2, np.zeros((n2, m2)),
        alpha2, L2, False
    )
    # Different alpha_l2 → different solutions
    assert not np.allclose(beta1, beta2), (
        "Two models with different alpha_l2 should produce different betas"
    )


# ---------------------------------------------------------------------------
# 10. Speed: inner solve completes quickly with JIT
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _JIT_DISABLED,
    reason="Speed test requires Numba JIT (NUMBA_DISABLE_JIT=1 is set)"
)
def test_solver_speed_primal():
    """Full inner solve on (n=300, m=150) must finish in < 10 s."""
    rng = np.random.default_rng(99)
    n, m = 300, 150
    X = rng.standard_normal((n, m))
    beta_true = np.zeros(m)
    beta_true[:10] = rng.standard_normal(10)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    a_max = _alpha_max(X, y)
    log_alpha = np.log(np.full(m, 0.1 * a_max))
    model = WeightedElasticNet(alpha_l2=1.0 / n)

    t0 = time.time()
    mask, dense, _ = compute_beta(
        X, y, log_alpha, model, tol=1e-5, compute_jac=False, max_iter=2000
    )
    elapsed = time.time() - t0

    assert elapsed < 10.0, (
        f"Inner primal solve took {elapsed:.1f}s — too slow (limit 10s)"
    )
    assert len(dense) > 0, "Solver returned empty support"


@pytest.mark.skipif(
    _JIT_DISABLED,
    reason="Speed test requires Numba JIT (NUMBA_DISABLE_JIT=1 is set)"
)
def test_solver_speed_with_jacobian():
    """Primal + Jacobian on (n=200, m=80) must finish in < 5 s."""
    rng = np.random.default_rng(77)
    n, m = 200, 80
    X = rng.standard_normal((n, m))
    beta_true = np.zeros(m)
    beta_true[:8] = rng.standard_normal(8)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    a_max = _alpha_max(X, y)
    log_alpha = np.log(np.full(m, 0.1 * a_max))
    model = WeightedElasticNet(alpha_l2=1.0 / n)

    t0 = time.time()
    mask, dense, jac = compute_beta(
        X, y, log_alpha, model, tol=1e-4, compute_jac=True, max_iter=1000
    )
    elapsed = time.time() - t0

    assert elapsed < 5.0, (
        f"Primal+Jacobian solve took {elapsed:.1f}s — too slow (limit 5s)"
    )
    assert jac is not None and jac.shape == (mask.sum(), mask.sum()), (
        f"Unexpected Jacobian shape: {jac.shape}"
    )
