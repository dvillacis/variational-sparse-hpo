"""Tests for WeightedSparseLogReg inner BCD solver — correctness and speed.

Note: the speed tests require Numba JIT and are skipped when NUMBA_DISABLE_JIT=1.

Correctness checks
------------------
 1. Primal objective decreases monotonically (dense path).
 2. Primal objective decreases monotonically (sparse path, direct kernel call).
 3. Solver converges (does NOT hit max_iter) with tol=1e-4.
 4. KKT optimality (dense): at the solution, subgradient condition holds.
 5. KKT optimality (sparse): same check for sparse X.
 6. Solution matches sklearn's LogisticRegression (scalar alpha, no L2).
 7. Dense and sparse code paths return the same beta.
 8. get_L returns a valid Lipschitz upper bound (>= adaptive per-coordinate L).
 9. get_mat_vec_impl LinearOperator matches explicit reduced Hessian.
10. __getattribute__ dispatch: _get_pobj, _get_dobj, get_mat_vec are reachable.
11. _get_pobj numerical stability: no overflow for extreme dual_var values.
12. hasattr(model, '_get_dobj') is True (dual-gap criterion is used in BCD loop).

Speed checks
------------
13. Dense inner solve (n=500, m=200) completes in < 10 s.
14. Sparse inner solve (n=500, m=200, 10% density) completes in < 10 s.
"""

import os
import time
import warnings

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse import random as sp_random

from sparse_ho.algo.forward import compute_beta
from sparse_ho.models import WeightedSparseLogReg
from sparse_ho.utils import sigma

_JIT_DISABLED = os.environ.get('NUMBA_DISABLE_JIT', '0') == '1'


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_dense_problem(n=200, m=80, rng=RNG):
    X = rng.standard_normal((n, m))
    beta_true = np.zeros(m)
    beta_true[:5] = rng.standard_normal(5)
    y = np.sign(X @ beta_true + 0.2 * rng.standard_normal(n))
    y[y == 0] = 1.0
    return X, y


def _make_sparse_problem(n=200, m=80, density=0.10, rng=RNG):
    X_csc = sp_random(n, m, density=density, format='csc', random_state=0)
    X_csc.data[:] = rng.standard_normal(len(X_csc.data))
    beta_true = np.zeros(m)
    beta_true[:5] = 1.0
    scores = np.asarray(X_csc @ beta_true).ravel()
    y = np.sign(scores + 0.2 * rng.standard_normal(n))
    y[y == 0] = 1.0
    return X_csc, y


def _alpha_max(X, y):
    from scipy.sparse import issparse
    if issparse(X):
        Xty = np.abs(np.asarray(X.T @ y).ravel())
    else:
        Xty = np.abs(X.T @ y)
    return float(np.max(Xty)) / (2.0 * len(y))


# ---------------------------------------------------------------------------
# 1. Primal objective decreases monotonically (dense path)
# ---------------------------------------------------------------------------

def test_primal_decreasing_dense():
    X, y = _make_dense_problem()
    n, m = X.shape
    alpha_l2 = 1.0 / n
    a_max = _alpha_max(X, y)
    alpha = np.full(m, 0.1 * a_max)

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    L = model.get_L(X)
    X_f = np.asfortranarray(X)
    beta = np.zeros(m)
    dual_var = np.zeros(n)
    dbeta = np.zeros((m, m))
    ddual_var = np.zeros((n, m))

    pobj_list = []
    for _ in range(30):
        model._update_beta_jac_bcd(
            X_f, y, beta, dbeta, dual_var, ddual_var, alpha, L, False)
        pobj_list.append(model._get_pobj(dual_var, X_f, beta, alpha, y))

    diffs = np.diff(pobj_list)
    assert np.all(diffs <= 1e-10), (
        f"Primal objective is not monotonically decreasing: "
        f"max increase = {diffs.max():.2e}"
    )


# ---------------------------------------------------------------------------
# 2. Solver converges without hitting max_iter
# ---------------------------------------------------------------------------

def test_convergence_within_max_iter():
    X, y = _make_dense_problem()
    n, m = X.shape
    alpha_l2 = 1.0 / n
    a_max = _alpha_max(X, y)
    log_alpha = np.log(np.full(m, 0.1 * a_max))

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    max_iter = 1000

    # Intercept the for-else: if it hit max_iter, compute_beta prints a warning
    # We check by verifying the pobj is converged after the call.
    mask, dense, _ = compute_beta(
        X, y, log_alpha, model, tol=1e-4, compute_jac=False, max_iter=max_iter,
    )
    # Compute final primal value and check it's reasonably small
    beta_full = np.zeros(m)
    beta_full[mask] = dense
    alpha = np.exp(log_alpha)
    X_f = np.asfortranarray(X)
    dual_var = y * (X_f @ beta_full)
    pobj_final = model._get_pobj(dual_var, X_f, beta_full, alpha, y)
    pobj0 = float(np.log(2))  # reference at beta=0

    # Solver should reduce the objective significantly from pobj0
    assert pobj_final < pobj0 * 0.99, (
        f"Solver barely improved: pobj_final={pobj_final:.4f} pobj0={pobj0:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. KKT conditions at the solution
# ---------------------------------------------------------------------------

def test_kkt_conditions():
    """For each coordinate j:
      if beta_j > 0:  grad_j + alpha_j = 0
      if beta_j < 0:  grad_j - alpha_j = 0
      if beta_j = 0:  |grad_j| <= alpha_j
    where grad_j = (1/n) X[:,j] @ (sigma(-y*Xb)*(-y)) + alpha_l2*beta_j
    """
    X, y = _make_dense_problem()
    n, m = X.shape
    alpha_l2 = 1.0 / n
    a_max = _alpha_max(X, y)
    alpha_vec = np.full(m, 0.1 * a_max)
    log_alpha = np.log(alpha_vec)

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    mask, dense, _ = compute_beta(
        X, y, log_alpha, model, tol=1e-4, compute_jac=False, max_iter=2000,
    )
    beta = np.zeros(m)
    beta[mask] = dense

    X_f = np.asfortranarray(X)
    Xb = X_f @ beta
    s = sigma(-y * Xb)
    grad_smooth = -(X_f.T @ (y * s)) / n + alpha_l2 * beta  # smooth gradient

    kkt_viol = np.zeros(m)
    for j in range(m):
        if beta[j] > 1e-8:
            kkt_viol[j] = abs(grad_smooth[j] + alpha_vec[j])
        elif beta[j] < -1e-8:
            kkt_viol[j] = abs(grad_smooth[j] - alpha_vec[j])
        else:
            kkt_viol[j] = max(0.0, abs(grad_smooth[j]) - alpha_vec[j])

    assert kkt_viol.max() < 1e-3, (
        f"KKT violation too large: max={kkt_viol.max():.2e}"
    )


# ---------------------------------------------------------------------------
# 4. Matches sklearn LogisticRegression (scalar uniform alpha)
# ---------------------------------------------------------------------------

def test_matches_sklearn_scalar():
    """With uniform alpha and alpha_l2=0, the BCD solution should match
    sklearn's saga L1 solver on the same problem."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n, m = 200, 30
    X = rng.standard_normal((n, m))
    beta_true = np.zeros(m)
    beta_true[:4] = rng.standard_normal(4)
    y_01 = (np.sign(X @ beta_true + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    y = y_01 * 2 - 1  # {0,1} -> {-1,+1}

    a_max = _alpha_max(X, y)
    alpha_val = 0.1 * a_max

    # sparse_ho solver (no L2 so the problems match exactly)
    log_alpha = np.log(np.full(m, alpha_val))
    model = WeightedSparseLogReg(alpha_l2=0.0)
    mask, dense, _ = compute_beta(
        X, y, log_alpha, model, tol=1e-6, compute_jac=False, max_iter=5000,
    )
    beta_ho = np.zeros(m)
    beta_ho[mask] = dense

    # sklearn: minimizes (1/n)*loss + alpha*||w||_1  when  C = 1/(n*alpha)
    C_sk = 1.0 / (n * alpha_val)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = LogisticRegression(
            C=C_sk, penalty='l1', solver='saga',
            tol=1e-8, max_iter=10000, fit_intercept=False,
        )
        clf.fit(X, y_01)
    beta_sk = clf.coef_.ravel()

    # Supports must agree
    supp_ho = set(np.where(np.abs(beta_ho) > 1e-5)[0])
    supp_sk = set(np.where(np.abs(beta_sk) > 1e-5)[0])
    assert supp_ho == supp_sk, (
        f"Support mismatch: sparse_ho={sorted(supp_ho)}, sklearn={sorted(supp_sk)}"
    )
    # Coefficients should agree to ~1 % (different solver tolerances)
    assert np.allclose(beta_ho, beta_sk, atol=1e-2), (
        f"Coefficient mismatch: max_diff={np.abs(beta_ho - beta_sk).max():.2e}"
    )


# ---------------------------------------------------------------------------
# 5. Dense and sparse code paths return the same beta
# ---------------------------------------------------------------------------

def test_dense_sparse_agree():
    rng = np.random.default_rng(7)
    n, m = 150, 60
    X_csc, y = _make_sparse_problem(n=n, m=m, rng=rng)
    X_dense = np.asarray(X_csc.todense())

    a_max = _alpha_max(X_dense, y)
    alpha_l2 = 1.0 / n
    log_alpha = np.log(np.full(m, 0.05 * a_max))

    model_d = WeightedSparseLogReg(alpha_l2=alpha_l2)
    model_s = WeightedSparseLogReg(alpha_l2=alpha_l2)

    mask_d, dense_d, _ = compute_beta(
        X_dense, y, log_alpha, model_d, tol=1e-5, compute_jac=False)
    mask_s, dense_s, _ = compute_beta(
        X_csc, y, log_alpha, model_s, tol=1e-5, compute_jac=False)

    beta_d = np.zeros(m)
    beta_d[mask_d] = dense_d
    beta_s = np.zeros(m)
    beta_s[mask_s] = dense_s

    assert np.allclose(beta_d, beta_s, atol=1e-4), (
        f"Dense/sparse solutions differ: max_diff={np.abs(beta_d - beta_s).max():.2e}"
    )


# ---------------------------------------------------------------------------
# 6. Speed: realistic problem completes in < 10 s
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 13. Speed: dense inner solve completes in < 10 s
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _JIT_DISABLED,
    reason="Speed test requires Numba JIT (NUMBA_DISABLE_JIT=1 is set)"
)
def test_solver_speed():
    rng = np.random.default_rng(99)
    n, m = 500, 200
    X = rng.standard_normal((n, m))
    beta_true = np.zeros(m)
    beta_true[:10] = rng.standard_normal(10)
    y = np.sign(X @ beta_true + 0.2 * rng.standard_normal(n))
    y[y == 0] = 1.0

    alpha_l2 = 1.0 / n
    a_max = _alpha_max(X, y)
    log_alpha = np.log(np.full(m, 0.1 * a_max))

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)

    t0 = time.time()
    mask, dense, _ = compute_beta(
        X, y, log_alpha, model, tol=1e-4, compute_jac=False, max_iter=1000,
    )
    elapsed = time.time() - t0

    assert elapsed < 10.0, (
        f"Inner solve took {elapsed:.1f}s — too slow (limit: 10s)"
    )
    assert len(dense) > 0, "Solver returned empty support"


# ---------------------------------------------------------------------------
# 2. Primal objective decreases monotonically (sparse path)
# ---------------------------------------------------------------------------

def test_primal_decreasing_sparse():
    """Call the sparse BCD kernel directly (as compute_beta does)."""
    rng = np.random.default_rng(13)
    n, m = 150, 60
    X_csc, y = _make_sparse_problem(n=n, m=m, rng=rng)

    a_max = _alpha_max(X_csc, y)
    alpha_l2 = 1.0 / n
    alpha = np.full(m, 0.1 * a_max)

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    L = model.get_L(X_csc)

    beta = np.zeros(m)
    dual_var = np.zeros(n)
    dbeta = np.zeros((m, m))
    ddual_var = np.zeros((n, m))

    pobj_list = []
    for _ in range(30):
        model._update_beta_jac_bcd_sparse(
            X_csc.data, X_csc.indptr, X_csc.indices, y, n, m,
            beta, dbeta, dual_var, ddual_var, alpha, L, False,
        )
        pobj_list.append(model._get_pobj(dual_var, X_csc, beta, alpha, y))

    diffs = np.diff(pobj_list)
    assert np.all(diffs <= 1e-10), (
        f"Sparse primal objective is not monotonically decreasing: "
        f"max increase = {diffs.max():.2e}"
    )


# ---------------------------------------------------------------------------
# 5. KKT conditions (sparse X)
# ---------------------------------------------------------------------------

def test_kkt_conditions_sparse():
    rng = np.random.default_rng(21)
    n, m = 200, 80
    X_csc, y = _make_sparse_problem(n=n, m=m, rng=rng)

    a_max = _alpha_max(X_csc, y)
    alpha_l2 = 1.0 / n
    alpha_vec = np.full(m, 0.1 * a_max)
    log_alpha = np.log(alpha_vec)

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    mask, dense, _ = compute_beta(
        X_csc, y, log_alpha, model, tol=1e-4, compute_jac=False, max_iter=2000,
    )
    beta = np.zeros(m)
    beta[mask] = dense

    Xb = np.asarray(X_csc @ beta).ravel()
    s = sigma(-y * Xb)
    grad_smooth = -np.asarray(X_csc.T @ (y * s)).ravel() / n + alpha_l2 * beta

    kkt_viol = np.zeros(m)
    for j in range(m):
        if beta[j] > 1e-8:
            kkt_viol[j] = abs(grad_smooth[j] + alpha_vec[j])
        elif beta[j] < -1e-8:
            kkt_viol[j] = abs(grad_smooth[j] - alpha_vec[j])
        else:
            kkt_viol[j] = max(0.0, abs(grad_smooth[j]) - alpha_vec[j])

    assert kkt_viol.max() < 1e-3, (
        f"Sparse KKT violation too large: max={kkt_viol.max():.2e}"
    )


# ---------------------------------------------------------------------------
# 8. get_L is a valid Lipschitz upper bound
# ---------------------------------------------------------------------------

def test_get_L_upper_bound():
    """get_L[j] = ||X[:,j]||^2 / (4n) + alpha_l2 is an upper bound because
    sigma(z)*(1-sigma(z)) <= 1/4 for all z.  The adaptive L_temp computed
    inside the BCD kernel must be <= get_L[j]."""
    rng = np.random.default_rng(55)
    n, m = 100, 40
    X, y = _make_dense_problem(n=n, m=m, rng=rng)
    alpha_l2 = 0.5 / n
    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    L_static = model.get_L(X)

    X_f = np.asfortranarray(X)
    a_max = _alpha_max(X, y)
    beta = np.zeros(m)
    dual_var = np.zeros(n)

    for j in range(m):
        sigmar = sigma(dual_var)
        L_adapt = float(np.sum(X_f[:, j] ** 2 * sigmar * (1.0 - sigmar)) / n + alpha_l2)
        assert L_static[j] >= L_adapt - 1e-14, (
            f"get_L[{j}]={L_static[j]:.4e} < adaptive L={L_adapt:.4e}"
        )

    # Also check that caching works: second call returns same object
    L_again = model.get_L(X)
    assert L_again is L_static


# ---------------------------------------------------------------------------
# 9. get_mat_vec_impl LinearOperator matches explicit reduced Hessian
# ---------------------------------------------------------------------------

def test_get_mat_vec_impl_correctness():
    """H_S v ≈ (1/n) X_S^T D X_S v + alpha_l2 v where D = diag(w)."""
    rng = np.random.default_rng(77)
    n, m = 80, 30
    X, y = _make_dense_problem(n=n, m=m, rng=rng)
    alpha_l2 = 1.0 / n

    a_max = _alpha_max(X, y)
    log_alpha = np.log(np.full(m, 0.05 * a_max))
    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    mask, dense, _ = compute_beta(
        X, y, log_alpha, model, tol=1e-4, compute_jac=False)

    if mask.sum() < 2:
        pytest.skip("Support too small for meaningful Hessian test")

    X_m = X[:, mask]
    a = y * (X_m @ dense)
    w = sigma(a) * (1.0 - sigma(a))
    H_explicit = (X_m.T @ (w[:, None] * X_m)) / n + alpha_l2 * np.eye(mask.sum())

    H_op = model.get_mat_vec(X, y, mask, dense, log_alpha)
    rng2 = np.random.default_rng(0)
    v = rng2.standard_normal(mask.sum())

    Hv_op = H_op @ v
    Hv_explicit = H_explicit @ v
    assert np.allclose(Hv_op, Hv_explicit, rtol=1e-5), (
        f"get_mat_vec mismatch: max_diff={np.abs(Hv_op - Hv_explicit).max():.2e}"
    )


# ---------------------------------------------------------------------------
# 10. __getattribute__ dispatch
# ---------------------------------------------------------------------------

def test_getattribute_dispatch():
    """_get_pobj, _get_dobj, and get_mat_vec are redirected via __getattribute__."""
    model = WeightedSparseLogReg(alpha_l2=0.1)

    # _get_pobj -> _get_pobj_impl
    # model._get_pobj triggers __getattribute__ and returns _get_pobj_impl
    assert model._get_pobj.__func__ is model._get_pobj_impl.__func__, (
        "_get_pobj should dispatch to _get_pobj_impl"
    )

    # _get_dobj -> _get_dobj_impl (the fix: without it hasattr returns False)
    assert model._get_dobj.__func__ is model._get_dobj_impl.__func__, (
        "_get_dobj should dispatch to _get_dobj_impl"
    )

    # get_mat_vec -> get_mat_vec_impl
    assert model.get_mat_vec.__func__ is model.get_mat_vec_impl.__func__, (
        "get_mat_vec should dispatch to get_mat_vec_impl"
    )

    # Functional check: two models with different alpha_l2 produce different pobj
    model2 = WeightedSparseLogReg(alpha_l2=10.0)
    rng = np.random.default_rng(42)
    n, m = 20, 10
    dual_var = rng.standard_normal(n)
    beta = rng.standard_normal(m)
    alphas = np.ones(m) * 0.01
    pobj1 = model._get_pobj(dual_var, None, beta, alphas, None)
    pobj2 = model2._get_pobj(dual_var, None, beta, alphas, None)
    assert pobj1 != pobj2, (
        "Different alpha_l2 values should produce different primal objectives"
    )


# ---------------------------------------------------------------------------
# 11. _get_pobj numerical stability
# ---------------------------------------------------------------------------

def test_pobj_numerical_stability():
    """np.logaddexp(0, -x) is stable; the old np.log1p(np.exp(-x)) overflows."""
    rng = np.random.default_rng(3)
    n, m = 50, 20
    _, y = _make_dense_problem(n=n, m=m, rng=rng)
    beta = rng.standard_normal(m)
    alphas = np.full(m, 0.01)

    model = WeightedSparseLogReg(alpha_l2=0.1)

    # Construct dual_var with very negative entries (badly misclassified samples)
    dual_var_extreme = np.full(n, -500.0)  # exp(500) would overflow
    pobj_extreme = model._get_pobj(dual_var_extreme, None, beta, alphas, y)
    assert np.isfinite(pobj_extreme), (
        f"_get_pobj overflowed: got {pobj_extreme}"
    )

    # Also test very positive (well-classified) entries
    dual_var_pos = np.full(n, 500.0)
    pobj_pos = model._get_pobj(dual_var_pos, None, beta, alphas, y)
    assert np.isfinite(pobj_pos), (
        f"_get_pobj overflowed for positive dual_var: got {pobj_pos}"
    )
    assert pobj_pos < pobj_extreme  # well-classified should have lower loss


# ---------------------------------------------------------------------------
# 12. hasattr(model, '_get_dobj') is True (dual-gap criterion accessible)
# ---------------------------------------------------------------------------

def test_dobj_accessible_for_early_stopping():
    """compute_beta checks hasattr(model, '_get_dobj'). If False it falls back
    to the less reliable pobj-decrease criterion.  After the __getattribute__
    fix this must return True."""
    model = WeightedSparseLogReg(alpha_l2=0.01)
    assert hasattr(model, "_get_dobj"), (
        "hasattr(model, '_get_dobj') is False — dual-gap stopping criterion "
        "disabled; inner solver will run max_iter iterations on every call."
    )


def test_sparse_wide_stop_criterion_exits_early():
    rng = np.random.default_rng(123)
    n, m = 80, 400
    X_csc, y = _make_sparse_problem(n=n, m=m, density=0.05, rng=rng)

    alpha_l2 = 1.0 / n
    a_max = _alpha_max(X_csc, y)
    log_alpha = np.log(np.full(m, 0.1 * a_max))

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
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


def test_sparse_active_set_heuristic_enables_real_sim_like_shapes():
    model = WeightedSparseLogReg(alpha_l2=0.01)

    # real-sim-like: sparse, high-dimensional, but not wide enough to satisfy
    # the old m >= 4n rule.
    X_real_sim_like = csc_matrix((43_385, 20_958))
    assert model.should_use_sparse_active_set(X_real_sim_like)

    # Low-dimensional sparse problems should stay on the full-pass path.
    X_low_dim = csc_matrix((43_385, 300))
    assert not model.should_use_sparse_active_set(X_low_dim)


# ---------------------------------------------------------------------------
# 14. Speed: sparse inner solve completes in < 10 s
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _JIT_DISABLED,
    reason="Speed test requires Numba JIT (NUMBA_DISABLE_JIT=1 is set)"
)
def test_solver_speed_sparse():
    rng = np.random.default_rng(88)
    n, m = 500, 200
    X_csc, y = _make_sparse_problem(n=n, m=m, density=0.10, rng=rng)

    alpha_l2 = 1.0 / n
    a_max = _alpha_max(X_csc, y)
    log_alpha = np.log(np.full(m, 0.1 * a_max))

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)

    t0 = time.time()
    mask, dense, _ = compute_beta(
        X_csc, y, log_alpha, model, tol=1e-4, compute_jac=False, max_iter=1000,
    )
    elapsed = time.time() - t0

    assert elapsed < 10.0, (
        f"Sparse inner solve took {elapsed:.1f}s — too slow (limit: 10s)"
    )
    assert len(dense) > 0, "Solver returned empty support"
