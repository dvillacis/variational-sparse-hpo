"""Experiment 2 — feature-wise vs. scalar elastic-net regularization.

Compares two methods on overparameterized synthetic regression:

  Scalar elastic-net CV
    Same WeightedElasticNet model with a uniform global penalty lambda,
    selected by grid search (100 values) on the validation set.

  Weighted elastic-net (ours)
    Full per-feature penalty vector x ∈ R^m learned by bilevel optimization
    (ImplicitVariational oracle + TrustRegion outer optimizer).

Both methods share the same alpha_l2 = 1/n_train fixed L2 term.

Usage
-----
    python run.py            # from within this directory, or
    python expes_fb/expe2_feature_resolution/run.py   # from repo root
"""

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# make data_gen importable when script is run from any directory
sys.path.insert(0, str(Path(__file__).parent))
from data_gen import make_three_group_dataset

from sparse_ho.models import WeightedElasticNet
from sparse_ho.criterion import HeldOutMSE
from sparse_ho.algo import ImplicitVariational, select_biactive_self_consistent
from sparse_ho.optimizers import TrustRegion
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from sparse_ho.algo.forward import compute_beta

# ---------------------------------------------------------------------------
# sweep configuration
# ---------------------------------------------------------------------------
M_VALUES       = [100, 200]       # number of features
# nm_ratio = n_total / m.  The sparse-recovery threshold for the inner problem
# is n_train > 2 * s_signal * ln(m / s_signal), where n_train = 0.6 * nm * m.
# The worst case across our configs is (m=200, sparsity=10%, s=20):
#   threshold = 2*20*ln(10) ≈ 92,  nm_min = 92 / (0.6*200) ≈ 0.768.
# nm=0.8 satisfies this with a ~4% margin; nm=1.0 gives ~30% margin.
# Both keep n_train < m (training remains overparameterized).
NM_RATIOS      = [0.8, 1.0]       # n / m (overparameterized but feasible)
SPARSITY_FRACS = [0.02, 0.05, 0.10]  # s_signal / m (monotone trend: advantage grows as sparsity decreases)
CORR           = 0.7              # signal ↔ corr-noise correlation
N_SEEDS        = 20
N_OUTER        = 50               # outer iterations for bilevel
N_GRID         = 100              # grid points for scalar baseline
TOL_INNER      = 1e-7             # inner solver tolerance

RESULTS_DIR  = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'

# --- smoke-test overrides: set VSHPO_SMOKE=1 for a fast, tiny run (CI / spot-check).
# When the variable is unset, the full configuration above is used unchanged.
import os
if os.environ.get('VSHPO_SMOKE'):
    M_VALUES       = [100]
    NM_RATIOS      = [0.8]
    SPARSITY_FRACS = [0.05]
    N_SEEDS        = 2
    N_OUTER        = 5
    N_GRID         = 10


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _solve_inner(model, X_tr, y_tr, log_alpha, tol=TOL_INNER):
    """Solve inner problem; return full coefficient vector."""
    mask, dense, _ = compute_beta(
        X_tr, y_tr, log_alpha, model, tol=tol, compute_jac=False)
    beta = np.zeros(X_tr.shape[1])
    beta[mask] = dense
    return beta


def compute_f1(beta_pred, beta_true, threshold=1e-10):
    """Support-recovery F1 score."""
    pred_supp = np.abs(beta_pred) > threshold
    true_supp = beta_true != 0
    tp = np.sum(pred_supp & true_supp)
    fp = np.sum(pred_supp & ~true_supp)
    fn = np.sum(~pred_supp & true_supp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# method 1 — scalar elastic-net: uniform lambda, selected by val MSE
# ---------------------------------------------------------------------------

def run_scalar(X, y, idx_train, idx_val, idx_test, alpha_l2, m):
    X_tr, y_tr   = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val],   y[idx_val]
    X_te, y_te   = X[idx_test],  y[idx_test]
    n_tr = len(idx_train)

    model = WeightedElasticNet(alpha_l2=alpha_l2)

    lambda_max  = np.max(np.abs(X_tr.T @ y_tr)) / n_tr
    log_lam_max = np.log(lambda_max * 0.9)
    log_lam_min = np.log(lambda_max * 1e-3)
    log_lambdas = np.linspace(log_lam_max, log_lam_min, N_GRID)

    best_val_mse  = np.inf
    best_log_lam  = log_lambdas[0]
    best_beta     = np.zeros(m)

    for log_lam in log_lambdas:
        log_alpha = log_lam * np.ones(m)
        beta = _solve_inner(model, X_tr, y_tr, log_alpha)
        val_mse = np.mean((X_val @ beta - y_val) ** 2)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_log_lam = log_lam
            best_beta    = beta.copy()

    test_mse       = np.mean((X_te @ best_beta - y_te) ** 2)
    penalty_profile = np.exp(best_log_lam) * np.ones(m)
    return best_beta, test_mse, penalty_profile


# ---------------------------------------------------------------------------
# method 2 — weighted elastic-net: bilevel via ImplicitVariational + TrustRegion
# ---------------------------------------------------------------------------

def run_weighted(X, y, idx_train, idx_val, idx_test, alpha_l2, m):
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_te, y_te = X[idx_test],  y[idx_test]
    n_tr = len(idx_train)

    model     = WeightedElasticNet(alpha_l2=alpha_l2)
    algo      = ImplicitVariational(
        policy=select_biactive_self_consistent,
        biactive_tol_rel=1e-8,
    )
    optimizer = TrustRegion(n_outer=N_OUTER, tol=1e-5)
    criterion = HeldOutMSE(idx_train, idx_val)

    # uniform initialisation at lambda_max / 5
    lambda_max = np.max(np.abs(X_tr.T @ y_tr)) / n_tr
    alpha0 = (lambda_max / 5.0) * np.ones(m)

    monitor = Monitor()
    t0 = time.time()
    grad_search(algo, criterion, model, optimizer, X, y, alpha0, monitor)
    elapsed = time.time() - t0

    # pick the alpha with the best validation objective seen during the run
    if monitor.alphas:
        best_iter   = int(np.argmin(monitor.objs))
        alpha_final = np.array(monitor.alphas[best_iter])
    else:
        alpha_final = alpha0.copy()

    log_alpha_final = np.log(alpha_final)
    beta_final = _solve_inner(model, X_tr, y_tr, log_alpha_final)
    test_mse   = np.mean((X_te @ beta_final - y_te) ** 2)

    return beta_final, test_mse, alpha_final, list(monitor.objs), elapsed


# ---------------------------------------------------------------------------
# per-configuration runner
# ---------------------------------------------------------------------------

def run_one(m, nm_ratio, sparsity_frac, seed):
    n        = int(nm_ratio * m)
    s_signal = max(int(sparsity_frac * m), 2)
    s_corr   = min(2 * s_signal, m - s_signal - 1)   # leave ≥1 pure-noise col

    rng = np.random.default_rng(seed)
    X, y, beta_true, groups, idx_train, idx_val, idx_test = (
        make_three_group_dataset(n, m, s_signal, s_corr, corr=CORR, rng=rng)
    )

    n_tr    = len(idx_train)
    alpha_l2 = 1.0 / n_tr

    # scalar baseline
    beta_sc, mse_sc, profile_sc = run_scalar(
        X, y, idx_train, idx_val, idx_test, alpha_l2, m)
    f1_sc = compute_f1(beta_sc, beta_true)

    # weighted bilevel
    beta_wt, mse_wt, profile_wt, objs_wt, t_wt = run_weighted(
        X, y, idx_train, idx_val, idx_test, alpha_l2, m)
    f1_wt = compute_f1(beta_wt, beta_true)

    base = dict(
        m=m, n=n, nm_ratio=nm_ratio, sparsity_frac=sparsity_frac,
        seed=seed, s_signal=s_signal, s_corr=s_corr,
    )
    return [
        dict(**base, method='scalar',
             test_mse=mse_sc, f1=f1_sc,
             penalty_profile=profile_sc),
        dict(**base, method='weighted',
             test_mse=mse_wt, f1=f1_wt,
             penalty_profile=profile_wt,
             val_objs=objs_wt, bilevel_time=t_wt),
    ]


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    configs = list(product(M_VALUES, NM_RATIOS, SPARSITY_FRACS))
    total   = len(configs) * N_SEEDS
    done    = 0

    rows = []
    for m, nm_ratio, sparsity_frac in configs:
        for seed in range(N_SEEDS):
            done += 1
            print(
                f"[{done:3d}/{total}] m={m:4d}  n/m={nm_ratio:.1f}"
                f"  sparsity={sparsity_frac:.0%}  seed={seed}",
                flush=True,
            )
            try:
                rows.extend(run_one(m, nm_ratio, sparsity_frac, seed))
            except Exception as exc:
                print(f"  ERROR: {exc}")

    df = pd.DataFrame(rows)
    df.to_pickle(RESULTS_PATH)
    print(f"\nSaved {len(df)} rows to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
