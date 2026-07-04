"""Experiment 4 — SOTA comparison: gradient starvation on degenerate instances.

Three methods are compared on the calibrated degenerate four-group dataset:

  Sparse-HO (scalar ℓ1)
  ─────────────────────
  A single global penalty x ∈ ℝ, tuned by grid search over 100 log-spaced
  values evaluated by validation MSE.  Per-feature targeting is structurally
  impossible.  Included to show that brute-force activation is not the right
  strategy.

  Sparse-HO (wℓ1)
  ────────────────
  Per-feature penalties x ∈ ℝ^m, optimised via the standard support-restricted
  adjoint (ImplicitVariational with null biactive policy) + NormalizedSubgradient
  outer loop.  Gradient starvation is exact and permanent for hidden features
  because they are biactive at the calibrated initialisation.

  NTRBA-wℓ1 (ours)
  ─────────────────
  Same parameterisation as above, but differentiated via the generalised support
  adjoint with descent-aligned biactive selection (ImplicitVariational + DA
  policy), updated with the nonsmooth trust-region outer solver (TrustRegion).

Dataset
-------
Uses the calibrated degenerate four-group dataset from
``expes_fb/shared/data_gen_degenerate.py``.

Sweep
-----
(n, m) ∈ {(100, 150), (200, 300), (500, 750)}
ρ      ∈ {0.90, 0.95, 0.98}
seeds  ∈ range(20)

→ 180 (n,m,ρ,seed) configurations × 3 methods = 540 rows.

Usage
-----
    python run.py            # from within this directory, or
    python expes_fb/expe4_sota_comparison/run.py   # from repo root
"""

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# make shared data_gen importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from data_gen_degenerate import make_degenerate_dataset

from sparse_ho.models import WeightedElasticNet
from sparse_ho.criterion import HeldOutMSE
from sparse_ho.algo import ImplicitVariational, select_biactive_self_consistent
from sparse_ho.optimizers import TrustRegion, NormalizedSubgradient
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from sparse_ho.algo.forward import compute_beta

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

NM_PAIRS = [(100, 150), (200, 300), (500, 750)]
RHO_VALUES = [0.90, 0.95, 0.98]
N_SEEDS = 20
N_OUTER = 60
INNER_TOL = 1e-8
N_GRID = 100           # scalar lambda grid points for sparseho_scalar

NBA_STEP_SIZE = 0.1    # fixed step for sparseho_wl1 (NBA)
TR_RADIUS0 = 0.1       # initial trust-region radius for ntrba_wl1
BIACTIVE_TOL_REL = 0.10
CALIBRATION_SLACK = 0.05

# Feature group fractions
EASY_FRAC = 0.04
DIST_FRAC = 0.04

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'

# --- smoke-test overrides: set VSHPO_SMOKE=1 for a fast, tiny run (CI / spot-check).
# When the variable is unset, the full configuration above is used unchanged.
import os
if os.environ.get('VSHPO_SMOKE'):
    # Use the plot's representative instance (n=200, m=300, rho=0.98) so the
    # convergence figure is exercised in smoke mode.
    NM_PAIRS   = [(200, 300)]
    RHO_VALUES = [0.98]
    N_SEEDS    = 2
    N_OUTER    = 5
    N_GRID     = 10

# Representative instance for the convergence figure
REP_N, REP_M, REP_RHO, REP_SEED = 200, 300, 0.98, 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_sizes(m):
    n_easy = max(int(EASY_FRAC * m), 5)
    n_dist = max(int(DIST_FRAC * m), 5)
    n_hidden = n_dist                      # 1:1 pairing
    return n_easy, n_dist, n_hidden


def _solve_inner(model, X_tr, y_tr, log_alpha, tol=INNER_TOL):
    """Solve inner WeightedElasticNet; return full coefficient vector."""
    mask, dense, _ = compute_beta(
        X_tr, y_tr, log_alpha, model, tol=tol, compute_jac=False)
    beta = np.zeros(X_tr.shape[1])
    beta[mask] = dense
    return beta


def compute_f1(beta_pred, beta_true, threshold=1e-10):
    """Support-recovery F1 over all features."""
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


def hidden_recall(beta_pred, groups, threshold=1e-10):
    """Fraction of hidden features with |β*| > threshold."""
    hid = groups['hidden']
    return float(np.mean(np.abs(beta_pred[hid]) > threshold))


# ---------------------------------------------------------------------------
# Sparse-HO (scalar): grid search over uniform penalty
# ---------------------------------------------------------------------------

def run_sparseho_scalar(X, y, beta_true, groups, idx_train, idx_val, idx_test,
                        alpha_l2, m):
    """Grid search over a scalar lambda; no hypergradient is computed."""
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_te, y_te = X[idx_test], y[idx_test]

    lam_max = float(np.max(np.abs(X_tr.T @ y_tr)) / len(idx_train))
    grid = np.logspace(
        np.log10(lam_max * 1e-3), np.log10(lam_max * 0.9), N_GRID
    )[::-1]   # descending: from large (sparse) to small (dense)

    model = WeightedElasticNet(alpha_l2=alpha_l2)

    best_val = np.inf
    best_beta = np.zeros(m)
    val_objs = []
    t0 = time.time()
    for lam in grid:
        log_alpha = np.log(lam) * np.ones(m)
        beta = _solve_inner(model, X_tr, y_tr, log_alpha)
        val_mse = float(np.mean((X_val @ beta - y_val) ** 2))
        val_objs.append(val_mse)
        if val_mse < best_val:
            best_val = val_mse
            best_beta = beta.copy()
    elapsed = time.time() - t0

    return dict(
        method='sparseho_scalar',
        val_objs=val_objs,
        # scalar method has no per-iteration gradient or alpha trajectory
        hidden_grad_norms=[],
        mean_hidden_alpha_traj=[],
        alphas_traj=grid.tolist(),      # list of N_GRID scalar lambdas
        best_val=float(best_val),
        beta_final=best_beta,
        test_mse=float(np.mean((X_te @ best_beta - y_te) ** 2)),
        elapsed=elapsed,
        t_per_iter=elapsed / N_GRID,
        n_iter=N_GRID,
        hidden_grad_norm_0=np.nan,
        termination='grid_search',
    )


# ---------------------------------------------------------------------------
# Gradient-based methods: sparseho_wl1, ntrba_wl1
# ---------------------------------------------------------------------------

def run_gradient_method(method_name, X, y, groups, idx_train, idx_val, idx_test,
                        log_alpha0, alpha_l2, m):
    """Run one gradient-based outer-loop method; extract per-iteration trajectories."""
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_te, y_te = X[idx_test], y[idx_test]
    hid = groups['hidden']

    if method_name == 'sparseho_wl1':
        algo = ImplicitVariational(
            policy=None, biactive_tol_rel=BIACTIVE_TOL_REL)
        optimizer = NormalizedSubgradient(
            n_outer=N_OUTER, step_size=NBA_STEP_SIZE, tol=INNER_TOL)
    else:  # ntrba_wl1
        algo = ImplicitVariational(
            policy=select_biactive_self_consistent,
            biactive_tol_rel=BIACTIVE_TOL_REL)
        optimizer = TrustRegion(
            n_outer=N_OUTER, radius0=TR_RADIUS0, tol=INNER_TOL)

    model = WeightedElasticNet(alpha_l2=alpha_l2)
    criterion = HeldOutMSE(idx_train, idx_val)
    monitor = Monitor()

    t0 = time.time()
    grad_search(algo, criterion, model, optimizer, X, y,
                np.exp(log_alpha0), monitor)
    elapsed = time.time() - t0

    # --- per-iteration derived quantities -----------------------------------
    hidden_grad_norms = []
    mean_hidden_alpha_traj = []
    alphas_traj = []

    for g, a in zip(monitor.grads, monitor.alphas):
        # hidden gradient norm
        if g is not None and hasattr(g, '__len__') and len(g) > max(hid):
            hidden_grad_norms.append(float(np.linalg.norm(g[hid])))
        else:
            hidden_grad_norms.append(np.nan)

        # mean hidden penalty and full alpha (for F1 post-hoc in plot.py)
        if a is not None and hasattr(a, '__len__') and len(a) > max(hid):
            mean_hidden_alpha_traj.append(float(np.mean(a[hid])))
            alphas_traj.append(a.copy())
        else:
            mean_hidden_alpha_traj.append(np.nan)
            alphas_traj.append(None)

    hidden_grad_norm_0 = (
        hidden_grad_norms[0] if hidden_grad_norms and
        np.isfinite(hidden_grad_norms[0]) else np.nan
    )

    # --- best alpha from validation trajectory ------------------------------
    if monitor.alphas:
        best_iter = int(np.argmin(monitor.objs))
        alpha_final = np.array(monitor.alphas[best_iter])
    else:
        alpha_final = np.exp(log_alpha0)

    log_alpha_final = np.log(np.maximum(alpha_final, 1e-300))
    beta_final = _solve_inner(model, X_tr, y_tr, log_alpha_final)

    return dict(
        method=method_name,
        val_objs=list(monitor.objs),
        hidden_grad_norms=hidden_grad_norms,
        mean_hidden_alpha_traj=mean_hidden_alpha_traj,
        alphas_traj=alphas_traj,      # list of m-dim arrays; used in plot.py
        best_val=float(min(monitor.objs)) if monitor.objs else np.nan,
        beta_final=beta_final,
        test_mse=float(np.mean((X_te @ beta_final - y_te) ** 2)),
        elapsed=elapsed,
        t_per_iter=elapsed / max(len(monitor.objs), 1),
        n_iter=len(monitor.objs),
        hidden_grad_norm_0=float(hidden_grad_norm_0),
        termination=getattr(optimizer, 'termination_reason_', None),
    )


# ---------------------------------------------------------------------------
# Per-configuration runner
# ---------------------------------------------------------------------------

def run_one(n, m, rho, seed):
    n_easy, n_dist, n_hidden = _group_sizes(m)
    alpha_l2 = 1.0 / int(0.6 * n)

    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X, y, beta_true, groups, idx_train, idx_val, idx_test, log_alpha0 = (
            make_degenerate_dataset(
                n, m, n_easy, n_dist, n_hidden,
                rho=rho, alpha_l2=alpha_l2,
                beta_std=1.0, noise_std=0.05,
                calibration_slack=CALIBRATION_SLACK,
                rng=rng,
            )
        )

    scalar_res = run_sparseho_scalar(
        X, y, beta_true, groups, idx_train, idx_val, idx_test, alpha_l2, m)
    wl1_res = run_gradient_method(
        'sparseho_wl1', X, y, groups, idx_train, idx_val, idx_test,
        log_alpha0, alpha_l2, m)
    ntrba_res = run_gradient_method(
        'ntrba_wl1', X, y, groups, idx_train, idx_val, idx_test,
        log_alpha0, alpha_l2, m)

    # reference: best validation loss achieved by ntrba_wl1
    ref_val = ntrba_res['best_val']

    rows = []
    for res in [scalar_res, wl1_res, ntrba_res]:
        beta_f = res['beta_final']
        rows.append(dict(
            n=n, m=m, rho=rho, seed=seed,
            method=res['method'],
            # per-iteration trajectory data (stored as lists)
            val_objs=res['val_objs'],
            hidden_grad_norms=res['hidden_grad_norms'],
            mean_hidden_alpha_traj=res['mean_hidden_alpha_traj'],
            alphas_traj=res['alphas_traj'],
            # scalar summary metrics
            val_loss_gap=float(res['best_val'] - ref_val),
            best_val_loss=float(res['best_val']),
            hidden_grad_norm_0=res['hidden_grad_norm_0'],
            hidden_recall=hidden_recall(beta_f, groups),
            f1=compute_f1(beta_f, beta_true),
            test_mse=res['test_mse'],
            elapsed=res['elapsed'],
            t_per_iter=res['t_per_iter'],
            n_iter=res['n_iter'],
            termination=res['termination'],
        ))
    return rows


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    configs = list(product(NM_PAIRS, RHO_VALUES, range(N_SEEDS)))
    total = len(configs)
    done = 0

    rows = []
    for (n, m), rho, seed in configs:
        done += 1
        print(
            f"[{done:3d}/{total}] n={n:4d} m={m:4d} rho={rho:.2f} seed={seed}",
            flush=True,
        )
        try:
            rows.extend(run_one(n, m, rho, seed))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(rows)
    df.to_pickle(RESULTS_PATH)
    print(f"\nSaved {len(df)} rows to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
