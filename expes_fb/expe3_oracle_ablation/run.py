"""Experiment 3 — 2×2 oracle × optimizer ablation on degenerate synthetic data.

Isolates the contributions of the oracle policy and the outer optimizer
independently by crossing two binary axes:

  Oracle policy
  ─────────────
  Null (baseline)     — working set S = strict primal support only.
                        Biactive features receive exactly zero hypergradient.
  DA (descent-aligned) — working set augmented with biactive coordinates
                          satisfying the descent-alignment condition.

  Outer optimizer
  ───────────────
  NBA   — projected normalized subgradient with fixed step size.
           x_{k+1} = proj(x_k − η · g_k / ‖g_k‖),  η fixed.
  NTRBA — nonsmooth trust-region with adaptive radius.

This yields four methods: NBA-null, NBA-SC, NTRBA-null, NTRBA-SC.

Dataset
-------
Uses the calibrated degenerate four-group dataset from
``expes_fb/shared/data_gen_degenerate.py``.  Hidden features are biactive
at the initial hyperparameter x_0 by construction, so both null and DA
oracles start from the same point with the null oracle providing zero
gradient for hidden features from the very first iteration.

Usage
-----
    python run.py            # from within this directory, or
    python expes_fb/expe3_oracle_ablation/run.py   # from repo root
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

M_VALUES  = [250, 500, 1000]     # number of features
N_RATIO   = 2 / 3                 # n_total / m  (overparameterized, n < m)
RHO       = 0.95                  # distractor–hidden Pearson correlation
N_SEEDS   = 20
N_OUTER   = 100                    # outer iterations (shared by all methods)
INNER_TOL = 1e-8                  # inner solver tolerance

NBA_STEP_SIZE    = 0.1    # fixed step for NBA (matches TR initial radius)
TR_RADIUS0       = 0.1    # initial trust-region radius for NTRBA
BIACTIVE_TOL_REL = 0.10   # 2 × calibration_slack → detects all hidden features
CALIBRATION_SLACK = 0.05

# Feature group fractions
EASY_FRAC   = 0.04  # fraction of m allocated to easy features
DIST_FRAC   = 0.04  # fraction allocated to distractors (= hidden)

RESULTS_DIR  = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'

# --- smoke-test overrides: set VSHPO_SMOKE=1 for a fast, tiny run (CI / spot-check).
# When the variable is unset, the full configuration above is used unchanged.
import os
if os.environ.get('VSHPO_SMOKE'):
    M_VALUES = [250]
    N_SEEDS  = 2
    N_OUTER  = 5


# ---------------------------------------------------------------------------
# Method definitions
# ---------------------------------------------------------------------------

METHODS = {
    'NBA-null':   dict(optimizer='nba',  policy=None),
    'NBA-SC':     dict(optimizer='nba',  policy='sc'),
    'NTRBA-null': dict(optimizer='ntrba', policy=None),
    'NTRBA-SC':   dict(optimizer='ntrba', policy='sc'),
}


def _make_optimizer(kind):
    if kind == 'nba':
        return NormalizedSubgradient(
            n_outer=N_OUTER, step_size=NBA_STEP_SIZE, tol=INNER_TOL)
    else:
        return TrustRegion(
            n_outer=N_OUTER, radius0=TR_RADIUS0, tol=INNER_TOL)


def _make_algo(policy_key):
    policy = select_biactive_self_consistent if policy_key == 'sc' else None
    return ImplicitVariational(
        policy=policy,
        biactive_tol_rel=BIACTIVE_TOL_REL,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_sizes(m):
    n_easy   = max(int(EASY_FRAC * m), 5)
    n_dist   = max(int(DIST_FRAC * m), 5)
    n_hidden = n_dist                          # 1:1 pairing
    return n_easy, n_dist, n_hidden


def _solve_inner(model, X_tr, y_tr, log_alpha, tol=INNER_TOL):
    """Solve inner problem; return full coefficient vector."""
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
# Single-run function
# ---------------------------------------------------------------------------

def run_method(method_name, X, y, groups, idx_train, idx_val, idx_test,
               log_alpha0, alpha_l2, m):
    """Run one method on a single (dataset, seed) instance.

    Returns a dict of per-run metrics.
    """
    cfg = METHODS[method_name]
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_te, y_te = X[idx_test],  y[idx_test]

    model     = WeightedElasticNet(alpha_l2=alpha_l2)
    algo      = _make_algo(cfg['policy'])
    optimizer = _make_optimizer(cfg['optimizer'])
    criterion = HeldOutMSE(idx_train, idx_val)

    monitor = Monitor()
    t0 = time.time()
    # grad_search expects actual penalty values (not log); log_alpha0 → exp
    grad_search(algo, criterion, model, optimizer, X, y, np.exp(log_alpha0), monitor)
    elapsed = time.time() - t0

    # ---- extract per-iteration trajectories --------------------------------
    val_objs = list(monitor.objs)
    grad_norms = [
        float(np.linalg.norm(g)) if g is not None else np.nan
        for g in monitor.grads
    ]

    # TR radius trajectory (only meaningful for NTRBA methods)
    tr_radii = []
    if hasattr(optimizer, 'history_') and optimizer.history_:
        tr_radii = [rec.get('radius_after', np.nan) for rec in optimizer.history_]

    # ---- best alpha from validation trajectory -----------------------------
    if monitor.alphas:
        best_iter   = int(np.argmin(monitor.objs))
        alpha_final = np.array(monitor.alphas[best_iter])
    else:
        alpha_final = np.exp(log_alpha0)

    log_alpha_final = np.log(np.maximum(alpha_final, 1e-300))

    # ---- final beta and metrics --------------------------------------------
    beta_final = _solve_inner(model, X_tr, y_tr, log_alpha_final)
    test_mse   = float(np.mean((X_te @ beta_final - y_te) ** 2))
    n_iter     = len(val_objs)
    t_per_iter = elapsed / max(n_iter, 1)

    return dict(
        method=method_name,
        val_objs=val_objs,
        grad_norms=grad_norms,
        tr_radii=tr_radii,
        alpha_final=alpha_final,
        beta_final=beta_final,
        test_mse=test_mse,
        elapsed=elapsed,
        t_per_iter=t_per_iter,
        n_iter=n_iter,
        termination=getattr(optimizer, 'termination_reason_', None),
    )


# ---------------------------------------------------------------------------
# Per-configuration runner
# ---------------------------------------------------------------------------

def run_one(m, seed):
    n        = int(N_RATIO * m)
    n_easy, n_dist, n_hidden = _group_sizes(m)
    alpha_l2 = 1.0 / int(0.6 * n)

    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X, y, beta_true, groups, idx_train, idx_val, idx_test, log_alpha0 = (
            make_degenerate_dataset(
                n, m, n_easy, n_dist, n_hidden,
                rho=RHO, alpha_l2=alpha_l2,
                beta_std=1.0, noise_std=0.05,
                calibration_slack=CALIBRATION_SLACK,
                rng=rng,
            )
        )

    rows = []
    method_results = {}
    for mname in METHODS:
        res = run_method(mname, X, y, groups, idx_train, idx_val, idx_test,
                         log_alpha0, alpha_l2, m)
        method_results[mname] = res

    # compute per-seed reference: best val loss achieved by NTRBA-SC
    ref_val = min(method_results['NTRBA-SC']['val_objs']) if method_results['NTRBA-SC']['val_objs'] else np.nan

    for mname, res in method_results.items():
        best_val = min(res['val_objs']) if res['val_objs'] else np.nan
        beta_f   = res['beta_final']

        rows.append(dict(
            m=m, seed=seed,
            method=mname,
            # per-iteration trajectories (stored as lists for later plotting)
            val_objs=res['val_objs'],
            grad_norms=res['grad_norms'],
            tr_radii=res['tr_radii'],
            # scalar metrics
            val_loss_gap=float(best_val - ref_val),
            best_val_loss=float(best_val),
            final_grad_norm=float(res['grad_norms'][-1]) if res['grad_norms'] else np.nan,
            f1=compute_f1(beta_f, beta_true),
            hidden_recall=hidden_recall(beta_f, groups),
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

    configs = list(product(M_VALUES, range(N_SEEDS)))
    total   = len(configs)
    done    = 0

    rows = []
    for m, seed in configs:
        done += 1
        print(
            f"[{done:3d}/{total}] m={m:5d}  seed={seed}",
            flush=True,
        )
        try:
            rows.extend(run_one(m, seed))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(rows)
    df.to_pickle(RESULTS_PATH)
    print(f"\nSaved {len(df)} rows to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
