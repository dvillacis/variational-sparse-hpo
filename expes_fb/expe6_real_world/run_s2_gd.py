"""Experiment 6 Setting 2 — Real-world benchmarks with NBA instead of NTRBA.

This variant mirrors ``run_s2.py`` but replaces the ``ntrba_wl1`` method with
``nba_wl1``:

  scalar_cv    Scalar l1 selected by LogisticRegressionCV.
  sparseho_wl1 WeightedSparseLogReg + Implicit + NormalizedSubgradient.
  nba_wl1      WeightedSparseLogReg + ImplicitVariational (DA policy)
               + NormalizedSubgradient.

Results are written under ``results/setting2_gd`` so they do not mix with the
trust-region experiment.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_s2 as base

from sparse_ho.algo import Implicit, ImplicitVariational, select_biactive_self_consistent
from sparse_ho.models import WeightedSparseLogReg
from sparse_ho.optimizers import NormalizedSubgradient
from sparse_ho.utils import Monitor

_BASE_RUN_ONE = base.run_one


RESULTS_DIR = Path(__file__).parent / "results" / "setting2_gd"
RESULTS_PATH = RESULTS_DIR / "results.pkl"
CHECKPOINT_PATH = RESULTS_DIR / "results_checkpoint.pkl"
CHECKPOINT_META_PATH = RESULTS_DIR / "results_checkpoint_meta.json"

METHOD_ORDER = ["scalar_cv", "sparseho_wl1", "nba_wl1"]


def run_gradient_method(
    method_name,
    X,
    y,
    idx_train,
    idx_val,
    idx_test,
    log_alpha0,
    alpha_l2,
    m,
    log_prefix=None,
    keep_debug=False,
):
    from scipy.sparse import issparse

    if issparse(X):
        X = X.tocsc()
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_te, y_te = X[idx_test], y[idx_test]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    if issparse(X_te):
        X_te = X_te.tocsc()

    if method_name == "sparseho_wl1":
        algo = Implicit()
    elif method_name == "nba_wl1":
        algo = ImplicitVariational(
            policy=select_biactive_self_consistent,
            biactive_tol_rel=base.BIACTIVE_TOL_REL,
        )
    else:
        raise ValueError(f"Unknown method {method_name!r}")

    optimizer = NormalizedSubgradient(
        n_outer=base.N_OUTER,
        step_size=base.NBA_STEP_SIZE,
        tol=base.INNER_TOL_TR,
    )

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    if keep_debug:
        model.inner_debug_records_ = []
        model.inner_debug_context_ = f"{method_name}:outer"
    criterion = base._HeldOutLogisticWithMaxIter(
        idx_train, idx_val, max_iter=base.INNER_MAX_ITER
    )
    algo_debug_records = []
    log_callback = base._make_monitor_callback(log_prefix)

    def _callback(obj, grad, mask, dense, alpha):
        if keep_debug and hasattr(algo, "last_run_info") and algo.last_run_info:
            algo_debug_records.append(dict(algo.last_run_info))
        log_callback(obj, grad, mask, dense, alpha)

    monitor = Monitor(callback=_callback)

    base._log(
        f"starting {method_name} with n_outer={base.N_OUTER} "
        f"alpha_l2={alpha_l2:.3e} inner_max_iter={base.INNER_MAX_ITER}",
        prefix=log_prefix,
    )
    t0 = time.time()
    base.grad_search(algo, criterion, model, optimizer, X, y, np.exp(log_alpha0), monitor)
    elapsed = time.time() - t0

    if monitor.alphas:
        best_iter = int(np.argmin(monitor.objs))
        alpha_final = np.array(monitor.alphas[best_iter])
    else:
        alpha_final = np.exp(log_alpha0)
    log_alpha_final = np.log(np.maximum(alpha_final, 1e-300))
    beta_final = base._solve_inner(
        model,
        X_tr,
        y_tr,
        log_alpha_final,
        debug_context=f"{method_name}:refit" if keep_debug else None,
        max_iter=base.INNER_MAX_ITER,
    )
    best_val = float(min(monitor.objs)) if monitor.objs else np.nan
    inner_debug_records = list(getattr(model, "inner_debug_records_", []))
    inner_debug_summary = base._summarize_inner_debug(inner_debug_records)
    algo_debug_summary = base._summarize_algo_debug(algo_debug_records)
    base._log(
        f"finished {method_name} in {elapsed:.1f}s "
        f"n_iter={len(monitor.objs)} best_val={best_val:.6f} "
        f"termination={getattr(optimizer, 'termination_reason_', None)}",
        prefix=log_prefix,
    )

    return dict(
        method=method_name,
        val_objs=list(monitor.objs),
        hidden_grad_norms=[],
        best_val=best_val,
        beta_final=beta_final,
        test_f1=base._test_f1(X_te, y_te, beta_final),
        sparsity=base._sparsity(beta_final),
        elapsed=elapsed,
        t_per_iter=elapsed / max(len(monitor.objs), 1),
        n_iter=len(monitor.objs),
        termination=getattr(optimizer, "termination_reason_", None),
        inner_debug_records=inner_debug_records,
        inner_debug_summary=inner_debug_summary,
        algo_debug=dict(getattr(algo, "last_run_info", {})),
        algo_debug_records=algo_debug_records,
        algo_debug_summary=algo_debug_summary,
    )


def run_one(
    dataset_name,
    X,
    y,
    seed,
    keep_debug=False,
    skip_methods=None,
    row_callback=None,
):
    """Delegate to the base runner after reapplying GD-specific overrides.

    This wrapper is important for joblib worker processes: workers may import
    ``run_s2`` afresh, so we must ensure they dispatch through the GD variant
    rather than the original NTRBA runner.
    """
    _configure_base_module()
    return _BASE_RUN_ONE(
        dataset_name,
        X,
        y,
        seed,
        keep_debug=keep_debug,
        skip_methods=skip_methods,
        row_callback=row_callback,
    )


def _configure_base_module():
    base.RESULTS_DIR = RESULTS_DIR
    base.RESULTS_PATH = RESULTS_PATH
    base.CHECKPOINT_PATH = CHECKPOINT_PATH
    base.CHECKPOINT_META_PATH = CHECKPOINT_META_PATH
    base.METHOD_ORDER = list(METHOD_ORDER)
    base.run_gradient_method = run_gradient_method
    base.run_one = run_one


_configure_base_module()


def main():
    print(
        "Setting2 GD methods: "
        + ", ".join(METHOD_ORDER),
        flush=True,
    )
    base.main()


if __name__ == "__main__":
    main()
