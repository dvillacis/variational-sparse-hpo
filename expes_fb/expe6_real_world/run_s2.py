"""Experiment 6 Setting 2 — Fully real-world classification benchmarks.

Three diagnostic datasets:

  mnist           0-vs-1 binary task from libsvmdata
  breast-cancer   binary LIBSVM dataset from libsvmdata
  leukemia        binary LIBSVM dataset from libsvmdata

Larger datasets such as ``rcv1``, ``rcv1.binary``, ``real-sim``, and
``news20.binary`` are still supported via ``get_dataset`` if needed later.

Three methods:
  scalar_cv      Scalar ℓ1 selected by LogisticRegressionCV.
                 Baseline with zero bilevel cost.
  sparseho_wl1   WeightedSparseLogReg + Implicit (original SparseHO)
                 + NormalizedSubgradient.
  ntrba_wl1      WeightedSparseLogReg + ImplicitVariational (DA policy)
                 + TrustRegion.

Metrics:  test F1, model sparsity (%), wall-clock time per outer iteration.

Usage
-----
    python run_s2.py
    python expes_fb/expe6_real_world/run_s2.py
    python expes_fb/expe6_real_world/run_s2.py --max-samples 5000
"""

import argparse
import sys
import time
import warnings
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from checkpointing import (
    load_dataframe_checkpoint,
    save_dataframe_checkpoint,
    completed_key_set,
)
from data_loaders import get_dataset

from sparse_ho.models import WeightedSparseLogReg
from sparse_ho.criterion import HeldOutLogistic
from sparse_ho.algo import Implicit, ImplicitVariational, select_biactive_self_consistent
from sparse_ho.optimizers import TrustRegion, NormalizedSubgradient
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from sparse_ho.algo.forward import compute_beta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# DATASETS = ['mnist', 'breast-cancer', 'leukemia', 'diabetes']
DATASETS = ['phishing', 'real-sim', 'news20.binary', 'rcv1.binary','mnist']
N_SEEDS  = 3
N_OUTER  = 60
INNER_TOL = 1e-4
INNER_MAX_ITER = 10000
INNER_TOL_TR = 1e-7

NBA_STEP_SIZE    = 0.1
TR_RADIUS0       = 0.1
BIACTIVE_TOL_REL = 1e-3

# scalar_cv settings
CV_FOLDS = 5
CV_N_ALPHAS = 30   # reduced from 100 for speed; enough for comparison

RESULTS_DIR  = Path(__file__).parent / 'results' / 'setting2'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'
CHECKPOINT_PATH = RESULTS_DIR / 'results_checkpoint.pkl'
CHECKPOINT_META_PATH = RESULTS_DIR / 'results_checkpoint_meta.json'

DATA_DIR = Path(__file__).parent / 'data'

LOG_OUTER_EVERY = 10
CV_HEARTBEAT_SECS = 30.0
METHOD_ORDER = ['scalar_cv', 'sparseho_wl1', 'ntrba_wl1', 'nba_wl1']
RUN_MAX_SAMPLES = None


def _log(msg, prefix=None):
    stamp = time.strftime('%H:%M:%S')
    if prefix:
        print(f"[{stamp}] [{prefix}] {msg}", flush=True)
    else:
        print(f"[{stamp}] {msg}", flush=True)


def _make_monitor_callback(prefix, every=LOG_OUTER_EVERY):
    state = {'k': 0}

    def _callback(obj, grad, mask, dense, alpha):
        state['k'] += 1
        k = state['k']
        if k == 1 or k % every == 0 or k == N_OUTER:
            grad_norm = float(np.linalg.norm(grad)) if grad is not None else np.nan
            _log(
                f"outer_iter={k:03d}/{N_OUTER} obj={obj:.6f} grad_norm={grad_norm:.3e}",
                prefix=prefix,
            )

    return _callback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiment 6 Setting 2."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional row subsample applied independently to each dataset "
            "before splitting. Defaults to the full dataset size."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=METHOD_ORDER,
        help=(
            "Subset of methods to run; the rest are skipped. Use to re-run "
            "only the band-dependent oracle (ntrba_wl1) while keeping the "
            "band-independent scalar_cv / sparseho_wl1 rows. Default: all."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to run. Default: all.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Redirect results to results/setting2_<tag>/ (avoids clobbering).",
    )
    return parser.parse_args()


def _run_with_heartbeat(func, prefix, label, interval=CV_HEARTBEAT_SECS):
    stop_event = threading.Event()
    start = time.time()

    def _heartbeat():
        while not stop_event.wait(interval):
            elapsed = time.time() - start
            _log(f"{label} still running elapsed={elapsed:.1f}s", prefix=prefix)

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    try:
        return func()
    finally:
        stop_event.set()
        thread.join(timeout=0.1)


def _make_cv_grid(n_alphas, dataset_name=None, X=None):
    """Return a scalar-CV grid adapted to the dataset regime.

    The original [1e-4, 1e4] grid spends a large amount of time in the
    weak-regularization tail on high-dimensional sparse datasets such as
    ``real-sim`` and ``news20.binary``. There, late-grid points become
    nearly dense and SAGA can take disproportionately long without changing
    the model-selection conclusion.
    """
    from scipy.sparse import issparse

    log10_min = -4.0
    log10_max = 4.0

    if X is not None and hasattr(X, "shape"):
        n_samples, n_features = X.shape
        is_high_dim_sparse = issparse(X) and n_features >= 5000
        if is_high_dim_sparse:
            log10_max = 1.5
        elif issparse(X) and n_features >= 1000:
            log10_max = 2.0
    elif dataset_name in {
        "real-sim", "news20.binary", "rcv1", "rcv1.binary", "rcv1_train.binary"
    }:
        log10_max = 1.5

    return np.logspace(log10_min, log10_max, int(n_alphas))


def _checkpoint_config():
    return {
        'datasets': list(DATASETS),
        'methods': list(METHOD_ORDER),
        'n_seeds': int(N_SEEDS),
        'n_outer': int(N_OUTER),
        'inner_tol': float(INNER_TOL),
        'inner_max_iter': int(INNER_MAX_ITER),
        'inner_tol_tr': float(INNER_TOL_TR),
        'cv_folds': int(CV_FOLDS),
        'cv_n_alphas': int(CV_N_ALPHAS),
        'max_samples': (
            None if RUN_MAX_SAMPLES is None else int(RUN_MAX_SAMPLES)
        ),
    }


def _load_checkpoint():
    return load_dataframe_checkpoint(
        CHECKPOINT_PATH, CHECKPOINT_META_PATH, _checkpoint_config(),
        log=_log,
    )


def _save_checkpoint(df: pd.DataFrame):
    save_dataframe_checkpoint(
        df, CHECKPOINT_PATH, CHECKPOINT_META_PATH, _checkpoint_config()
    )


def _completed_methods(df, dataset_name, seed):
    if df.empty:
        return set()
    subset = df[(df.dataset == dataset_name) & (df.seed == seed)]
    return set(subset.method.tolist()) if not subset.empty else set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_split(n_total, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    n_tr  = int(0.60 * n_total)
    n_val = int(0.20 * n_total)
    return idx[:n_tr], idx[n_tr:n_tr + n_val], idx[n_tr + n_val:]


def _preprocess_features_for_split(dataset_name, X, idx_train):
    """Apply Setting-2 feature preprocessing without train/val leakage."""
    from scipy.sparse import issparse

    X_proc = X
    notes = []

    # libsvmdata breast-cancer includes an ID-like first column with values
    # several orders of magnitude larger than the diagnostic features.
    if dataset_name == 'breast-cancer' and X_proc.shape[1] > 1:
        X_proc = X_proc[:, 1:]
        notes.append("dropped column 0 (ID-like feature)")

    if not issparse(X_proc):
        X_arr = np.asarray(X_proc, dtype=float)
        scaler = StandardScaler()
        scaler.fit(X_arr[idx_train])
        X_proc = np.asfortranarray(scaler.transform(X_arr))
        notes.append("standardized dense features using train split stats")
    else:
        X_proc = X_proc.tocsc()

    return X_proc, notes


def _solve_inner(model, X_tr, y_tr, log_alpha, debug_context=None, max_iter=None):
    from scipy.sparse import issparse
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    if debug_context is not None:
        model.inner_debug_context_ = debug_context
    if max_iter is None:
        max_iter = INNER_MAX_ITER
    mask, dense, _ = compute_beta(
        X_tr, y_tr, log_alpha, model, tol=INNER_TOL, compute_jac=False,
        max_iter=max_iter)
    m = X_tr.shape[1]
    beta = np.zeros(m)
    beta[mask] = dense
    return beta


def _test_f1(X_te, y_te, beta):
    from scipy.sparse import issparse
    if issparse(X_te):
        scores = np.asarray(X_te @ beta).ravel()
    else:
        scores = X_te @ beta
    y_pred = np.sign(scores)
    y_pred[y_pred == 0] = 1.0
    return float(f1_score(y_te, y_pred, average='binary', zero_division=0))


def _sparsity(beta, thr=1e-10):
    """Fraction of features that are active (>threshold)."""
    return float(np.mean(np.abs(beta) > thr)) * 100.0


def _summarize_inner_debug(records):
    if not records:
        return {}
    elapsed = np.array([rec["elapsed"] for rec in records], dtype=float)
    passes = np.array([rec["n_passes"] for rec in records], dtype=float)
    active_means = np.array([
        rec["mean_active_size"] for rec in records
        if np.isfinite(rec.get("mean_active_size", np.nan))
    ], dtype=float)
    slowest = records[int(np.argmax(elapsed))]
    stop_reasons = {}
    for rec in records:
        reason = rec.get("stop_reason", "unknown")
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    return {
        "n_calls": int(len(records)),
        "total_elapsed": float(elapsed.sum()),
        "mean_elapsed": float(elapsed.mean()),
        "max_elapsed": float(elapsed.max()),
        "total_passes": int(passes.sum()),
        "mean_passes": float(passes.mean()),
        "max_passes": int(passes.max()),
        "stop_reasons": stop_reasons,
        "used_dense_active_set": bool(any(
            rec.get("used_dense_active_set", False) for rec in records
        )),
        "mean_active_size": (
            float(active_means.mean()) if active_means.size else np.nan
        ),
        "max_active_size": int(max(
            rec.get("max_active_size", 0) for rec in records
        )),
        "slowest_call": {
            "call_index": int(slowest["call_index"]),
            "context": slowest.get("context"),
            "elapsed": float(slowest["elapsed"]),
            "n_passes": int(slowest["n_passes"]),
            "support_size": int(slowest["support_size"]),
            "stop_reason": slowest.get("stop_reason"),
            "full_passes": int(slowest.get("full_passes", 0)),
            "restricted_passes": int(slowest.get("restricted_passes", 0)),
            "mean_active_size": float(slowest.get("mean_active_size", np.nan)),
        },
    }


def _summarize_algo_debug(records):
    if not records:
        return {}
    biactive = np.array(
        [rec.get("biactive_size", np.nan) for rec in records], dtype=float
    )
    selected = np.array(
        [rec.get("selected_biactive_size", np.nan) for rec in records], dtype=float
    )
    support = np.array(
        [rec.get("selected_support_size", np.nan) for rec in records], dtype=float
    )
    return {
        "n_calls": int(len(records)),
        "mean_biactive_size": float(np.nanmean(biactive)),
        "max_biactive_size": int(np.nanmax(biactive)),
        "mean_selected_biactive_size": float(np.nanmean(selected)),
        "max_selected_biactive_size": int(np.nanmax(selected)),
        "mean_selected_support_size": float(np.nanmean(support)),
        "max_selected_support_size": int(np.nanmax(support)),
    }


def _alpha_max(X_tr, y_tr):
    from scipy.sparse import issparse
    Xty = np.abs(np.asarray(X_tr.T @ y_tr).ravel()) if issparse(X_tr) else np.abs(X_tr.T @ y_tr)
    return float(np.max(Xty)) / (2.0 * len(y_tr))


class _HeldOutLogisticWithMaxIter(HeldOutLogistic):
    """Setting-2 wrapper to override the inner-solver iteration cap."""

    def __init__(self, idx_train, idx_val, max_iter):
        super().__init__(idx_train, idx_val)
        self.max_iter = int(max_iter)

    def get_val_grad(
            self, model, X, y, log_alpha, compute_beta_grad, max_iter=10000,
            tol=1e-5, monitor=None):
        return super().get_val_grad(
            model, X, y, log_alpha, compute_beta_grad,
            max_iter=self.max_iter, tol=tol, monitor=monitor)


# ---------------------------------------------------------------------------
# Method: scalar_cv
# ---------------------------------------------------------------------------

def run_scalar_cv(X, y, idx_train, idx_val, idx_test, m, log_prefix=None,
                  dataset_name=None):
    """Logistic regression with cross-validated global penalty."""
    from scipy.sparse import issparse

    X_tr_val = X[np.concatenate([idx_train, idx_val])]
    y_tr_val  = y[np.concatenate([idx_train, idx_val])]
    X_te      = X[idx_test]
    y_te      = y[idx_test]
    if issparse(X_tr_val):
        X_tr_val = X_tr_val.tocsc()
    if issparse(X_te):
        X_te = X_te.tocsc()

    t0 = time.time()
    n_trainval = len(idx_train) + len(idx_val)
    cs_grid = _make_cv_grid(CV_N_ALPHAS, dataset_name=dataset_name, X=X_tr_val)

    if issparse(X_tr_val):
        nnz = int(X_tr_val.nnz)
        density = nnz / (X_tr_val.shape[0] * X_tr_val.shape[1])
        _log(
            f"scalar_cv data n={n_trainval:,} m={m:,} nnz={nnz:,} "
            f"density={density:.3e} approx_fits={CV_FOLDS * len(cs_grid)} "
            f"C_range=[{cs_grid[0]:.1e}, {cs_grid[-1]:.1e}]",
            prefix=log_prefix,
        )
    else:
        _log(
            f"scalar_cv data n={n_trainval:,} m={m:,} "
            f"approx_fits={CV_FOLDS * len(cs_grid)} "
            f"C_range=[{cs_grid[0]:.1e}, {cs_grid[-1]:.1e}]",
            prefix=log_prefix,
        )
    _log(
        f"starting scalar_cv with cv={CV_FOLDS} Cs={len(cs_grid)}",
        prefix=log_prefix,
    )
    y_01 = ((y_tr_val + 1) / 2).astype(int)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=False)
    mean_scores = []
    best_score = -np.inf
    best_c = None

    def _eval_grid():
        nonlocal best_score, best_c
        for i, c in enumerate(cs_grid, start=1):
            fold_scores = []
            for tr_idx, va_idx in cv.split(X_tr_val, y_01):
                clf = LogisticRegression(
                    C=float(c),
                    solver='saga',
                    penalty='l1',
                    max_iter=300,
                    tol=1e-4,
                    n_jobs=1,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    clf.fit(X_tr_val[tr_idx], y_01[tr_idx])
                fold_scores.append(float(clf.score(X_tr_val[va_idx], y_01[va_idx])))
            mean_score = float(np.mean(fold_scores))
            mean_scores.append(mean_score)
            if mean_score > best_score:
                best_score = mean_score
                best_c = float(c)
            if i == 1 or i % 10 == 0 or i == len(cs_grid):
                _log(
                    f"cv_point={i:03d}/{len(cs_grid)} C={float(c):.3e} "
                    f"mean_acc={mean_score:.4f} best_C={best_c:.3e} "
                    f"best_acc={best_score:.4f}",
                    prefix=log_prefix,
                )

    _eval_grid()

    clf = LogisticRegression(
        C=best_c,
        solver='saga',
        penalty='l1',
        max_iter=300,
        tol=1e-4,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf.fit(X_tr_val, y_01)
    elapsed = time.time() - t0
    _log(
        f"finished scalar_cv in {elapsed:.1f}s best_C={best_c:.3e} "
        f"best_cv_acc={best_score:.4f}",
        prefix=log_prefix,
    )

    beta = clf.coef_.ravel()
    if issparse(X_te):
        scores = np.asarray(X_te @ beta).ravel() + clf.intercept_[0]
    else:
        scores = X_te @ beta + clf.intercept_[0]
    y_pred = np.sign(scores)
    y_pred[y_pred == 0] = 1.0
    test_f1 = float(f1_score(y_te, y_pred, average='binary', zero_division=0))

    return dict(
        method='scalar_cv',
        val_objs=[], hidden_grad_norms=[],
        best_val=np.nan,
        beta_final=beta,
        test_f1=test_f1,
        sparsity=_sparsity(beta),
        elapsed=elapsed,
        t_per_iter=elapsed / CV_FOLDS,
        n_iter=CV_FOLDS,
        termination='cv',
    )


# ---------------------------------------------------------------------------
# Method: gradient-based wl1 methods
# ---------------------------------------------------------------------------

def run_gradient_method(method_name, X, y, idx_train, idx_val, idx_test,
                        log_alpha0, alpha_l2, m, log_prefix=None,
                        keep_debug=False):
    from scipy.sparse import issparse
    if issparse(X):
        X = X.tocsc()
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_te, y_te = X[idx_test], y[idx_test]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    if issparse(X_te):
        X_te = X_te.tocsc()

    if method_name == 'sparseho_wl1':
        algo = Implicit()
        optimizer = NormalizedSubgradient(
            n_outer=N_OUTER, step_size=NBA_STEP_SIZE, tol=INNER_TOL_TR)
    elif method_name == 'nba_wl1':
        # variational oracle + normalized-subgradient optimizer: isolates the
        # oracle cost from the trust-region's extra per-iteration trial solve.
        algo = ImplicitVariational(
            policy=select_biactive_self_consistent,
            biactive_tol_rel=BIACTIVE_TOL_REL)
        optimizer = NormalizedSubgradient(
            n_outer=N_OUTER, step_size=NBA_STEP_SIZE, tol=INNER_TOL_TR)
    else:  # ntrba_wl1
        algo = ImplicitVariational(
            policy=select_biactive_self_consistent,
            biactive_tol_rel=BIACTIVE_TOL_REL)
        optimizer = TrustRegion(
            n_outer=N_OUTER, radius0=TR_RADIUS0, tol=INNER_TOL_TR)

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    if keep_debug:
        model.inner_debug_records_ = []
        model.inner_debug_context_ = f"{method_name}:outer"
    criterion = _HeldOutLogisticWithMaxIter(
        idx_train, idx_val, max_iter=INNER_MAX_ITER)
    algo_debug_records = []
    log_callback = _make_monitor_callback(log_prefix)

    def _callback(obj, grad, mask, dense, alpha):
        if keep_debug and hasattr(algo, "last_run_info") and algo.last_run_info:
            algo_debug_records.append(dict(algo.last_run_info))
        log_callback(obj, grad, mask, dense, alpha)

    monitor = Monitor(callback=_callback)

    _log(
        f"starting {method_name} with n_outer={N_OUTER} alpha_l2={alpha_l2:.3e} "
        f"inner_max_iter={INNER_MAX_ITER}",
        prefix=log_prefix,
    )
    t0 = time.time()
    grad_search(algo, criterion, model, optimizer, X, y,
                np.exp(log_alpha0), monitor)
    elapsed = time.time() - t0

    if monitor.alphas:
        best_iter = int(np.argmin(monitor.objs))
        alpha_final = np.array(monitor.alphas[best_iter])
    else:
        alpha_final = np.exp(log_alpha0)
    log_alpha_final = np.log(np.maximum(alpha_final, 1e-300))
    beta_final = _solve_inner(
        model, X_tr, y_tr, log_alpha_final,
        debug_context=f"{method_name}:refit" if keep_debug else None,
        max_iter=INNER_MAX_ITER,
    )
    best_val = float(min(monitor.objs)) if monitor.objs else np.nan
    inner_debug_records = list(getattr(model, "inner_debug_records_", []))
    inner_debug_summary = _summarize_inner_debug(inner_debug_records)
    algo_debug_summary = _summarize_algo_debug(algo_debug_records)
    _log(
        f"finished {method_name} in {elapsed:.1f}s "
        f"n_iter={len(monitor.objs)} best_val={best_val:.6f} "
        f"termination={getattr(optimizer, 'termination_reason_', None)}",
        prefix=log_prefix,
    )

    return dict(
        method=method_name,
        val_objs=list(monitor.objs),
        hidden_grad_norms=[],   # no ground truth in Setting 2
        best_val=best_val,
        beta_final=beta_final,
        test_f1=_test_f1(X_te, y_te, beta_final),
        sparsity=_sparsity(beta_final),
        elapsed=elapsed,
        t_per_iter=elapsed / max(len(monitor.objs), 1),
        n_iter=len(monitor.objs),
        termination=getattr(optimizer, 'termination_reason_', None),
        inner_debug_records=inner_debug_records,
        inner_debug_summary=inner_debug_summary,
        algo_debug=dict(getattr(algo, 'last_run_info', {})),
        algo_debug_records=algo_debug_records,
        algo_debug_summary=algo_debug_summary,
    )


# ---------------------------------------------------------------------------
# Per-dataset, per-seed runner
# ---------------------------------------------------------------------------

def run_one(dataset_name, X, y, seed, keep_debug=False, skip_methods=None,
            row_callback=None):
    run_prefix = f"s2 {dataset_name} seed={seed}"
    skip_methods = set(skip_methods or [])
    n_total = X.shape[0]
    idx_train, idx_val, idx_test = _make_split(n_total, seed)
    X, prep_notes = _preprocess_features_for_split(dataset_name, X, idx_train)
    m = X.shape[1]
    _log(
        f"split sizes train/val/test={len(idx_train)}/{len(idx_val)}/{len(idx_test)}",
        prefix=run_prefix,
    )
    for note in prep_notes:
        _log(note, prefix=run_prefix)

    X_tr, y_tr = X[idx_train], y[idx_train]
    from scipy.sparse import issparse
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    n_tr = len(idx_train)
    alpha_l2 = 1.0 / n_tr
    a_max = _alpha_max(X_tr, y_tr)
    _log(f"alpha_max={a_max:.3e} alpha_l2={alpha_l2:.3e}", prefix=run_prefix)

    # Initialize log_alpha at 10% of alpha_max (gives moderate sparsity)
    log_alpha0 = np.log(0.10 * a_max) * np.ones(m)

    rows = []
    for method_name in METHOD_ORDER:
        if method_name in skip_methods:
            _log(f"skipping completed method {method_name}", prefix=run_prefix)
            continue

        if method_name == 'scalar_cv':
            res = run_scalar_cv(
                X, y, idx_train, idx_val, idx_test, m,
                log_prefix=f"{run_prefix} scalar_cv",
                dataset_name=dataset_name,
            )
        else:
            res = run_gradient_method(
                method_name, X, y, idx_train, idx_val, idx_test,
                log_alpha0, alpha_l2, m, log_prefix=f"{run_prefix} {method_name}",
                keep_debug=keep_debug)

        row = dict(
            dataset=dataset_name, seed=seed,
            method=res['method'],
            test_f1=res['test_f1'],
            sparsity=res['sparsity'],
            elapsed=res['elapsed'],
            t_per_iter=res['t_per_iter'],
            n_iter=res['n_iter'],
            best_val=res.get('best_val', np.nan),
            termination=res['termination'],
        )
        if keep_debug:
            row.update(
                val_objs=list(res.get('val_objs', [])),
                beta_final=np.array(res.get('beta_final', np.array([])), copy=True),
                alpha_max=float(a_max),
                alpha_l2=float(alpha_l2),
                n_train=len(idx_train),
                n_val=len(idx_val),
                n_test=len(idx_test),
                inner_debug_records=list(res.get('inner_debug_records', [])),
                inner_debug_summary=dict(res.get('inner_debug_summary', {})),
                algo_debug=dict(res.get('algo_debug', {})),
                algo_debug_records=list(res.get('algo_debug_records', [])),
                algo_debug_summary=dict(res.get('algo_debug_summary', {})),
            )
        rows.append(row)
        if row_callback is not None:
            row_callback(dict(row))
    _log("completed all methods", prefix=run_prefix)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    global RUN_MAX_SAMPLES, RESULTS_DIR, RESULTS_PATH, CHECKPOINT_PATH
    global CHECKPOINT_META_PATH
    RUN_MAX_SAMPLES = args.max_samples
    if args.tag:
        RESULTS_DIR = RESULTS_DIR.parent / f'setting2_{args.tag}'
        RESULTS_PATH = RESULTS_DIR / 'results.pkl'
        CHECKPOINT_PATH = RESULTS_DIR / 'results_checkpoint.pkl'
        CHECKPOINT_META_PATH = RESULTS_DIR / 'results_checkpoint_meta.json'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    skip_base = (set(METHOD_ORDER) - set(args.methods)) if args.methods else set()
    datasets = args.datasets if args.datasets else DATASETS

    checkpoint_df = _load_checkpoint()
    all_rows = checkpoint_df.to_dict(orient='records') if not checkpoint_df.empty else []

    for dname in datasets:
        _log("=" * 60)
        _log(f"dataset={dname}")
        try:
            X, y = get_dataset(dname, DATA_DIR)
        except FileNotFoundError as e:
            _log(f"skip: {e}", prefix=dname)
            continue
        from scipy.sparse import issparse
        if issparse(X):
            X = X.tocsc()

        if args.max_samples is not None and X.shape[0] > args.max_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], args.max_samples, replace=False)
            X = X[idx]
            y = y[idx]
            _log(f"subsampled to n={X.shape[0]:,}", prefix=dname)

        n, m = X.shape
        pos_rate = float(np.mean(y == 1))
        _log(f"n={n:,} m={m:,} pos_rate={pos_rate:.2f}", prefix=dname)

        pending = []
        checkpoint_view = pd.DataFrame(all_rows)
        done_seed_keys = completed_key_set(checkpoint_view, ['dataset', 'seed'])
        for seed in range(N_SEEDS):
            if (dname, seed) in done_seed_keys:
                _log("skipping completed seed from checkpoint", prefix=f"{dname} seed={seed}")
                continue
            done_methods = _completed_methods(checkpoint_view, dname, seed)
            pending.append((seed, set(done_methods) | skip_base))

        if not pending:
            continue

        from joblib import Parallel, delayed

        results_iter = Parallel(
            n_jobs=min(N_SEEDS, len(pending)),
            backend='loky',
            verbose=0,
            return_as='generator_unordered',
        )(
            delayed(run_one)(
                dname, X, y, seed,
                keep_debug=False,
                skip_methods=done_methods,
            )
            for seed, done_methods in pending
        )

        for rows in results_iter:
            if not rows:
                continue
            all_rows.extend(rows)
            df_ckpt = pd.DataFrame(all_rows).sort_values(
                ['dataset', 'seed', 'method']
            ).reset_index(drop=True)
            _save_checkpoint(df_ckpt)
            seed = rows[0]['seed']
            _log("checkpointed completed seed", prefix=f"{dname} seed={seed}")
            for r in rows:
                _log(
                    f"{r['method']:16s} F1={r['test_f1']:.3f} "
                    f"sparsity={r['sparsity']:.1f}% t={r['elapsed']:.1f}s "
                    f"termination={r['termination']}",
                    prefix=f"{dname} seed={seed}",
                )

    if not all_rows:
        print("No results to save.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(['dataset', 'seed', 'method']).reset_index(drop=True)
    df.to_pickle(RESULTS_PATH)
    _save_checkpoint(df)
    print(f"\nSaved {len(df)} rows to {RESULTS_PATH}")

    summary = df.groupby(['dataset', 'method'])[
        ['test_f1', 'sparsity', 't_per_iter']
    ].agg(['mean', 'std'])
    print(summary.to_string())


if __name__ == '__main__':
    main()
