"""Experiment 6 Setting 1 — Semi-synthetic sparse-text benchmark.

Purpose
-------
Bridge the gap between Experiments 3/4 (fully synthetic) and Setting 2
(fully real-world): use real sparse classification features to form the
correlation structure, but inject a known sparse signal so that support
recovery can be measured.

Feature layout
--------------
  [0 : N_EASY]                         Easy (synthetic Gaussian, always active)
  [N_EASY : N_EASY+N_PAIRS]            Distractor (real rcv1.binary, correlated w/ hidden)
  [N_EASY+N_PAIRS : N_EASY+2*N_PAIRS]  Hidden (real rcv1.binary, biactive at init)
  [N_EASY+2*N_PAIRS : m]               Background (real rcv1.binary noise)

Signal model
------------
  logit_i = X_i^T β_true + σ_noise · ε_i,   ε_i ~ N(0,1)
  y_i     = sign(logit_i)                    ∈ {-1, +1}

β_true is nonzero on easy and hidden features only.  Distractor and
background features have β_true = 0.

Biactive calibration
---------------------
Same two-pass procedure as ``make_degenerate_dataset``, adapted for the
logistic smooth gradient:

  Pass 1  Solve with large sentinel penalty on hidden features → β_S
          (hidden forced to zero).
  Pass 2  g_hid_j = get_grad_smooth(X_tr, y_tr, β_S)[hidden_j].
          Set exp(x0_hid_j) = |g_hid_j| * (1 + CALIBRATION_SLACK).

Methods
-------
  sparseho_scalar  Scalar ℓ1 selected by grid search over 100 λ values.
  sparseho_wl1     Implicit (original SparseHO) + NormalizedSubgradient.
  ntrba_wl1        ImplicitVariational (DA policy)   + TrustRegion.

Usage
-----
    python run_s1.py            # from within this directory, or
    python expes_fb/expe6_real_world/run_s1.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, issparse, hstack

sys.path.insert(0, str(Path(__file__).parent))
from checkpointing import (
    load_dataframe_checkpoint,
    save_dataframe_checkpoint,
    completed_key_set,
)
from data_loaders import (
    get_dataset, top_variance_features, correlation_matrix,
    top_correlated_pairs,
)

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

N_BACKGROUND = 500     # cap on top-variance real features used to build the benchmark
N_EASY       = 10      # injected Gaussian easy features
N_PAIRS      = 15      # distractor/hidden pairs from rcv1.binary
BETA_STD     = 1.0     # ±BETA_STD for easy/hidden coefficients
NOISE_STD    = 0.20    # label noise σ

N_SEEDS  = 5
N_OUTER  = 60
INNER_TOL = 1e-4

NBA_STEP_SIZE    = 0.1
TR_RADIUS0       = 0.1
BIACTIVE_TOL_REL = 0.10
CALIBRATION_SLACK = 0.05
N_GRID           = 100    # scalar lambda grid points

# Penalty fractions relative to alpha_max
EASY_FRAC = 0.05          # exp(x_easy) = EASY_FRAC * alpha_max  → active
DIST_FRAC = 0.05          # exp(x_dist) = DIST_FRAC * alpha_max  → active
BG_FRAC   = 0.70          # exp(x_bg)   = BG_FRAC   * alpha_max  → inactive

N_MAX_SAMPLES = None     # subsample large datasets for speed (use None for all)
DATASETS = ['rcv1.binary', 'real-sim', 'w8a', 'news20.binary']

RESULTS_DIR  = Path(__file__).parent / 'results' / 'setting1'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'
CHECKPOINT_PATH = RESULTS_DIR / 'results_checkpoint.pkl'
CHECKPOINT_META_PATH = RESULTS_DIR / 'results_checkpoint_meta.json'

DATA_DIR = Path(__file__).parent / 'data'

LOG_GRID_EVERY = 20
LOG_OUTER_EVERY = 10


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


def _checkpoint_config():
    return {
        'datasets': list(DATASETS),
        'n_seeds': int(N_SEEDS),
        'n_outer': int(N_OUTER),
        'inner_tol': float(INNER_TOL),
        'n_grid': int(N_GRID),
        'n_background': int(N_BACKGROUND),
        'n_easy': int(N_EASY),
        'n_pairs': int(N_PAIRS),
        'beta_std': float(BETA_STD),
        'noise_std': float(NOISE_STD),
        'n_max_samples': (
            None if N_MAX_SAMPLES is None else int(N_MAX_SAMPLES)
        ),
    }


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_dataset(X_base, rng):
    """Build the semi-synthetic feature matrix and beta_true.

    Returns
    -------
    X_all : csc_matrix, shape (n_total, m)
    beta_true : ndarray, shape (m,)
    groups : dict  (maps 'easy','distractor','hidden','background' to index arrays)
    """
    n_total, n_features = X_base.shape
    n_background = min(N_BACKGROUND, n_features)
    if n_background < 2 * N_PAIRS:
        raise ValueError(
            f"Need at least {2 * N_PAIRS} usable real features to form "
            f"{N_PAIRS} distractor/hidden pairs, got {n_background}."
        )

    # --- select background features ---
    bg_feat_idx = top_variance_features(X_base, n_background)
    X_bg = X_base[:, bg_feat_idx]
    if issparse(X_bg):
        X_bg = X_bg.tocsc()
    else:
        X_bg = csc_matrix(X_bg)

    # --- find correlated pairs ---
    corr = correlation_matrix(X_bg)
    pairs = top_correlated_pairs(corr, N_PAIRS)

    dist_local = [p[0] for p in pairs]   # local indices into X_bg
    hid_local  = [p[1] for p in pairs]

    # remaining columns → background noise
    used = set(dist_local) | set(hid_local)
    bg_local = [i for i in range(n_background) if i not in used]

    # --- inject easy features (synthetic Gaussian) ---
    X_easy = csc_matrix(rng.standard_normal((n_total, N_EASY)))

    # --- assemble X ---
    X_dist = X_bg[:, dist_local]
    X_hid  = X_bg[:, hid_local]
    X_noise = X_bg[:, bg_local]

    X_all = hstack([X_easy, X_dist, X_hid, X_noise], format='csc')

    # --- column index groups ---
    i_easy  = np.arange(N_EASY)
    i_dist  = np.arange(N_EASY, N_EASY + N_PAIRS)
    i_hid   = np.arange(N_EASY + N_PAIRS, N_EASY + 2 * N_PAIRS)
    i_noise = np.arange(N_EASY + 2 * N_PAIRS, X_all.shape[1])
    groups = dict(easy=i_easy, distractor=i_dist, hidden=i_hid, background=i_noise)

    # --- ground truth ---
    m = X_all.shape[1]
    beta_true = np.zeros(m)
    signs_easy = rng.choice([-1.0, 1.0], N_EASY)
    signs_hid  = rng.choice([-1.0, 1.0], N_PAIRS)
    beta_true[i_easy] = signs_easy * BETA_STD
    beta_true[i_hid]  = signs_hid  * BETA_STD

    return X_all, beta_true, groups


def _make_split(n_total, rng):
    """Return (idx_train, idx_val, idx_test) for a 60/20/20 split."""
    idx = rng.permutation(n_total)
    n_tr  = int(0.60 * n_total)
    n_val = int(0.20 * n_total)
    return idx[:n_tr], idx[n_tr:n_tr + n_val], idx[n_tr + n_val:]


def _make_labels(X, beta_true, rng):
    """Generate binary labels y = sign(X β + noise)."""
    logit = np.asarray(X @ beta_true).ravel()
    logit += NOISE_STD * rng.standard_normal(X.shape[0])
    y = np.sign(logit)
    y[y == 0] = 1.0
    return y


# ---------------------------------------------------------------------------
# Biactive calibration (two-pass, logistic)
# ---------------------------------------------------------------------------

def _calibrate_biactive(X_tr, y_tr, groups, alpha_max, alpha_l2, rng):
    """Calibrate initial log_alpha so hidden features are biactive.

    Returns log_alpha0 : ndarray, shape (m,)
    """
    m = X_tr.shape[1]
    model = WeightedSparseLogReg(alpha_l2=alpha_l2)

    # Base initialization
    log_alpha = np.zeros(m)
    log_alpha[groups['easy']]       = np.log(EASY_FRAC * alpha_max)
    log_alpha[groups['distractor']] = np.log(DIST_FRAC * alpha_max)
    log_alpha[groups['hidden']]     = np.log(alpha_max)   # sentinel (large)
    log_alpha[groups['background']] = np.log(BG_FRAC   * alpha_max)

    # Pass 1: solve with large hidden penalty
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask, dense, _ = compute_beta(
            X_tr, y_tr, log_alpha, model, tol=INNER_TOL, compute_jac=False)
    beta_S = np.zeros(m)
    beta_S[mask] = dense

    # Pass 2: compute logistic gradient for hidden features at beta_S
    g_full = model.get_grad_smooth(X_tr, y_tr, beta_S)   # shape (m,)
    g_hid  = g_full[groups['hidden']]

    exp_x0_hid = np.abs(g_hid) * (1.0 + CALIBRATION_SLACK)
    near_zero = exp_x0_hid < 1e-12
    if np.any(near_zero):
        warnings.warn(
            f"Setting 1: {near_zero.sum()} hidden features have near-zero "
            "calibrated penalty; replaced with sentinel.",
            RuntimeWarning,
        )
        exp_x0_hid[near_zero] = alpha_max * 0.5

    log_alpha[groups['hidden']] = np.log(exp_x0_hid)
    return log_alpha


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _solve_inner(model, X_tr, y_tr, log_alpha):
    mask, dense, _ = compute_beta(
        X_tr, y_tr, log_alpha, model, tol=INNER_TOL, compute_jac=False)
    beta = np.zeros(X_tr.shape[1])
    beta[mask] = dense
    return beta


def _f1(beta, beta_true, thr=1e-10):
    pred = np.abs(beta) > thr
    true = beta_true != 0
    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _hidden_recall(beta, groups, thr=1e-10):
    return float(np.mean(np.abs(beta[groups['hidden']]) > thr))


def _active_feature_percent(beta, thr=1e-10):
    return 100.0 * float(np.mean(np.abs(beta) > thr))


def _test_logloss(X_te, y_te, beta):
    scores = np.asarray(X_te @ beta).ravel() if issparse(X_te) else X_te @ beta
    return float(np.mean(np.log1p(np.exp(-y_te * scores))))


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_sparseho_scalar(X, y, beta_true, groups, idx_train, idx_val, idx_test,
                        alpha_l2, alpha_max, m, log_prefix=None):
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_te, y_te = X[idx_test], y[idx_test]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    if issparse(X_val):
        X_val = X_val.tocsc()
    if issparse(X_te):
        X_te = X_te.tocsc()

    grid = np.logspace(
        np.log10(alpha_max * 1e-3), np.log10(alpha_max * 0.9), N_GRID
    )[::-1]
    model = WeightedSparseLogReg(alpha_l2=alpha_l2)

    best_val, best_beta = np.inf, np.zeros(m)
    val_objs = []
    t0 = time.time()
    _log(f"starting scalar grid search with {len(grid)} lambdas", prefix=log_prefix)
    for i, lam in enumerate(grid, start=1):
        log_alpha = np.log(lam) * np.ones(m)
        beta = _solve_inner(model, X_tr, y_tr, log_alpha)
        if issparse(X_val):
            scores = np.asarray(X_val @ beta).ravel()
        else:
            scores = X_val @ beta
        val_ll = float(np.mean(np.log1p(np.exp(-y_val * scores))))
        val_objs.append(val_ll)
        if val_ll < best_val:
            best_val = val_ll
            best_beta = beta.copy()
        if i == 1 or i % LOG_GRID_EVERY == 0 or i == len(grid):
            _log(
                f"grid_iter={i:03d}/{len(grid)} lambda={lam:.3e} "
                f"val_logloss={val_ll:.6f} best={best_val:.6f}",
                prefix=log_prefix,
            )
    elapsed = time.time() - t0
    _log(
        f"finished scalar grid search in {elapsed:.1f}s best_val={best_val:.6f}",
        prefix=log_prefix,
    )

    return dict(
        method='sparseho_scalar',
        val_objs=val_objs, hidden_grad_norms=[], mean_hidden_alpha_traj=[],
        best_val=float(best_val), beta_final=best_beta,
        test_logloss=_test_logloss(X_te, y_te, best_beta),
        elapsed=elapsed, t_per_iter=elapsed / N_GRID, n_iter=N_GRID,
        hidden_grad_norm_0=np.nan, termination='grid_search',
    )


def run_gradient_method(method_name, X, y, groups, idx_train, idx_val, idx_test,
                        log_alpha0, alpha_l2, m, log_prefix=None):
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_te, y_te = X[idx_test], y[idx_test]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    if issparse(X_te):
        X_te = X_te.tocsc()
    hid = groups['hidden']

    if method_name == 'sparseho_wl1':
        algo = Implicit()
        optimizer = NormalizedSubgradient(
            n_outer=N_OUTER, step_size=NBA_STEP_SIZE, tol=INNER_TOL)
    elif method_name == 'null_tr_wl1':
        # oracle-isolated control: null (support-restricted) selection with the
        # SAME trust-region optimizer as NTRBA, so null_tr vs ntrba isolates the
        # SC oracle's contribution from the optimizer (answers the confounding
        # concern that the Setting-1 gap mixes oracle and optimizer effects).
        algo = ImplicitVariational(
            policy=None, biactive_tol_rel=BIACTIVE_TOL_REL)
        optimizer = TrustRegion(
            n_outer=N_OUTER, radius0=TR_RADIUS0, tol=INNER_TOL)
    else:  # ntrba_wl1
        algo = ImplicitVariational(
            policy=select_biactive_self_consistent,
            biactive_tol_rel=BIACTIVE_TOL_REL)
        optimizer = TrustRegion(
            n_outer=N_OUTER, radius0=TR_RADIUS0, tol=INNER_TOL)

    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    criterion = HeldOutLogistic(idx_train, idx_val)
    monitor = Monitor(callback=_make_monitor_callback(log_prefix))

    _log(
        f"starting {method_name} with n_outer={N_OUTER} alpha_l2={alpha_l2:.3e}",
        prefix=log_prefix,
    )
    t0 = time.time()
    grad_search(algo, criterion, model, optimizer, X, y,
                np.exp(log_alpha0), monitor)
    elapsed = time.time() - t0

    hidden_grad_norms, mean_hidden_alpha_traj = [], []
    for g, a in zip(monitor.grads, monitor.alphas):
        if g is not None and hasattr(g, '__len__') and len(g) > max(hid):
            hidden_grad_norms.append(float(np.linalg.norm(g[hid])))
        else:
            hidden_grad_norms.append(np.nan)
        if a is not None and hasattr(a, '__len__') and len(a) > max(hid):
            mean_hidden_alpha_traj.append(float(np.mean(a[hid])))
        else:
            mean_hidden_alpha_traj.append(np.nan)

    hidden_grad_norm_0 = (
        hidden_grad_norms[0] if hidden_grad_norms
        and np.isfinite(hidden_grad_norms[0]) else np.nan
    )

    if monitor.alphas:
        best_iter = int(np.argmin(monitor.objs))
        alpha_final = np.array(monitor.alphas[best_iter])
    else:
        alpha_final = np.exp(log_alpha0)
    log_alpha_final = np.log(np.maximum(alpha_final, 1e-300))
    beta_final = _solve_inner(model, X_tr, y_tr, log_alpha_final)
    best_val = float(min(monitor.objs)) if monitor.objs else np.nan
    _log(
        f"finished {method_name} in {elapsed:.1f}s "
        f"n_iter={len(monitor.objs)} best_val={best_val:.6f} "
        f"termination={getattr(optimizer, 'termination_reason_', None)}",
        prefix=log_prefix,
    )

    return dict(
        method=method_name,
        val_objs=list(monitor.objs),
        hidden_grad_norms=hidden_grad_norms,
        mean_hidden_alpha_traj=mean_hidden_alpha_traj,
        best_val=best_val,
        beta_final=beta_final,
        test_logloss=_test_logloss(X_te, y_te, beta_final),
        elapsed=elapsed,
        t_per_iter=elapsed / max(len(monitor.objs), 1),
        n_iter=len(monitor.objs),
        hidden_grad_norm_0=float(hidden_grad_norm_0),
        termination=getattr(optimizer, 'termination_reason_', None),
    )


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def run_one(X_all, y_all, beta_true, groups, seed):
    seed_prefix = f"s1 seed={seed}"
    rng_split = np.random.default_rng(seed + 100)
    n_total = X_all.shape[0]
    m = X_all.shape[1]
    _log(f"starting run with n={n_total} m={m}", prefix=seed_prefix)
    idx_train, idx_val, idx_test = _make_split(n_total, rng_split)

    X_tr = X_all[idx_train]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    y_tr = y_all[idx_train]
    n_tr = len(idx_train)
    alpha_l2 = 1.0 / n_tr

    # lambda_max for logistic
    Xty = np.abs(X_tr.T @ y_tr) if not issparse(X_tr) else np.abs(
        np.asarray(X_tr.T @ y_tr).ravel())
    alpha_max = float(np.max(Xty)) / (2.0 * n_tr)
    _log(
        f"split sizes train/val/test={len(idx_train)}/{len(idx_val)}/{len(idx_test)} "
        f"alpha_max={alpha_max:.3e}",
        prefix=seed_prefix,
    )

    # Calibrate biactive initialization
    rng_cal = np.random.default_rng(seed + 200)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _log("calibrating biactive initialization", prefix=seed_prefix)
        log_alpha0 = _calibrate_biactive(
            X_tr, y_tr, groups, alpha_max, alpha_l2, rng_cal)
    _log("finished calibration", prefix=seed_prefix)

    # Run methods
    scalar_res = run_sparseho_scalar(
        X_all, y_all, beta_true, groups,
        idx_train, idx_val, idx_test, alpha_l2, alpha_max, m,
        log_prefix=f"{seed_prefix} scalar")
    wl1_res = run_gradient_method(
        'sparseho_wl1', X_all, y_all, groups,
        idx_train, idx_val, idx_test, log_alpha0, alpha_l2, m,
        log_prefix=f"{seed_prefix} sparseho_wl1")
    null_tr_res = run_gradient_method(
        'null_tr_wl1', X_all, y_all, groups,
        idx_train, idx_val, idx_test, log_alpha0, alpha_l2, m,
        log_prefix=f"{seed_prefix} null_tr_wl1")
    ntrba_res = run_gradient_method(
        'ntrba_wl1', X_all, y_all, groups,
        idx_train, idx_val, idx_test, log_alpha0, alpha_l2, m,
        log_prefix=f"{seed_prefix} ntrba_wl1")

    ref_val = ntrba_res['best_val']

    rows = []
    for res in [scalar_res, wl1_res, null_tr_res, ntrba_res]:
        bf = res['beta_final']
        rows.append(dict(
            dataset=None,
            seed=seed,
            method=res['method'],
            val_objs=res['val_objs'],
            hidden_grad_norms=res['hidden_grad_norms'],
            mean_hidden_alpha_traj=res['mean_hidden_alpha_traj'],
            val_loss_gap=float(res['best_val'] - ref_val),
            best_val_loss=float(res['best_val']),
            hidden_grad_norm_0=res['hidden_grad_norm_0'],
            hidden_recall=_hidden_recall(bf, groups),
            active_features_pct=_active_feature_percent(bf),
            f1=_f1(bf, beta_true),
            test_logloss=res['test_logloss'],
            elapsed=res['elapsed'],
            t_per_iter=res['t_per_iter'],
            n_iter=res['n_iter'],
            termination=res['termination'],
        ))
    _log("completed all methods", prefix=seed_prefix)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    from joblib import Parallel, delayed

    checkpoint_df = load_dataframe_checkpoint(
        CHECKPOINT_PATH, CHECKPOINT_META_PATH, _checkpoint_config(),
        log=_log,
    )
    rows = checkpoint_df.to_dict(orient='records') if not checkpoint_df.empty else []
    for dataset in DATASETS:
        _log("=" * 60)
        _log(f"dataset={dataset}")
        try:
            X_full, _ = get_dataset(dataset, DATA_DIR)
        except FileNotFoundError as exc:
            _log(f"[{dataset}] skip: {exc}")
            continue

        if N_MAX_SAMPLES is not None and X_full.shape[0] > N_MAX_SAMPLES:
            rng_sub = np.random.default_rng(0)
            idx_sub = rng_sub.choice(X_full.shape[0], N_MAX_SAMPLES, replace=False)
            X_full = X_full[idx_sub]
            _log(f"[{dataset}] subsampled to n={X_full.shape[0]}")

        rng_build = np.random.default_rng(0)
        X_all, beta_true, groups = _build_dataset(X_full, rng_build)

        rng_label = np.random.default_rng(1)
        y_all = _make_labels(X_all, beta_true, rng_label)

        m = X_all.shape[1]
        _log(
            f"[{dataset}] feature layout easy={len(groups['easy'])} "
            f"dist={len(groups['distractor'])} hidden={len(groups['hidden'])} "
            f"bg={len(groups['background'])} total_m={m}"
        )

        done_keys = completed_key_set(pd.DataFrame(rows), ['dataset', 'seed'])
        pending_seeds = [seed for seed in range(N_SEEDS) if (dataset, seed) not in done_keys]
        for seed in range(N_SEEDS):
            if (dataset, seed) in done_keys:
                _log("skipping completed seed from checkpoint", prefix=f"{dataset} seed={seed}")

        if not pending_seeds:
            continue

        def _run_seed(seed):
            seed_rows = run_one(X_all, y_all, beta_true, groups, seed)
            for row in seed_rows:
                row['dataset'] = dataset
            return seed_rows

        results_iter = Parallel(
            n_jobs=min(N_SEEDS, len(pending_seeds)),
            backend='loky',
            verbose=0,
            return_as='generator_unordered',
        )(
            delayed(_run_seed)(s) for s in pending_seeds
        )
        for seed_rows in results_iter:
            if not seed_rows:
                continue
            rows.extend(seed_rows)
            df_ckpt = pd.DataFrame(rows).sort_values(
                ['dataset', 'seed', 'method']
            ).reset_index(drop=True)
            save_dataframe_checkpoint(
                df_ckpt, CHECKPOINT_PATH, CHECKPOINT_META_PATH,
                _checkpoint_config(),
            )
            seed = seed_rows[0]['seed']
            _log("checkpointed completed seed", prefix=f"{dataset} seed={seed}")

    if not rows:
        _log("No results to save.")
        return

    df = pd.DataFrame(rows).sort_values(['dataset', 'seed', 'method']).reset_index(drop=True)
    df.to_pickle(RESULTS_PATH)
    save_dataframe_checkpoint(df, CHECKPOINT_PATH, CHECKPOINT_META_PATH, _checkpoint_config())
    print(f"\nSaved {len(df)} rows to {RESULTS_PATH}")

    summary = df.groupby(['dataset', 'method'])[
        ['hidden_grad_norm_0', 'hidden_recall', 'active_features_pct',
         'f1', 'test_logloss', 'elapsed']
    ].agg(['mean', 'std'])
    print(summary)


if __name__ == '__main__':
    main()
