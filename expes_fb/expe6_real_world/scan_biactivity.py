"""Screen for NATURAL biactivity on degeneracy-prone datasets.

Question: does the biactive set B (at an honest, scale-free detection band) become
non-empty at penalty locations a practitioner would actually visit? If so, the
sign-consistent biactive selection has a natural (not calibrated) justification.

Biactivity is a support-transition phenomenon. The null (support-restricted) oracle
freezes every inactive coordinate, so a *fixed-point* audit is the least likely place
to see it. We therefore audit B at four locations:

  (a) alpha_max boundary   -- the argmax feature is EXACTLY at its kink by the
                              definition of alpha_max; near-duplicate top features
                              are biactive together (coupled case, Remark 3).
  (b) path ladder t*a_max   -- feature-entry breakpoints along the regularization path.
  (c) scalar CV-optimal a   -- minima of the piecewise-smooth validation curve often
                              sit at breakpoints.
  (d) Sparse-HO fixed point -- the standard-init endpoint (control; expected B~0).

For each we report, at the SCALE-FREE band (and legacy for contrast): support size,
B, SC-selected count, how genuinely near-kink the biactive coords are (relative gap),
a coupling score (largest cluster of biactive coords with tied |v|, i.e. duplicate
features), and whether SC changes the hypergradient (||g_sc - g_null||).

Usage
-----
    python scan_biactivity.py                       # full candidate list
    python scan_biactivity.py --datasets leukemia gisette --max-samples 4000
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).parent))
from data_loaders import get_dataset
from run_s2 import (
    _make_split, _preprocess_features_for_split, _alpha_max,
    _HeldOutLogisticWithMaxIter,
)
from inner_helpers import _solve_beta, _phi_val, _best_log_alpha

from sparse_ho.models import WeightedSparseLogReg
from sparse_ho.algo import (
    Implicit, ImplicitVariational, select_biactive_self_consistent)
from sparse_ho.algo.implicit_variational import _resolve_gamma, _resolve_lambdas
from sparse_ho.optimizers import NormalizedSubgradient
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor

RESULTS_DIR = Path(__file__).parent / 'results' / 'biactivity_scan'

CANDIDATES = [
    # (name, mechanism)
    ('leukemia', 'microarray'),
    ('colon-cancer', 'microarray'),
    ('duke breast-cancer', 'microarray'),
    ('gisette', 'engineered-probes'),
    ('madelon', 'engineered-redundant'),
    ('a9a', 'onehot-categorical'),
    ('dna', 'onehot-categorical'),
    ('splice', 'onehot-categorical'),
    ('svmguide1', 'low-dim-dense'),
    ('sonar', 'low-dim-dense'),
]

INNER_TOL = 1e-9
INNER_MAX_ITER = 20000
EPS_BANDS = [1e-3, 1e-2]
LADDER = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
ACTIVE_THR = 1e-10


def _log(msg, prefix=None):
    stamp = time.strftime('%H:%M:%S')
    print(f"[{stamp}]" + (f" [{prefix}]" if prefix else "") + f" {msg}",
          flush=True)


def _partition_audit(model, X_tr, y_tr, log_alpha, eps_B, scale_floor):
    """Return partition diagnostics at log_alpha, mirroring Phase 1 exactly."""
    beta, mask, dense = _solve_beta(
        model, X_tr, y_tr, log_alpha, INNER_TOL, INNER_MAX_ITER)
    m = X_tr.shape[1]
    gamma = _resolve_gamma(None, model, X_tr)
    lambdas = _resolve_lambdas(model, log_alpha, m)
    if hasattr(model, 'set_variational_alpha'):
        model.set_variational_alpha(np.exp(log_alpha))
    grad_F = np.asarray(model.get_grad_smooth(X_tr, y_tr, beta), float)
    v = beta - gamma * grad_F
    u = gamma * lambdas
    abs_v = np.abs(v)
    gap = np.abs(abs_v - u)
    scale = np.maximum(abs_v, u)
    if scale_floor > 0.0:
        scale = np.maximum(scale, scale_floor)
    tol = eps_B * scale
    biactive = (gap <= tol) & (u > 1e-14)
    support = np.abs(beta) > ACTIVE_THR
    bi_idx = np.flatnonzero(biactive & ~support)  # biactive-and-inactive
    rel_gap = gap / np.maximum(np.maximum(abs_v, u), 1e-300)

    # coupling: largest cluster of biactive coords whose |v| are tied within 1%
    coupling = 0
    if bi_idx.size:
        vv = np.sort(abs_v[bi_idx])
        # scan sorted |v|, count max run within 1% relative spread
        run = 1
        best = 1
        for k in range(1, vv.size):
            if vv[k] <= vv[k - 1] * 1.01 + 1e-300:
                run += 1
                best = max(best, run)
            else:
                run = 1
        coupling = int(best)

    return dict(
        support=int(support.sum()),
        B=int(bi_idx.size),
        near_kink_min=float(rel_gap[bi_idx].min()) if bi_idx.size else np.nan,
        near_kink_med=float(np.median(rel_gap[bi_idx])) if bi_idx.size else np.nan,
        coupling=coupling,
        beta=beta, mask=mask, dense=dense,
    )


def _sc_vs_null(model, X, y, idx_train, idx_val, log_alpha, eps_B, scale_floor):
    """||g_sc - g_null|| and SC-selected biactive count at log_alpha."""
    orc_null = ImplicitVariational(
        policy=None, biactive_tol_rel=eps_B,
        biactive_scale_floor=scale_floor, tol_lin_sys=1e-8)
    orc_sc = ImplicitVariational(
        policy=select_biactive_self_consistent, biactive_tol_rel=eps_B,
        biactive_scale_floor=scale_floor, tol_lin_sys=1e-8)
    crit = _HeldOutLogisticWithMaxIter(idx_train, idx_val, INNER_MAX_ITER)
    _, g_null = crit.get_val_grad(
        model, X, y, log_alpha, orc_null.compute_beta_grad, tol=INNER_TOL)
    crit2 = _HeldOutLogisticWithMaxIter(idx_train, idx_val, INNER_MAX_ITER)
    _, g_sc = crit2.get_val_grad(
        model, X, y, log_alpha, orc_sc.compute_beta_grad, tol=INNER_TOL)
    info = dict(getattr(orc_sc, 'last_run_info', {}) or {})
    return (float(np.linalg.norm(np.asarray(g_sc) - np.asarray(g_null))),
            int(info.get('selected_biactive_size', 0)))


def _cv_optimal_t(model, X_tr, y_tr, X_val, y_val, a_max, m):
    """Scalar penalty t*a_max minimizing held-out logistic loss (quick grid)."""
    best_t, best_v = None, np.inf
    for t in np.geomspace(0.02, 0.99, 12):
        la = np.log(t * a_max) * np.ones(m)
        v = _phi_val(model, X_tr, y_tr, X_val, y_val, la, INNER_TOL,
                     INNER_MAX_ITER)
        if v < best_v:
            best_v, best_t = v, t
    return float(best_t)


def scan_one(name, mech, args):
    X, y = get_dataset(name, Path(__file__).parent / 'data')
    if issparse(X):
        X = X.tocsc()
    if args.max_samples and X.shape[0] > args.max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], args.max_samples, replace=False)
        X, y = X[idx], y[idx]
    n, m = X.shape
    idx_train, idx_val, idx_test = _make_split(n, args.seed)
    X, _ = _preprocess_features_for_split(name, X, idx_train)
    if issparse(X):
        X = X.tocsc()
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    alpha_l2 = 1.0 / len(idx_train)
    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    a_max = _alpha_max(X_tr, y_tr)
    _log(f"n={n} m={m} a_max={a_max:.3e} mech={mech}", prefix=name)

    # Sparse-HO fixed point from a moderate init
    log_alpha0 = np.log(0.10 * a_max) * np.ones(m)
    mon0 = Monitor()
    grad_search(
        Implicit(),
        _HeldOutLogisticWithMaxIter(idx_train, idx_val, INNER_MAX_ITER),
        model,
        NormalizedSubgradient(n_outer=args.n_outer, step_size=0.1, tol=1e-8),
        X, y, np.exp(log_alpha0), mon0)
    x_star = _best_log_alpha(mon0)

    t_cv = _cv_optimal_t(model, X_tr, y_tr, X_val, y_val, a_max, m)

    locations = [('alpha_max', np.log(a_max) * np.ones(m))]
    for t in LADDER:
        locations.append((f'path_{t}', np.log(t * a_max) * np.ones(m)))
    locations.append((f'cv_opt_{t_cv:.3g}', np.log(t_cv * a_max) * np.ones(m)))
    locations.append(('fixed_point', x_star))

    rows = []
    for loc_name, la in locations:
        for eps_B in EPS_BANDS:
            sf = _partition_audit(model, X_tr, y_tr, la, eps_B, scale_floor=0.0)
            lg = _partition_audit(model, X_tr, y_tr, la, eps_B, scale_floor=1.0)
            gdisc, sc_sel = (np.nan, 0)
            if sf['B'] > 0:
                gdisc, sc_sel = _sc_vs_null(
                    model, X, y, idx_train, idx_val, la, eps_B, scale_floor=0.0)
            rows.append(dict(
                dataset=name, mech=mech, location=loc_name, eps_B=eps_B,
                support=sf['support'], B_scalefree=sf['B'], B_legacy=lg['B'],
                sc_selected=sc_sel, coupling=sf['coupling'],
                near_kink_min=sf['near_kink_min'],
                near_kink_med=sf['near_kink_med'],
                hgrad_disc=gdisc, m=m, n=n))
            tag = '  <== NATURAL B>0' if sf['B'] > 0 else ''
            _log(f"{loc_name:16s} eps={eps_B:.0e} supp={sf['support']:4d} "
                 f"B(sf)={sf['B']:4d} B(leg)={lg['B']:5d} "
                 f"sc_sel={sc_sel:3d} coup={sf['coupling']:3d}"
                 f"{tag}", prefix=name)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--max-samples', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-outer', type=int, default=40)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    todo = CANDIDATES
    if args.datasets:
        todo = [(n, m) for (n, m) in CANDIDATES if n in args.datasets]
        known = {n for n, _ in CANDIDATES}
        for n in args.datasets:
            if n not in known:
                todo.append((n, 'user'))

    all_rows = []
    for name, mech in todo:
        _log("=" * 64)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                all_rows.extend(scan_one(name, mech, args))
        except Exception as e:  # noqa: BLE001
            _log(f"SKIP ({type(e).__name__}: {e})", prefix=name)
            continue
        sfx = f"_{args.tag}" if args.tag else ""
        pd.DataFrame(all_rows).to_pickle(RESULTS_DIR / f'scan{sfx}.pkl')

    if not all_rows:
        _log("no results.")
        return
    df = pd.DataFrame(all_rows)
    _log("=" * 64)
    _log("NATURAL biactivity (scale-free band) summary — rows with B>0:")
    hits = df[df.B_scalefree > 0]
    if hits.empty:
        _log("  none: no dataset/location shows biactivity at an honest band.")
    else:
        cols = ['dataset', 'mech', 'location', 'eps_B', 'support',
                'B_scalefree', 'sc_selected', 'coupling', 'near_kink_med',
                'hgrad_disc']
        print(hits[cols].to_string(index=False))
    _log(f"saved -> {RESULTS_DIR}")


if __name__ == '__main__':
    main()
