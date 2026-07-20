"""Experiment 7 — Soundness of the stationarity certificate along the path.

Replaces the old Section 5.5 "natural incidence of biactivity" diagnostic. At a
sequence of regularization-path locations (headline: alpha_max, the deterministic
warm start of every path solver), we evaluate both selections against an
independent ground truth Phi'(x;-e_i) (model-free finite difference of the real
solver) and record whether the support-restricted (null) stopping test
`||h||=0` is a FALSE stationarity certificate.

Two-sided by design:
  - microarray (leukemia/colon/duke): natural biactive clusters -> false
    certificate expected at alpha_max;
  - text/categorical (madelon/a9a/splice): biactive set empty at cv-opt ->
    the two oracles coincide (parity), the no-harm prediction.
  - real text designs shared with Experiment 6 Setting 2 (rcv1/real-sim/news20):
    false certificate at alpha_max on the very datasets Setting 2 tunes, and an
    empty-or-negligible biactive set at cv-opt (near-parity where it operates).

Sensitivity sweeps make the headline referee-proof (all pointwise, cheap):
  band eps_B, finite-difference step (Richardson), and split seed.

Usage
-----
    uv run python expes_fb/expe7_certificate_soundness/run.py
    uv run python expes_fb/expe7_certificate_soundness/run.py \
        --datasets leukemia --seeds 0 1 2 --quick
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from certificate_audit import (                                       # noqa: E402
    audit_point, load_split, partition_biactive,
    DEFAULT_EPS_B, DEFAULT_FD_STEP)

DATA_DIR = HERE.parent / 'expe6_real_world' / 'data'
RESULTS_DIR = HERE / 'results'

# Three blocks, ordered: natural-biactivity (microarray), engineered/categorical,
# then the real high-dimensional text designs shared with Experiment 6 Setting 2
# (rcv1/real-sim/news20). The text block ties the anchor diagnostic to the very
# datasets tuned in Setting 2: false certificate at alpha_max, near-parity at cv-opt.
DATASETS = [
    ('leukemia', 'microarray'),
    ('colon-cancer', 'microarray'),
    ('duke breast-cancer', 'microarray'),
    ('madelon', 'engineered'),
    ('a9a', 'categorical'),
    ('splice', 'categorical'),
    ('rcv1.binary', 'text'),
    ('real-sim', 'text'),
    ('news20.binary', 'text'),
]

# Path locations as fractions of alpha_max; cv-opt is appended per dataset.
PATH_TS = [1.0, 0.9, 0.7, 0.5]

# Sensitivity grids
EPS_B_GRID = [1e-1, 1e-2, 1e-3]
FD_STEP_GRID = [1e-1, 1e-2, 1e-3]


def _log(msg, prefix=None):
    stamp = time.strftime('%H:%M:%S')
    print(f"[{stamp}]" + (f" [{prefix}]" if prefix else "") + f" {msg}",
          flush=True)


def _cv_opt_t(model, X, y, idx_tr, idx_val, a_max, m):
    """Scalar penalty t*a_max minimizing held-out logistic loss (quick grid)."""
    from certificate_audit import fd_directional  # reuse phi0 machinery
    from inner_helpers import _phi_val
    from scipy.sparse import issparse
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_val], y[idx_val]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    if issparse(X_val):
        X_val = X_val.tocsc()
    best_t, best_v = None, np.inf
    for t in np.geomspace(0.02, 0.99, 12):
        la = np.log(t * a_max) * np.ones(m)
        v = _phi_val(model, X_tr, y_tr, X_val, y_val, la, 1e-9, 20000)
        if v < best_v:
            best_v, best_t = v, float(t)
    return best_t


def audit_dataset(name, mech, seed, args):
    d = load_split(name, seed, DATA_DIR)
    model, X, y = d['model'], d['X'], d['y']
    idx_tr, idx_val, a_max, m = d['idx_tr'], d['idx_val'], d['a_max'], d['m']
    _log(f"n={d['n']} m={m} a_max={a_max:.3e} mech={mech} seed={seed}",
         prefix=name)

    t_cv = _cv_opt_t(model, X, y, idx_tr, idx_val, a_max, m)
    locations = [('alpha_max', 1.0)]
    if not args.quick:
        locations += [(f'path_{t:g}', t) for t in PATH_TS[1:]]
    locations += [(f'cv_opt', t_cv)]

    rows = []
    for loc_name, t in locations:
        log_alpha = np.log(t * a_max) * np.ones(m)
        # headline audit at the default band / fd-step
        r = audit_point(name, loc_name, model, X, y, idx_tr, idx_val,
                        log_alpha, eps_B=DEFAULT_EPS_B, scale_floor=0.0,
                        fd_step=DEFAULT_FD_STEP)
        r.update(mech=mech, seed=seed, t=t, m=m, sweep='headline')
        rows.append(r)
        tag = ('  <== FALSE CERTIFICATE' if r['false_certificate']
               else ('  (parity: B=0)' if r['B'] == 0 else ''))
        _log(f"{loc_name:12s} t={t:.3g} supp={r['support']:4d} B={r['B']:3d} "
             f"||g_null||={r['g_null_norm']:.2e} cert={r['null_certifies_stationary']} "
             f"i*_gt={r['istar_gt']:+.3e} sc={r['istar_sc_pred']:+.3e} "
             f"rel={r['istar_match_rel']:.1e}{tag}", prefix=name)

        # sensitivity sweeps only at alpha_max (the headline location)
        if loc_name == 'alpha_max' and not args.no_sweep:
            for eps_B in EPS_B_GRID:
                rr = audit_point(name, loc_name, model, X, y, idx_tr, idx_val,
                                 log_alpha, eps_B=eps_B, scale_floor=0.0,
                                 fd_step=DEFAULT_FD_STEP)
                rr.update(mech=mech, seed=seed, t=t, m=m, sweep=f'band_{eps_B:g}')
                rows.append(rr)
            for h in FD_STEP_GRID:
                rr = audit_point(name, loc_name, model, X, y, idx_tr, idx_val,
                                 log_alpha, eps_B=DEFAULT_EPS_B, scale_floor=0.0,
                                 fd_step=h)
                rr.update(mech=mech, seed=seed, t=t, m=m, sweep=f'fd_{h:g}')
                rows.append(rr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--quick', action='store_true',
                    help='only alpha_max + cv_opt (skip interior path rows)')
    ap.add_argument('--no-sweep', action='store_true',
                    help='skip band/fd sensitivity sweeps')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    todo = DATASETS
    if args.datasets:
        todo = [(n, m) for (n, m) in DATASETS if n in args.datasets]
        known = {n for n, _ in DATASETS}
        todo += [(n, 'user') for n in args.datasets if n not in known]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, mech in todo:
        for seed in args.seeds:
            _log("=" * 68)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    all_rows.extend(audit_dataset(name, mech, seed, args))
            except Exception as e:  # noqa: BLE001
                _log(f"SKIP ({type(e).__name__}: {e})", prefix=name)
                import traceback
                traceback.print_exc()
                continue
            sfx = f"_{args.tag}" if args.tag else ""
            pd.DataFrame(all_rows).to_pickle(RESULTS_DIR / f'results{sfx}.pkl')

    if not all_rows:
        _log("no results.")
        return
    df = pd.DataFrame(all_rows)
    head = df[df.sweep == 'headline']
    _log("=" * 68)
    _log("HEADLINE (alpha_max) — false-certificate witnesses:")
    amax = head[head.location == 'alpha_max']
    cols = ['dataset', 'seed', 'support', 'B', 'g_null_norm',
            'null_certifies_stationary', 'istar_gt', 'istar_sc_pred',
            'istar_match_rel', 'false_certificate']
    print(amax.reindex(columns=cols).to_string(index=False))
    _log(f"false certificate at alpha_max on: "
         f"{sorted(set(amax[amax.false_certificate].dataset))}")
    _log(f"saved -> {RESULTS_DIR}")


if __name__ == '__main__':
    main()
