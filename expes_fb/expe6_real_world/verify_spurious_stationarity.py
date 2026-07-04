"""Empirical keystone for the SC-selection justification (correctness, not speed).

Claim: at the alpha_max reference penalty, the support-restricted (Sparse-HO) oracle
returns h = 0 -- a stationarity certificate -- yet Phi is NOT stationary: decreasing
the penalty on the biactive top feature strictly decreases the held-out objective.
The SC oracle detects exactly this (h_i^SC != 0), and by Proposition 15 its component
equals -Phi'(x_bar; -e_i). We verify the descent by finite differences and check the
slope matches -h_i^SC (SC) versus 0 (null).

If FD slope < 0 ~= -h_i^SC while the null oracle reports 0, then Sparse-HO's zero is a
FALSE stationarity certificate and SC's nonzero selection is the correct one.

Usage
-----
    python verify_spurious_stationarity.py --datasets leukemia colon-cancer duke-breast-cancer
"""

import argparse
import sys
import time
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
from run_s3_counterfactual import _solve_beta, _phi_val

from sparse_ho.models import WeightedSparseLogReg
from sparse_ho.algo import ImplicitVariational, select_biactive_self_consistent
from sparse_ho.criterion import HeldOutLogistic

RESULTS_DIR = Path(__file__).parent / 'results' / 'spurious_stationarity'
INNER_TOL = 1e-10
INNER_MAX_ITER = 30000
EPS_B = 1e-2
FD_STEPS = [0.05, 0.1, 0.25, 0.5]
ALIASES = {'duke-breast-cancer': 'duke breast-cancer'}


def _log(msg, prefix=None):
    stamp = time.strftime('%H:%M:%S')
    print(f"[{stamp}]" + (f" [{prefix}]" if prefix else "") + f" {msg}",
          flush=True)


def _hgrad(policy, model, X, y, idx_tr, idx_val, x_log):
    orc = ImplicitVariational(
        policy=policy, biactive_tol_rel=EPS_B, biactive_scale_floor=0.0,
        tol_lin_sys=1e-9)
    crit = _HeldOutLogisticWithMaxIter(idx_tr, idx_val, INNER_MAX_ITER)
    val, grad = crit.get_val_grad(
        model, X, y, x_log, orc.compute_beta_grad, tol=INNER_TOL)
    return float(val), np.asarray(grad, float), dict(
        getattr(orc, 'last_run_info', {}) or {})


def run_one(name, args):
    loader = ALIASES.get(name, name)
    X, y = get_dataset(loader, Path(__file__).parent / 'data')
    if issparse(X):
        X = X.tocsc()
    if args.max_samples and X.shape[0] > args.max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], args.max_samples, replace=False)
        X, y = X[idx], y[idx]
    rows = []
    for seed in range(args.n_seeds):
        n, m = X.shape
        idx_tr, idx_val, idx_te = _make_split(n, seed)
        Xp, _ = _preprocess_features_for_split(loader, X, idx_tr)
        if issparse(Xp):
            Xp = Xp.tocsc()
        X_tr, y_tr = Xp[idx_tr], y[idx_tr]
        X_val, y_val = Xp[idx_val], y[idx_val]
        model = WeightedSparseLogReg(alpha_l2=1.0 / len(idx_tr))
        a_max = _alpha_max(X_tr, y_tr)
        x_star = np.log(a_max) * np.ones(m)  # the alpha_max reference penalty

        phi0 = _phi_val(model, X_tr, y_tr, X_val, y_val, x_star, INNER_TOL,
                        INNER_MAX_ITER)
        v_n, g_null, _ = _hgrad(None, model, Xp, y, idx_tr, idx_val, x_star)
        v_s, g_sc, info = _hgrad(select_biactive_self_consistent, model, Xp, y,
                                 idx_tr, idx_val, x_star)
        sel = np.flatnonzero(np.abs(g_sc - g_null) > 1e-12)
        _log(f"seed={seed} a_max={a_max:.3e} |g_null|={np.linalg.norm(g_null):.3e} "
             f"|g_sc|={np.linalg.norm(g_sc):.3e} B={info.get('biactive_size')} "
             f"sc_sel={info.get('selected_biactive_size')} probing {sel.size} coords",
             prefix=name)

        for i in sel:
            h_sc_i = float(g_sc[i])
            h_null_i = float(g_null[i])
            # finite-difference one-sided slope along -e_i (penalty decrease)
            slopes = []
            for t in FD_STEPS:
                xm = x_star.copy()
                xm[i] -= t
                phi_m = _phi_val(model, X_tr, y_tr, X_val, y_val, xm, INNER_TOL,
                                 INNER_MAX_ITER)
                slopes.append((phi_m - phi0) / t)   # ~ Phi'(x; -e_i)
            fd = float(np.median(slopes))
            rows.append(dict(
                dataset=name, seed=seed, coord=int(i), phi0=phi0,
                gnull_norm=float(np.linalg.norm(g_null)),
                gsc_norm=float(np.linalg.norm(g_sc)),
                biactive=int(info.get('biactive_size', 0)),
                sc_selected=int(info.get('selected_biactive_size', 0)),
                a_max=float(a_max), m=int(m),
                h_sc_i=h_sc_i, h_null_i=h_null_i,
                pred_slope_sc=-h_sc_i, pred_slope_null=-h_null_i,
                fd_slope=fd, fd_min=float(min(slopes)), fd_max=float(max(slopes)),
                descent_confirmed=bool(fd < -1e-9),
                sc_correct_sign=bool(np.sign(fd) == np.sign(-h_sc_i) and fd < 0),
            ))
            _log(f"  coord {int(i):5d}: FD slope={fd:+.3e}  -h_sc={-h_sc_i:+.3e} "
                 f"(SC pred) | null pred=0  => "
                 f"{'DESCENT (null certificate FALSE)' if fd < -1e-9 else 'flat'}",
                 prefix=name)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['leukemia', 'colon-cancer', 'duke-breast-cancer'])
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--max-samples', type=int, default=6000)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name in args.datasets:
        _log("=" * 60)
        _log(f"dataset={name}")
        try:
            all_rows.extend(run_one(name, args))
        except Exception as e:  # noqa: BLE001
            _log(f"SKIP ({type(e).__name__}: {e})", prefix=name)
            continue
        sfx = f"_{args.tag}" if args.tag else ""
        pd.DataFrame(all_rows).to_pickle(RESULTS_DIR / f'spurious{sfx}.pkl')

    if not all_rows:
        _log("no biactive coords probed.")
        return
    df = pd.DataFrame(all_rows)
    _log("=" * 60)
    n_desc = int(df.descent_confirmed.sum())
    _log(f"biactive coords probed at alpha_max: {len(df)}")
    _log(f"  FD-confirmed strict descent (null certificate FALSE): {n_desc}/{len(df)}")
    _log(f"  SC predicted-slope sign matches FD descent: "
         f"{int(df.sc_correct_sign.sum())}/{len(df)}")
    g = df.groupby('dataset').agg(
        n_probe=('coord', 'size'),
        n_descent=('descent_confirmed', 'sum'),
        med_fd=('fd_slope', 'median'),
        med_pred_sc=('pred_slope_sc', 'median')).round(4)
    print(g.to_string())
    _log(f"saved -> {RESULTS_DIR}")


if __name__ == '__main__':
    main()
