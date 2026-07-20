"""Negative-control column for Table tab:expe5_setting2:
   ||g_sc - g_null|| at the hyperparameter Sparse-HO RETURNS under standard init.

For each Setting-2 dataset: run Sparse-HO (null) to convergence from the standard
0.10*alpha_max init, take its returned x_T (log_alpha_final), and evaluate the
biactive set B(x_T) and the oracle gap ||g_sc(x_T) - g_null(x_T)||. If B=0 the two
selections are bit-identical there -> the sign-consistent construction costs
nothing where biactivity is absent (the honest negative control).

Usage
-----
    uv run python expes_fb/expe7_certificate_soundness/setting2_oracle_gap.py \
        --datasets phishing rcv1.binary real-sim
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from certificate_audit import (load_split, oracle_grads,               # noqa: E402
                               partition_biactive, DEFAULT_EPS_B)
sys.path.insert(0, str(HERE.parent / 'expe6_real_world'))
from run_s2 import run_gradient_method                                 # noqa: E402

RESULTS_DIR = HERE / 'results'
S2 = ['mnist', 'news20.binary', 'phishing', 'rcv1.binary', 'real-sim']


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=S2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.datasets:
        try:
            d = load_split(name, args.seed, HERE.parent / 'expe6_real_world' / 'data')
            model, X, y = d['model'], d['X'], d['y']
            idx_tr, idx_val, idx_te = d['idx_tr'], d['idx_val'], d['idx_te']
            m, a_max = d['m'], d['a_max']
            alpha_l2 = 1.0 / len(idx_tr)
            log_alpha0 = np.log(0.10 * a_max) * np.ones(m)
            res = run_gradient_method(
                'sparseho_wl1', X, y, idx_tr, idx_val, idx_te, log_alpha0,
                alpha_l2, m, log_prefix=f"s2gap {name}")
            xT = np.asarray(res['log_alpha_final'], float)
            X_tr = X[idx_tr]
            from scipy.sparse import issparse
            if issparse(X_tr):
                X_tr = X_tr.tocsc()
            bi, supp = partition_biactive(model, X_tr, y[idx_tr], xT, DEFAULT_EPS_B)
            og = oracle_grads(model, X, y, idx_tr, idx_val, xT, DEFAULT_EPS_B)
            row = dict(dataset=name, support=supp, B=len(bi),
                       g_disc=og['g_disc'], g_sc_norm=og['g_sc_norm'],
                       g_null_norm=og['g_null_norm'],
                       test_f1=res['test_f1'], sparsity=res['sparsity'],
                       termination=res['termination'], n_iter=res['n_iter'])
            rows.append(row)
            _log(f"{name}: supp={supp} B={len(bi)} "
                 f"||g_sc-g_null||={og['g_disc']:.3e} "
                 f"(||g_sc||={og['g_sc_norm']:.3e}) term={res['termination']}")
        except Exception as e:  # noqa: BLE001
            _log(f"SKIP {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        sfx = f"_{args.tag}" if args.tag else ""
        pd.DataFrame(rows).to_pickle(RESULTS_DIR / f'setting2_gap{sfx}.pkl')

    if rows:
        _log("=" * 60)
        print(pd.DataFrame(rows)[['dataset', 'support', 'B', 'g_disc',
                                  'test_f1', 'termination']].to_string(index=False))


if __name__ == '__main__':
    main()
