"""Experiment 5 Setting 2 re-run with an outer CONVERGENCE stop (Option A).

Motivation
----------
In the fixed-budget Setting-2 run, NTRBA's validation objective plateaus by outer
iteration ~20 on the large sparse datasets, but its step/stationarity stopping
test does not fire on the flat validation valleys, so it idles to the full
60-iteration budget. That inflates its reported total wall-clock ~3x and makes it
look slower than Sparse-HO, even though it has converged.

This runner re-runs the two gradient methods with a convergence-based outer stop
(``PLATEAU_PATIENCE`` / ``PLATEAU_RTOL``) applied IDENTICALLY to both, capped at
N_OUTER. Sparse-HO (subgradient, still descending) runs to the cap; NTRBA stops
when its objective plateaus. The oracle/band/matrix-free config is inherited
unchanged from run_s2 (scale-free band, auto matrix-free for |S|>=48), so the
per-iteration cost matches the paper table -- only n_iter and total time change.

Outputs (tag ``setting2_convergence``): per (dataset, seed, method) row with
test_f1, sparsity, elapsed, t_per_iter, n_iter, termination, and the full
``val_objs`` trajectory (for the convergence figure). scalar_cv is NOT re-run
(band-independent); its rows are taken from the paper run at table-build time.

Usage
-----
    python run_s2_convergence.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import argparse                                                     # noqa: E402
import run_s2  # noqa: E402
from run_s2 import (run_gradient_method, _make_split,               # noqa: E402
                    _preprocess_features_for_split, _alpha_max,
                    DATASETS, N_SEEDS, DATA_DIR, _log)
from data_loaders import get_dataset                                # noqa: E402

METHODS = ['sparseho_wl1', 'ntrba_wl1']
# Order fast/validating datasets first (rcv1 is the canonical case where NTRBA
# previously idled to 61); the slow ones (real-sim, news20) run last.
DATASET_ORDER = ['rcv1.binary', 'phishing', 'mnist', 'real-sim', 'news20.binary']


def _one(dname, X, y, seed):
    n_total = X.shape[0]
    idx_tr, idx_val, idx_te = _make_split(n_total, seed, y=y)
    Xp, notes = _preprocess_features_for_split(dname, X, idx_tr)
    if issparse(Xp):
        Xp = Xp.tocsc()
    m = Xp.shape[1]
    X_tr, y_tr = Xp[idx_tr], y[idx_tr]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    alpha_l2 = 1.0 / len(idx_tr)
    a_max = _alpha_max(X_tr, y_tr)
    log_alpha0 = np.log(0.10 * a_max) * np.ones(m)

    rows = []
    for mth in METHODS:
        res = run_gradient_method(
            mth, Xp, y, idx_tr, idx_val, idx_te, log_alpha0, alpha_l2, m,
            log_prefix=f"s2conv {dname} seed={seed} {mth}")
        rows.append(dict(
            dataset=dname, seed=seed, method=mth,
            test_f1=res['test_f1'], sparsity=res['sparsity'],
            elapsed=res['elapsed'], t_per_iter=res['t_per_iter'],
            n_iter=res['n_iter'], best_val=res.get('best_val', np.nan),
            termination=res['termination'],
            val_objs=list(res.get('val_objs', [])),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=int, default=60,
                    help='outer-iteration cap N_OUTER')
    ap.add_argument('--tag', default='setting2_convergence',
                    help='results subdirectory tag')
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--rtol', type=float, default=1e-4)
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='Override the dataset list '
                         '(default: run_s2.DATASETS, fast-first order).')
    ap.add_argument('--seeds', type=int, default=None,
                    help='Number of random-split seeds (default: run_s2.N_SEEDS).')
    args = ap.parse_args()

    # apply config to the shared run_s2 module (read at optimizer construction)
    run_s2.N_OUTER = args.cap
    run_s2.PLATEAU_PATIENCE = args.patience
    run_s2.PLATEAU_RTOL = args.rtol
    out = HERE / 'results' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    results_pkl = out / 'results.pkl'

    if args.datasets:
        datasets = list(args.datasets)
    else:
        datasets = [d for d in DATASET_ORDER if d in DATASETS] + \
            [d for d in DATASETS if d not in DATASET_ORDER]
    n_seeds = args.seeds if args.seeds is not None else N_SEEDS
    _log(f"re-run: cap={args.cap} patience={args.patience} rtol={args.rtol} "
         f"tag={args.tag} datasets={datasets} seeds={n_seeds}")
    all_rows = []
    if results_pkl.exists():
        all_rows = pd.read_pickle(results_pkl).to_dict('records')
        done = {(r['dataset'], r['seed'], r['method']) for r in all_rows}
        _log(f"resuming: {len(done)} rows already present")
    else:
        done = set()

    for dname in datasets:
        need = [(s, m) for s in range(n_seeds) for m in METHODS
                if (dname, s, m) not in done]
        if not need:
            _log(f"{dname}: all seeds/methods done, skipping load")
            continue
        _log(f"{dname}: loading dataset")
        X, y = get_dataset(dname, DATA_DIR)
        for seed in range(n_seeds):
            if all((dname, seed, m) in done for m in METHODS):
                continue
            t0 = time.time()
            try:
                rows = _one(dname, X, y, seed)
            except Exception as e:  # noqa: BLE001
                _log(f"ERROR {dname} seed={seed}: {e!r}")
                continue
            all_rows = [r for r in all_rows
                        if not (r['dataset'] == dname and r['seed'] == seed
                                and r['method'] in METHODS)]
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_pickle(results_pkl)   # checkpoint
            for r in rows:
                _log(f"  saved {dname} seed={seed} {r['method']}: "
                     f"n_iter={r['n_iter']} term={r['termination']} "
                     f"t/it={r['t_per_iter']:.2f} total={r['elapsed']:.1f} "
                     f"F1={r['test_f1']:.3f}")
            _log(f"{dname} seed={seed} done in {time.time()-t0:.1f}s")
    _log(f"FINISHED: {len(all_rows)} rows -> {results_pkl}")


if __name__ == '__main__':
    main()
