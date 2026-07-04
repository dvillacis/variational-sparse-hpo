"""Decisive test for justifying biactive selection in a Setting-2-style run.

Runs the FULL bilevel optimization twice with the SAME oracle infrastructure and the
SAME trust-region optimizer, changing ONLY the biactive selection policy:
  SC  : select_biactive_self_consistent   (the proposed rule)
  null: policy=None                        (support-restricted, = Sparse-HO's selection)

Logs the biactive count B_k and the selected count at EVERY outer iteration (not just the
fixed point), and compares the two runs' final models. Answers: does biactive selection
ever fire along a real trajectory on large data, and does it change the outcome?

If SC never diverges from null (B_k=0 throughout, identical final model), biactive
selection is provably inert on that dataset — honest does-no-harm. leukemia (tiny,
coupled biactive) is the positive control that SHOULD diverge.

Usage
-----
    python run_sc_vs_null_trajectory.py --datasets a5a w3a rcv1.binary leukemia
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).parent))
from data_loaders import get_dataset, load_libsvmdata_binary
from run_s2 import (_make_split, _preprocess_features_for_split, _alpha_max,
                    _test_f1, _HeldOutLogisticWithMaxIter)
from run_s3_counterfactual import _solve_beta

from sparse_ho.models import WeightedSparseLogReg
from sparse_ho.algo import ImplicitVariational, select_biactive_self_consistent
from sparse_ho.optimizers import TrustRegion
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor

RESULTS_DIR = Path(__file__).parent / 'results' / 'sc_vs_null'
INNER_TOL = 1e-7
INNER_MAX_ITER = 10000
N_OUTER = 60
EPS_B = 1e-3
ALIASES = {'duke-breast-cancer': 'duke breast-cancer'}


def _log(msg, prefix=None):
    print(f"[{time.strftime('%H:%M:%S')}]" + (f" [{prefix}]" if prefix else "")
          + f" {msg}", flush=True)


def _run_policy(policy, model, X, y, idx_tr, idx_val, x0, m):
    algo = ImplicitVariational(policy=policy, biactive_tol_rel=EPS_B,
                               biactive_scale_floor=0.0)
    crit = _HeldOutLogisticWithMaxIter(idx_tr, idx_val, INNER_MAX_ITER)
    opt = TrustRegion(n_outer=N_OUTER, radius0=0.1, tol=1e-7)
    traj = []

    def cb(obj, grad, mask=None, dense=None, alpha=None):
        info = dict(getattr(algo, 'last_run_info', {}) or {})
        traj.append((int(info.get('biactive_size', 0)),
                     int(info.get('selected_biactive_size', 0))))
    mon = Monitor(callback=cb)
    grad_search(algo, crit, model, opt, X, y, np.exp(x0), mon)
    la = np.log(np.maximum(np.asarray(mon.alphas[int(np.argmin(mon.objs))],
                                      float), 1e-300)) if mon.objs else x0
    return mon, traj, la


def run_one(name, args):
    loader = ALIASES.get(name, name)
    try:
        X, y = get_dataset(loader, Path(__file__).parent / 'data')
    except ValueError:
        X, y = load_libsvmdata_binary(loader)
    if issparse(X):
        X = X.tocsc()
    if args.max_samples and X.shape[0] > args.max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], args.max_samples, replace=False)
        X, y = X[idx], y[idx]
    n, m = X.shape
    idx_tr, idx_val, idx_te = _make_split(n, 0)
    Xp, _ = _preprocess_features_for_split(loader, X, idx_tr)
    if issparse(Xp):
        Xp = Xp.tocsc()
    X_tr, y_tr = Xp[idx_tr], y[idx_tr]
    X_te, y_te = Xp[idx_te], y[idx_te]
    model = WeightedSparseLogReg(alpha_l2=1.0 / len(idx_tr))
    a_max = _alpha_max(X_tr, y_tr)
    x0 = np.log(args.init_frac * a_max) * np.ones(m)

    mon_s, traj_s, la_s = _run_policy(select_biactive_self_consistent, model, Xp,
                                      y, idx_tr, idx_val, x0, m)
    mon_n, traj_n, la_n = _run_policy(None, model, Xp, y, idx_tr, idx_val, x0, m)

    B_s = np.array([b for b, _ in traj_s])
    sel_s = np.array([s for _, s in traj_s])

    def _beta(la):
        b, _, _ = _solve_beta(model, X_tr, y_tr, la, INNER_TOL, INNER_MAX_ITER)
        return b
    b_s, b_n = _beta(la_s), _beta(la_n)
    f1_s, f1_n = _test_f1(X_te, y_te, b_s), _test_f1(X_te, y_te, b_n)
    supp_s = int((np.abs(b_s) > 1e-10).sum())
    supp_n = int((np.abs(b_n) > 1e-10).sum())
    model_div = float(np.linalg.norm(b_s - b_n))
    row = dict(
        dataset=name, n=n, m=m,
        iters_B_pos=int((B_s > 0).sum()), max_B=int(B_s.max() if len(B_s) else 0),
        total_selected=int(sel_s.sum()),
        f1_sc=f1_s, f1_null=f1_n, d_f1=float(f1_s - f1_n),
        supp_sc=supp_s, supp_null=supp_n, model_divergence=model_div,
        phi_sc=float(min(mon_s.objs)) if mon_s.objs else np.nan,
        phi_null=float(min(mon_n.objs)) if mon_n.objs else np.nan,
    )
    _log(f"iters_with_B>0={row['iters_B_pos']}/{len(B_s)} maxB={row['max_B']} "
         f"selected_total={row['total_selected']} | dF1={row['d_f1']:+.4f} "
         f"model_div={model_div:.2e} supp {supp_n}->{supp_s}", prefix=name)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['a5a', 'w3a', 'rcv1.binary', 'leukemia'])
    ap.add_argument('--max-samples', type=int, default=6000)
    ap.add_argument('--init-frac', type=float, default=0.10, dest='init_frac',
                    help="uniform init penalty as a fraction of alpha_max.")
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.datasets:
        _log("=" * 56)
        _log(f"dataset={name}")
        try:
            rows.append(run_one(name, args))
        except Exception as e:  # noqa: BLE001
            _log(f"SKIP ({type(e).__name__}: {e})", prefix=name)
        sfx = f"_{args.tag}" if args.tag else ""
        pd.DataFrame(rows).to_pickle(RESULTS_DIR / f'sc_vs_null{sfx}.pkl')
    if rows:
        df = pd.DataFrame(rows)
        _log("=" * 56)
        _log("SC vs null (biactive selection isolated): does SC ever diverge?")
        print(df[['dataset', 'n', 'm', 'iters_B_pos', 'max_B', 'total_selected',
                  'd_f1', 'model_divergence']].to_string(index=False))


if __name__ == '__main__':
    main()
