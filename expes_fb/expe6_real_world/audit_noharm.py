"""Panel C — does-no-harm: where biactivity is absent (sparse high-dim text at an
honest scale-free band), the SC oracle is IDENTICAL to Sparse-HO. Audit B and the
hypergradient discrepancy ||g_sc - g_null|| at the CV-optimal penalty.

Usage
-----
    python audit_noharm.py --datasets real-sim news20.binary rcv1.binary phishing
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
from run_s3_counterfactual import _phi_val

from sparse_ho.models import WeightedSparseLogReg
from sparse_ho.algo import ImplicitVariational, select_biactive_self_consistent

RESULTS_DIR = Path(__file__).parent / 'results' / 'biactivity_scan'
INNER_TOL = 1e-9
INNER_MAX_ITER = 20000
EPS_B = 1e-3          # honest band for sparse data


def _log(msg, prefix=None):
    print(f"[{time.strftime('%H:%M:%S')}]" + (f" [{prefix}]" if prefix else "")
          + f" {msg}", flush=True)


def _hgrad(policy, model, X, y, idx_tr, idx_val, x_log):
    orc = ImplicitVariational(policy=policy, biactive_tol_rel=EPS_B,
                              biactive_scale_floor=0.0, tol_lin_sys=1e-8)
    crit = _HeldOutLogisticWithMaxIter(idx_tr, idx_val, INNER_MAX_ITER)
    _, g = crit.get_val_grad(model, X, y, x_log, orc.compute_beta_grad,
                             tol=INNER_TOL)
    return np.asarray(g, float), dict(orc.last_run_info or {})


def run_one(name, args):
    X, y = get_dataset(name, Path(__file__).parent / 'data')
    if issparse(X):
        X = X.tocsc()
    if args.max_samples and X.shape[0] > args.max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], args.max_samples, replace=False)
        X, y = X[idx], y[idx]
    n, m = X.shape
    idx_tr, idx_val, idx_te = _make_split(n, 0)
    Xp, _ = _preprocess_features_for_split(name, X, idx_tr)
    if issparse(Xp):
        Xp = Xp.tocsc()
    X_tr, y_tr = Xp[idx_tr], y[idx_tr]
    X_val, y_val = Xp[idx_val], y[idx_val]
    model = WeightedSparseLogReg(alpha_l2=1.0 / len(idx_tr))
    a_max = _alpha_max(X_tr, y_tr)
    # CV-optimal scalar penalty
    best_t, best_v = None, np.inf
    for t in np.geomspace(0.02, 0.99, 10):
        la = np.log(t * a_max) * np.ones(m)
        v = _phi_val(model, X_tr, y_tr, X_val, y_val, la, INNER_TOL, INNER_MAX_ITER)
        if v < best_v:
            best_v, best_t = v, t
    x_star = np.log(best_t * a_max) * np.ones(m)
    g_null, _ = _hgrad(None, model, Xp, y, idx_tr, idx_val, x_star)
    g_sc, info = _hgrad(select_biactive_self_consistent, model, Xp, y, idx_tr,
                        idx_val, x_star)
    disc = float(np.linalg.norm(g_sc - g_null))
    row = dict(dataset=name, n=n, m=m, cv_t=best_t,
               biactive=int(info.get('biactive_size', 0)),
               sc_selected=int(info.get('selected_biactive_size', 0)),
               hgrad_discrepancy=disc)
    _log(f"m={m} cv_t={best_t:.3g} B={row['biactive']} "
         f"sc_sel={row['sc_selected']} ||g_sc-g_null||={disc:.2e}", prefix=name)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['real-sim', 'news20.binary', 'rcv1.binary',
                             'phishing'])
    ap.add_argument('--max-samples', type=int, default=8000)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.datasets:
        _log("=" * 56)
        try:
            rows.append(run_one(name, args))
        except Exception as e:  # noqa: BLE001
            _log(f"SKIP ({type(e).__name__}: {e})", prefix=name)
        sfx = f"_{args.tag}" if args.tag else ""
        pd.DataFrame(rows).to_pickle(RESULTS_DIR / f'noharm{sfx}.pkl')
    if rows:
        df = pd.DataFrame(rows)
        _log("=" * 56)
        _log("DOES-NO-HARM (B=0 => SC identical to Sparse-HO):")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
