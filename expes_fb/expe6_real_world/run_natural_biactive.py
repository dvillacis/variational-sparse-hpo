"""Natural-biactivity counterfactual: does the SC oracle recover descent that the
support-restricted (Sparse-HO) oracle misses, at STANDARD initializations on real
correlated-feature data?

Unlike the calibrated Exp 3/4 (planted biactive coordinates), here the initialization
is a plain uniform penalty at a fraction of alpha_max — a standard sparse-init choice.
Biactivity arises naturally: near alpha_max the top (correlated) features sit at their
soft-thresholding kink. We continue from that init under {null, SC} oracles with the
SAME optimizer and compare the held-out objective trajectory.

The sharp case is init = alpha_max exactly: the support is empty, so the
support-restricted hypergradient is IDENTICALLY ZERO (no active coordinate contributes
and biactive coordinates are given zero) and Sparse-HO cannot even start; the SC oracle
assigns descent to the biactive top features.

Usage
-----
    python run_natural_biactive.py --datasets leukemia colon-cancer duke-breast-cancer
    python run_natural_biactive.py --datasets leukemia --init-fracs 1.0 0.95 0.8 \
        --optimizers nba trust_region
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
    _make_split, _preprocess_features_for_split, _alpha_max, _test_f1,
    _HeldOutLogisticWithMaxIter,
)
from run_s3_counterfactual import _solve_beta, _best_log_alpha, _make_optimizer

from sparse_ho.models import WeightedSparseLogReg
from sparse_ho.algo import (
    ImplicitVariational, select_biactive_self_consistent)
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor

RESULTS_DIR = Path(__file__).parent / 'results' / 'natural_biactive'
INNER_TOL = 1e-9
INNER_MAX_ITER = 20000
EPS_B = 1e-2          # honest scale-free band (within 1% of the kink)
ACTIVE_THR = 1e-10

# canonical names (argparse-friendly aliases → loader names)
ALIASES = {'duke-breast-cancer': 'duke breast-cancer'}


def _log(msg, prefix=None):
    stamp = time.strftime('%H:%M:%S')
    print(f"[{stamp}]" + (f" [{prefix}]" if prefix else "") + f" {msg}",
          flush=True)


def _oracle(policy):
    return ImplicitVariational(
        policy=policy, biactive_tol_rel=EPS_B, biactive_scale_floor=0.0,
        tol_lin_sys=1e-8)


def _continue(oracle, model, X, y, idx_train, idx_val, x0_log, n_outer,
              opt_kind, step):
    crit = _HeldOutLogisticWithMaxIter(idx_train, idx_val, INNER_MAX_ITER)
    if opt_kind == 'trust_region':
        opt = _make_optimizer('trust_region', n_outer, tol=1e-8)
    else:
        opt = _make_optimizer('nba', n_outer, tol=1e-8)
        opt.step_size = step
    mon = Monitor()
    grad_search(oracle, crit, model, opt, X, y, np.exp(x0_log), mon)
    return mon


def _grad_norm_at(oracle, model, X, y, idx_train, idx_val, x_log):
    crit = _HeldOutLogisticWithMaxIter(idx_train, idx_val, INNER_MAX_ITER)
    val, grad = crit.get_val_grad(
        model, X, y, x_log, oracle.compute_beta_grad, tol=INNER_TOL)
    info = dict(getattr(oracle, 'last_run_info', {}) or {})
    return float(val), float(np.linalg.norm(grad)), info


def run_one(name, args):
    loader_name = ALIASES.get(name, name)
    X, y = get_dataset(loader_name, Path(__file__).parent / 'data')
    if issparse(X):
        X = X.tocsc()
    if args.max_samples and X.shape[0] > args.max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], args.max_samples, replace=False)
        X, y = X[idx], y[idx]
    n, m = X.shape
    rows = []
    for seed in range(args.n_seeds):
        idx_tr, idx_val, idx_te = _make_split(n, seed)
        Xp, _ = _preprocess_features_for_split(loader_name, X, idx_tr)
        if issparse(Xp):
            Xp = Xp.tocsc()
        X_tr, y_tr = Xp[idx_tr], y[idx_tr]
        X_te, y_te = Xp[idx_te], y[idx_te]
        model = WeightedSparseLogReg(alpha_l2=1.0 / len(idx_tr))
        a_max = _alpha_max(X_tr, y_tr)

        for frac in args.init_fracs:
            x0 = np.log(frac * a_max) * np.ones(m)
            orc_null, orc_sc = _oracle(None), _oracle(
                select_biactive_self_consistent)
            v0, gn_null, info_null = _grad_norm_at(
                orc_null, model, Xp, y, idx_tr, idx_val, x0)
            _, gn_sc, info_sc = _grad_norm_at(
                orc_sc, model, Xp, y, idx_tr, idx_val, x0)
            B = int(info_sc.get('biactive_size', 0))
            sc_sel = int(info_sc.get('selected_biactive_size', 0))

            for opt_kind in args.optimizers:
                monN = _continue(orc_null, model, Xp, y, idx_tr, idx_val, x0,
                                 args.n_outer, opt_kind, args.step)
                monS = _continue(orc_sc, model, Xp, y, idx_tr, idx_val, x0,
                                 args.n_outer, opt_kind, args.step)
                phiN = float(min(monN.objs)) if monN.objs else np.nan
                phiS = float(min(monS.objs)) if monS.objs else np.nan
                dN = float(monN.objs[0] - phiN) if monN.objs else np.nan
                dS = float(monS.objs[0] - phiS) if monS.objs else np.nan

                def _beta(mon):
                    la = _best_log_alpha(mon)
                    if la is None:
                        return np.zeros(m)
                    b, _, _ = _solve_beta(model, X_tr, y_tr, la, INNER_TOL,
                                          INNER_MAX_ITER)
                    return b
                bN, bS = _beta(monN), _beta(monS)
                f1N = _test_f1(X_te, y_te, bN)
                f1S = _test_f1(X_te, y_te, bS)
                suppN = int((np.abs(bN) > ACTIVE_THR).sum())
                suppS = int((np.abs(bS) > ACTIVE_THR).sum())
                rows.append(dict(
                    dataset=name, seed=seed, init_frac=frac, optimizer=opt_kind,
                    n=n, m=m, biactive=B, sc_selected=sc_sel,
                    gnorm_null0=gn_null, gnorm_sc0=gn_sc, phi0=float(v0),
                    dPhi_null=dN, dPhi_sc=dS, phi_null=phiN, phi_sc=phiS,
                    d_phi=float(phiN - phiS),  # >0 means SC reaches lower loss
                    test_f1_null=f1N, test_f1_sc=f1S,
                    d_test_f1=float(f1S - f1N),
                    support_null=suppN, support_sc=suppS,
                    n_iter_null=len(monN.objs), n_iter_sc=len(monS.objs)))
                _log(f"seed={seed} frac={frac:.3g} [{opt_kind}] B={B} "
                     f"sc_sel={sc_sel} |g_null0|={gn_null:.2e} "
                     f"|g_sc0|={gn_sc:.2e} dPhi_null={dN:.3e} dPhi_sc={dS:.3e} "
                     f"phi_null={phiN:.5f} phi_sc={phiS:.5f} "
                     f"dF1={f1S - f1N:+.4f}", prefix=name)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['leukemia', 'colon-cancer', 'duke-breast-cancer'])
    ap.add_argument('--init-fracs', nargs='+', type=float,
                    default=[1.0, 0.95, 0.8], dest='init_fracs')
    ap.add_argument('--optimizers', nargs='+',
                    default=['nba', 'trust_region'])
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--n-outer', type=int, default=30)
    ap.add_argument('--step', type=float, default=0.1)
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
        pd.DataFrame(all_rows).to_pickle(RESULTS_DIR / f'natural{sfx}.pkl')

    if not all_rows:
        _log("no results.")
        return
    df = pd.DataFrame(all_rows)
    _log("=" * 60)
    _log("SUMMARY (mean over seeds) — d_phi>0 & dPhi_sc>dPhi_null ⇒ SC wins:")
    g = df.groupby(['dataset', 'init_frac', 'optimizer'])[
        ['biactive', 'sc_selected', 'gnorm_null0', 'gnorm_sc0',
         'dPhi_null', 'dPhi_sc', 'd_phi', 'd_test_f1']].mean()
    print(g.to_string())
    _log(f"saved -> {RESULTS_DIR}")


if __name__ == '__main__':
    main()
