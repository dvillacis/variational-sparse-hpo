"""T3.2 + T3.3 — certificate audit at the RETURNED solution (and at alpha_max).

Pre-registered, report-either-way audit. This is NOT a fourth attempt to manufacture
an SC advantage (see memory `biactivity-knife-edge-finding.md`); its purpose is to bound
the certificate claim, whichever way the numbers fall.

Question T3.2: on microarray data under STANDARD init (0.10*alpha_max), does the
support-restricted (null / Sparse-HO) method RETURN a solution x_T at which
  (i)  it certifies stationarity (grad_norm < tol -> termination_reason_ == 'stationary'),
       or would under a gradient stopping test, AND
  (ii) a biactive penalty-decrease strictly reduces the held-out (validation) loss?
If both -> false certificate at a returned solution (keystone). If (i) fails
(nonempty support => ||g_null|| not ~0, method still moving), the phenomenon is confined
to the empty-support regime -> honest scope bound. Reported per dataset, no selection.

Question T3.3: initialize BOTH methods at alpha_max (the canonical path anchor). The
null method has ||g_null|| == 0 there (empty support) and should terminate 'stationary'
at iteration 0 (a fixed point of its own update); the SC/NTRBA method should move and
descend. Deterministic, no seeds.

Usage
-----
    uv run python expes_fb/expe6_real_world/audit_terminal_certificate.py
    uv run python expes_fb/expe6_real_world/audit_terminal_certificate.py \
        --datasets leukemia colon-cancer --fd-step 0.1 --seed 0
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from data_loaders import get_dataset                                   # noqa: E402
from run_s2 import (                                                   # noqa: E402
    _make_split, _preprocess_features_for_split, _alpha_max,
    _HeldOutLogisticWithMaxIter, run_gradient_method,
    INNER_TOL, INNER_MAX_ITER, BIACTIVE_TOL_REL, INNER_TOL_TR)
from inner_helpers import _solve_beta, _phi_val                        # noqa: E402

from sparse_ho.models import WeightedSparseLogReg                      # noqa: E402
from sparse_ho.algo import (                                           # noqa: E402
    ImplicitVariational, select_biactive_self_consistent)
from sparse_ho.algo.implicit_variational import (                     # noqa: E402
    _resolve_gamma, _resolve_lambdas)

DATA_DIR = HERE / 'data'
RESULTS_DIR = HERE / 'results' / 'terminal_certificate'
MICROARRAY = ['leukemia', 'colon-cancer', 'duke breast-cancer']
ACTIVE_THR = 1e-10
STATIONARY_TOL = INNER_TOL_TR   # 1e-7: the grad-norm test in NormalizedSubgradient


def _log(msg, prefix=None):
    stamp = time.strftime('%H:%M:%S')
    print(f"[{stamp}]" + (f" [{prefix}]" if prefix else "") + f" {msg}", flush=True)


def _biactive_indices(model, X_tr, y_tr, log_alpha, eps_B, scale_floor=0.0):
    """Biactive-and-inactive coordinate indices at log_alpha (mirrors Phase 1)."""
    beta, _, _ = _solve_beta(model, X_tr, y_tr, log_alpha, INNER_TOL, INNER_MAX_ITER)
    m = X_tr.shape[1]
    gamma = _resolve_gamma(None, model, X_tr)
    lambdas = _resolve_lambdas(model, log_alpha, m)
    if hasattr(model, 'set_variational_alpha'):
        model.set_variational_alpha(np.exp(log_alpha))
    grad_F = np.asarray(model.get_grad_smooth(X_tr, y_tr, beta), float)
    v = beta - gamma * grad_F
    u = gamma * lambdas
    gap = np.abs(np.abs(v) - u)
    scale = np.maximum(np.abs(v), u)
    if scale_floor > 0.0:
        scale = np.maximum(scale, scale_floor)
    biactive = (gap <= eps_B * scale) & (u > 1e-14)
    support = np.abs(beta) > ACTIVE_THR
    return np.flatnonzero(biactive & ~support), int(support.sum())


def _oracle_norms(model, X, y, idx_train, idx_val, log_alpha, eps_B, scale_floor=0.0):
    """||g_null||, ||g_sc||, ||g_sc - g_null|| and SC-selected count at log_alpha."""
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
    g_null = np.asarray(g_null, float)
    g_sc = np.asarray(g_sc, float)
    return dict(
        g_null_norm=float(np.linalg.norm(g_null)),
        g_sc_norm=float(np.linalg.norm(g_sc)),
        g_disc=float(np.linalg.norm(g_sc - g_null)),
        sc_selected=int(info.get('selected_biactive_size', 0)),
        g_sc_vec=g_sc, g_null_vec=g_null)


def _fd_descent(model, X, y, idx_train, idx_val, log_alpha, bi_idx, fd_step):
    """Held-out (validation) loss change under a penalty decrease on each biactive i.

    Moves along -e_i (reduce log-penalty): Phi'(x;-e_i) < 0 means descent.
    """
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    if issparse(X_val):
        X_val = X_val.tocsc()
    phi0 = _phi_val(model, X_tr, y_tr, X_val, y_val, log_alpha,
                    INNER_TOL, INNER_MAX_ITER)
    recs = []
    for i in bi_idx:
        la = np.asarray(log_alpha, float).copy()
        la[i] = la[i] - fd_step
        phi1 = _phi_val(model, X_tr, y_tr, X_val, y_val, la,
                        INNER_TOL, INNER_MAX_ITER)
        recs.append(dict(coord=int(i), phi0=float(phi0), phi1=float(phi1),
                         delta=float(phi1 - phi0),
                         slope=float((phi1 - phi0) / fd_step),
                         descent=bool(phi1 < phi0 - 1e-12)))
    return float(phi0), recs


def _audit_point(tag, model, X, y, idx_tr, idx_val, log_alpha, fd_step, name):
    """Full certificate audit at one hyperparameter point."""
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    on = _oracle_norms(model, X, y, idx_tr, idx_val, log_alpha, BIACTIVE_TOL_REL)
    bi_idx, supp = _biactive_indices(model, X_tr, y_tr, log_alpha, BIACTIVE_TOL_REL)
    phi0, fd = _fd_descent(model, X, y, idx_tr, idx_val, log_alpha, bi_idx, fd_step)
    n_desc = int(sum(r['descent'] for r in fd))
    certifies = bool(on['g_null_norm'] < STATIONARY_TOL)

    # Ground-truth reconciliation at each biactive coordinate:
    #   FD slope        ~ Phi'(x; -e_i)            (model-free ground truth)
    #   SC prediction   = -h_sc,i = -g_sc[i]       (manuscript identity Phi'(x;-e_i) = -h_i)
    #   null prediction = -g_null[i] = 0           (false: certifies flat)
    g_sc_vec, g_null_vec = on['g_sc_vec'], on['g_null_vec']
    for r in fd:
        i = r['coord']
        r['gt_dderiv'] = r['slope']
        r['sc_pred'] = float(-g_sc_vec[i])
        r['null_pred'] = float(-g_null_vec[i])
        r['sc_match_err'] = float(abs(r['gt_dderiv'] - r['sc_pred']))
        r['sc_match_rel'] = float(r['sc_match_err'] / max(abs(r['gt_dderiv']), 1e-12))
    istar = min(fd, key=lambda r: r['gt_dderiv']) if fd else None

    _log(f"[{tag}] supp={supp} B={len(bi_idx)} "
         f"||g_null||={on['g_null_norm']:.3e} ||g_sc||={on['g_sc_norm']:.3e} "
         f"disc={on['g_disc']:.3e} null_cert_stationary={certifies} "
         f"FD-descent={n_desc}/{len(bi_idx)}", prefix=name)
    if istar is not None:
        _log(f"   [i*={istar['coord']}] Phi'(x;-e_i*)_FD={istar['gt_dderiv']:.4e}  "
             f"SC(-h_sc)={istar['sc_pred']:.4e}  null={istar['null_pred']:.1e}  "
             f"|FD-SC|={istar['sc_match_err']:.2e} (rel {istar['sc_match_rel']:.1e})",
             prefix=name)
    return dict(
        dataset=name, point=tag, support=supp, B=len(bi_idx),
        g_null_norm=on['g_null_norm'], g_sc_norm=on['g_sc_norm'],
        g_disc=on['g_disc'], sc_selected=on['sc_selected'],
        null_certifies_stationary=certifies,
        fd_n_descent=n_desc, fd_records=fd, phi0=phi0,
        strongest_descent=float(min([r['delta'] for r in fd], default=0.0)),
        istar_gt=float(istar['gt_dderiv']) if istar else np.nan,
        istar_sc_pred=float(istar['sc_pred']) if istar else np.nan,
        istar_null_pred=float(istar['null_pred']) if istar else np.nan,
        istar_match_err=float(istar['sc_match_err']) if istar else np.nan,
        istar_match_rel=float(istar['sc_match_rel']) if istar else np.nan)


def audit_one(name, seed, fd_step, n_outer_amax):
    X, y = get_dataset(name, DATA_DIR)
    if issparse(X):
        X = X.tocsc()
    n = X.shape[0]
    idx_tr, idx_val, idx_te = _make_split(n, seed, y=y)
    X, _ = _preprocess_features_for_split(name, X, idx_tr)
    if issparse(X):
        X = X.tocsc()
    m = X.shape[1]
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    alpha_l2 = 1.0 / len(idx_tr)
    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    a_max = _alpha_max(X_tr, y_tr)
    _log(f"n={n} m={m} a_max={a_max:.3e} "
         f"split={len(idx_tr)}/{len(idx_val)}/{len(idx_te)}", prefix=name)

    rows, trajs = [], []

    # ---- T3.2: standard-init terminal iterate of Sparse-HO (null) ----
    log_alpha0 = np.log(0.10 * a_max) * np.ones(m)
    res = run_gradient_method(
        'sparseho_wl1', X, y, idx_tr, idx_val, idx_te, log_alpha0, alpha_l2, m,
        log_prefix=f"audit {name} sparseho std-init")
    xT = np.asarray(res['log_alpha_final'], float)
    r = _audit_point('sho_terminal_stdinit', model, X, y, idx_tr, idx_val,
                     xT, fd_step, name)
    r.update(termination=res['termination'], n_iter=res['n_iter'],
             test_f1=res['test_f1'], sparsity=res['sparsity'],
             init='0.10*a_max')
    rows.append(r)
    trajs.append(dict(dataset=name, point='sho_terminal_stdinit',
                      method='sparseho_wl1', init='0.10*a_max',
                      val_objs=list(res.get('val_objs', [])),
                      termination=res['termination']))

    # ---- T3.3: alpha_max operating point + end-to-end from alpha_max ----
    x_amax = np.log(a_max) * np.ones(m)
    r2 = _audit_point('alpha_max', model, X, y, idx_tr, idx_val,
                      x_amax, fd_step, name)
    r2.update(init='a_max')
    rows.append(r2)

    for mth in ('sparseho_wl1', 'ntrba_wl1'):
        res_m = run_gradient_method(
            mth, X, y, idx_tr, idx_val, idx_te, x_amax, alpha_l2, m,
            log_prefix=f"audit {name} {mth} amax-init")
        trajs.append(dict(dataset=name, point='amax_init_run', method=mth,
                          init='a_max', val_objs=list(res_m.get('val_objs', [])),
                          termination=res_m['termination'],
                          n_iter=res_m['n_iter'], test_f1=res_m['test_f1'],
                          sparsity=res_m['sparsity']))
        _log(f"[amax-init end-to-end] {mth}: term={res_m['termination']} "
             f"n_iter={res_m['n_iter']} test_f1={res_m['test_f1']:.3f} "
             f"sparsity={res_m['sparsity']:.3f}", prefix=name)

    return rows, trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=MICROARRAY)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fd-step', type=float, default=0.1)
    ap.add_argument('--n-outer-amax', type=int, default=20)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, all_trajs = [], []
    for name in args.datasets:
        _log("=" * 68)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rows, trajs = audit_one(name, args.seed, args.fd_step,
                                        args.n_outer_amax)
            all_rows.extend(rows)
            all_trajs.extend(trajs)
        except Exception as e:  # noqa: BLE001
            _log(f"SKIP ({type(e).__name__}: {e})", prefix=name)
            import traceback
            traceback.print_exc()
            continue
        sfx = f"_{args.tag}" if args.tag else ""
        pd.DataFrame(all_rows).to_pickle(RESULTS_DIR / f'audit{sfx}.pkl')
        pd.DataFrame(all_trajs).to_pickle(RESULTS_DIR / f'trajs{sfx}.pkl')

    if not all_rows:
        _log("no results.")
        return

    df = pd.DataFrame(all_rows)
    _log("=" * 68)
    _log("CERTIFICATE AUDIT SUMMARY (pre-registered, report either way):")
    cols = ['dataset', 'point', 'support', 'B', 'g_null_norm', 'g_sc_norm',
            'null_certifies_stationary', 'fd_n_descent', 'termination']
    show = df.reindex(columns=cols)
    print(show.to_string(index=False))
    _log("-" * 68)
    # verdict per the pre-registered decision rule
    key = df[(df.point == 'sho_terminal_stdinit')]
    keystone = key[(key.null_certifies_stationary) & (key.fd_n_descent > 0)]
    if len(keystone):
        _log(f"KEYSTONE: null method RETURNS a certified-stationary solution with a "
             f"biactive descent direction on: {list(keystone.dataset)}")
    else:
        _log("No keystone at standard-init terminals: null certificate is not "
             "'stationary' at the returned interior solutions (support nonempty). "
             "=> phenomenon confined to empty-support regime (path top); report as scope bound.")
    amax = df[df.point == 'alpha_max']
    amax_false = amax[(amax.null_certifies_stationary) & (amax.fd_n_descent > 0)]
    if len(amax_false):
        _log(f"alpha_max false certificate confirmed on: {list(amax_false.dataset)} "
             f"(||g_null||<{STATIONARY_TOL:.0e} yet biactive FD-descent exists).")
    _log(f"saved -> {RESULTS_DIR}")


if __name__ == '__main__':
    main()
