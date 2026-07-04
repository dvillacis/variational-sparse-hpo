"""Experiment 6 Setting 3 — natural-degeneracy / gradient-starvation counterfactual.

Tests, on the FULLY REAL Setting-2 datasets (standard uniform init, no planted
signal, band decoupled from any calibration), whether biactive gradient starvation
arises naturally and is consequential. Design is a falsifiable adjudicator; see
``Revision2_23062026/EXP_NATURAL_STARVATION_SPEC.md``.

Three sub-experiments per (dataset, seed):
  A. warm-start counterfactual: from the Sparse-HO fixed point x*, continue with the
     SAME optimizer (NBA) under {null, SC} oracles. Predict dPhi_null ~= 0,
     dPhi_SC > 0 iff the SC-selected biactive set is non-empty. Plus held-out TEST F1.
  B. tau x eps_B invariance: at x*, re-audit across an inner-tol ladder and an eps_B
     plateau. B(tau) must not shrink (true kinks, not under-solve); dPhi plateau vs eps_B.
  C. branch-stability + Prop-15: perturb -/+ t e_i, confirm activation, and match the
     finite-difference slope of Phi against -h_i^SC.

Usage
-----
    python run_s3_counterfactual.py                       # full frozen config
    python run_s3_counterfactual.py --smoke               # quick sanity on cheap data
    python run_s3_counterfactual.py --datasets phishing mnist --n-seeds 2 \
        --max-samples 3000 --n-outer-converge 15 --n-outer-cont 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).parent))
import config_s3 as C
from data_loaders import get_dataset
# reuse the pure helpers from the Setting-2 runner (no main() runs on import)
from run_s2 import (
    _make_split,
    _preprocess_features_for_split,
    _test_f1,
    _alpha_max,
    _HeldOutLogisticWithMaxIter,
)

from sparse_ho.models import WeightedSparseLogReg
from sparse_ho.criterion import HeldOutLogistic
from sparse_ho.algo import (
    Implicit, ImplicitVariational, make_select_biactive_self_consistent_topM,
    select_biactive_self_consistent)
from sparse_ho.algo.forward import compute_beta
from sparse_ho.optimizers import NormalizedSubgradient, TrustRegion
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor

RESULTS_DIR = Path(__file__).parent / 'results' / 'setting3'


def _log(msg, prefix=None):
    stamp = time.strftime('%H:%M:%S')
    print(f"[{stamp}]" + (f" [{prefix}]" if prefix else "") + f" {msg}", flush=True)


# ---------------------------------------------------------------------------
# stateless inner-solve / objective helpers (no criterion warm-start caching)
# ---------------------------------------------------------------------------

def _solve_beta(model, X_tr, y_tr, log_alpha, tol, max_iter):
    """Full beta vector at a fixed hyperparameter (stateless)."""
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    mask, dense, _ = compute_beta(
        X_tr, y_tr, log_alpha, model, tol=tol, compute_jac=False,
        max_iter=max_iter)
    m = X_tr.shape[1]
    beta = np.zeros(m)
    beta[mask] = dense
    return beta, mask, dense


def _phi_val(model, X_tr, y_tr, X_val, y_val, log_alpha, tol, max_iter):
    """Upper-level objective Phi(x) = held-out logistic loss (stateless)."""
    _, mask, dense = _solve_beta(model, X_tr, y_tr, log_alpha, tol, max_iter)
    return float(HeldOutLogistic.get_val_outer(X_val, y_val, mask, dense))


def _audit(oracle, model, X, y, idx_train, idx_val, x_log, tol, max_iter):
    """Single-point hypergradient + biactive incidence at x_log.

    Uses a FRESH criterion so the inner solve is not warm-started from a
    previous probe (rigor: the biactive partition must reflect x_log alone).
    Returns (phi, grad, info) where info carries biactive_size / selected sizes.
    """
    crit = _HeldOutLogisticWithMaxIter(idx_train, idx_val, max_iter)
    val, grad = crit.get_val_grad(
        model, X, y, x_log, oracle.compute_beta_grad, tol=tol)
    info = dict(getattr(oracle, 'last_run_info', {}) or {})
    return float(val), np.asarray(grad, dtype=float), info


def _make_optimizer(kind, n_outer, tol):
    if kind == 'trust_region':
        return TrustRegion(n_outer=n_outer, radius0=C.NBA_STEP_SIZE, tol=tol)
    return NormalizedSubgradient(
        n_outer=n_outer, step_size=C.NBA_STEP_SIZE, tol=tol)


def _continue(oracle, model, X, y, idx_train, idx_val, x_start_log,
              n_outer, tol, opt_kind='nba', fallback_oracle=None):
    """Warm-start continuation from x_start_log under a FIXED optimizer.

    Returns (Monitor, optimizer); the optimizer carries the fallback
    counters when `fallback_oracle` is given (trust_region only).
    """
    crit = _HeldOutLogisticWithMaxIter(idx_train, idx_val, C.INNER_MAX_ITER)
    opt = _make_optimizer(opt_kind, n_outer, tol)
    mon = Monitor()
    grad_search(oracle, crit, model, opt, X, y, np.exp(x_start_log), mon,
                fallback_algo=fallback_oracle)
    return mon, opt


def _best_log_alpha(mon):
    if not mon.objs:
        return None
    k = int(np.argmin(mon.objs))
    return np.log(np.maximum(np.asarray(mon.alphas[k], dtype=float), 1e-300))


# ---------------------------------------------------------------------------
# per (dataset, seed) driver
# ---------------------------------------------------------------------------

def run_one(dname, X, y, seed, args):
    pfx = f"s3 {dname} seed={seed}"
    n_total = X.shape[0]
    idx_train, idx_val, idx_test = _make_split(n_total, seed)
    X, _ = _preprocess_features_for_split(dname, X, idx_train)
    if issparse(X):
        X = X.tocsc()
    m = X.shape[1]
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_te, y_te = X[idx_test], y[idx_test]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    alpha_l2 = 1.0 / len(idx_train)
    a_max = _alpha_max(X_tr, y_tr)
    log_alpha0 = np.log(C.INIT_FRACTION * a_max) * np.ones(m)
    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    max_iter = C.INNER_MAX_ITER
    eps_ref = C.BIACTIVE_TOL_REL_REF
    tref = C.INNER_TOL_REF

    _log(f"m={m} alpha_max={a_max:.3e} converging Sparse-HO "
         f"({args.n_outer_converge} outer)...", prefix=pfx)

    # ---- 1. Sparse-HO fixed point --------------------------------------
    mon0 = Monitor()
    grad_search(
        Implicit(),
        _HeldOutLogisticWithMaxIter(idx_train, idx_val, max_iter),
        model,
        NormalizedSubgradient(
            n_outer=args.n_outer_converge, step_size=C.NBA_STEP_SIZE, tol=tref),
        X, y, np.exp(log_alpha0), mon0)
    x_star = _best_log_alpha(mon0)
    phi_star = float(min(mon0.objs))
    phi_star_fd = _phi_val(model, X_tr, y_tr, X_val, y_val, x_star, tref, max_iter)
    _log(f"fixed point reached: phi*={phi_star:.6f}", prefix=pfx)

    # ---- 2. audits at x_star -------------------------------------------
    sf = 1.0 if args.legacy_band else 0.0
    orc_null = ImplicitVariational(
        policy=None, biactive_tol_rel=eps_ref, biactive_scale_floor=sf,
        tol_lin_sys=C.LIN_SYS_TOL)
    orc_sc = ImplicitVariational(
        policy=select_biactive_self_consistent,
        biactive_tol_rel=eps_ref, biactive_scale_floor=sf,
        tol_lin_sys=C.LIN_SYS_TOL)
    _, g_null, info_null = _audit(
        orc_null, model, X, y, idx_train, idx_val, x_star, tref, max_iter)
    _, g_sc, info_sc = _audit(
        orc_sc, model, X, y, idx_train, idx_val, x_star, tref, max_iter)
    B = float(info_sc.get('biactive_size', np.nan))
    m_sel = float(info_sc.get('selected_biactive_size', np.nan))
    hgrad_disc = float(np.linalg.norm(g_sc - g_null))
    _log(f"audit: biactive B={B:.0f} selected m_sel={m_sel:.0f} "
         f"||g_sc-g_null||={hgrad_disc:.3e}", prefix=pfx)

    # ---- 3. continuations (optimizer held fixed) -----------------------
    ck = args.continuation_optimizer
    monA, _ = _continue(orc_null, model, X, y, idx_train, idx_val, x_star,
                        args.n_outer_cont, tref, opt_kind=ck)
    monB, _ = _continue(orc_sc, model, X, y, idx_train, idx_val, x_star,
                        args.n_outer_cont, tref, opt_kind=ck)
    monAp, _ = _continue(Implicit(), model, X, y, idx_train, idx_val, x_star,
                         args.n_outer_cont, tref, opt_kind=ck)
    dPhi_A = float(monA.objs[0] - min(monA.objs)) if monA.objs else np.nan
    dPhi_B = float(monB.objs[0] - min(monB.objs)) if monB.objs else np.nan
    dPhi_Ap = float(monAp.objs[0] - min(monAp.objs)) if monAp.objs else np.nan

    # ---- 3b. SC + descent-verified fallback arm (safeguard) ------------
    monC, optC = (None, None)
    dPhi_C = np.nan
    if args.fallback_arm:
        monC, optC = _continue(
            orc_sc, model, X, y, idx_train, idx_val, x_star,
            args.n_outer_cont, tref, opt_kind=ck, fallback_oracle=orc_null)
        dPhi_C = float(monC.objs[0] - min(monC.objs)) if monC.objs else np.nan

    # ---- 5. consequence (test F1 + support delta) ----------------------
    def _beta_of(mon):
        la = _best_log_alpha(mon)
        if la is None:
            return np.zeros(m)
        beta, _, _ = _solve_beta(model, X_tr, y_tr, la, tref, max_iter)
        return beta
    betaA, betaB = _beta_of(monA), _beta_of(monB)
    f1A, f1B = _test_f1(X_te, y_te, betaA), _test_f1(X_te, y_te, betaB)
    suppA = set(np.where(np.abs(betaA) > C.ACTIVE_THR)[0])
    suppB = set(np.where(np.abs(betaB) > C.ACTIVE_THR)[0])
    cf_row = dict(
        dataset=dname, seed=seed, m=m, phi_star=phi_star,
        biactive=B, selected_biactive=m_sel, hgrad_discrepancy=hgrad_disc,
        dPhi_null=dPhi_A, dPhi_sc=dPhi_B, dPhi_sparseho=dPhi_Ap,
        phi0_null=float(monA.objs[0]) if monA.objs else np.nan,
        phi0_sc=float(monB.objs[0]) if monB.objs else np.nan,
        test_f1_null=f1A, test_f1_sc=f1B, d_test_f1=float(f1B - f1A),
        support_added=int(len(suppB - suppA)),
        n_iter_null=len(monA.objs), n_iter_sc=len(monB.objs),
    )
    _log(f"counterfactual [{args.continuation_optimizer}]: "
         f"dPhi_null={dPhi_A:.3e} dPhi_sc={dPhi_B:.3e} "
         f"d_test_f1={f1B - f1A:+.4f}", prefix=pfx)

    if args.fallback_arm:
        betaC = _beta_of(monC)
        f1C = _test_f1(X_te, y_te, betaC)
        suppC = set(np.where(np.abs(betaC) > C.ACTIVE_THR)[0])
        cf_row.update(
            dPhi_sc_fb=dPhi_C,
            phi0_sc_fb=float(monC.objs[0]) if monC.objs else np.nan,
            test_f1_sc_fb=f1C, d_test_f1_fb=float(f1C - f1A),
            support_added_fb=int(len(suppC - suppA)),
            n_iter_sc_fb=len(monC.objs),
            n_fb_tried=int(getattr(optC, 'n_fallback_tried_', 0)),
            n_fb_accepted=int(getattr(optC, 'n_fallback_accepted_', 0)),
        )
        _log(f"fallback arm [{args.continuation_optimizer}]: "
             f"dPhi_sc_fb={dPhi_C:.3e} d_test_f1_fb={f1C - f1A:+.4f} "
             f"fb_tried={cf_row['n_fb_tried']} "
             f"fb_accepted={cf_row['n_fb_accepted']}", prefix=pfx)

    # ---- 3c. capped sign-consistent arms (top-M seed + SC pruning) ------
    for M in args.topm_arms:
        orc_M = ImplicitVariational(
            policy=make_select_biactive_self_consistent_topM(M),
            biactive_tol_rel=eps_ref, biactive_scale_floor=sf,
            tol_lin_sys=C.LIN_SYS_TOL)
        # one-step descent audit at x_star along the capped-SC direction
        v_M, g_M, info_M = _audit(
            orc_M, model, X, y, idx_train, idx_val, x_star, tref, max_iter)
        m_sel_M = float(info_M.get('selected_biactive_size', np.nan))
        dphi1_M = np.nan
        g_M_norm = float(np.linalg.norm(g_M))
        if g_M_norm > 0:
            crit_p = _HeldOutLogisticWithMaxIter(idx_train, idx_val, max_iter)
            x_new = crit_p.proj_hyperparam(
                model, X, y, x_star - C.NBA_STEP_SIZE * g_M / g_M_norm)
            phi_new = _phi_val(
                model, X_tr, y_tr, X_val, y_val, x_new, tref, max_iter)
            dphi1_M = float(v_M - phi_new)
        # continuation WITHOUT fallback: does the capped direction descend?
        monM, optM = _continue(
            orc_M, model, X, y, idx_train, idx_val, x_star,
            args.n_outer_cont, tref, opt_kind=ck)
        dPhi_M = float(monM.objs[0] - min(monM.objs)) if monM.objs else np.nan
        betaM = _beta_of(monM)
        f1M = _test_f1(X_te, y_te, betaM)
        hist = getattr(optM, 'history_', [])
        n_acc_M = sum(
            1 for r in hist
            if int(r.get('accepted', 0)) == 1
            and np.isfinite(r.get('rho', np.nan)))
        cf_row.update({
            f'msel_top{M}': m_sel_M,
            f'dphi1_top{M}': dphi1_M,
            f'dPhi_sc_top{M}': dPhi_M,
            f'test_f1_top{M}': f1M,
            f'd_test_f1_top{M}': float(f1M - f1A),
            f'n_acc_top{M}': int(n_acc_M),
            f'n_iter_top{M}': len(monM.objs),
        })
        _log(f"topM arm M={M} [{ck}]: m_sel={m_sel_M:.0f} "
             f"dphi1={dphi1_M:+.3e} dPhi={dPhi_M:.3e} acc={n_acc_M} "
             f"d_test_f1={f1M - f1A:+.4f}", prefix=pfx)

    if args.counterfactual_only:
        return cf_row, [], []

    # ---- sub-experiment B: tau x eps_B invariance (audit-only + 1-step) --
    inv_rows = []
    for tau in args.tau_ladder:
        for eps in args.eps_grid:
            alg_sc = ImplicitVariational(
                policy=select_biactive_self_consistent,
                biactive_tol_rel=eps, biactive_scale_floor=sf,
                tol_lin_sys=C.LIN_SYS_TOL)
            alg_nu = ImplicitVariational(
                policy=None, biactive_tol_rel=eps, biactive_scale_floor=sf,
                tol_lin_sys=C.LIN_SYS_TOL)
            v_sc, gg_sc, inf = _audit(
                alg_sc, model, X, y, idx_train, idx_val, x_star, tau, max_iter)
            _, gg_nu, _ = _audit(
                alg_nu, model, X, y, idx_train, idx_val, x_star, tau, max_iter)
            bi_norm = float(np.linalg.norm(gg_sc - gg_nu))
            dphi1 = np.nan
            if C.COMPUTE_DPHI_GRID and np.linalg.norm(gg_sc) > 0:
                crit_p = _HeldOutLogisticWithMaxIter(idx_train, idx_val, max_iter)
                x_new = crit_p.proj_hyperparam(
                    model, X, y,
                    x_star - C.NBA_STEP_SIZE * gg_sc / np.linalg.norm(gg_sc))
                phi_new = _phi_val(
                    model, X_tr, y_tr, X_val, y_val, x_new, tau, max_iter)
                dphi1 = float(v_sc - phi_new)
            inv_rows.append(dict(
                dataset=dname, seed=seed, tau=tau, eps_B=eps,
                biactive=float(inf.get('biactive_size', np.nan)),
                selected_biactive=float(inf.get('selected_biactive_size', np.nan)),
                bi_hgrad_norm=bi_norm, dphi_1step=dphi1))
    _log("invariance grid done", prefix=pfx)

    # ---- sub-experiment C: branch-stability + Prop-15 finite difference --
    br_rows = []
    sel = np.where(np.abs(g_sc - g_null) > C.DIFF_THR)[0]
    if sel.size > C.MAX_BRANCH_COORDS:
        # probe the strongest SC contributions
        order = np.argsort(-np.abs((g_sc - g_null)[sel]))
        sel = sel[order[:C.MAX_BRANCH_COORDS]]
    for i in sel:
        h_i = float(g_sc[i])
        for t in C.BRANCH_T:
            la_m = x_star.copy(); la_m[i] -= t
            la_p = x_star.copy(); la_p[i] += t
            beta_m, mask_m, dense_m = _solve_beta(
                model, X_tr, y_tr, la_m, tref, max_iter)
            beta_p, _, _ = _solve_beta(model, X_tr, y_tr, la_p, tref, max_iter)
            active_m = bool(abs(beta_m[i]) > C.ACTIVE_THR)
            active_p = bool(abs(beta_p[i]) > C.ACTIVE_THR)
            phi_m = float(HeldOutLogistic.get_val_outer(
                X_val, y_val, mask_m, dense_m))
            fd_slope = (phi_m - phi_star_fd) / t   # ~ Phi'(x; -e_i) = -h_i
            br_rows.append(dict(
                dataset=dname, seed=seed, coord=int(i), t=t,
                branch_ok=bool(active_m and not active_p),
                active_minus=active_m, active_plus=active_p,
                fd_slope=fd_slope, minus_h_i=-h_i, h_i=h_i))
    _log(f"branch check done on {sel.size} coords", prefix=pfx)

    return cf_row, inv_rows, br_rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 6 Setting 3.")
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--n-seeds', type=int, default=None)
    p.add_argument('--n-outer-converge', type=int, default=C.N_OUTER_CONVERGE)
    p.add_argument('--n-outer-cont', type=int, default=C.N_OUTER_CONT)
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--continuation-optimizer', choices=['nba', 'trust_region'],
                   default='nba', dest='continuation_optimizer',
                   help="optimizer for the warm-start continuation arms.")
    p.add_argument('--counterfactual-only', action='store_true',
                   dest='counterfactual_only',
                   help="skip the invariance grid and branch check (fast probe).")
    p.add_argument('--fallback-arm', action='store_true', dest='fallback_arm',
                   help="add a continuation arm: SC oracle with null-oracle "
                        "fallback on trust-region rejection (safeguard). "
                        "Only meaningful with --continuation-optimizer "
                        "trust_region.")
    p.add_argument('--topm-arms', nargs='+', type=int, default=[],
                   dest='topm_arms', metavar='M',
                   help="add continuation arms with the top-M capped "
                        "sign-consistent policy (no fallback), one per M; "
                        "also records a one-step descent audit at x*.")
    p.add_argument('--legacy-band', action='store_true', dest='legacy_band',
                   help="use the legacy biactive detection scale "
                        "max(|v|,u,1) instead of the scale-free max(|v|,u); "
                        "reproduces results from tags recorded before the "
                        "scale-floor fix (e.g. mnist_tr, mnist_tr_fb).")
    p.add_argument('--tag', default='',
                   help="suffix for output pkls (avoids clobbering prior runs).")
    p.add_argument('--smoke', action='store_true',
                   help="fast sanity run: phishing+mnist, 2 seeds, tiny budgets.")
    args = p.parse_args()
    if args.smoke:
        args.datasets = args.datasets or ['phishing', 'mnist']
        args.n_seeds = args.n_seeds or 2
        args.n_outer_converge = min(args.n_outer_converge, 15)
        args.n_outer_cont = min(args.n_outer_cont, 10)
        args.max_samples = args.max_samples or 3000
    args.datasets = args.datasets or C.DATASETS
    args.n_seeds = args.n_seeds or C.N_SEEDS
    # smoke uses a reduced invariance grid for speed
    args.tau_ladder = [1e-4, 1e-8] if args.smoke else C.TAU_LADDER
    args.eps_grid = [1e-4, 1e-3, 1e-1] if args.smoke else C.EPS_B_PLATEAU
    return args


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"datasets={args.datasets} n_seeds={args.n_seeds} "
         f"converge={args.n_outer_converge} cont={args.n_outer_cont} "
         f"max_samples={args.max_samples} smoke={args.smoke}")

    cf_rows, inv_rows, br_rows = [], [], []
    for dname in args.datasets:
        _log("=" * 60)
        _log(f"dataset={dname}")
        try:
            X, y = get_dataset(dname, Path(__file__).parent / 'data')
        except Exception as e:  # noqa: BLE001
            _log(f"skip ({e})", prefix=dname)
            continue
        if issparse(X):
            X = X.tocsc()
        if args.max_samples is not None and X.shape[0] > args.max_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], args.max_samples, replace=False)
            X, y = X[idx], y[idx]
            _log(f"subsampled to n={X.shape[0]:,}", prefix=dname)
        _log(f"n={X.shape[0]:,} m={X.shape[1]:,}", prefix=dname)

        seeds = list(range(args.n_seeds))
        if args.n_jobs == 1:
            outs = [run_one(dname, X, y, s, args) for s in seeds]
        else:
            from joblib import Parallel, delayed
            outs = Parallel(n_jobs=min(args.n_jobs, len(seeds)), backend='loky')(
                delayed(run_one)(dname, X, y, s, args) for s in seeds)

        for cf, inv, br in outs:
            cf_rows.append(cf)
            inv_rows.extend(inv)
            br_rows.extend(br)

        # incremental save after each dataset (tagged to avoid clobbering)
        sfx = f"_{args.tag}" if args.tag else ""
        pd.DataFrame(cf_rows).to_pickle(RESULTS_DIR / f'results_counterfactual{sfx}.pkl')
        if inv_rows:
            pd.DataFrame(inv_rows).to_pickle(RESULTS_DIR / f'results_invariance{sfx}.pkl')
        if br_rows:
            pd.DataFrame(br_rows).to_pickle(RESULTS_DIR / f'results_branch{sfx}.pkl')
        _log(f"saved partial results after {dname}", prefix=dname)

    if not cf_rows:
        _log("no results produced.")
        return
    df = pd.DataFrame(cf_rows)
    _log("=" * 60)
    _log("SUMMARY (mean over seeds):")
    summ = df.groupby('dataset')[
        ['biactive', 'selected_biactive', 'dPhi_null', 'dPhi_sc',
         'd_test_f1']].mean()
    print(summ.to_string())
    _log(f"saved -> {RESULTS_DIR}")


if __name__ == '__main__':
    main()
