"""Prototype: guarantee descent for the SC oracle when correlated (coupled) biactive
coordinates appear.

Diagnosis. The SC descent guarantee (Prop 15) is PER-COORDINATE: sigma_i q_i > 0 makes
-e_i a descent direction. When several biactive coordinates are correlated, the reduced
adjoint block H_S = gamma [nabla^2 F]_{S,S} has strong off-diagonals, so the JOINT
direction -h need not be descent even though each coordinate passes the coordinate-wise
test. We measure Phi'(x; -h) by finite differences and compare selection rules:

  null      : biactive excluded (Sparse-HO).
  sc        : plain sign-consistent pruning loop (current).
  greedy    : forward-select biactive coords, keep one only if it IMPROVES the joint
              directional derivative Phi'(x; -h) (descent-certified; guarantees descent).
  hess_dec  : pick biactive representatives that are near-orthogonal in the Hessian
              metric |H_ij|/sqrt(H_ii H_jj) < rho (decouples the block; Prop 15 lifts).

Usage
-----
    python improve_sc_coupled.py --case leukemia_cvopt
    python improve_sc_coupled.py --case mnist_legacy
    python improve_sc_coupled.py --case all
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
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
from sparse_ho.algo.implicit_variational import (
    _resolve_gamma, _resolve_lambdas, _partition_coordinates,
    _build_reduced_hessian_block)

INNER_TOL = 1e-10
INNER_MAX_ITER = 30000
FD_STEPS = [1e-3, 3e-3, 1e-2, 3e-2]
ALIASES = {'duke': 'duke breast-cancer'}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Point:
    """All Phase-1 quantities at a fixed hyperparameter x* (log-alpha)."""

    def __init__(self, model, X_tr, y_tr, X_val, y_val, x_star, scale_floor):
        self.model, self.X_tr, self.y_tr = model, X_tr, y_tr
        self.X_val, self.y_val = X_val, y_val
        self.x_star = x_star
        m = X_tr.shape[1]
        beta, mask, dense = _solve_beta(model, X_tr, y_tr, x_star, INNER_TOL,
                                        INNER_MAX_ITER)
        self.beta = beta
        self.gamma = _resolve_gamma(None, model, X_tr)
        self.alpha = np.exp(x_star)
        lambdas = _resolve_lambdas(model, x_star, m)
        if hasattr(model, 'set_variational_alpha'):
            model.set_variational_alpha(self.alpha)
        grad_F = np.asarray(model.get_grad_smooth(X_tr, y_tr, beta), float)
        v = beta - self.gamma * grad_F
        u = self.gamma * lambdas
        Ip, Im, A, Bp, Bm = _partition_coordinates(
            v, u, biactive_tol_rel=self.eps_b, scale_floor=scale_floor)
        self.Ip, self.Im, self.A, self.Bp, self.Bm = Ip, Im, A, Bp, Bm
        # z* = grad of held-out logistic loss wrt beta (no ridge)
        Xb = np.asarray(X_val @ beta).ravel() if issparse(X_val) else X_val @ beta
        s = _sigmoid(-y_val * Xb)
        nval = X_val.shape[0]
        zt = -(np.asarray(X_val.T @ (y_val * s)).ravel() if issparse(X_val)
               else X_val.T @ (y_val * s)) / nval
        self.z_star = zt
        self.phi0 = _phi_val(model, X_tr, y_tr, X_val, y_val, x_star, INNER_TOL,
                             INNER_MAX_ITER)
        self.m = m

    eps_b = 1e-2

    def direction(self, sel_bp, sel_bm):
        """Full hypergradient h for selection S = I ∪ sel (biactive subsets)."""
        S = self.Ip | self.Im | sel_bp | sel_bm
        if not np.any(S):
            return np.zeros(self.m)
        sigma = np.zeros(self.m)
        sigma[self.Ip | sel_bp] = +1.0
        sigma[self.Im | sel_bm] = -1.0
        H_S = _build_reduced_hessian_block(S, self.gamma, self.model, self.X_tr,
                                           self.y_tr, self.beta)
        nS = H_S.shape[0]
        H_S.flat[::nS + 1] += 1e-8
        rhs = -self.z_star[S]
        try:
            q_S = cho_solve(cho_factor(H_S, lower=True, check_finite=False), rhs,
                            check_finite=False)
        except LinAlgError:
            q_S = np.linalg.lstsq(H_S, rhs, rcond=None)[0]
        q = np.zeros(self.m)
        q[S] = q_S
        return self.alpha * sigma * self.gamma * q

    def fd_slope(self, h):
        """Phi'(x*; -h/||h||) by one-sided finite differences (descent if < 0)."""
        nh = norm(h)
        if nh == 0:
            return np.nan
        d = -h / nh
        slopes = []
        for t in FD_STEPS:
            xt = self.x_star + t * d
            phit = _phi_val(self.model, self.X_tr, self.y_tr, self.X_val,
                            self.y_val, xt, INNER_TOL, INNER_MAX_ITER)
            slopes.append((phit - self.phi0) / t)
        return float(np.median(slopes))


def _cand(pt):
    """Descent-aligned biactive candidates, ranked by |z*| descending."""
    cp = np.flatnonzero(pt.Bp & (pt.z_star < 0.0))
    cm = np.flatnonzero(pt.Bm & (pt.z_star > 0.0))
    cand = np.concatenate([cp, cm])
    order = np.argsort(-np.abs(pt.z_star[cand]))
    return cand[order], set(cp.tolist())


def sel_plain_sc(pt):
    orc = ImplicitVariational(
        policy=select_biactive_self_consistent, biactive_tol_rel=pt.eps_b,
        biactive_scale_floor=(1.0 if pt.legacy else 0.0), tol_lin_sys=1e-9)
    crit = _HeldOutLogisticWithMaxIter(pt.idx_tr, pt.idx_val, INNER_MAX_ITER)
    crit.get_val_grad(pt.model, pt.Xfull, pt.yfull, pt.x_star,
                      orc.compute_beta_grad, tol=INNER_TOL)
    info = dict(orc.last_run_info or {})
    mask = info.get('selected_biactive_mask', np.zeros(pt.m, bool))
    return (mask & pt.Bp), (mask & pt.Bm)


def sel_greedy(pt, cap=60):
    """Descent-certified forward selection: keep a biactive coord only if it makes
    the JOINT directional derivative more negative."""
    cand, cp_set = _cand(pt)
    cand = cand[:cap]
    sel_bp = np.zeros(pt.m, bool)
    sel_bm = np.zeros(pt.m, bool)
    best = pt.fd_slope(pt.direction(sel_bp, sel_bm))  # null direction slope
    if not np.isfinite(best):
        best = np.inf   # null direction is zero (starved): any descent improves
    for i in cand:
        tp, tm = sel_bp.copy(), sel_bm.copy()
        if i in cp_set:
            tp[i] = True
        else:
            tm[i] = True
        s = pt.fd_slope(pt.direction(tp, tm))
        if np.isfinite(s) and s < best - 1e-9:
            sel_bp, sel_bm, best = tp, tm, s
    return sel_bp, sel_bm


def sel_hess_decorrelated(pt, rho=0.3):
    """Pick biactive reps near-orthogonal in the Hessian metric (decouple block)."""
    cand, cp_set = _cand(pt)
    if cand.size == 0:
        return np.zeros(pt.m, bool), np.zeros(pt.m, bool)
    # Hessian diagonal + needed columns via matvec
    picked = []
    hcols = {}

    def hcol(j):
        if j not in hcols:
            e = np.zeros(pt.m)
            e[j] = 1.0
            hcols[j] = np.asarray(
                pt.model.get_hess_smooth(pt.X_tr, pt.y_tr, pt.beta, e), float)
        return hcols[j]
    diag = np.array([hcol(j)[j] for j in cand])
    for k, j in enumerate(cand):
        ok = True
        for p in picked:
            hij = hcol(j)[p]
            denom = np.sqrt(max(hcol(j)[j], 1e-30) * max(hcol(p)[p], 1e-30))
            if abs(hij) / denom > rho:
                ok = False
                break
        if ok:
            picked.append(j)
    sel_bp = np.zeros(pt.m, bool)
    sel_bm = np.zeros(pt.m, bool)
    for j in picked:
        if j in cp_set:
            sel_bp[j] = True
        else:
            sel_bm[j] = True
    return sel_bp, sel_bm


def _make_synth_correlated(k_dup=12, n=60, p=80, rho_noise=0.02, seed=0,
                           adversarial=False):
    """X with a near-perfectly-correlated relevant cluster (features 0..k_dup-1)
    plus independent noise features. The cluster shares one latent direction, so at
    a uniform penalty its members hit the soft-threshold kink TOGETHER (coupled
    biactivity) and the reduced Hessian block over them is near rank-1.

    adversarial=True flips the sign of half the cluster columns, so the coupled
    coordinates carry CONFLICTING gradient signs -- the case that most stresses the
    sign-consistency pruning loop."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)                      # shared latent
    Xc = z[:, None] + rho_noise * rng.standard_normal((n, k_dup))  # ~duplicates
    if adversarial:
        flips = rng.choice([-1.0, 1.0], size=k_dup)
        flips[0] = 1.0
        Xc = Xc * flips[None, :]                     # conflicting signs, |corr|~1
    Xn = rng.standard_normal((n, p - k_dup))        # noise features
    X = np.hstack([Xc, Xn])
    X /= (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
    y = np.sign(z + 0.1 * rng.standard_normal(n))   # label depends on the cluster
    y[y == 0] = 1.0
    return X, y


def build_case(case):
    if case == 'mnist_legacy':
        name, frac, legacy = 'mnist', 0.10, True
    elif case == 'leukemia_cvopt':
        name, frac, legacy = 'leukemia', None, False
    elif case == 'duke_cvopt':
        name, frac, legacy = 'duke', None, False
    elif case == 'leukemia_near':
        name, frac, legacy = 'leukemia', 0.97, False
    elif case.startswith('synth'):
        # synthetic correlated cluster: sweep the penalty to the regime where the
        # cluster is BIACTIVE (at the kink) rather than active or inactive.
        X, y = _make_synth_correlated()
        n, m = X.shape
        idx_tr, idx_val, idx_te = _make_split(n, 0)
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]
        model = WeightedSparseLogReg(alpha_l2=1.0 / len(idx_tr))
        a_max = _alpha_max(X_tr, y_tr)
        best = None
        for frac in np.geomspace(0.999, 0.80, 30):
            x_star = np.log(frac * a_max) * np.ones(m)
            pt = Point(model, X_tr, y_tr, X_val, y_val, x_star, scale_floor=0.0)
            B = int((pt.Bp | pt.Bm).sum())
            supp = int((np.abs(pt.beta) > 1e-10).sum())
            # want the cluster biactive and few active (biactive-dominated)
            if B > 0 and (best is None or B > best[0]):
                best = (B, frac, pt)
            if B >= 8 and supp <= 2:
                best = (B, frac, pt)
                break
        if best is None:
            frac = 0.95
            x_star = np.log(frac * a_max) * np.ones(m)
            pt = Point(model, X_tr, y_tr, X_val, y_val, x_star, scale_floor=0.0)
        else:
            _, frac, pt = best
        pt.legacy = False
        pt.idx_tr, pt.idx_val = idx_tr, idx_val
        pt.Xfull, pt.yfull = X, y
        return case, 'synth-correlated', frac, pt
    else:
        raise ValueError(case)
    loader = ALIASES.get(name, name)
    X, y = get_dataset(loader, Path(__file__).parent / 'data')
    if issparse(X):
        X = X.tocsc()
    if X.shape[0] > 6000:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], 6000, replace=False)
        X, y = X[idx], y[idx]
    n, m = X.shape
    idx_tr, idx_val, idx_te = _make_split(n, 0)
    Xp, _ = _preprocess_features_for_split(loader, X, idx_tr)
    if issparse(Xp):
        Xp = Xp.tocsc()
    X_tr, y_tr = Xp[idx_tr], y[idx_tr]
    X_val, y_val = Xp[idx_val], y[idx_val]
    model = WeightedSparseLogReg(alpha_l2=1.0 / len(idx_tr))
    a_max = _alpha_max(X_tr, y_tr)
    if frac is None:  # CV-optimal scalar penalty
        best_t, best_v = None, np.inf
        for t in np.geomspace(0.02, 0.99, 12):
            la = np.log(t * a_max) * np.ones(m)
            vv = _phi_val(model, X_tr, y_tr, X_val, y_val, la, INNER_TOL,
                          INNER_MAX_ITER)
            if vv < best_v:
                best_v, best_t = vv, t
        frac = best_t
    x_star = np.log(frac * a_max) * np.ones(m)
    pt = Point(model, X_tr, y_tr, X_val, y_val, x_star,
               scale_floor=(1.0 if legacy else 0.0))
    pt.legacy = legacy
    pt.idx_tr, pt.idx_val = idx_tr, idx_val
    pt.Xfull, pt.yfull = Xp, y
    return case, name, frac, pt


def _stress_sweep():
    """Try hard to break SC joint descent on genuinely-biactive coupled configs:
    vary correlation strength, cluster size, sample count, penalty."""
    _log("STRESS SWEEP — searching for a genuinely-biactive config where plain SC "
         "is NOT descent")
    n_broken = 0
    n_total = 0
    for adversarial in [False, True]:
      for rho_noise in [0.001, 0.01, 0.05]:
        for k_dup in [6, 12, 25]:
            for n in [40, 80]:
                for seed in range(3):
                    X, y = _make_synth_correlated(
                        k_dup=k_dup, n=n, p=max(80, 2 * k_dup),
                        rho_noise=rho_noise, seed=seed,
                        adversarial=adversarial)
                    idx_tr, idx_val, _ = _make_split(X.shape[0], 0)
                    model = WeightedSparseLogReg(alpha_l2=1.0 / len(idx_tr))
                    X_tr, y_tr = X[idx_tr], y[idx_tr]
                    X_val, y_val = X[idx_val], y[idx_val]
                    a_max = _alpha_max(X_tr, y_tr)
                    # find a biactive-dominated penalty
                    chosen = None
                    for frac in np.geomspace(0.999, 0.9, 20):
                        xs = np.log(frac * a_max) * np.ones(X.shape[1])
                        pt = Point(model, X_tr, y_tr, X_val, y_val, xs,
                                   scale_floor=0.0)
                        B = int((pt.Bp | pt.Bm).sum())
                        supp = int((np.abs(pt.beta) > 1e-10).sum())
                        if B >= 3 and supp <= max(2, k_dup // 2):
                            chosen = pt
                            break
                    if chosen is None:
                        continue
                    chosen.idx_tr, chosen.idx_val = idx_tr, idx_val
                    chosen.Xfull, chosen.yfull = X, y
                    chosen.legacy = False
                    sp, sm = sel_plain_sc(chosen)
                    slope = chosen.fd_slope(chosen.direction(sp, sm))
                    n_total += 1
                    is_desc = np.isfinite(slope) and slope < -1e-9
                    adv = 'ADV' if adversarial else 'pos'
                    if not is_desc:
                        n_broken += 1
                        gp, gm = sel_greedy(chosen)
                        gs = chosen.fd_slope(chosen.direction(gp, gm))
                        hp, hm = sel_hess_decorrelated(chosen)
                        hs = chosen.fd_slope(chosen.direction(hp, hm))
                        _log(f"  BROKEN [{adv}] rho={rho_noise} k={k_dup} n={n} "
                             f"seed={seed} B={int((chosen.Bp|chosen.Bm).sum())} "
                             f"sc_slope={slope:+.3e} greedy={gs:+.3e} "
                             f"hess_dec={hs:+.3e}")
                    else:
                        _log(f"  ok [{adv}] rho={rho_noise} k={k_dup} n={n} "
                             f"seed={seed} sc_slope={slope:+.3e}")
    _log(f"STRESS SWEEP done: {n_broken}/{n_total} configs where plain SC was NOT "
         f"descent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='all',
                    choices=['all', 'mnist_legacy', 'leukemia_cvopt',
                             'duke_cvopt', 'leukemia_near', 'synth', 'stress'])
    args = ap.parse_args()
    if args.case == 'stress':
        _stress_sweep()
        return
    cases = (['synth', 'leukemia_near', 'leukemia_cvopt']
             if args.case == 'all' else [args.case])
    for case in cases:
        _log("=" * 64)
        cse, name, frac, pt = build_case(case)
        B = int((pt.Bp | pt.Bm).sum())
        _log(f"CASE {cse}: {name} m={pt.m} frac={frac:.3g} "
             f"support={int((np.abs(pt.beta)>1e-10).sum())} biactive B={B}")
        variants = {
            'null': (np.zeros(pt.m, bool), np.zeros(pt.m, bool)),
            'sc': sel_plain_sc(pt),
            'greedy': sel_greedy(pt),
            'hess_dec': sel_hess_decorrelated(pt),
        }
        for vn, (sp, sm) in variants.items():
            h = pt.direction(sp, sm)
            slope = pt.fd_slope(h)
            nsel = int(sp.sum() + sm.sum())
            flag = ('DESCENT' if slope < -1e-9 else
                    ('~flat' if abs(slope) <= 1e-9 else 'ASCENT (not descent)'))
            _log(f"  {vn:9s} |sel|={nsel:4d}  Phi'(x;-h)={slope:+.4e}  "
                 f"||h||={norm(h):.3e}  -> {flag}")


if __name__ == '__main__':
    main()
