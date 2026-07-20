"""Core primitives for the certificate-soundness experiment (Experiment 7).

Self-contained audit functions. We import only STABLE shared helpers from the
Experiment-6 package (dataset loaders, train/val split, inner solver, held-out
objective) and the sparse_ho oracles; all audit logic lives here so the
experiment stands alone and is reproducible without the scratch scripts.

The object under test: at a hyperparameter point x, the support-restricted
("null") selection and the sign-consistent ("SC") selection are both valid
elements of the residual-enlarged subdifferential, but only one yields a sound
stationarity certificate. We adjudicate soundness against an INDEPENDENT ground
truth — the one-sided directional derivative Phi'(x; -e_i) of the held-out
objective, obtained by a model-free finite difference of the actual solver — and
NEVER by "SC agrees with SC."

Identity under test (manuscript Prop. sign_consistency_descent):
    Phi'(x; -e_i) = -h_i          (h_i = i-th component of the SC hypergradient)
so a biactive coordinate i is a FALSE-CERTIFICATE witness when
    g_null_i = 0   (null certifies flat)   while   Phi'(x;-e_i) < 0   (descent exists),
and the SC prediction -g_sc_i reproduces Phi'(x;-e_i).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import issparse

# stable shared helpers from the Experiment-6 package
_E6 = Path(__file__).parent.parent / 'expe6_real_world'
sys.path.insert(0, str(_E6))
from data_loaders import get_dataset                                   # noqa: E402,F401
from run_s2 import (                                                   # noqa: E402
    _make_split, _preprocess_features_for_split, _alpha_max,
    _HeldOutLogisticWithMaxIter)
from inner_helpers import _solve_beta, _phi_val                        # noqa: E402

from sparse_ho.models import WeightedSparseLogReg                      # noqa: E402,F401
from sparse_ho.algo import (                                           # noqa: E402
    ImplicitVariational, select_biactive_self_consistent)
from sparse_ho.algo.implicit_variational import (                     # noqa: E402
    _resolve_gamma, _resolve_lambdas)

# Tight inner tolerance for the audit: partition and FD ground truth need an
# accurate beta, independent of the looser tolerance used for outer HPO runs.
INNER_TOL = 1e-9
INNER_MAX_ITER = 20000
ACTIVE_THR = 1e-10
DEFAULT_EPS_B = 1e-3
DEFAULT_FD_STEP = 1e-2
# NormalizedSubgradient (Sparse-HO's optimizer here) calls a point stationary
# when the outer grad-norm falls below this; used to decide "null certifies".
STATIONARY_TOL = 1e-7


def partition_biactive(model, X_tr, y_tr, log_alpha, eps_B=DEFAULT_EPS_B,
                       scale_floor=0.0):
    """Biactive-and-inactive coordinate indices + support size at log_alpha.

    Mirrors Phase 1 of the subgradient oracle: coordinate i is biactive when
    | |v_i| - u_i | <= eps_B * max(|v_i|, u_i), with v = beta - gamma*grad_F,
    u = gamma*lambda, and it is outside the primal support (beta_i ~ 0).
    """
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    beta, _, _ = _solve_beta(model, X_tr, y_tr, log_alpha, INNER_TOL,
                             INNER_MAX_ITER)
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


def oracle_grads(model, X, y, idx_train, idx_val, log_alpha,
                 eps_B=DEFAULT_EPS_B, scale_floor=0.0):
    """Full null and SC hypergradient vectors at log_alpha (shared inner solve)."""
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
        g_null=g_null, g_sc=g_sc,
        g_null_norm=float(np.linalg.norm(g_null)),
        g_sc_norm=float(np.linalg.norm(g_sc)),
        g_disc=float(np.linalg.norm(g_sc - g_null)),
        sc_selected=int(info.get('selected_biactive_size', 0)))


def fd_directional(model, X, y, idx_train, idx_val, log_alpha, coords, fd_step):
    """Phi'(x; -e_i) by a one-sided finite difference of the real solver.

    Returns phi0 and, per coordinate i in ``coords``, the slope
    (Phi(x - fd_step e_i) - Phi(x)) / fd_step  ~  Phi'(x; -e_i).
    """
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    if issparse(X_val):
        X_val = X_val.tocsc()
    phi0 = _phi_val(model, X_tr, y_tr, X_val, y_val, log_alpha,
                    INNER_TOL, INNER_MAX_ITER)
    out = {}
    for i in coords:
        la = np.asarray(log_alpha, float).copy()
        la[i] = la[i] - fd_step
        phi1 = _phi_val(model, X_tr, y_tr, X_val, y_val, la,
                        INNER_TOL, INNER_MAX_ITER)
        out[int(i)] = float((phi1 - phi0) / fd_step)
    return float(phi0), out


def audit_point(dataset, location, model, X, y, idx_tr, idx_val, log_alpha,
                eps_B=DEFAULT_EPS_B, scale_floor=0.0, fd_step=DEFAULT_FD_STEP):
    """One row: partition + oracle norms + ground-truth reconciliation at i*.

    i* = the biactive coordinate with the most negative Phi'(x;-e_i) (strongest
    certified descent). At i* we report the FD ground truth, the SC prediction
    -g_sc[i*], the null prediction -g_null[i*] (=0), and the match error.
    """
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    bi_idx, supp = partition_biactive(model, X_tr, y_tr, log_alpha, eps_B,
                                      scale_floor)
    og = oracle_grads(model, X, y, idx_tr, idx_val, log_alpha, eps_B, scale_floor)
    phi0, slopes = fd_directional(model, X, y, idx_tr, idx_val, log_alpha,
                                  bi_idx, fd_step)
    recs = []
    for i in bi_idx:
        gt = slopes[int(i)]                       # Phi'(x; -e_i), ground truth
        recs.append(dict(
            coord=int(i), gt_dderiv=gt,
            sc_pred=float(-og['g_sc'][i]),        # -h_sc,i  (SC certificate)
            null_pred=float(-og['g_null'][i]),    # -h_null,i = 0
            descent=bool(gt < -1e-9),
            sc_match_err=float(abs(gt - (-og['g_sc'][i]))),
            sc_match_rel=float(abs(gt - (-og['g_sc'][i])) /
                               max(abs(gt), 1e-12))))
    n_desc = int(sum(r['descent'] for r in recs))
    istar = min(recs, key=lambda r: r['gt_dderiv']) if recs else None
    null_certifies = bool(og['g_null_norm'] < STATIONARY_TOL)
    false_certificate = bool(null_certifies and n_desc > 0)
    row = dict(
        dataset=dataset, location=location, eps_B=eps_B, fd_step=fd_step,
        scale_floor=scale_floor, support=supp, B=len(bi_idx),
        g_null_norm=og['g_null_norm'], g_sc_norm=og['g_sc_norm'],
        g_disc=og['g_disc'], sc_selected=og['sc_selected'],
        null_certifies_stationary=null_certifies,
        fd_n_descent=n_desc, false_certificate=false_certificate,
        phi0=phi0, records=recs)
    if istar is not None:
        row.update(
            istar=istar['coord'], istar_gt=istar['gt_dderiv'],
            istar_sc_pred=istar['sc_pred'], istar_null_pred=istar['null_pred'],
            istar_match_err=istar['sc_match_err'],
            istar_match_rel=istar['sc_match_rel'])
    else:
        row.update(istar=-1, istar_gt=np.nan, istar_sc_pred=np.nan,
                   istar_null_pred=np.nan, istar_match_err=np.nan,
                   istar_match_rel=np.nan)
    return row


def load_split(name, seed, data_dir):
    """Load a dataset, make the standard split, preprocess. Returns everything
    the audit needs (model, X, y, indices, alpha_max)."""
    X, y = get_dataset(name, data_dir)
    if issparse(X):
        X = X.tocsc()
    n = X.shape[0]
    idx_tr, idx_val, idx_te = _make_split(n, seed, y=y)
    X, _ = _preprocess_features_for_split(name, X, idx_tr)
    if issparse(X):
        X = X.tocsc()
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    if issparse(X_tr):
        X_tr = X_tr.tocsc()
    alpha_l2 = 1.0 / len(idx_tr)
    model = WeightedSparseLogReg(alpha_l2=alpha_l2)
    a_max = _alpha_max(X_tr, y_tr)
    return dict(model=model, X=X, y=y, idx_tr=idx_tr, idx_val=idx_val,
                idx_te=idx_te, a_max=a_max, m=X.shape[1], n=n)
