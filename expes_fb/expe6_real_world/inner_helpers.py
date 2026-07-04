"""Stateless inner-solve / objective helpers shared by the Experiment 6 scripts.

These are pure functions (no criterion warm-start caching, no experiment config)
used by the biactivity diagnostic (``scan_biactivity.py``). They were factored out
of the Setting-3 runner so the diagnostic does not depend on it.
"""

import numpy as np
from scipy.sparse import issparse

from sparse_ho.algo.forward import compute_beta
from sparse_ho.criterion import HeldOutLogistic


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


def _best_log_alpha(mon):
    if not mon.objs:
        return None
    k = int(np.argmin(mon.objs))
    return np.log(np.maximum(np.asarray(mon.alphas[k], dtype=float), 1e-300))
