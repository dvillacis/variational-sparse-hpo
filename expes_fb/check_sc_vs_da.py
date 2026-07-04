"""Compare DA vs SC biactive selection policies on a single problem instance.

Checks whether the SC fixed-point refinement actually removes any biactive
coordinates relative to the DA initialization, and how much the hypergradients
and adjoint vectors differ.

Run from the repo root:
    python expes_fb/check_sc_vs_da.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'shared'))
from data_gen_degenerate import make_degenerate_dataset

from sparse_ho.models import WeightedElasticNet
from sparse_ho.algo import (
    ImplicitVariational,
    select_biactive_by_zstar_sign,
    select_biactive_self_consistent,
)
from sparse_ho.algo.forward import compute_beta
from sparse_ho.algo.implicit_variational import (
    _partition_coordinates,
    _resolve_gamma,
    _resolve_lambdas,
    _make_full_beta,
)

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

M = 300
N_RATIO = 2 / 3
RHO = 0.95
BIACTIVE_TOL_REL = 0.10
CALIBRATION_SLACK = 0.05
INNER_TOL = 1e-8

rng = np.random.default_rng(42)
n = int(N_RATIO * M)
n_easy  = max(int(0.04 * M), 5)
n_dist  = max(int(0.04 * M), 5)
n_hid   = n_dist
alpha_l2 = 1.0 / int(0.6 * n)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    X, y, beta_true, groups, idx_train, idx_val, idx_test, log_alpha0 = (
        make_degenerate_dataset(
            n, M, n_easy, n_dist, n_hid,
            rho=RHO, alpha_l2=alpha_l2,
            calibration_slack=CALIBRATION_SLACK,
            rng=rng,
        )
    )

X_tr, y_tr = X[idx_train], y[idx_train]
X_val, y_val = X[idx_val], y[idx_val]
model = WeightedElasticNet(alpha_l2=alpha_l2)

def get_grad_outer(mask, dense):
    """z* = ∇_y L = 2/n_val · X_val[:,S]ᵀ (X_val[:,S] dense - y_val)."""
    X_val_m = X_val[:, mask]
    return 2 * (X_val_m.T @ (X_val_m @ dense - y_val)) / len(y_val)

# ---------------------------------------------------------------------------
# Compute β at initial hyperparameter
# ---------------------------------------------------------------------------

mask, dense, _ = compute_beta(
    X_tr, y_tr, log_alpha0, model, tol=INNER_TOL, compute_jac=False)
n_features = X_tr.shape[1]
beta = _make_full_beta(mask, dense, n_features)

gamma = _resolve_gamma(None, model, X_tr)
lambdas = _resolve_lambdas(model, log_alpha0, n_features)
grad_F = model.get_grad_smooth(X_tr, y_tr, beta)
v_bar = beta - gamma * grad_F
u_bar = gamma * lambdas

I_plus, I_minus, A, B_plus, B_minus = _partition_coordinates(
    v_bar, u_bar,
    biactive_tol_abs=0.0,
    biactive_tol_rel=BIACTIVE_TOL_REL,
)

# z* = ∇_y L(x, ȳ) — validation gradient (full dimension)
mask_full = np.ones(n_features, dtype=bool)
z_star = np.asarray(get_grad_outer(mask_full, beta), dtype=float)

# ---------------------------------------------------------------------------
# Apply both policies
# ---------------------------------------------------------------------------

MB_da_plus, MB_da_minus = select_biactive_by_zstar_sign(
    I_plus, I_minus, A, B_plus, B_minus, z_star=z_star
)

MB_sc_plus, MB_sc_minus = select_biactive_self_consistent(
    I_plus, I_minus, A, B_plus, B_minus,
    z_star=z_star, gamma=gamma, model=model, X=X_tr, y=y_tr, beta=beta,
)

S_da = I_plus | I_minus | MB_da_plus | MB_da_minus
S_sc = I_plus | I_minus | MB_sc_plus | MB_sc_minus

# ---------------------------------------------------------------------------
# Compute full hypergradients via ImplicitVariational
# ---------------------------------------------------------------------------

def _hypergrad(policy):
    algo = ImplicitVariational(policy=policy, biactive_tol_rel=BIACTIVE_TOL_REL)
    _, _, jac_v, _, sets = algo.compute_beta_grad(
        X_tr, y_tr, log_alpha0, model, get_grad_outer,
        gamma=gamma,
        return_sets=True,
    )
    return jac_v, sets

hg_da, sets_da = _hypergrad(select_biactive_by_zstar_sign)
hg_sc, sets_sc = _hypergrad(select_biactive_self_consistent)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

n_strict_active = int(np.sum(I_plus | I_minus))
n_biactive       = int(np.sum(B_plus | B_minus))
n_hid_group      = len(groups['hidden'])

print(f"\n{'='*60}")
print(f"  Problem:  n={n}, m={M}, rho={RHO}, n_hidden={n_hid_group}")
print(f"  γ = {gamma:.4e}")
print(f"{'='*60}")
print(f"\nCoordinate partition at log_alpha0:")
print(f"  Strict active (I+ ∪ I-) : {n_strict_active}")
print(f"  Biactive   (B+ ∪ B-)    : {n_biactive}")
print(f"    of which in hidden group: "
      f"{int(np.sum((B_plus | B_minus)[groups['hidden']]))}/{n_hid_group}")

print(f"\nSelected biactives:")
print(f"  DA : {int(np.sum(MB_da_plus | MB_da_minus))} "
      f"(B+: {np.sum(MB_da_plus)}, B-: {np.sum(MB_da_minus)})")
print(f"  SC : {int(np.sum(MB_sc_plus | MB_sc_minus))} "
      f"(B+: {np.sum(MB_sc_plus)}, B-: {np.sum(MB_sc_minus)})")

removed = (MB_da_plus | MB_da_minus) & ~(MB_sc_plus | MB_sc_minus)
print(f"  Removed by SC refinement: {int(np.sum(removed))}")

print(f"\nWorking set |S|:")
print(f"  DA : {int(np.sum(S_da))}")
print(f"  SC : {int(np.sum(S_sc))}")

print(f"\nHypergradient comparison (scalar = sum over features):")
hg_da_arr = np.atleast_1d(hg_da)
hg_sc_arr = np.atleast_1d(hg_sc)
diff = hg_sc_arr - hg_da_arr
print(f"  ‖h_SC‖         = {np.linalg.norm(hg_sc_arr):.6e}")
print(f"  ‖h_DA‖         = {np.linalg.norm(hg_da_arr):.6e}")
print(f"  ‖h_SC - h_DA‖  = {np.linalg.norm(diff):.6e}")
rel = np.linalg.norm(diff) / max(np.linalg.norm(hg_da_arr), 1e-14)
print(f"  Relative diff  = {rel:.4%}")

# Check sign consistency of DA solution (are any DA adjoint signs wrong?)
p_da = sets_da.get('p', None)
if p_da is not None:
    sigma_da = np.zeros(n_features)
    sigma_da[I_plus | MB_da_plus] = +1.0
    sigma_da[I_minus | MB_da_minus] = -1.0
    biact_da = MB_da_plus | MB_da_minus
    inconsistent_da = biact_da & (sigma_da * p_da <= 0.0)
    print(f"\nSign consistency of DA adjoint on biactive coords:")
    print(f"  DA biactives in S: {int(np.sum(biact_da))}")
    print(f"  Inconsistent (σ·q ≤ 0): {int(np.sum(inconsistent_da))}")
    if np.any(inconsistent_da):
        print("  → SC will remove these (this is the difference between DA and SC)")
    else:
        print("  → DA solution is already self-consistent; SC = DA for this instance")

print()
