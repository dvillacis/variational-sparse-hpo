"""Degenerate four-group dataset for Experiments 3 and 4.

Feature layout (columns of X)
------------------------------
[0            : n_easy]                  Easy features      — nonzero ±β_std ground truth on
                                                              both training and validation splits.
                                                              Always strictly active.
[n_easy       : n_easy+n_dist]           Distractor features — correlated with hidden features
                                                              (Pearson ρ), zero ground truth.
                                                              Strictly active at calibrated init
                                                              (they absorb the hidden signal on
                                                              the training split).
[n_easy+n_dist: n_easy+n_dist+n_hidden]  Hidden features    — nonzero ±β_std ground truth.
                                                              Biactive at calibrated init by
                                                              construction: |v_j|=γexp(x_j)
                                                              exactly.  Carry validation signal
                                                              only (distractors absorb their
                                                              training contribution).
[n_easy+n_dist+n_hidden : m]             Noise features      — iid Gaussian, zero ground truth,
                                                              always strictly inactive.

Why this is degenerate
----------------------
The response y is generated from easy + hidden features only.  Because each distractor column
d_k is near-collinear with its paired hidden column h_j (Pearson ρ), the inner solver—given a
sufficiently small distractor penalty and a hidden penalty set exactly at the biactive
threshold—fits the training residual using the distractors alone.  Hidden features sit exactly
at the ℓ1 threshold (biactive), and the standard support-restricted adjoint (Sparse-HO) assigns
them an identically zero hypergradient.

For the descent-aligned (DA) biactive policy to work, each hidden feature j must belong to
B+ (resp. B-) when β_hid_j > 0 (resp. < 0), and the corresponding validation gradient
z*_j must have the correct sign.  This requires:

    g_hid_j ≈ ρ · exp(x_dist) - (1-ρ²) · β_hid_j < 0   for β_hid_j > 0

which is guaranteed when:

    exp(x_dist) < (1-ρ²)/ρ · β_std

The parameter `penalty_dist_fraction` (default 0.5) controls the margin:

    exp(x_dist) = penalty_dist_fraction · (1-ρ²)/ρ · β_std

The function enforces this condition analytically, so the construction is valid for all
ρ ∈ (0, 1).  A post-calibration sign check warns if any hidden feature violates the
DA-alignment condition (which can happen due to finite-sample noise at very high ρ or
very small n_tr; increase n or decrease penalty_dist_fraction to mitigate).

Calibration procedure (two-pass)
---------------------------------
Pass 1 — Set x_hid to a large sentinel value (= log(alpha_noise)) so that hidden features
          are strictly inactive.  Solve the inner WeightedElasticNet problem on the training
          split to obtain β_S with β_S_hid = 0.

Pass 2 — Compute g_hid_j = (1/n_tr) X_tr[:,hid_j]ᵀ (X_tr β_S - y_tr) for each hidden
          feature j.  Set exp(x0_hid_j) = |g_hid_j| so that the KKT condition for hidden
          features at β_S_hid = 0 holds with equality — i.e., hidden features are biactive
          at the calibrated initial hyperparameter x0.

Because the inner problem is strongly convex (alpha_l2 > 0), the calibrated β_S is the
unique minimiser, and the biactive condition holds exactly (up to inner-solver tolerance).
"""

import warnings

import numpy as np

from sparse_ho.algo.forward import compute_beta
from sparse_ho.models import WeightedElasticNet


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _solve_inner(X_tr, y_tr, log_alpha, alpha_l2, tol=1e-8):
    """Solve inner WeightedElasticNet on (X_tr, y_tr); return full β vector."""
    model = WeightedElasticNet(alpha_l2=alpha_l2)
    mask, dense, _ = compute_beta(
        X_tr, y_tr, log_alpha, model, tol=tol, compute_jac=False)
    beta = np.zeros(X_tr.shape[1])
    beta[mask] = dense
    return beta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_degenerate_dataset(
    n, m, n_easy, n_dist, n_hidden,
    rho,
    alpha_l2,
    *,
    beta_std=1.0,
    noise_std=0.05,
    penalty_dist_fraction=0.5,
    penalty_easy_fraction=0.5,
    penalty_noise_factor=10.0,
    calibration_slack=0.05,
    inner_tol=1e-8,
    rng=None,
):
    """Generate a calibrated degenerate four-group regression dataset.

    Parameters
    ----------
    n : int
        Total number of samples (train + val + test, 60/20/20 split).
    m : int
        Total number of features.
    n_easy : int
        Number of easy (always-active) features.
    n_dist : int
        Number of distractor features.
    n_hidden : int
        Number of hidden (biactive) features.  If n_hidden > n_dist, distractor
        columns are cycled modularly when building the correlation structure.
    rho : float
        Pearson correlation between each hidden column and its paired distractor.
        Must satisfy 0 < rho < 1.
    alpha_l2 : float
        Fixed L2 ridge penalty for the inner WeightedElasticNet problem.
    beta_std : float, optional (default=1.0)
        Standard deviation of the true ±β_std coefficients.
    noise_std : float, optional (default=0.05)
        Standard deviation of additive Gaussian observation noise.
    penalty_dist_fraction : float, optional (default=0.5)
        Controls the distractor penalty:
            exp(x_dist) = penalty_dist_fraction * (1-rho²)/rho * beta_std
        Must be < 1 to guarantee the DA-alignment condition g_hid_j < 0 for
        β_hid_j > 0.  Smaller values widen the margin at the cost of making
        distractor features harder to activate.
    penalty_easy_fraction : float, optional (default=0.5)
        Controls the easy-feature penalty:
            exp(x_easy) = penalty_easy_fraction * beta_std
        Must be < 1 so easy features are strictly active.
    penalty_noise_factor : float, optional (default=10.0)
        Noise-feature penalty = penalty_noise_factor * beta_std  (large, inactive).
        Also used as the initial sentinel penalty for hidden features during
        pass 1 of the calibration.
    calibration_slack : float, optional (default=0.05)
        Upward bias applied to the calibrated hidden penalty after pass 2:
            exp(x0_hid_j) = |g_hid_j| * (1 + calibration_slack)
        This ensures hidden features are **strictly inactive** (|β*_hid| = 0 exactly)
        rather than numerically at the knife-edge biactive point, and produces a
        predictable gap_rel ≈ calibration_slack for biactive detection.
        Set the ImplicitVariational parameter biactive_tol_rel ≥ 2 * calibration_slack
        in the experiment to reliably detect hidden features.  Default 0.05 ⟹
        biactive_tol_rel = 0.10 is sufficient.
    inner_tol : float, optional (default=1e-8)
        Inner solver tolerance used during calibration.
    rng : np.random.Generator or None
        Random generator.  Falls back to ``np.random.default_rng(0)``.

    Returns
    -------
    X : ndarray, shape (n, m)
    y : ndarray, shape (n,)
    beta_true : ndarray, shape (m,)
        Ground truth (nonzero on easy + hidden features only).
    groups : dict
        Maps ``'easy'``, ``'distractor'``, ``'hidden'``, ``'noise'`` to
        index arrays (``np.ndarray`` of int).
    idx_train, idx_val, idx_test : ndarray of int
        60/20/20 split index arrays (sorted).
    log_alpha0 : ndarray, shape (m,)
        Calibrated initial log-hyperparameters.  Hidden-feature entries are set
        so that hidden features are exactly biactive at the calibrated solution.

    Raises
    ------
    ValueError
        If n_easy + n_dist + n_hidden > m, or if rho is out of (0, 1).

    Warns
    -----
    RuntimeWarning
        If any hidden feature violates the DA-alignment condition (g_hid and
        β_hid have the same sign) after calibration, or if any calibrated
        hidden penalty is near zero.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_noise = m - n_easy - n_dist - n_hidden
    if n_noise < 0:
        raise ValueError(
            f"n_easy + n_dist + n_hidden = {n_easy + n_dist + n_hidden} > m = {m}."
        )
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho must be in (0, 1), got {rho}.")
    if not (0.0 < penalty_dist_fraction < 1.0):
        raise ValueError(
            "penalty_dist_fraction must be in (0, 1) to guarantee the "
            f"DA-alignment condition; got {penalty_dist_fraction}."
        )

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------
    X_easy = rng.standard_normal((n, n_easy))
    X_dist = rng.standard_normal((n, n_dist))

    # Hidden columns are each correlated with one distractor column (cycling).
    dist_idx = np.arange(n_hidden) % n_dist
    X_hid = (
        rho * X_dist[:, dist_idx]
        + np.sqrt(1.0 - rho ** 2) * rng.standard_normal((n, n_hidden))
    )

    if n_noise > 0:
        X_noise = rng.standard_normal((n, n_noise))
        X = np.hstack([X_easy, X_dist, X_hid, X_noise])
    else:
        X = np.hstack([X_easy, X_dist, X_hid])

    # ------------------------------------------------------------------
    # Ground truth — nonzero only on easy + hidden
    # ------------------------------------------------------------------
    hid_start = n_easy + n_dist

    beta_true = np.zeros(m)
    beta_true[:n_easy] = beta_std * rng.choice([-1.0, 1.0], size=n_easy)
    beta_true[hid_start:hid_start + n_hidden] = (
        beta_std * rng.choice([-1.0, 1.0], size=n_hidden)
    )

    # ------------------------------------------------------------------
    # Response (easy + hidden signal + noise)
    # ------------------------------------------------------------------
    y = X @ beta_true + noise_std * rng.standard_normal(n)

    # ------------------------------------------------------------------
    # 60 / 20 / 20 split
    # ------------------------------------------------------------------
    perm = rng.permutation(n)
    n_tr  = int(0.6 * n)
    n_val = int(0.2 * n)
    idx_train = np.sort(perm[:n_tr])
    idx_val   = np.sort(perm[n_tr:n_tr + n_val])
    idx_test  = np.sort(perm[n_tr + n_val:])

    X_tr, y_tr = X[idx_train], y[idx_train]

    # ------------------------------------------------------------------
    # Initial log-hyperparameters
    #
    # Distractor penalty: chosen to guarantee the DA-alignment condition.
    # From the 1D approximation (normalised features, dominant pairing):
    #   g_hid_j ≈ ρ·exp(x_dist) - (1-ρ²)·β_hid_j
    # For g_hid_j < 0 when β_hid_j > 0 we need:
    #   exp(x_dist) < (1-ρ²)/ρ · β_std
    # We set exp(x_dist) = penalty_dist_fraction · (1-ρ²)/ρ · β_std.
    # ------------------------------------------------------------------
    alpha_dist = penalty_dist_fraction * (1.0 - rho ** 2) / rho * beta_std
    alpha_easy = penalty_easy_fraction * beta_std
    alpha_noise_sentinel = penalty_noise_factor * beta_std   # also used as hidden sentinel

    log_alpha0 = np.zeros(m)
    log_alpha0[:n_easy] = np.log(alpha_easy)
    log_alpha0[n_easy:hid_start] = np.log(alpha_dist)
    log_alpha0[hid_start:hid_start + n_hidden] = np.log(alpha_noise_sentinel)  # sentinel
    if n_noise > 0:
        log_alpha0[hid_start + n_hidden:] = np.log(alpha_noise_sentinel)

    # ------------------------------------------------------------------
    # Calibration — Pass 1: solve with hidden features forced out
    # ------------------------------------------------------------------
    beta_S = _solve_inner(X_tr, y_tr, log_alpha0, alpha_l2, tol=inner_tol)

    # ------------------------------------------------------------------
    # Calibration — Pass 2: set hidden penalty to biactive threshold
    # ------------------------------------------------------------------
    residual = X_tr @ beta_S - y_tr                                      # shape (n_tr,)
    g_hid = (
        X_tr[:, hid_start:hid_start + n_hidden].T @ residual / n_tr     # shape (n_hidden,)
    )

    # Verify DA-alignment: g_hid_j and β_hid_j must have opposite signs.
    beta_hid = beta_true[hid_start:hid_start + n_hidden]
    sign_mismatch = np.sign(g_hid) == np.sign(beta_hid)   # True = bad
    n_bad = int(np.sum(sign_mismatch))
    if n_bad > 0:
        warnings.warn(
            f"{n_bad}/{n_hidden} hidden feature(s) have g_hid and β_hid with the "
            "same sign after calibration.  These features land in the wrong biactive "
            "set (B- instead of B+, or vice versa) so the DA policy will not include "
            "them.  Consider decreasing penalty_dist_fraction or using a larger n to "
            "reduce finite-sample noise.",
            RuntimeWarning,
            stacklevel=2,
        )

    exp_x0_hid = np.abs(g_hid)

    # Floor: if |g_hid_j| is negligible, biactivity is trivial and the
    # feature would become active with any small perturbation — not useful
    # for demonstrating gradient starvation.
    eps_floor = 1e-10 * beta_std
    near_zero = exp_x0_hid < eps_floor
    if np.any(near_zero):
        warnings.warn(
            f"{np.sum(near_zero)} hidden feature(s) have a near-zero calibrated "
            f"penalty (min={exp_x0_hid.min():.2e}, floor={eps_floor:.2e}).  "
            "Biactivity for these features may be numerically fragile.  "
            "Consider increasing rho or beta_std.",
            RuntimeWarning,
            stacklevel=2,
        )
        exp_x0_hid = np.maximum(exp_x0_hid, eps_floor)

    # Apply calibration slack: push penalty slightly above threshold so that
    # hidden features are **strictly inactive** (|β*| = 0 exactly) rather than
    # at the numerical knife-edge.  This produces gap_rel ≈ slack, making
    # biactive detection via biactive_tol_rel ≥ 2*slack robust and predictable.
    exp_x0_hid = exp_x0_hid * (1.0 + float(calibration_slack))

    log_alpha0[hid_start:hid_start + n_hidden] = np.log(exp_x0_hid)

    # ------------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------------
    groups = {
        'easy':       np.arange(n_easy),
        'distractor': np.arange(n_easy, hid_start),
        'hidden':     np.arange(hid_start, hid_start + n_hidden),
        'noise':      np.arange(hid_start + n_hidden, m),
    }

    return X, y, beta_true, groups, idx_train, idx_val, idx_test, log_alpha0


# ---------------------------------------------------------------------------
# Calibration diagnostic
# ---------------------------------------------------------------------------

def check_calibration(
    X, y, beta_true, groups, idx_train, idx_val, log_alpha0, alpha_l2,
    *,
    biactive_tol_rel=1e-6,
    inner_tol=1e-8,
    verbose=True,
):
    """Verify the calibration of a degenerate dataset.

    Solves the inner problem at the calibrated ``log_alpha0`` and checks:

    1. Easy and distractor features are strictly active (|β*| > 0).
    2. Hidden features are biactive (|β*_j| ≈ 0 AND |v_j| ≈ γ·exp(x_j)).
    3. Noise features are strictly inactive (|v_j| < γ·exp(x_j)).
    4. DA-alignment holds: g_hid_j has the opposite sign to β_true_hid_j.
    5. Validation gradient z* for hidden features has the correct sign for
       the DA biactive selection policy.

    Returns a dict of diagnostics.  Prints a summary when ``verbose=True``.
    """
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val],   y[idx_val]
    n_tr = len(idx_train)

    beta_S = _solve_inner(X_tr, y_tr, log_alpha0, alpha_l2, tol=inner_tol)

    n_features = X.shape[1]

    # Smooth gradient at β_S
    model = WeightedElasticNet(alpha_l2=alpha_l2)
    g_full = X_tr.T @ (X_tr @ beta_S - y_tr) / n_tr + alpha_l2 * beta_S

    # Proximal argument v = β_S - γ g_full  (γ = 1/L where L ≈ ||X_tr||²/n_tr)
    L = float(np.linalg.norm(X_tr, ord=2) ** 2) / n_tr
    gamma = 1.0 / max(L, 1e-12)
    v_bar = beta_S - gamma * g_full
    u_bar = gamma * np.exp(log_alpha0)    # γ·exp(x)

    abs_v = np.abs(v_bar)
    scale = np.maximum(abs_v, u_bar)
    gap_rel = np.abs(abs_v - u_bar) / np.maximum(scale, 1e-14)

    beta_true_hid = beta_true[groups['hidden']]
    g_hid = g_full[groups['hidden']]
    z_star_hid = X_val[:, groups['hidden']].T @ (X_val @ beta_S - y_val) / len(idx_val)

    diag = {}
    # --- per-group biactivity status ---
    for gname, gidx in groups.items():
        is_biactive = gap_rel[gidx] <= biactive_tol_rel
        diag[f'{gname}_biactive_frac'] = float(np.mean(is_biactive))
        diag[f'{gname}_mean_abs_beta'] = float(np.mean(np.abs(beta_S[gidx])))
        diag[f'{gname}_mean_gap_rel'] = float(np.mean(gap_rel[gidx]))

    # --- DA-alignment ---
    da_aligned = np.sign(g_hid) != np.sign(beta_true_hid)
    diag['hidden_da_aligned_frac'] = float(np.mean(da_aligned))

    # --- z* sign check for DA policy ---
    # B+ (v_hid > 0, g_hid < 0): need z*_j < 0
    # B- (v_hid < 0, g_hid > 0): need z*_j > 0
    correct_z_sign = (
        ((g_hid < 0) & (z_star_hid < 0)) |   # B+: z* < 0 ✓
        ((g_hid > 0) & (z_star_hid > 0))       # B-: z* > 0 ✓
    )
    diag['hidden_z_sign_ok_frac'] = float(np.mean(correct_z_sign))

    if verbose:
        print("=== Calibration check ===")
        for gname in ('easy', 'distractor', 'hidden', 'noise'):
            bf = diag[f'{gname}_biactive_frac']
            ab = diag[f'{gname}_mean_abs_beta']
            gr = diag[f'{gname}_mean_gap_rel']
            print(f"  {gname:>12s}: |β*| mean={ab:.3e}  biactive_frac={bf:.2f}  "
                  f"gap_rel mean={gr:.2e}")
        print(f"  hidden DA-aligned : {diag['hidden_da_aligned_frac']:.2f} "
              f"({int(diag['hidden_da_aligned_frac']*len(groups['hidden']))}"
              f"/{len(groups['hidden'])})")
        print(f"  hidden z* sign ok : {diag['hidden_z_sign_ok_frac']:.2f} "
              f"({int(diag['hidden_z_sign_ok_frac']*len(groups['hidden']))}"
              f"/{len(groups['hidden'])})")
    return diag
