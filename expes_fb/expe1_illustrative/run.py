"""Experiment 1 — illustrative sanity checks from ``experimental_strategy.md``.

Panel A
-------
Constructs the 2D diagonal weighted-L1 instance from the draft and stores
contour grids for:
  - the true nonsmooth lower-level objective,
  - two Berkovier-Engelman (BE) smoothings,
  - the forward-backward envelope (FBE).

Panel B
-------
Constructs the 3-feature diagonal counterexample from the draft and stores
outer-loop trajectories for the null oracle and the descent-aligned (DA)
oracle.  To keep the numbers hand-verifiable, the outer update acts only on
the second hyperparameter:

    x_2^{k+1} = x_2^k - eta * h_2(x^k).

The lower-level solution and the validation loss are available in closed form.
At every step we cross-check the analytic hypergradient against the live
``ImplicitVariational`` implementation.

Usage
-----
    python run.py
    python expes_fb/expe1_illustrative/run.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from sparse_ho.algo import ImplicitVariational, select_biactive_self_consistent
from sparse_ho.models import WeightedElasticNet

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "results.pkl"

PANEL_A_CFG = {
    "d": np.array([0.8, 0.5], dtype=float),
    "lambdas": np.array([0.6, 0.6], dtype=float),
    "be_gammas": (0.1, 0.5),
    "fbe_gamma": 0.5,
    "grid_y1": (-0.15, 0.95, 401),
    "grid_y2": (-0.25, 0.65, 401),
    "phi_alpha_grid": (0.05, 1.2, 500),
    "phi_alpha_weight": 0.4,
}

PANEL_B_CFG = {
    "d_train": np.array([0.4, 1.2, 0.0], dtype=float),
    "d_val": np.array([0.0, 1.2, 0.0], dtype=float),
    "alpha0": np.array([0.35, 1.20, 2.00], dtype=float),
    "step_size": 1.0,
    "n_outer": 10,
    "tol": 1e-12,
}


def soft_threshold(v, thresh):
    """Element-wise soft thresholding."""
    return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)


def true_cost(y, d, lambdas):
    """True lower-level objective."""
    y = np.asarray(y, dtype=float)
    return 0.5 * np.sum((y - d) ** 2) + np.sum(lambdas * np.abs(y))


def be_cost(y, d, lambdas, smooth_gamma):
    """BE-smoothed lower-level objective."""
    y = np.asarray(y, dtype=float)
    return 0.5 * np.sum((y - d) ** 2) + np.sum(
        lambdas * np.sqrt(y ** 2 + smooth_gamma ** 2)
    )


def fbe_cost(y, d, lambdas, gamma):
    """Forward-backward envelope of ``f(y) + g(y)`` with ``gamma < 1``.

    Here ``f(y) = 0.5 * ||y - d||^2`` and ``g(y) = sum_j lambda_j |y_j|``.
    """
    y = np.asarray(y, dtype=float)
    grad_f = y - d
    prox_arg = y - gamma * grad_f
    prox = soft_threshold(prox_arg, gamma * lambdas)
    return (
        0.5 * np.sum((y - d) ** 2)
        + np.dot(grad_f, prox - y)
        + 0.5 * np.sum((prox - y) ** 2) / gamma
        + np.sum(lambdas * np.abs(prox))
    )


def _be_coordinate_minimizer(d_j, lambda_j, smooth_gamma):
    """Minimize the 1D BE-smoothed objective for a positive coordinate."""
    if d_j == 0.0:
        return 0.0

    sign = 1.0 if d_j >= 0.0 else -1.0
    d_abs = abs(float(d_j))

    def deriv(y_abs):
        return y_abs - d_abs + lambda_j * y_abs / np.sqrt(
            y_abs ** 2 + smooth_gamma ** 2
        )

    root = brentq(deriv, 0.0, d_abs)
    return sign * root


def _grid_costs(panel_a_cfg):
    d = panel_a_cfg["d"]
    lambdas = panel_a_cfg["lambdas"]
    gamma_fbe = float(panel_a_cfg["fbe_gamma"])

    y1 = np.linspace(*panel_a_cfg["grid_y1"])
    y2 = np.linspace(*panel_a_cfg["grid_y2"])
    Y1, Y2 = np.meshgrid(y1, y2)

    true_grid = (
        0.5 * ((Y1 - d[0]) ** 2 + (Y2 - d[1]) ** 2)
        + lambdas[0] * np.abs(Y1)
        + lambdas[1] * np.abs(Y2)
    )

    be_grids = {}
    for smooth_gamma in panel_a_cfg["be_gammas"]:
        be_grids[f"be_{smooth_gamma:.1f}"] = (
            0.5 * ((Y1 - d[0]) ** 2 + (Y2 - d[1]) ** 2)
            + lambdas[0] * np.sqrt(Y1 ** 2 + smooth_gamma ** 2)
            + lambdas[1] * np.sqrt(Y2 ** 2 + smooth_gamma ** 2)
        )

    grad1 = Y1 - d[0]
    grad2 = Y2 - d[1]
    prox_arg1 = Y1 - gamma_fbe * grad1
    prox_arg2 = Y2 - gamma_fbe * grad2
    prox1 = soft_threshold(prox_arg1, gamma_fbe * lambdas[0])
    prox2 = soft_threshold(prox_arg2, gamma_fbe * lambdas[1])
    fbe_grid = (
        0.5 * ((Y1 - d[0]) ** 2 + (Y2 - d[1]) ** 2)
        + grad1 * (prox1 - Y1)
        + grad2 * (prox2 - Y2)
        + 0.5 * ((prox1 - Y1) ** 2 + (prox2 - Y2) ** 2) / gamma_fbe
        + lambdas[0] * np.abs(prox1)
        + lambdas[1] * np.abs(prox2)
    )

    return y1, y2, true_grid, be_grids, fbe_grid


def _panel_a_upper_objective(y2, alpha2, *, rho):
    """Illustrative outer loss for the 1D ``x_2`` slice.

    Phi(alpha_2) = |y_2^*(alpha_2)| + rho * alpha_2.
    """
    return abs(float(y2)) + float(rho) * float(alpha2)


def _make_panel_a_phi_slice(panel_a_cfg):
    """Build the optional upper-level slice from line 48 of the strategy."""
    d2 = float(panel_a_cfg["d"][1])
    rho = float(panel_a_cfg["phi_alpha_weight"])
    alpha_grid = np.linspace(*panel_a_cfg["phi_alpha_grid"])
    x2_grid = np.log(alpha_grid)

    def y2_true(alpha2):
        return max(d2 - alpha2, 0.0)

    def y2_be(alpha2, smooth_gamma):
        return _be_coordinate_minimizer(d2, alpha2, smooth_gamma)

    def phi_from_y(y2, alpha2):
        return _panel_a_upper_objective(y2, alpha2, rho=rho)

    curves = {
        "true": np.array(
            [phi_from_y(y2_true(alpha2), alpha2) for alpha2 in alpha_grid],
            dtype=float,
        ),
    }
    for smooth_gamma in panel_a_cfg["be_gammas"]:
        curves[f"be_{smooth_gamma:.1f}"] = np.array(
            [
                phi_from_y(y2_be(alpha2, smooth_gamma), alpha2)
                for alpha2 in alpha_grid
            ],
            dtype=float,
        )

    minima = {
        "true": {
            "alpha2": d2,
            "x2": float(np.log(d2)),
            "phi": phi_from_y(y2_true(d2), d2),
        }
    }
    for smooth_gamma in panel_a_cfg["be_gammas"]:
        key = f"be_{smooth_gamma:.1f}"
        result = minimize_scalar(
            lambda alpha2: phi_from_y(y2_be(alpha2, smooth_gamma), alpha2),
            bounds=(alpha_grid[0], alpha_grid[-1]),
            method="bounded",
        )
        minima[key] = {
            "alpha2": float(result.x),
            "x2": float(np.log(result.x)),
            "phi": float(result.fun),
        }

    assert minima["be_0.1"]["alpha2"] < d2
    assert minima["be_0.5"]["alpha2"] < minima["be_0.1"]["alpha2"]

    return {
        "config": {
            "alpha_weight": rho,
            "alpha_grid_min": float(alpha_grid[0]),
            "alpha_grid_max": float(alpha_grid[-1]),
            "kink_alpha2": d2,
            "kink_x2": float(np.log(d2)),
            "formula": "Phi(alpha2) = |y2*(alpha2)| + rho * alpha2",
        },
        "alpha2_grid": alpha_grid,
        "x2_grid": x2_grid,
        "curves": curves,
        "minima": minima,
    }


def make_panel_a_results():
    """Generate contour grids and minima for Panel A."""
    cfg = PANEL_A_CFG
    d = cfg["d"]
    lambdas = cfg["lambdas"]
    gamma_fbe = float(cfg["fbe_gamma"])

    true_min = soft_threshold(d, lambdas)
    np.testing.assert_allclose(true_min, np.array([0.2, 0.0]), atol=1e-12)

    y1, y2, true_grid, be_grids, fbe_grid = _grid_costs(cfg)

    minima = {
        "true": true_min,
        "fbe": true_min.copy(),
    }
    for smooth_gamma in cfg["be_gammas"]:
        key = f"be_{smooth_gamma:.1f}"
        minima[key] = np.array(
            [
                _be_coordinate_minimizer(d[0], lambdas[0], smooth_gamma),
                _be_coordinate_minimizer(d[1], lambdas[1], smooth_gamma),
            ],
            dtype=float,
        )

    assert minima["be_0.1"][1] > 0.0
    assert minima["be_0.5"][1] > minima["be_0.1"][1]
    np.testing.assert_allclose(
        fbe_cost(true_min, d, lambdas, gamma_fbe),
        true_cost(true_min, d, lambdas),
        atol=1e-12,
    )
    phi_slice = _make_panel_a_phi_slice(cfg)

    return {
        "config": {
            "d": d,
            "lambdas": lambdas,
            "be_gammas": cfg["be_gammas"],
            "fbe_gamma": gamma_fbe,
        },
        "grid": {
            "y1": y1,
            "y2": y2,
            "true": true_grid,
            "fbe": fbe_grid,
            **be_grids,
        },
        "minima": minima,
        "phi_slice": phi_slice,
    }


def _make_panel_b_problem():
    """Return the exact diagonal train/validation problem used in Panel B."""
    d_train = PANEL_B_CFG["d_train"]
    d_val = PANEL_B_CFG["d_val"]
    sqrt3 = np.sqrt(3.0)
    X = np.vstack([sqrt3 * np.eye(3), sqrt3 * np.eye(3)])
    y = np.concatenate([sqrt3 * d_train, sqrt3 * d_val])
    idx_train = np.arange(3)
    idx_val = np.arange(3, 6)
    return X, y, idx_train, idx_val


def beta_closed_form(log_alpha, d_train):
    """Lower-level solution for the diagonal design in Panel B."""
    alpha = np.exp(log_alpha)
    return soft_threshold(d_train, alpha)


def val_loss_closed_form(beta, d_val):
    """Validation loss 0.5 ||beta - d_val||^2."""
    return 0.5 * np.sum((beta - d_val) ** 2)


def analytic_hypergradient(log_alpha, *, use_sc_policy):
    """Closed-form hypergradient for Panel B.

    The smooth Hessian is the identity and all active coordinates are positive.
    The SC policy adds feature 2 when it is biactive and ``z_2 < 0``
    (diagonal Hessian, so SC reduces to the initialization step).
    """
    d_train = PANEL_B_CFG["d_train"]
    d_val = PANEL_B_CFG["d_val"]
    alpha = np.exp(log_alpha)
    beta = beta_closed_form(log_alpha, d_train)
    z_star = beta - d_val

    strict_support = beta > 0.0
    selected_support = strict_support.copy()

    # For this diagonal construction with gamma = 1, the prox argument is
    # exactly d_train, hence the biactive test reduces to |d_train_j| = alpha_j
    # on zero coordinates.
    biactive_plus = (
        (beta == 0.0)
        & (d_train > 0.0)
        & np.isclose(np.abs(d_train), alpha, atol=1e-12, rtol=1e-12)
    )
    if use_sc_policy:
        selected_support |= biactive_plus & (z_star < 0.0)  # diagonal ⟹ SC = init step

    p = np.zeros_like(beta)
    p[selected_support] = -z_star[selected_support]

    hypergrad = np.zeros_like(beta)
    hypergrad[selected_support] = alpha[selected_support] * p[selected_support]

    return {
        "beta": beta,
        "z_star": z_star,
        "p": p,
        "strict_support": strict_support,
        "biactive_plus": biactive_plus,
        "selected_support": selected_support,
        "hypergrad": hypergrad,
    }


def library_hypergradient(log_alpha, *, policy):
    """Hypergradient from the live ``ImplicitVariational`` implementation."""
    X, y, idx_train, idx_val = _make_panel_b_problem()
    X_train = X[idx_train]
    y_train = y[idx_train]
    X_val = X[idx_val]
    y_val = y[idx_val]

    model = WeightedElasticNet(alpha_l2=0.0)
    algo = ImplicitVariational(policy=policy, biactive_tol_rel=PANEL_B_CFG["tol"])

    def get_grad_outer(mask, dense):
        Xv = X_val[:, mask]
        return (Xv.T @ (Xv @ dense - y_val)) / len(y_val)

    mask, dense, hypergrad, _, sets = algo.compute_beta_grad(
        X_train,
        y_train,
        log_alpha,
        model,
        get_grad_outer,
        tol=PANEL_B_CFG["tol"],
        max_iter=10_000,
        full_jac_v=True,
        return_sets=True,
    )

    beta = np.zeros(X_train.shape[1], dtype=float)
    beta[mask] = dense
    return {
        "beta": beta,
        "hypergrad": np.asarray(hypergrad, dtype=float),
        "sets": sets,
    }


def run_panel_b_method(method_name, *, policy):
    """Run the coordinate outer loop for one oracle policy."""
    step_size = float(PANEL_B_CFG["step_size"])
    d_val = PANEL_B_CFG["d_val"]
    log_alpha = np.log(PANEL_B_CFG["alpha0"]).copy()

    rows = []
    max_grad_err = 0.0
    oracle_check = None

    for k in range(PANEL_B_CFG["n_outer"] + 1):
        analytic = analytic_hypergradient(
            log_alpha, use_sc_policy=(policy is not None)
        )
        library = library_hypergradient(log_alpha, policy=policy)

        np.testing.assert_allclose(
            library["beta"], analytic["beta"], atol=1e-9, rtol=1e-9
        )
        np.testing.assert_allclose(
            library["hypergrad"], analytic["hypergrad"], atol=1e-7, rtol=1e-7
        )

        max_grad_err = max(
            max_grad_err,
            float(np.max(np.abs(library["hypergrad"] - analytic["hypergrad"]))),
        )

        if oracle_check is None:
            sets = library["sets"]
            oracle_check = {
                "strict_support": np.flatnonzero(analytic["strict_support"]).tolist(),
                "biactive_plus": np.flatnonzero(analytic["biactive_plus"]).tolist(),
                "selected_support": np.flatnonzero(analytic["selected_support"]).tolist(),
                "selected_biactive": np.flatnonzero(
                    sets["M_B_plus"] | sets["M_B_minus"]
                ).tolist(),
                "z_star": analytic["z_star"].tolist(),
                "p": analytic["p"].tolist(),
                "hypergrad_analytic": analytic["hypergrad"].tolist(),
                "hypergrad_library": library["hypergrad"].tolist(),
            }

        beta = analytic["beta"]
        rows.append(
            {
                "iteration": k,
                "x2": float(log_alpha[1]),
                "alpha2": float(np.exp(log_alpha[1])),
                "beta2": float(beta[1]),
                "val_loss": float(val_loss_closed_form(beta, d_val)),
                "hypergrad2": float(analytic["hypergrad"][1]),
            }
        )

        if k < PANEL_B_CFG["n_outer"]:
            log_alpha[1] -= step_size * analytic["hypergrad"][1]

    return {
        "method": method_name,
        "step_size": step_size,
        "rows": rows,
        "oracle_check": oracle_check,
        "max_abs_grad_err": max_grad_err,
    }


def make_panel_b_results():
    """Generate Panel B trajectories and oracle checks."""
    methods = {
        "null": run_panel_b_method("null", policy=None),
        "sc": run_panel_b_method("sc", policy=select_biactive_self_consistent),
    }

    null_check = methods["null"]["oracle_check"]
    sc_check = methods["sc"]["oracle_check"]

    assert null_check["selected_support"] == [0]
    assert sc_check["selected_support"] == [0, 1]
    assert null_check["hypergrad_analytic"][1] == 0.0
    assert sc_check["hypergrad_analytic"][1] > 0.0

    return {
        "config": PANEL_B_CFG.copy(),
        "methods": methods,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    results = {
        "panel_a": make_panel_a_results(),
        "panel_b": make_panel_b_results(),
    }

    with RESULTS_PATH.open("wb") as f:
        pickle.dump(results, f)

    panel_b = results["panel_b"]["methods"]
    print(f"Saved results to {RESULTS_PATH}")
    print(
        "Panel B checks:"
        f" null h2={panel_b['null']['oracle_check']['hypergrad_analytic'][1]:.6f},"
        f" da h2={panel_b['sc']['oracle_check']['hypergrad_analytic'][1]:.6f}"
    )
    print(
        "Gradient verification max abs error:"
        f" null={panel_b['null']['max_abs_grad_err']:.3e},"
        f" da={panel_b['sc']['max_abs_grad_err']:.3e}"
    )


if __name__ == "__main__":
    main()
