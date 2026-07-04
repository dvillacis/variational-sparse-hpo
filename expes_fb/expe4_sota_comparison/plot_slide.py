"""Plot Experiment 4 — slide version (convergence panels, horizontal layout).

Produces a single wide figure suited for a conference slide:

  fig_experiment4_slide.pdf

The 4 convergence panels (validation loss gap, hidden-feature gradient norm,
mean hidden penalty, support recovery F1) are arranged in a single row.

Usage
-----
    python plot_slide.py
    python expes_fb/expe4_sota_comparison/plot_slide.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expes_fb.shared.data_gen_degenerate import make_degenerate_dataset
from expes_fb.shared.plotting import (
    PALETTE,
    REFERENCE_COLORS,
    add_shared_legend,
    apply_plot_style,
    get_method_style,
)

from sparse_ho.models import WeightedElasticNet
from sparse_ho.algo.forward import compute_beta

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'
FIG_PATH = RESULTS_DIR / 'fig_experiment4_slide.pdf'

METHODS = ['sparseho_scalar', 'sparseho_wl1', 'ntrba_wl1']
METHOD_LABELS = {
    'sparseho_scalar': 'Scalar ℓ1',
    'sparseho_wl1':    'SparseHO (wℓ1)',
    'ntrba_wl1':       'NTRBA-wℓ1 (ours)',
}
METHOD_STYLES = {
    'sparseho_scalar': {
        **get_method_style('sparseho_scalar'),
        'color': PALETTE['slate'],
    },
    'sparseho_wl1': {
        **get_method_style('sparseho_wl1'),
        'color': PALETTE['charcoal'],
    },
    'ntrba_wl1': {
        **get_method_style('ntrba_wl1'),
        'color': PALETTE['ink'],
    },
}
METHOD_COLORS = {key: METHOD_STYLES[key]['color'] for key in METHODS}
METHOD_LS = {key: METHOD_STYLES[key]['linestyle'] for key in METHODS}

REP_N, REP_M, REP_RHO, REP_SEED = 200, 300, 0.98, 0
EASY_FRAC = 0.04
DIST_FRAC = 0.04
INNER_TOL = 1e-8
CALIBRATION_SLACK = 0.05

apply_plot_style()


def _group_sizes(m):
    n_easy = max(int(EASY_FRAC * m), 5)
    n_dist = max(int(DIST_FRAC * m), 5)
    n_hidden = n_dist
    return n_easy, n_dist, n_hidden


def _solve_inner_f1(X_tr, y_tr, alpha, alpha_l2, beta_true, tol=INNER_TOL):
    model = WeightedElasticNet(alpha_l2=alpha_l2)
    log_alpha = np.log(np.maximum(np.asarray(alpha), 1e-300))
    mask, dense, _ = compute_beta(
        X_tr, y_tr, log_alpha, model, tol=tol, compute_jac=False)
    beta = np.zeros(X_tr.shape[1])
    beta[mask] = dense
    pred_supp = np.abs(beta) > 1e-10
    true_supp = beta_true != 0
    tp = np.sum(pred_supp & true_supp)
    fp = np.sum(pred_supp & ~true_supp)
    fn = np.sum(~pred_supp & true_supp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _compute_f1_trajectory(alphas_traj, X_tr, y_tr, beta_true, alpha_l2):
    f1s = []
    for a in alphas_traj:
        if a is None:
            f1s.append(np.nan)
        else:
            f1s.append(_solve_inner_f1(X_tr, y_tr, a, alpha_l2, beta_true))
    return f1s


def plot_convergence_horizontal(df):
    n, m, rho = REP_N, REP_M, REP_RHO
    seed = REP_SEED

    sel = df[
        (df.n == n) & (df.m == m) &
        np.isclose(df.rho, rho) & (df.seed == seed)
    ]
    if sel.empty:
        print(f"plot_convergence: no rows for n={n}, m={m}, rho={rho}, seed={seed}, skipping.")
        return

    n_easy, n_dist, n_hidden = _group_sizes(m)
    alpha_l2 = 1.0 / int(0.6 * n)
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X_rep, y_rep, beta_true_rep, _, idx_train_rep, _, _, _ = (
            make_degenerate_dataset(
                n, m, n_easy, n_dist, n_hidden,
                rho=rho, alpha_l2=alpha_l2,
                beta_std=1.0, noise_std=0.05,
                calibration_slack=CALIBRATION_SLACK,
                rng=rng,
            )
        )
    X_tr_rep = X_rep[idx_train_rep]
    y_tr_rep = y_rep[idx_train_rep]

    ntrba_row = sel[sel.method == 'ntrba_wl1']
    ref_val = (
        float(min(ntrba_row.iloc[0]['val_objs']))
        if not ntrba_row.empty and ntrba_row.iloc[0]['val_objs']
        else 0.0
    )

    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.8))
    ax_gap, ax_hgrad, ax_hpen, ax_f1 = axes

    scalar_row = sel[sel.method == 'sparseho_scalar']
    scalar_best = (
        float(min(scalar_row.iloc[0]['val_objs']))
        if not scalar_row.empty else np.nan
    )

    for mname in ['sparseho_wl1', 'ntrba_wl1']:
        rows = sel[sel.method == mname]
        if rows.empty:
            continue
        row = rows.iloc[0]

        val_objs = row['val_objs']
        hg_norms = row['hidden_grad_norms']
        mean_hpen = row['mean_hidden_alpha_traj']
        alphas_traj = row['alphas_traj']

        iters = np.arange(len(val_objs))
        color = METHOD_COLORS[mname]
        ls = METHOD_LS[mname]
        lw = 1.6
        label = METHOD_LABELS[mname]

        if val_objs:
            gap = [v - ref_val for v in val_objs]
            ax_gap.plot(iters, gap, color=color, linestyle=ls,
                        linewidth=lw, label=label,
                        marker=METHOD_STYLES[mname]['marker'],
                        markevery=max(len(gap) // 8, 1), markersize=3.0)

        if hg_norms:
            ax_hgrad.plot(iters[:len(hg_norms)], hg_norms,
                          color=color, linestyle=ls, linewidth=lw, label=label,
                          marker=METHOD_STYLES[mname]['marker'],
                          markevery=max(len(hg_norms) // 8, 1), markersize=3.0)

        if mean_hpen:
            ax_hpen.plot(iters[:len(mean_hpen)], mean_hpen,
                         color=color, linestyle=ls, linewidth=lw, label=label,
                         marker=METHOD_STYLES[mname]['marker'],
                         markevery=max(len(mean_hpen) // 8, 1), markersize=3.0)

        if alphas_traj and any(a is not None for a in alphas_traj):
            f1_traj = _compute_f1_trajectory(
                alphas_traj, X_tr_rep, y_tr_rep, beta_true_rep, alpha_l2)
            ax_f1.plot(iters[:len(f1_traj)], f1_traj,
                       color=color, linestyle=ls, linewidth=lw, label=label,
                       marker=METHOD_STYLES[mname]['marker'],
                       markevery=max(len(f1_traj) // 8, 1), markersize=3.0)

    if np.isfinite(scalar_best):
        scalar_gap = scalar_best - ref_val
        ax_gap.axhline(scalar_gap, color=METHOD_COLORS['sparseho_scalar'],
                       linestyle=METHOD_LS['sparseho_scalar'], linewidth=1.4,
                       label=METHOD_LABELS['sparseho_scalar'])

    ax_gap.set_xlabel('Outer iteration')
    ax_gap.set_ylabel('Validation loss gap  Φ − Φ*')
    ax_gap.set_title('Validation loss gap')
    ax_gap.grid(axis='y', color=REFERENCE_COLORS['grid'], linewidth=0.5, linestyle='--')

    ax_hgrad.set_xlabel('Outer iteration')
    ax_hgrad.set_ylabel('‖∇$_{x_{\\mathrm{hid}}}$Φ‖')
    ax_hgrad.set_title('Hidden-feature gradient norm')
    ax_hgrad.set_yscale('symlog', linthresh=1e-6)
    ax_hgrad.grid(axis='y', color=REFERENCE_COLORS['grid'], linewidth=0.5, linestyle='--')

    ax_hpen.set_xlabel('Outer iteration')
    ax_hpen.set_ylabel('Mean hidden penalty')
    ax_hpen.set_title('Mean hidden penalty')
    ax_hpen.grid(axis='y', color=REFERENCE_COLORS['grid'], linewidth=0.5, linestyle='--')

    ax_f1.set_xlabel('Outer iteration')
    ax_f1.set_ylabel('Support recovery F1')
    ax_f1.set_title('F1 trajectory')
    ax_f1.set_ylim(-0.05, 1.05)
    ax_f1.grid(axis='y', color=REFERENCE_COLORS['grid'], linewidth=0.5, linestyle='--')

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    add_shared_legend(
        fig, list(axes), ncol=3, bbox_to_anchor=(0.5, 0.97)
    )
    fig.suptitle(
        f'n={n},  p={m},  ρ={rho},  seed={seed}',
        fontsize=9, y=1.0,
    )
    fig.savefig(FIG_PATH, bbox_inches='tight')
    print(f"Saved {FIG_PATH}")
    plt.close(fig)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results not found at {RESULTS_PATH}.  Run run.py first.")
    df = pd.read_pickle(RESULTS_PATH)
    plot_convergence_horizontal(df)


if __name__ == '__main__':
    main()
