"""Plot results for Experiment 4 — SOTA comparison: gradient starvation.

Produces two figures saved to the results/ subdirectory:

  fig_metrics.pdf
      4-row × 3-column bar chart.  Rows: val_loss_gap, hidden_grad_norm_0,
      hidden_recall, F1.  Columns: ρ ∈ {0.90, 0.95, 0.98} (averaged over
      problem sizes and seeds).  Bars: three methods.

  fig_convergence.pdf
      Four-panel convergence plot for the representative instance
      (n=200, m=300, ρ=0.98, seed=0):
        (a) Validation loss gap  Φ − Φ*  over outer iterations.
        (b) Hidden-feature gradient magnitude ‖∇_{x_hid}Φ‖ (the proof panel).
        (c) Mean hidden penalty  mean_j exp(x_j)  over outer iterations.
        (d) Support recovery F1 over outer iterations (computed post-hoc from
            stored alpha trajectories by re-solving the inner problem).

Usage
-----
    python plot.py           # from within this directory, or
    python expes_fb/expe4_sota_comparison/plot.py   # from repo root
"""

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
    BAR_ERROR_KW,
    PALETTE,
    REFERENCE_COLORS,
    add_shared_legend,
    apply_plot_style,
    figure_size,
    grid_figure_size,
    get_method_style,
)

from sparse_ho.models import WeightedElasticNet
from sparse_ho.algo.forward import compute_beta

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'

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
        'facecolor': PALETTE['fog'],
        'edgecolor': PALETTE['slate'],
        'hatch': None,
    },
    'sparseho_wl1': {
        **get_method_style('sparseho_wl1'),
        'color': PALETTE['charcoal'],
        'facecolor': PALETTE['charcoal'],
        'edgecolor': PALETTE['charcoal'],
        'hatch': '///',
    },
    'ntrba_wl1': {
        **get_method_style('ntrba_wl1'),
        'color': PALETTE['ink'],
        'facecolor': PALETTE['ink'],
        'edgecolor': PALETTE['ink'],
        'hatch': None,
    },
}
METHOD_COLORS = {key: METHOD_STYLES[key]['color'] for key in METHODS}
METHOD_LS = {key: METHOD_STYLES[key]['linestyle'] for key in METHODS}

# Representative instance
REP_N, REP_M, REP_RHO, REP_SEED = 200, 300, 0.98, 0
EASY_FRAC = 0.04
DIST_FRAC = 0.04
INNER_TOL = 1e-8
CALIBRATION_SLACK = 0.05

apply_plot_style()


# ---------------------------------------------------------------------------
# Figure 1 — scalar metrics bar chart
# ---------------------------------------------------------------------------

SCALAR_METRICS = [
    ('val_loss_gap',       'Val. loss gap  Δℓ',                  False),
    ('hidden_grad_norm_0', '‖∇_{x_hid}Φ‖  at k=0',              True),
    ('hidden_recall',      'Hidden-feature recall',               False),
    ('f1',                 'Support recovery F1',                  False),
]

RHO_VALUES = [0.90, 0.95, 0.98]


def plot_metrics(df):
    n_metrics = len(SCALAR_METRICS)
    n_rho = len(RHO_VALUES)

    fig, axes = plt.subplots(
        n_metrics, n_rho,
        figsize=grid_figure_size(
            n_metrics, n_rho, width='twocol', panel_aspect=0.92, extra_height=0.9
        ),
        sharey='row',
    )
    if n_rho == 1:
        axes = axes[:, np.newaxis]

    x = np.arange(len(METHODS))
    width = 0.62

    for row, (metric, ylabel, log_scale) in enumerate(SCALAR_METRICS):
        for col, rho in enumerate(RHO_VALUES):
            ax = axes[row, col]
            sub = df[np.isclose(df.rho, rho)]
            means, errs = [], []
            for mname in METHODS:
                vals = sub[sub.method == mname][metric].values.astype(float)
                means.append(np.nanmean(vals) if len(vals) else np.nan)
                errs.append(np.nanstd(vals)   if len(vals) else 0.0)

            for xpos, mean, err, mname in zip(x, means, errs, METHODS):
                style = METHOD_STYLES[mname]
                ax.bar(
                    xpos,
                    mean,
                    width,
                    yerr=err,
                    capsize=3,
                    color=style['facecolor'],
                    edgecolor=style['edgecolor'],
                    linewidth=0.8,
                    hatch=style['hatch'],
                    error_kw=BAR_ERROR_KW,
                )
            ax.set_xticks(x)
            if row == n_metrics - 1:
                ax.set_xticklabels(['scalar', 'wℓ1', 'ntrba-wℓ1'], rotation=18, ha='right')
            else:
                ax.set_xticklabels([])
            if log_scale:
                # shift any non-positive values before log scale
                ax.set_yscale('symlog', linthresh=1e-6)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f'ρ = {rho}')
            ax.set_xlim(-0.6, len(METHODS) - 0.4)
            ax.grid(axis='y', color=REFERENCE_COLORS['grid'], linewidth=0.5, linestyle='--')

    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=METHOD_STYLES[mname]['facecolor'],
            edgecolor=METHOD_STYLES[mname]['edgecolor'],
            hatch=METHOD_STYLES[mname]['hatch'],
            label=METHOD_LABELS[mname],
        )
        for mname in METHODS
    ]
    add_shared_legend(
        fig,
        axes.ravel(),
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out = RESULTS_DIR / 'fig_metrics.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Post-hoc F1 trajectory for the convergence figure
# ---------------------------------------------------------------------------

def _group_sizes(m):
    n_easy = max(int(EASY_FRAC * m), 5)
    n_dist = max(int(DIST_FRAC * m), 5)
    n_hidden = n_dist
    return n_easy, n_dist, n_hidden


def _solve_inner_f1(X_tr, y_tr, alpha, alpha_l2, beta_true, tol=INNER_TOL):
    """Solve inner problem at a given alpha vector; return F1 against beta_true."""
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
    """Compute F1 at each outer iteration from stored alpha trajectories."""
    f1s = []
    for a in alphas_traj:
        if a is None:
            f1s.append(np.nan)
        else:
            f1s.append(_solve_inner_f1(X_tr, y_tr, a, alpha_l2, beta_true))
    return f1s


# ---------------------------------------------------------------------------
# Figure 2 — convergence for representative instance
# ---------------------------------------------------------------------------

def plot_convergence(df):
    n, m, rho = REP_N, REP_M, REP_RHO
    seed = REP_SEED

    sel = df[
        (df.n == n) & (df.m == m) &
        np.isclose(df.rho, rho) & (df.seed == seed)
    ]
    if sel.empty:
        print(f"plot_convergence: no rows for n={n}, m={m}, rho={rho}, seed={seed}, skipping.")
        return

    # Regenerate the representative dataset (same seed) for post-hoc F1
    n_easy, n_dist, n_hidden = _group_sizes(m)
    alpha_l2 = 1.0 / int(0.6 * n)
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X_rep, y_rep, beta_true_rep, groups_rep, idx_train_rep, _, _, _ = (
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

    # reference (best val for ntrba_wl1)
    ntrba_row = sel[sel.method == 'ntrba_wl1']
    ref_val = (
        float(min(ntrba_row.iloc[0]['val_objs']))
        if not ntrba_row.empty and ntrba_row.iloc[0]['val_objs']
        else 0.0
    )

    fig, axes = plt.subplots(2, 2, figsize=figure_size('twocol', aspect=0.9))
    ax_gap, ax_hgrad, ax_hpen, ax_f1 = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])

    # Collect scalar best val for sparseho_scalar horizontal line
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

        # (a) validation loss gap
        if val_objs:
            gap = [v - ref_val for v in val_objs]
            ax_gap.plot(iters, gap, color=color, linestyle=ls,
                        linewidth=lw, label=label,
                        marker=METHOD_STYLES[mname]['marker'],
                        markevery=max(len(gap) // 8, 1), markersize=3.0)

        # (b) hidden gradient norm
        if hg_norms:
            ax_hgrad.plot(iters[:len(hg_norms)], hg_norms,
                          color=color, linestyle=ls, linewidth=lw, label=label,
                          marker=METHOD_STYLES[mname]['marker'],
                          markevery=max(len(hg_norms) // 8, 1), markersize=3.0)

        # (c) mean hidden penalty
        if mean_hpen:
            ax_hpen.plot(iters[:len(mean_hpen)], mean_hpen,
                         color=color, linestyle=ls, linewidth=lw, label=label,
                         marker=METHOD_STYLES[mname]['marker'],
                         markevery=max(len(mean_hpen) // 8, 1), markersize=3.0)

        # (d) F1 per iteration (post-hoc)
        if alphas_traj and any(a is not None for a in alphas_traj):
            f1_traj = _compute_f1_trajectory(
                alphas_traj, X_tr_rep, y_tr_rep, beta_true_rep, alpha_l2)
            ax_f1.plot(iters[:len(f1_traj)], f1_traj,
                       color=color, linestyle=ls, linewidth=lw, label=label,
                       marker=METHOD_STYLES[mname]['marker'],
                       markevery=max(len(f1_traj) // 8, 1), markersize=3.0)

    # Horizontal reference lines for scalar method
    if np.isfinite(scalar_best):
        scalar_gap = scalar_best - ref_val
        ax_gap.axhline(scalar_gap, color=METHOD_COLORS['sparseho_scalar'],
                       linestyle=METHOD_LS['sparseho_scalar'], linewidth=1.4,
                       label=METHOD_LABELS['sparseho_scalar'])

    # Formatting
    ax_gap.set_xlabel('Outer iteration')
    ax_gap.set_ylabel('Validation loss gap  Φ − Φ*')
    ax_gap.set_title(f'Validation loss gap  (n={n}, p={m}, ρ={rho}, seed={seed})')
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

    add_shared_legend(
        fig, [ax_gap, ax_hgrad, ax_hpen, ax_f1], ncol=3, bbox_to_anchor=(0.5, 0.995)
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out = RESULTS_DIR / 'fig_convergence.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results not found at {RESULTS_PATH}.  Run run.py first.")
    df = pd.read_pickle(RESULTS_PATH)
    print(f"Loaded {len(df)} rows from {RESULTS_PATH}")
    print(f"Methods:  {sorted(df.method.unique())}")
    print(f"(n,m):    {sorted(set(zip(df.n, df.m)))}")
    print(f"rho:      {sorted(df.rho.unique())}")
    print(f"Seeds:    {sorted(df.seed.unique())}")

    plot_metrics(df)
    plot_convergence(df)


if __name__ == '__main__':
    main()
