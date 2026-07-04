"""Plot results for Experiment 6 Setting 1 — Semi-synthetic sparse-text.

Produces two figures:

  fig_s1_metrics.pdf
      Bar chart: hidden_grad_norm_0, hidden_recall, F1, test_logloss
      for the three methods, averaged over seeds.

  fig_s1_convergence.pdf
      Convergence for seed 0:
        (a) Validation log-loss over outer iterations.
        (b) Hidden-feature gradient norm ‖∇_{x_hid}Φ‖ over iterations.

Usage
-----
    python plot_s1.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expes_fb.shared.plotting import (
    BAR_ERROR_KW,
    add_shared_legend,
    apply_plot_style,
    figure_size,
    grid_figure_size,
    get_method_style,
)

RESULTS_DIR = Path(__file__).parent / 'results' / 'setting1'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'

METHODS = ['sparseho_scalar', 'sparseho_wl1', 'ntrba_wl1']
METHOD_LABELS = {
    'sparseho_scalar': 'Scalar ℓ1 (grid)',
    'sparseho_wl1':    'SparseHO (wℓ1)',
    'ntrba_wl1':       'NTRBA-wℓ1 (ours)',
}
METHOD_STYLES = {key: get_method_style(key) for key in METHODS}
METHOD_COLORS = {key: METHOD_STYLES[key]['color'] for key in METHODS}
METHOD_LS = {key: METHOD_STYLES[key]['linestyle'] for key in METHODS}

apply_plot_style()

SCALAR_METRICS = [
    ('hidden_grad_norm_0', '‖∇$_{x_{hid}}$Φ‖ at k=0', True),
    ('hidden_recall',      'Hidden-feature recall',     False),
    ('f1',                 'Support recovery F1',        False),
    ('test_logloss',       'Test log-loss',               False),
]


def plot_metrics(df):
    n_m = len(SCALAR_METRICS)
    fig, axes = plt.subplots(
        1, n_m,
        figsize=grid_figure_size(1, n_m, width='twocol', panel_aspect=0.88, extra_height=0.45),
    )
    x = np.arange(len(METHODS))
    width = 0.55

    for col, (metric, ylabel, log_scale) in enumerate(SCALAR_METRICS):
        ax = axes[col]
        means, errs = [], []
        for mname in METHODS:
            vals = df[df.method == mname][metric].values.astype(float)
            means.append(np.nanmean(vals) if len(vals) else np.nan)
            errs.append(np.nanstd(vals) if len(vals) else 0.0)
        ax.bar(x, means, width,
               yerr=errs, capsize=3,
               color=[METHOD_COLORS[m_] for m_ in METHODS],
               error_kw=BAR_ERROR_KW)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m_] for m_ in METHODS],
                           rotation=25, ha='right')
        if log_scale:
            ax.set_yscale('symlog', linthresh=1e-6)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)

    fig.suptitle('Setting 1: Semi-synthetic rcv1.binary (mean ± std over seeds)')
    fig.tight_layout()
    out = RESULTS_DIR / 'fig_s1_metrics.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


def plot_convergence(df, seed=0):
    sel = df[df.seed == seed]
    if sel.empty:
        print(f"plot_convergence: no rows for seed={seed}, skipping.")
        return

    fig, (ax_val, ax_hg) = plt.subplots(1, 2, figsize=figure_size('twocol', aspect=0.5))

    scalar_row = sel[sel.method == 'sparseho_scalar']
    scalar_best = (float(min(scalar_row.iloc[0]['val_objs']))
                   if not scalar_row.empty and scalar_row.iloc[0]['val_objs']
                   else np.nan)

    for mname in ['sparseho_wl1', 'ntrba_wl1']:
        row_sel = sel[sel.method == mname]
        if row_sel.empty:
            continue
        row = row_sel.iloc[0]
        objs = row['val_objs']
        hg = row['hidden_grad_norms']
        iters = np.arange(len(objs))
        color, ls = METHOD_COLORS[mname], METHOD_LS[mname]
        label = METHOD_LABELS[mname]

        if objs:
            ax_val.plot(iters, objs, color=color, linestyle=ls,
                        linewidth=1.6, label=label,
                        marker=METHOD_STYLES[mname]['marker'],
                        markevery=max(len(objs) // 8, 1), markersize=3.0)
        if hg:
            ax_hg.plot(np.arange(len(hg)), hg, color=color, linestyle=ls,
                       linewidth=1.6, label=label,
                       marker=METHOD_STYLES[mname]['marker'],
                       markevery=max(len(hg) // 8, 1), markersize=3.0)

    if np.isfinite(scalar_best):
        ax_val.axhline(scalar_best,
                       color=METHOD_COLORS['sparseho_scalar'],
                       linestyle=METHOD_LS['sparseho_scalar'],
                       linewidth=1.4,
                       label=METHOD_LABELS['sparseho_scalar'])

    ax_val.set_xlabel('Outer iteration')
    ax_val.set_ylabel('Validation log-loss')
    ax_val.set_title(f'(a) Convergence  (seed={seed})')

    ax_hg.set_yscale('symlog', linthresh=1e-6)
    ax_hg.set_xlabel('Outer iteration')
    ax_hg.set_ylabel('‖∇$_{x_{hid}}$Φ‖')
    ax_hg.set_title('(b) Hidden-feature gradient norm')

    add_shared_legend(fig, [ax_val, ax_hg], ncol=3, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out = RESULTS_DIR / 'fig_s1_convergence.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results not found at {RESULTS_PATH}.  Run run_s1.py first.")
    df = pd.read_pickle(RESULTS_PATH)
    print(f"Loaded {len(df)} rows.")
    print(df.groupby('method')[
        ['hidden_grad_norm_0', 'hidden_recall', 'f1', 'test_logloss']
    ].mean().to_string())
    plot_metrics(df)
    plot_convergence(df, seed=0)


if __name__ == '__main__':
    main()
