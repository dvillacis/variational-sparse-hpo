"""Plot results for Experiment 2.

Produces the manuscript figure, saved to the results/ subdirectory:

  fig_penalty_profile.pdf   — learned penalty exp(x_j) per feature, colored
                              by group (signal / corr noise / pure noise).

Usage
-----
    python plot.py           # from within this directory, or
    python expes_fb/expe2_feature_resolution/plot.py   # from repo root
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expes_fb.shared.plotting import (
    BAR_ERROR_KW,
    PALETTE,
    REFERENCE_COLORS,
    apply_plot_style,
    add_shared_legend,
    figure_size,
    grid_figure_size,
    get_method_style,
)

RESULTS_DIR  = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'

METHOD_COLORS = {
    'scalar': get_method_style('scalar')['color'],
    'weighted': get_method_style('weighted')['color'],
}
METHOD_LABELS = {
    'scalar':   'Scalar (CV)',
    'weighted': 'Weighted (ours)',
}
METHOD_STYLES = {
    key: get_method_style(key) for key in METHOD_LABELS
}
GROUP_STYLES = {
    'signal': {
        'facecolor': PALETTE['ink'],
        'edgecolor': PALETTE['ink'],
        'hatch': None,
        'label': 'Signal',
    },
    'corr_noise': {
        'facecolor': PALETTE['charcoal'],
        'edgecolor': PALETTE['charcoal'],
        'hatch': '///',
        'label': 'Corr. noise',
    },
    'pure_noise': {
        'facecolor': PALETTE['fog'],
        'edgecolor': PALETTE['slate'],
        'hatch': None,
        'label': 'Pure noise',
    },
}

apply_plot_style()


# ---------------------------------------------------------------------------
# Figure A — penalty profile
# ---------------------------------------------------------------------------

def plot_penalty_profile(df, m=100, nm_ratio=0.2, sparsity_frac=0.05, seed=0):
    """Side-by-side bar chart of exp(x_j) for scalar vs. weighted."""
    sel = df[
        (df.m == m) & (df.nm_ratio == nm_ratio) &
        (df.sparsity_frac == sparsity_frac) & (df.seed == seed)
    ]
    if sel.empty:
        print("plot_penalty_profile: no matching rows, skipping.")
        return

    row0 = sel.iloc[0]
    s_signal = int(row0['s_signal'])
    s_corr   = int(row0['s_corr'])

    fig, axes = plt.subplots(
        1, 2, figsize=grid_figure_size(1, 2, width='twocol', panel_aspect=0.56, extra_height=0.32)
    )
    xs = np.arange(m)
    group_slices = [
        ('signal', slice(0, s_signal)),
        ('corr_noise', slice(s_signal, s_signal + s_corr)),
        ('pure_noise', slice(s_signal + s_corr, m)),
    ]

    for ax, method in zip(axes, ['scalar', 'weighted']):
        rows = sel[sel.method == method]
        if rows.empty:
            ax.set_title(f'{METHOD_LABELS[method]} — no data')
            continue
        profile = np.asarray(rows.iloc[0]['penalty_profile'])
        for group_name, sl in group_slices:
            if sl.start >= sl.stop:
                continue
            style = GROUP_STYLES[group_name]
            ax.bar(
                xs[sl],
                profile[sl],
                width=1.0,
                linewidth=0.35,
                color=style['facecolor'],
                edgecolor=style['edgecolor'],
                hatch=style['hatch'],
            )
        ax.set_xlabel('Feature index')
        ax.set_ylabel('Penalty weight  $\\exp(x_j)$')
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlim(-0.5, m - 0.5)

    legend_patches = [
        mpatches.Patch(
            facecolor=style['facecolor'],
            edgecolor=style['edgecolor'],
            hatch=style['hatch'],
            label=style['label'],
        )
        for style in GROUP_STYLES.values()
    ]
    add_shared_legend(
        fig,
        axes,
        handles=legend_patches,
        labels=[patch.get_label() for patch in legend_patches],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
    )
    fig.suptitle(
        f'p={m}, n/p={nm_ratio}, sparsity={sparsity_frac:.0%}, seed={seed}',
        y=0.96,
    )
    fig.subplots_adjust(top=0.80, bottom=0.40, wspace=0.18)
    fig.tight_layout()
    out = RESULTS_DIR / 'fig_penalty_profile.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results not found at {RESULTS_PATH}. Run run.py first.")
    df = pd.read_pickle(RESULTS_PATH)
    print(f"Loaded {len(df)} rows from {RESULTS_PATH}")

    # use the smallest config as the representative for the penalty profile
    m_rep  = int(df.m.min())
    nm_rep = float(df.nm_ratio.min())
    sp_rep = float(df.sparsity_frac.max())

    plot_penalty_profile(df, m=m_rep, nm_ratio=nm_rep, sparsity_frac=sp_rep)


if __name__ == '__main__':
    main()
