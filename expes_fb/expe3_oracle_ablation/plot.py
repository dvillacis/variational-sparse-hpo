"""Plot results for Experiment 3 — oracle × optimizer ablation.

Produces the manuscript figure, saved to the results/ subdirectory:

  fig_convergence.pdf   — Two-panel dynamics plot for the representative
                          instance (m=500, seed=0):
                            (a) Validation loss trajectory (all 4 methods).
                            (b) Trust-region radius Δ_k (NTRBA-null, NTRBA-SC).

Usage
-----
    python plot.py           # from within this directory, or
    python expes_fb/expe3_oracle_ablation/plot.py   # from repo root
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
    add_shared_legend,
    apply_plot_style,
    figure_size,
    grid_figure_size,
    get_method_style,
)

RESULTS_DIR  = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'

# Method ordering and visual style
METHODS = ['NBA-null', 'NBA-SC', 'NTRBA-null', 'NTRBA-SC']

METHOD_LABELS = {
    'NBA-null':   'NBA-null',
    'NBA-SC':     'NBA-SC',
    'NTRBA-null': 'NTRBA-null',
    'NTRBA-SC':   'NTRBA-SC (ours)',
}
METHOD_STYLES = {
    'NBA-null': {
        **get_method_style('NBA-null'),
        'color': PALETTE['slate'],
        'facecolor': PALETTE['fog'],
        'edgecolor': PALETTE['slate'],
        'hatch': None,
    },
    'NBA-SC': {
        **get_method_style('NBA-SC'),
        'color': PALETTE['charcoal'],
        'facecolor': PALETTE['charcoal'],
        'edgecolor': PALETTE['charcoal'],
        'hatch': '///',
    },
    'NTRBA-null': {
        **get_method_style('NTRBA-null'),
        'color': PALETTE['charcoal'],
        'facecolor': 'white',
        'edgecolor': PALETTE['charcoal'],
        'hatch': '...',
    },
    'NTRBA-SC': {
        **get_method_style('NTRBA-SC'),
        'color': PALETTE['ink'],
        'facecolor': PALETTE['ink'],
        'edgecolor': PALETTE['ink'],
        'hatch': None,
    },
}

apply_plot_style()


# ---------------------------------------------------------------------------
# Figure — convergence for representative instance
# ---------------------------------------------------------------------------

def plot_convergence(df, m_rep=500, seed_rep=0):
    sel = df[(df.m == m_rep) & (df.seed == seed_rep)]
    if sel.empty:
        print(f"plot_convergence: no rows for m={m_rep}, seed={seed_rep}, skipping.")
        return

    fig, (ax_val, ax_rad) = plt.subplots(1, 2, figsize=figure_size('twocol', aspect=0.48))

    # panel (a): validation loss trajectory
    for mname in METHODS:
        rows = sel[sel.method == mname]
        if rows.empty:
            continue
        objs = rows.iloc[0]['val_objs']
        if not objs:
            continue
        ax_val.plot(
            objs,
            color=METHOD_STYLES[mname]['color'],
            label=METHOD_LABELS[mname],
            linewidth=1.5,
            linestyle=METHOD_STYLES[mname]['linestyle'],
            marker=METHOD_STYLES[mname]['marker'],
            markevery=max(len(objs) // 8, 1),
            markersize=3.2,
        )
    ax_val.set_xlabel('Outer iteration')
    ax_val.set_ylabel('Validation loss')
    ax_val.set_title(f'Validation loss  (p={m_rep}, seed={seed_rep})')
    ax_val.grid(axis='y', color=REFERENCE_COLORS['grid'], linewidth=0.5, linestyle='--')

    # panel (b): trust-region radius for NTRBA methods
    found_any = False
    for mname in ['NTRBA-null', 'NTRBA-SC']:
        rows = sel[sel.method == mname]
        if rows.empty:
            continue
        radii = rows.iloc[0]['tr_radii']
        if not radii:
            continue
        ax_rad.plot(
            radii,
            color=METHOD_STYLES[mname]['color'],
            label=METHOD_LABELS[mname],
            linewidth=1.5,
            linestyle=METHOD_STYLES[mname]['linestyle'],
            marker=METHOD_STYLES[mname]['marker'],
            markevery=max(len(radii) // 8, 1),
            markersize=3.2,
        )
        found_any = True
    if found_any:
        ax_rad.set_yscale('log')
        ax_rad.set_xlabel('Outer iteration')
        ax_rad.set_ylabel(r'Trust-region radius  $\Delta_k$')
        ax_rad.set_title('Trust-region radius')
        ax_rad.grid(axis='y', color=REFERENCE_COLORS['grid'], linewidth=0.5, linestyle='--')
    else:
        ax_rad.set_visible(False)

    add_shared_legend(fig, [ax_val, ax_rad], ncol=4, bbox_to_anchor=(0.5, 0.995))
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
    print(f"Methods: {sorted(df.method.unique())}")
    print(f"m values: {sorted(df.m.unique())}")
    print(f"Seeds:    {sorted(df.seed.unique())}")

    m_rep = 500 if 500 in df.m.values else int(df.m.median())
    plot_convergence(df, m_rep=m_rep, seed_rep=0)


if __name__ == '__main__':
    main()
