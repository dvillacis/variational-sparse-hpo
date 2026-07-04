"""Plot results for Experiment 5 — Oracle scalability.

Produces two figures saved to the results/ subdirectory:

  fig_scaling.pdf
      Log-log plot of wall-clock time vs m for the three oracle variants
      (full_dense, null, da).  Reference lines with slope 1, 2, 3 are drawn
      in the background.  The primary message: null and da overlap while
      full_dense is many orders of magnitude slower.

  fig_sparsity.pdf
      Log-log plot of wall-clock time vs ρ_s for the DA oracle at fixed
      m = 10^4.  A slope-3 reference confirms O(|S|^3) ∝ O(ρ_s^3) scaling.

Usage
-----
    python plot.py           # from within this directory, or
    python expes_fb/expe5_scalability/plot.py   # from repo root
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
    PALETTE,
    REFERENCE_COLORS,
    apply_plot_style,
    figure_size,
    get_method_style,
)

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'

ORACLE_LABELS = {
    'dense': 'Full dense  $O(m^3)$',
    'null':  'Support-reduced (null)',
    'sc':    'Support-reduced (SC)',
}
ORACLE_STYLES = {key: get_method_style(key) for key in ORACLE_LABELS}
ORACLE_COLORS = {key: ORACLE_STYLES[key]['color'] for key in ORACLE_LABELS}
ORACLE_LS = {key: ORACLE_STYLES[key]['linestyle'] for key in ORACLE_LABELS}
ORACLE_MARKER = {key: ORACLE_STYLES[key]['marker'] for key in ORACLE_LABELS}

apply_plot_style()

# ---------------------------------------------------------------------------
# Reference slope lines
# ---------------------------------------------------------------------------

def _draw_slope_lines(ax, x_range, ref_point, slopes=(1, 2, 3)):
    """Draw reference lines with given slopes anchored at ref_point."""
    x0, y0 = ref_point
    x_arr = np.array(x_range)
    for slope in slopes:
        y_arr = y0 * (x_arr / x0) ** slope
        ax.plot(x_arr, y_arr,
                color=REFERENCE_COLORS['guide'], linestyle=':', linewidth=0.8, zorder=0)
        # label at the right end
        ax.annotate(
            f'slope {slope}',
            xy=(x_arr[-1], y_arr[-1]),
            fontsize=6, color=REFERENCE_COLORS['guide'],
            ha='left', va='center',
        )


# ---------------------------------------------------------------------------
# Figure 1 — primary sweep: time vs m
# ---------------------------------------------------------------------------

def plot_scaling(df):
    sub = df[df.sweep == 'primary']
    if sub.empty:
        print("plot_scaling: no primary sweep data.")
        return

    fig, ax = plt.subplots(figsize=figure_size('onecol', aspect=0.72))

    for oracle in ['dense', 'null', 'sc']:
        d = sub[sub.oracle == oracle].sort_values('m')
        d = d.dropna(subset=['t_median'])
        if d.empty:
            continue
        ms = d['m'].values
        ts = d['t_median'].values
        ax.loglog(
            ms, ts,
            color=ORACLE_COLORS[oracle],
            linestyle=ORACLE_LS[oracle],
            linewidth=1.8,
            marker=ORACLE_MARKER[oracle],
            markersize=5,
            label=ORACLE_LABELS[oracle],
        )

    # Reference slope lines anchored at (m=100, t=1e-5)
    m_all = sorted(sub['m'].unique())
    x_range = [min(m_all), max(m_all)]
    _draw_slope_lines(ax, x_range, ref_point=(100, 1e-5), slopes=(1, 2, 3))

    ax.set_xlabel('Number of features  $p$')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Oracle scalability: adjoint Hessian + Cholesky solve')
    ax.legend(loc='upper left')
    ax.set_xticks(m_all)
    ax.set_xticklabels([f'$10^{{{np.log10(m):.1f}}}$' for m in m_all])

    fig.tight_layout()
    out = RESULTS_DIR / 'fig_scaling.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — secondary sweep: time vs rho_s
# ---------------------------------------------------------------------------

def plot_sparsity(df):
    sub = df[(df.sweep == 'secondary') & (df.oracle == 'sc')].sort_values('rho_s')
    if sub.empty:
        print("plot_sparsity: no secondary sweep data.")
        return

    m = int(sub['m'].iloc[0])
    rho_vals = sub['rho_s'].values
    t_vals = sub['t_median'].values

    fig, ax = plt.subplots(figsize=figure_size('onecol', aspect=0.7))

    ax.loglog(
        rho_vals, t_vals,
        color=ORACLE_COLORS['sc'],
        linestyle=ORACLE_LS['sc'],
        linewidth=1.8,
        marker=ORACLE_MARKER['sc'],
        markersize=5,
        label=f'DA oracle  ($m = 10^{{{np.log10(m):.0f}}}$)',
    )

    # slope-3 reference anchored at first data point
    rho_range = np.array([rho_vals[0], rho_vals[-1]])
    t0 = t_vals[0]
    rho0 = rho_vals[0]
    ax.loglog(
        rho_range,
        t0 * (rho_range / rho0) ** 3,
        color=REFERENCE_COLORS['guide'], linestyle=':', linewidth=0.8, zorder=0,
    )
    ax.annotate(
        'slope 3',
        xy=(rho_range[-1], t0 * (rho_range[-1] / rho0) ** 3),
        fontsize=6, color=REFERENCE_COLORS['guide'], ha='left', va='center',
    )

    ax.set_xlabel('Sparsity density  $\\rho_s$')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title(f'Sparsity sensitivity  ($m = 10^{{{np.log10(m):.0f}}}$)')
    ax.legend()
    ax.set_xticks(rho_vals)
    ax.set_xticklabels([f'{int(r*100)}%' for r in rho_vals])

    fig.tight_layout()
    out = RESULTS_DIR / 'fig_sparsity.pdf'
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
    print(df.to_string())

    plot_scaling(df)
    plot_sparsity(df)


if __name__ == '__main__':
    main()
