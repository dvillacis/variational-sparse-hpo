"""Experiment 7 — figure: the false-certificate zone sits at the top of the path.

Two panels sharing the x-axis t = alpha / alpha_max (1 -> cv-opt): support size
and biactive count B along the path, with the false-certificate zone
(||g_null|| = 0 AND Phi' < 0) marked at the anchor. One microarray panel
(natural biactive cluster) and one text panel (clean). Reads results/results.pkl.

Usage
-----
    uv run python expes_fb/expe7_certificate_soundness/plot.py
    uv run python expes_fb/expe7_certificate_soundness/plot.py --copy-to /path/to/paper
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / 'results'
sys.path.insert(0, str(HERE.parent / 'shared'))

try:
    from plotting import apply_plot_style, figure_size, PALETTE
    _HAVE_STYLE = True
except Exception:  # noqa: BLE001
    _HAVE_STYLE = False
    PALETTE = {'ink': '#1A1A1A', 'blue': '#0072B2', 'vermillion': '#D55E00',
               'fog': '#BDBDBD', 'slate': '#737373'}

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def _panel(ax, d, title):
    d = d[d.sweep == 'headline'].sort_values('t', ascending=False)
    t = d['t'].to_numpy(float)
    supp = d['support'].to_numpy(float)
    B = d['B'].to_numpy(float)
    gnull = d['g_null_norm'].to_numpy(float)
    false_cert = d['false_certificate'].to_numpy(bool)

    ax.plot(t, supp, color=PALETTE['slate'], marker='o', ms=3, lw=1.2,
            label=r'support $|\mathcal{I}|$')
    ax.plot(t, B, color=PALETTE['blue'], marker='^', ms=4, lw=1.4,
            label=r'biactive $B$')
    # mark false-certificate points (null certifies yet descent exists)
    if false_cert.any():
        ax.scatter(t[false_cert], B[false_cert], s=90, facecolors='none',
                   edgecolors=PALETTE['vermillion'], linewidths=1.6, zorder=5,
                   label='false certificate')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r'$t=\alpha/\alpha_{\max}$')
    ax.set_xlim(max(t) + 0.03, min(t) - 0.03)   # path descends left->right
    ax.set_ylabel('count')
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default=str(RESULTS_DIR / 'results.pkl'))
    ap.add_argument('--micro', default='leukemia')
    ap.add_argument('--text', default='splice')
    ap.add_argument('--copy-to', default=None)
    args = ap.parse_args()

    if _HAVE_STYLE:
        apply_plot_style()
        figsize = figure_size('twocol', aspect=0.42)
    else:
        figsize = (6.85, 2.9)

    df = pd.read_pickle(args.pkl)
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    _panel(axes[0], df[df.dataset == args.micro],
           f'{args.micro} (natural biactivity)')
    _panel(axes[1], df[df.dataset == args.text],
           f'{args.text} (clean)')
    axes[0].legend(fontsize=7, loc='upper left', framealpha=0.9)

    out = RESULTS_DIR / 'fig_certificate_soundness.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'saved -> {out}')
    if args.copy_to:
        dest = Path(args.copy_to) / 'fig_certificate_soundness.pdf'
        fig.savefig(dest, bbox_inches='tight')
        print(f'copied -> {dest}')


if __name__ == '__main__':
    main()
