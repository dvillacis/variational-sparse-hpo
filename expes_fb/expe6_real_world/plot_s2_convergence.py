"""Convergence figure for Experiment 5 Setting 2 (Option A re-run).

For each dataset, plots the held-out (validation) objective vs outer iteration for
Sparse-HO (subgradient) and NTRBA (trust region), both under the shared plateau
stop. It visualizes the key point: NTRBA's objective plateaus and it STOPS early
(marker), while the subgradient is still descending at the budget cap.

Reads results/<TAG>/results.pkl (needs `val_objs`); TAG below.

Usage
-----
    python plot_s2_convergence.py
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
from expes_fb.shared.plotting import (               # noqa: E402
    apply_plot_style, figure_size, get_method_style)

HERE = Path(__file__).parent
TAG = 'setting2_cap100'
PKL = HERE / 'results' / TAG / 'results.pkl'
OUT = HERE / 'results' / TAG / 'fig_s2_convergence.pdf'

DATASETS = ['rcv1.binary', 'real-sim', 'news20.binary', 'mnist', 'phishing']
DLABEL = {'rcv1.binary': 'rcv1', 'real-sim': 'real-sim',
          'news20.binary': 'news20', 'mnist': 'mnist (0/1)', 'phishing': 'phishing'}
METHODS = ['sparseho_wl1', 'ntrba_wl1']
MLABEL = {'sparseho_wl1': 'Sparse-HO', 'ntrba_wl1': 'NTRBA (ours)'}

apply_plot_style()


def _pad(seqs, length, fill='last'):
    out = np.full((len(seqs), length), np.nan)
    for i, s in enumerate(seqs):
        s = np.asarray(s, float)
        n = min(len(s), length)
        out[i, :n] = s[:n]
        if fill == 'last' and len(s) < length and len(s):
            out[i, len(s):] = s[-1]
    return out


def main():
    df = pd.read_pickle(PKL)
    present = [d for d in DATASETS if d in set(df.dataset.unique())]
    ncol = len(present)
    fig, axes = plt.subplots(
        1, ncol, figsize=figure_size('twocol', aspect=0.42),
        squeeze=False)
    axes = axes[0]

    for ax, d in zip(axes, present):
        for m in METHODS:
            blk = df[(df.dataset == d) & (df.method == m)]
            if not len(blk):
                continue
            seqs = [list(v) for v in blk.val_objs.values if len(v)]
            if not seqs:
                continue
            stop = float(np.mean([len(s) for s in seqs]))       # mean stop iter
            L = max(len(s) for s in seqs)
            arr = _pad(seqs, L, fill='last')
            mean = np.nanmean(arr, axis=0)
            it = np.arange(1, L + 1)
            st = get_method_style(m)
            # solid up to the (mean) stop; faint dotted afterwards for NTRBA
            k = max(1, int(round(stop)))
            ax.plot(it[:k], mean[:k], color=st['color'], lw=1.4,
                    label=MLABEL[m])
            if m == 'ntrba_wl1':
                ax.plot(it[k - 1], mean[k - 1], marker='o', ms=4.5,
                        color=st['color'], zorder=5)
                ax.annotate(f'stop $\\approx${k}', (it[k - 1], mean[k - 1]),
                            textcoords='offset points', xytext=(4, 6),
                            fontsize=6, color=st['color'])
        ax.set_title(DLABEL[d], fontsize=8)
        ax.set_xlabel('outer iteration')
        ax.set_yscale('log')
    axes[0].set_ylabel('validation objective')
    axes[0].legend(frameon=False, fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight')
    print('saved', OUT)


if __name__ == '__main__':
    main()
