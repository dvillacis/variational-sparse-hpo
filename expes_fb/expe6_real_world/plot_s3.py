"""Figures for Experiment 6 Setting 3 — natural-degeneracy counterfactual.

  fig_expe6_invariance.pdf     (a) B, m_sel vs eps_B ; (b) B vs tau per dataset
  fig_expe6_counterfactual.pdf (a) dPhi null vs SC per dataset ; (b) Prop-15 scatter

Reuses the shared paper style (expes_fb/shared/plotting.py). Saves to the local
results dir; pass --paper to also copy PDFs into the manuscript folder.

Usage
-----
    python plot_s3.py
    python plot_s3.py --paper
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from expes_fb.shared.plotting import (  # noqa: E402
    apply_plot_style, figure_size, add_shared_legend, PALETTE)

RESULTS_DIR = Path(__file__).parent / 'results' / 'setting3'
# Paper folder (active revision); resolved from the repo CLAUDE.md pointer.
PAPER_DIR = Path(
    "/Users/davidvillacis/Google Drive/Mi unidad/PUBLICATIONS/"
    "65525240e179c735d533dfb0/Revision2_23062026")

DATASET_DISPLAY = {
    'mnist': 'mnist', 'rcv1.binary': 'rcv1', 'rcv1_train.binary': 'rcv1',
    'real-sim': 'real-sim', 'news20.binary': 'news20', 'phishing': 'phishing',
}


def _save(fig, name, to_paper):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / name
    fig.savefig(out, bbox_inches='tight')
    print('saved', out)
    if to_paper and PAPER_DIR.exists():
        dst = PAPER_DIR / name
        shutil.copyfile(out, dst)
        print('copied ->', dst)
    elif to_paper:
        print(f'(paper dir not found, skipped: {PAPER_DIR})')


def plot_invariance(inv, to_paper):
    fig, axes = plt.subplots(1, 2, figsize=figure_size('twocol', aspect=0.42))
    eps_vals = sorted(inv.eps_B.unique())
    tau_min = inv.tau.min()

    # (a) B and m_sel vs eps_B at tightest tau, averaged over datasets/seeds
    sub = inv[np.isclose(inv.tau, tau_min)]
    B = [sub[np.isclose(sub.eps_B, e)].biactive.mean() for e in eps_vals]
    S = [sub[np.isclose(sub.eps_B, e)].selected_biactive.mean() for e in eps_vals]
    ax = axes[0]
    ax.plot(eps_vals, B, marker='o', color=PALETTE['ink'], label=r'biactive $B$')
    ax.plot(eps_vals, S, marker='^', color=PALETTE['blue'],
            label=r'SC-selected $m_{\mathrm{sel}}$')
    ax.axvline(0.10, color=PALETTE['vermillion'], ls=':', lw=1.0)
    ax.text(0.10, ax.get_ylim()[1], r' $2\delta$ (planted band)',
            color=PALETTE['vermillion'], fontsize=6, va='top', ha='left')
    ax.set_xscale('log'); ax.set_xlabel(r'detection band $\varepsilon_B$')
    ax.set_ylabel('count (mean over data)')
    ax.set_title('(a) incidence is band-controlled')

    # (b) B vs tau per dataset at eps_B = 1e-3 (flat => not a solver artifact)
    ax = axes[1]
    eps_ref = 1e-3 if any(np.isclose(inv.eps_B, 1e-3)) else eps_vals[len(eps_vals) // 2]
    subb = inv[np.isclose(inv.eps_B, eps_ref)]
    taus = sorted(subb.tau.unique())
    for d in sorted(subb.dataset.unique()):
        sd = subb[subb.dataset == d]
        vals = [sd[np.isclose(sd.tau, t)].biactive.mean() for t in taus]
        ax.plot(taus, vals, marker='s', label=DATASET_DISPLAY.get(d, d))
    ax.set_xscale('log'); ax.set_xlabel(r'inner-solver tolerance $\tau$')
    ax.set_ylabel(r'biactive $B$')
    ax.set_title(rf'(b) $\tau$-invariance ($\varepsilon_B={eps_ref:g}$)')

    add_shared_legend(fig, axes)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, 'fig_expe6_invariance.pdf', to_paper)
    plt.close(fig)


def plot_counterfactual(cf, br, to_paper):
    has_branch = br is not None and not br.empty
    ncol = 2 if has_branch else 1
    fig, axes = plt.subplots(
        1, ncol, figsize=figure_size('twocol' if has_branch else 'onecol',
                                     aspect=0.5 if has_branch else 0.7))
    axes = np.atleast_1d(axes)

    # (a) dPhi null vs SC per dataset
    ax = axes[0]
    datasets = sorted(cf.dataset.unique())
    x = np.arange(len(datasets)); w = 0.38
    mn = [cf[cf.dataset == d].dPhi_null.mean() for d in datasets]
    en = [cf[cf.dataset == d].dPhi_null.std() for d in datasets]
    ms = [cf[cf.dataset == d].dPhi_sc.mean() for d in datasets]
    es = [cf[cf.dataset == d].dPhi_sc.std() for d in datasets]
    ax.bar(x - w / 2, mn, w, yerr=en, label='null oracle',
           color=PALETTE['slate'])
    ax.bar(x + w / 2, ms, w, yerr=es, label='sign-consistent',
           color=PALETTE['ink'])
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in datasets],
                       rotation=30, ha='right')
    ax.set_ylabel(r'$\Delta\Phi$ (held-out decrease)')
    ax.set_title('(a) warm-start counterfactual')

    # (b) Prop-15 validation: fd_slope vs -h_i
    if has_branch:
        ax = axes[1]
        ax.scatter(br.minus_h_i, br.fd_slope, s=10, alpha=0.6,
                   color=PALETTE['blue'])
        lo = float(np.nanmin([br.minus_h_i.min(), br.fd_slope.min()]))
        hi = float(np.nanmax([br.minus_h_i.max(), br.fd_slope.max()]))
        ax.plot([lo, hi], [lo, hi], ls='--', color=PALETTE['charcoal'], lw=1.0)
        ax.set_xlabel(r'$-h_i^{\mathrm{SC}}$ (predicted slope)')
        ax.set_ylabel(r"finite-diff $\Phi'(x;-e_i)$")
        ax.set_title('(b) Prop.~15 check')

    add_shared_legend(fig, [axes[0]])
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, 'fig_expe6_counterfactual.pdf', to_paper)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--paper', action='store_true',
                   help='also copy PDFs into the manuscript folder')
    args = p.parse_args()
    apply_plot_style()

    cf = pd.read_pickle(RESULTS_DIR / 'results_counterfactual.pkl')
    inv_path = RESULTS_DIR / 'results_invariance.pkl'
    br_path = RESULTS_DIR / 'results_branch.pkl'
    inv = pd.read_pickle(inv_path) if inv_path.exists() else pd.DataFrame()
    br = pd.read_pickle(br_path) if br_path.exists() else pd.DataFrame()

    if not inv.empty:
        plot_invariance(inv, args.paper)
    plot_counterfactual(cf, br, args.paper)


if __name__ == '__main__':
    main()
