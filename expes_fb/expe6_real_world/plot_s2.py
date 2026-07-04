"""Plot results for Experiment 6 Setting 2 — Real-world benchmarks.

Produces one figure:

  fig_s2_results.pdf
      3-row × n_datasets bar chart.
        Row 1 — Test F1.
        Row 2 — Active features (% non-zero coefficients).
        Row 3 — Wall-clock time per outer iteration (s).
      Error bars show ± std over seeds.

Usage
-----
    python plot_s2.py
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
    grid_figure_size,
    get_method_style,
)

RESULTS_DIR = Path(__file__).parent / 'results' / 'setting2'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'

METHODS = ['scalar_cv', 'sparseho_wl1', 'ntrba_wl1']
METHOD_LABELS = {
    'scalar_cv':      'Scalar ℓ1 (CV)',
    'sparseho_wl1':   'SparseHO (wℓ1)',
    'ntrba_wl1':      'NTRBA-wℓ1 (ours)',
}
METHOD_STYLES = {key: get_method_style(key) for key in METHODS}
METHOD_COLORS = {key: METHOD_STYLES[key]['color'] for key in METHODS}

apply_plot_style()

METRICS = [
    ('test_f1',    'Test F1',                       False),
    ('sparsity',   'Active features (%)',            False),
    ('t_per_iter', 'Time per outer iteration (s)',   True),
]

DATASET_DISPLAY = {
    'mnist': 'mnist (0/1)',
    'breast-cancer': 'breast-cancer',
    'leukemia': 'leukemia',
    'rcv1': 'rcv1',
    'rcv1.binary': 'rcv1.binary',
    'rcv1_train.binary': 'rcv1.binary',
    'real-sim': 'real-sim',
    'news20.binary': 'news20',
}


def plot_results(df):
    datasets = sorted(df.dataset.unique())
    n_ds = len(datasets)
    n_met = len(METRICS)

    if n_ds == 0:
        print("No datasets found in results.")
        return

    fig, axes = plt.subplots(
        n_met, n_ds,
        figsize=grid_figure_size(
            n_met, n_ds, width='twocol', panel_aspect=0.9, extra_height=0.75
        ),
        sharey='row',
        squeeze=False,
    )

    x = np.arange(len(METHODS))
    width = 0.55

    for row, (metric, ylabel, log_scale) in enumerate(METRICS):
        for col, dname in enumerate(datasets):
            ax = axes[row, col]
            sub = df[df.dataset == dname]
            means, errs = [], []
            for mname in METHODS:
                vals = sub[sub.method == mname][metric].values.astype(float)
                means.append(np.nanmean(vals) if len(vals) else np.nan)
                errs.append(np.nanstd(vals)   if len(vals) else 0.0)

            ax.bar(x, means, width,
                   yerr=errs, capsize=3,
                   color=[METHOD_COLORS[m_] for m_ in METHODS],
                   error_kw=BAR_ERROR_KW)
            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_LABELS[m_] for m_ in METHODS],
                               rotation=25, ha='right')
            if log_scale:
                ax.set_yscale('log')
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(DATASET_DISPLAY.get(dname, dname))
            ax.set_xlim(-0.6, len(METHODS) - 0.4)

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
               for m in METHODS]

    add_shared_legend(
        fig,
        axes.ravel(),
        handles=patches,
        labels=[METHOD_LABELS[m] for m in METHODS],
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out = RESULTS_DIR / 'fig_s2_results.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results not found at {RESULTS_PATH}.  Run run_s2.py first.")
    df = pd.read_pickle(RESULTS_PATH)
    print(f"Loaded {len(df)} rows.")
    summary = df.groupby(['dataset', 'method'])[
        ['test_f1', 'sparsity', 't_per_iter']
    ].agg(['mean', 'std'])
    print(summary.to_string())
    plot_results(df)


if __name__ == '__main__':
    main()
