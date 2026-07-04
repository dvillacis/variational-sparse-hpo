"""Generate LaTeX table for Experiment 6 Setting 1.

Produces a grouped long-form table indexed by (dataset, method), with
mean ± std over seeds for the semi-synthetic sparse-text benchmark.

Usage
-----
    python table_s1.py
    python expes_fb/expe6_real_world/table_s1.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / 'results' / 'setting1'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'
TABLE_PATH = RESULTS_DIR / 'table_s1_metrics.tex'

DATASETS = ['rcv1.binary', 'real-sim', 'w8a', 'news20.binary']
DATASET_LABELS = {
    'rcv1.binary': r'\textsc{RCV1}',
    'real-sim': r'\textsc{Real-sim}',
    'w8a': r'\textsc{W8A}',
    'news20.binary': r'\textsc{News20}',
}

METHODS = ['sparseho_scalar', 'sparseho_wl1', 'ntrba_wl1']
METHOD_LABELS = {
    'sparseho_scalar': r'Scalar $\ell_1$ (grid)',
    'sparseho_wl1': r'\texttt{Sparse-HO} ($w\ell_1$)',
    'ntrba_wl1': r'\texttt{NTRBA}-$w\ell_1$ (ours)',
}

METRICS = [
    ('hidden_recall', r'Hidden recall', False, '.3f'),
    ('active_features_pct', r'Act. feat. (\%)', True, '.1f'),
    ('f1', r'F1', False, '.3f'),
    ('test_logloss', r'Log-loss', True, '.3f'),
    ('elapsed', r'Runtime (s)', True, '.1f'),
]


def _fmt(mu, sd, spec):
    if np.isnan(mu):
        return r'\text{---}'
    return rf'{mu:{spec}}\,\pm\,{sd:{spec}}'


def _shade(cell):
    return rf'\cellcolor{{gray!15}} ${cell}$'


def _present_datasets(df: pd.DataFrame):
    present = list(df['dataset'].dropna().unique())
    ordered = [name for name in DATASETS if name in present]
    extras = sorted(name for name in present if name not in ordered)
    return ordered + extras


def build_table(df: pd.DataFrame) -> str:
    datasets = _present_datasets(df)

    stats = {}
    for dataset in datasets:
        for method in METHODS:
            block = df[(df.dataset == dataset) & (df.method == method)]
            for col, _, _, _ in METRICS:
                vals = block[col].values.astype(float)
                stats[(dataset, method, col)] = (
                    np.nanmean(vals) if len(vals) else np.nan,
                    np.nanstd(vals) if len(vals) else np.nan,
                )

    best = {}
    for dataset in datasets:
        for col, _, lower_is_better, _ in METRICS:
            mus = [stats[(dataset, method, col)][0] for method in METHODS]
            valid = [v for v in mus if not np.isnan(v)]
            if valid:
                best[(dataset, col)] = min(valid) if lower_is_better else max(valid)

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \setlength{\tabcolsep}{4pt}')
    lines.append(r'  \caption{%')
    lines.append(r'    Experiment~6 Setting~1: semi-synthetic sparse-text benchmark')
    lines.append(r'    built from real sparse design matrices, reported as')
    lines.append(r'    mean\,\textpm\,std over 5 train/val/test splits.')
    lines.append(r'    Active features is the percentage of non-zero coefficients')
    lines.append(r'    in the final inner solution; lower is better.')
    lines.append(r'    Gray shading marks the best value within each dataset block.')
    # lines.append(r'    Requires \texttt{\textbackslash usepackage\{multirow\}} and')
    # lines.append(r'    \texttt{\textbackslash usepackage[table]\{xcolor\}}.')
    lines.append(r'  }')
    lines.append(r'  \label{tab:expe6_s1}')
    lines.append(r'  \begin{tabular}{llrrrrr}')
    lines.append(r'    \toprule')
    lines.append(r'    & & Recovery & Sparsity & Accuracy & Accuracy & Efficiency \\')
    lines.append(r'    \cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7}')
    lines.append(
        r'    Dataset & Method & '
        + ' & '.join(label for _, label, _, _ in METRICS)
        + r' \\'
    )
    lines.append(r'    \midrule')

    for dataset_idx, dataset in enumerate(datasets):
        for method_idx, method in enumerate(METHODS):
            cells = []
            if method_idx == 0:
                cells.append(
                    rf'\multirow{{{len(METHODS)}}}{{*}}{{{DATASET_LABELS.get(dataset, dataset)}}}'
                )
            else:
                cells.append('')
            cells.append(METHOD_LABELS[method])

            for col, _, _, spec in METRICS:
                mu, sd = stats[(dataset, method, col)]
                cell = _fmt(mu, sd, spec)
                best_v = best.get((dataset, col), np.nan)
                if (not np.isnan(mu) and not np.isnan(best_v)
                        and np.isclose(mu, best_v, rtol=1e-6)):
                    cells.append(_shade(cell))
                else:
                    cells.append(f'${cell}$')

            lines.append('    ' + ' & '.join(cells) + r' \\')

        if dataset_idx != len(datasets) - 1:
            lines.append(r'    \midrule')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f'Results not found at {RESULTS_PATH}. Run run_s1.py first.')

    df = pd.read_pickle(RESULTS_PATH)
    required_cols = {'dataset', 'active_features_pct', 'elapsed'}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            'Setting 1 results are from the old schema and cannot be rendered '
            f'by the updated table. Missing columns: {missing}. '
            'Rerun expes_fb/expe6_real_world/run_s1.py first.'
        )

    print(f'Loaded {len(df)} rows from {RESULTS_PATH}')
    print(f'Datasets: {sorted(df.dataset.unique())}')
    print(f'Methods : {sorted(df.method.unique())}')
    print(f'Seeds   : {sorted(df.seed.unique())}')
    print()

    table = build_table(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table + '\n')
    print(f'Saved {TABLE_PATH}')
    print()
    print(table)


if __name__ == '__main__':
    main()
