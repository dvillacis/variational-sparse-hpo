"""Generate LaTeX table for Experiment 6 Setting 2 — Real-world benchmarks.

Produces a compact long-form table (mean ± std over seeds) indexed by
dataset and method, rather than a wide dataset-by-columns layout.

Table structure
---------------
  Rows   : grouped by dataset using ``\\multirow``, then method
  Columns: Dataset | Method | F1 ↑ | Active Features (%) ↓ | t/iter (s) ↓

Usage
-----
    python table_s2.py            # from within this directory, or
    python expes_fb/expe6_real_world/table_s2.py   # from repo root
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR  = Path(__file__).parent / 'results' / 'setting2'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'
TABLE_PATH   = RESULTS_DIR / 'table_s2_results.tex'

METHODS = ['scalar_cv', 'sparseho_wl1', 'ntrba_wl1']
METHOD_LABELS = {
    'scalar_cv':     r'Scalar $\ell_1$ (CV)',
    'sparseho_wl1':  r'\texttt{Sparse-HO} ($w\ell_1$)',
    'ntrba_wl1':     r'\texttt{NTRBA}-$w\ell_1$ (ours)',
}

# (column key, short header, lower_is_better, printf format)
METRICS = [
    ('test_f1',    r'F1 $\uparrow$',                    False, '.3f'),
    ('sparsity',   r'Active features (\%) $\downarrow$', True,  '.3f'),
    ('t_per_iter', r't/iter (s) $\downarrow$',          True,  '.2f'),
]

# Pretty dataset names for the header
DATASET_DISPLAY = {
    'mnist':         r'\textsc{mnist} (0/1)',
    'breast-cancer': r'\textsc{breast-cancer}',
    'leukemia':      r'\textsc{leukemia}',
    'rcv1':          r'\textsc{rcv1}',
    'rcv1.binary':   r'\textsc{rcv1.binary}',
    'rcv1_train.binary': r'\textsc{rcv1.binary}',
    'real-sim':      r'\textsc{real-sim}',
    'news20.binary': r'\textsc{news20}',
}


def _fmt(mu, sd, spec):
    if np.isnan(mu):
        return r'\text{---}'
    return rf'{mu:{spec}}\,\pm\,{sd:{spec}}'


def _shade(cell):
    return rf'\cellcolor{{gray!15}} ${cell}$'


def build_table(df: pd.DataFrame) -> str:
    datasets = sorted(df.dataset.unique())
    methods = [m for m in METHODS if m in set(df.method.unique())]

    # pre-compute (mean, std) per (method, dataset, metric)
    stats = {}
    for method in methods:
        for dname in datasets:
            for col, _, _, _ in METRICS:
                vals = df[(df.method == method) & (df.dataset == dname)][col].values.astype(float)
                stats[(method, dname, col)] = (
                    np.nanmean(vals) if len(vals) else np.nan,
                    np.nanstd(vals)  if len(vals) else np.nan,
                )

    # best method per (dataset, metric)
    best = {}
    for dname in datasets:
        for col, _, lower_is_better, _ in METRICS:
            mus   = [stats[(m, dname, col)][0] for m in methods]
            valid = [v for v in mus if not np.isnan(v)]
            if valid:
                best[(dname, col)] = min(valid) if lower_is_better else max(valid)

    col_spec = r'llrrr'

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \setlength{\tabcolsep}{4pt}')
    lines.append(r'  \caption{%')
    lines.append(r'    Experiment~6 Setting~2: real-world classification benchmarks')
    lines.append(r'    (mean\,\textpm\,std over random splits).')
    lines.append(r'    Active features is the fraction of features with non-zero weight at convergence.')
    lines.append(r'    t/iter is wall-clock time per outer iteration.')
    lines.append(r'    Gray shading marks the best value within each dataset block.')
    lines.append(r'    Requires \texttt{\textbackslash usepackage\{multirow\}} and')
    lines.append(r'    \texttt{\textbackslash usepackage[table]\{xcolor\}}.')
    lines.append(r'  }')
    lines.append(r'  \label{tab:expe6_s2}')
    lines.append(rf'  \begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Dataset & Method & F1 $\uparrow$ & Active features (\%) $\downarrow$ & t/iter (s) $\downarrow$ \\'
    )
    lines.append(r'    \midrule')

    # data rows grouped by dataset
    for ds_idx, dname in enumerate(datasets):
        rows_for_dataset = []
        for method in methods:
            metric_cells = []
            for col, _, _, spec in METRICS:
                mu, sd = stats[(method, dname, col)]
                cell = _fmt(mu, sd, spec)
                best_v = best.get((dname, col), np.nan)
                if (not np.isnan(mu) and not np.isnan(best_v)
                        and np.isclose(mu, best_v, rtol=1e-6)):
                    metric_cells.append(_shade(cell))
                else:
                    metric_cells.append(f'${cell}$')
            rows_for_dataset.append((method, metric_cells))

        n_rows = len(rows_for_dataset)
        ds_label = DATASET_DISPLAY.get(dname, dname)
        for row_idx, (method, metric_cells) in enumerate(rows_for_dataset):
            if row_idx == 0:
                prefix = rf'\multirow{{{n_rows}}}{{*}}{{{ds_label}}} & '
            else:
                prefix = '    & '
            line = prefix + METHOD_LABELS[method] + ' & ' + ' & '.join(metric_cells) + r' \\'
            lines.append('    ' + line if row_idx == 0 else line)
        if ds_idx != len(datasets) - 1:
            lines.append(r'    \midrule')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f'Results not found at {RESULTS_PATH}. Run run_s2.py first.')

    df = pd.read_pickle(RESULTS_PATH)
    print(f'Loaded {len(df)} rows from {RESULTS_PATH}')
    print(f'Methods  : {sorted(df.method.unique())}')
    print(f'Datasets : {sorted(df.dataset.unique())}')
    print(f'Seeds    : {sorted(df.seed.unique())}')
    print()

    table = build_table(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table + '\n')
    print(f'Saved {TABLE_PATH}')
    print()
    print(table)


if __name__ == '__main__':
    main()
