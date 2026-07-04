"""Generate LaTeX table for Experiment 4 — SOTA comparison.

The table emphasizes both mechanism and downstream outcome:

  - hidden-gradient norm at k=0   : starvation diagnostic,
  - best validation loss          : optimization criterion,
  - F1                            : support quality,
  - test MSE                      : predictive performance.

Rows are grouped by ``(rho, (n, m))`` and averaged over seeds only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR  = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'
TABLE_PATH   = RESULTS_DIR / 'table_sota.tex'

METHODS = ['sparseho_scalar', 'sparseho_wl1', 'ntrba_wl1']
METHOD_LABELS = {
    'sparseho_scalar': r'Scalar $\ell_1$',
    'sparseho_wl1':    r'\texttt{Sparse-HO} ($w\ell_1$)',
    'ntrba_wl1':       r'\texttt{NTRBA}-$w\ell_1$ (ours)',
}

# (column key, header, lower_is_better, printf format)
METRICS = [
    ('hidden_grad_norm_0', r'$\|\partial_{x_{\mathrm{hid}}}\Phi\|_{k=0}$ $\uparrow$', False, '.2f'),
    ('best_val_loss',      r'Best val. loss $\downarrow$',      True,  '.3f'),
    ('f1',                 r'F1 $\uparrow$',                    False, '.3f'),
    ('test_mse',           r'Test MSE $\downarrow$',            True,  '.2f'),
]

N_LABEL_COLS = 2   # ρ | method
N_TOTAL_COLS = N_LABEL_COLS + len(METRICS)   # 6
ROWS_PER_RHO = len(METHODS)                  # 3


def _fmt(mu, sd, spec):
    if np.isnan(mu):
        return r'\text{---}'
    return rf'{mu:{spec}}\,\pm\,{sd:{spec}}'


def _shade(cell):
    return rf'\cellcolor{{gray!15}} ${cell}$'


def _present_rho_values(df: pd.DataFrame):
    return sorted(float(rho) for rho in df.rho.dropna().unique())


def _present_nm_pairs(df: pd.DataFrame):
    pairs = {
        (int(n), int(m))
        for n, m in zip(df.n.values, df.m.values)
        if pd.notna(n) and pd.notna(m)
    }
    return sorted(pairs)


def build_table(df: pd.DataFrame) -> str:
    rho_values = _present_rho_values(df)
    nm_pairs = _present_nm_pairs(df)

    # pre-compute (mean, std) for every (method, metric, rho, n, m) cell,
    # averaging over seeds only
    stats = {}
    for method in METHODS:
        for col, _, _, _ in METRICS:
            for rho in rho_values:
                for n, m in nm_pairs:
                    sub = df[
                        np.isclose(df.rho, rho)
                        & (df.n == n)
                        & (df.m == m)
                        & (df.method == method)
                    ]
                    vals = sub[col].values.astype(float)
                    stats[(method, col, rho, n, m)] = (
                        np.nanmean(vals) if len(vals) else np.nan,
                        np.nanstd(vals) if len(vals) else np.nan,
                    )

    # best method per (metric, rho, n, m) — for shading
    best = {}
    for col, _, lower_is_better, _ in METRICS:
        for rho in rho_values:
            for n, m in nm_pairs:
                mus = [stats[(method, col, rho, n, m)][0] for method in METHODS]
                valid = [v for v in mus if not np.isnan(v)]
                if valid:
                    best[(col, rho, n, m)] = (
                        min(valid) if lower_is_better else max(valid)
                    )

    col_spec = 'l' * 3 + 'r' * len(METRICS)

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \setlength{\tabcolsep}{4pt}')
    lines.append(r'  \caption{%')
    lines.append(r'    Comparison on degenerate synthetic instances')
    lines.append(r'    (mean\,\textpm\,std over 20 seeds for each $(\rho, (n,p))$ setting).')
    lines.append(r'    $\|\partial_{x_{\mathrm{hid}}}\Phi\|_{k=0}$ is the hidden-feature subgradient norm')
    lines.append(r'    at the first outer iteration (not defined for the scalar baseline).')
    lines.append(r'    Gray shading marks the best value in each $(\rho, (n,p), \text{metric})$ block.')
    lines.append(r'  }')
    lines.append(r'  \label{tab:expe4_sota}')
    lines.append(rf'  \begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \toprule')

    # single header row
    metric_headers = ' & '.join(label for _, label, _, _ in METRICS)
    lines.append(rf'    $\rho$ & $(n,p)$ & Method & {metric_headers} \\')
    lines.append(r'    \midrule')

    # data rows
    for rho_idx, rho in enumerate(rho_values):
        rho_pairs = [
            (n, m) for (n, m) in nm_pairs
            if np.isclose(df[(df.n == n) & (df.m == m)].rho, rho).any()
        ]
        if not rho_pairs:
            continue

        if rho_idx != 0:
            lines.append(r'    \midrule')

        rho_cell = rf'\multirow{{{len(rho_pairs) * len(METHODS)}}}{{*}}{{${rho}$}}'

        for pair_idx, (n, m) in enumerate(rho_pairs):
            nm_cell = rf'\multirow{{{len(METHODS)}}}{{*}}{{$(%d,%d)$}}' % (n, m)

            if pair_idx != 0:
                lines.append(r'    \cmidrule(lr){2-7}')

            for method_idx, method in enumerate(METHODS):
                rho_col = rho_cell if pair_idx == 0 and method_idx == 0 else ''
                nm_col = nm_cell if method_idx == 0 else ''
                method_col = METHOD_LABELS[method]

                data_cells = []
                for col, _, lower_is_better, spec in METRICS:
                    mu, sd = stats[(method, col, rho, n, m)]
                    cell = _fmt(mu, sd, spec)
                    best_val = best.get((col, rho, n, m), np.nan)
                    if (not np.isnan(mu) and not np.isnan(best_val)
                            and np.isclose(mu, best_val, rtol=1e-6)):
                        data_cells.append(_shade(cell))
                    else:
                        data_cells.append(f'${cell}$')

                lines.append(
                    '    ' + ' & '.join([rho_col, nm_col, method_col] + data_cells) + r' \\'
                )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f'Results not found at {RESULTS_PATH}. Run run.py first.')

    df = pd.read_pickle(RESULTS_PATH)
    print(f'Loaded {len(df)} rows from {RESULTS_PATH}')
    print(f'Methods : {sorted(df.method.unique())}')
    print(f'(n,m)   : {sorted(set(zip(df.n, df.m)))}')
    print(f'rho     : {sorted(df.rho.unique())}')
    print(f'Seeds   : {sorted(df.seed.unique())}')
    print()

    table = build_table(df)

    RESULTS_DIR.mkdir(exist_ok=True)
    TABLE_PATH.write_text(table + '\n')
    print(f'Saved {TABLE_PATH}')
    print()
    print(table)


if __name__ == '__main__':
    main()
