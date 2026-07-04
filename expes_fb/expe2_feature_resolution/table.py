"""Generate LaTeX table for Experiment 2.

Produces the manuscript table (tab:expe2_metrics; printed to stdout and saved
to results/) summarising support-recovery F1 and test MSE for both methods
across all configurations.

Usage
-----
    python table.py            # from within this directory, or
    python expes_fb/expe2_feature_resolution/table.py   # from repo root
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR  = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'
TABLE_PATH   = RESULTS_DIR / 'table_metrics.tex'


def build_table(df: pd.DataFrame) -> str:
    ms      = sorted(df.m.unique())
    nm_rats = sorted(df.nm_ratio.unique())
    spars   = sorted(df.sparsity_frac.unique())
    configs = list(product(ms, nm_rats, spars))

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{%')
    lines.append(r'    Support-recovery F1 and test MSE (mean\,\textpm\,std over 10 seeds)')
    lines.append(r'    for scalar elastic-net CV vs.\ weighted elastic-net (ours)')
    lines.append(r'    across all synthetic configurations.')
    lines.append(r'  }')
    lines.append(r'  \label{tab:expe2_metrics}')
    lines.append(r'  \setlength{\tabcolsep}{5pt}')
    lines.append(r'  \begin{tabular}{cccrrrr}')
    lines.append(r'    \toprule')
    lines.append(r'    & & &'
                 r' \multicolumn{2}{c}{F1 $\uparrow$} &'
                 r' \multicolumn{2}{c}{Test MSE $\downarrow$} \\')
    lines.append(r'    \cmidrule(lr){4-5} \cmidrule(lr){6-7}')
    lines.append(r'    $p$ & $n/p$ & sparsity'
                 r' & Scalar & Weighted (ours)'
                 r' & Scalar & Weighted (ours) \\')
    lines.append(r'    \midrule')

    # pre-compute row span sizes for \multirow
    n_spars  = len(spars)     # rows per (m, nm_ratio) block
    n_nm     = len(nm_rats)   # nm_ratio values per m block
    rows_per_m  = n_nm * n_spars
    rows_per_nm = n_spars

    def fmt(mu, sd, decimals=3):
        if np.isnan(mu):
            return r'\text{---}'
        return rf'{mu:.{decimals}f}\,\pm\,{sd:.{decimals}f}'

    def as_cell(cell):
        return f'${cell}$'

    prev_m      = None
    prev_nm     = None
    for m, nm_ratio, sparsity_frac in configs:
        sub = df[(df.m == m) & (df.nm_ratio == nm_ratio) &
                 (df.sparsity_frac == sparsity_frac)]

        def stats(method, col):
            vals = sub[sub.method == method][col].values.astype(float)
            if len(vals) == 0:
                return np.nan, np.nan
            return vals.mean(), vals.std()

        f1_sc_mu,  f1_sc_sd  = stats('scalar',   'f1')
        f1_wt_mu,  f1_wt_sd  = stats('weighted', 'f1')
        mse_sc_mu, mse_sc_sd = stats('scalar',   'test_mse')
        mse_wt_mu, mse_wt_sd = stats('weighted', 'test_mse')

        f1_sc_str  = fmt(f1_sc_mu,  f1_sc_sd)
        f1_wt_str  = fmt(f1_wt_mu,  f1_wt_sd)
        mse_sc_str = fmt(mse_sc_mu, mse_sc_sd)
        mse_wt_str = fmt(mse_wt_mu, mse_wt_sd)

        # midrule between m blocks
        if prev_m is not None and m != prev_m:
            lines.append(r'    \midrule')

        # \multirow cells: emit on first row of each block, blank otherwise
        nm_col = f'{nm_ratio:.2f}'.rstrip('0').rstrip('.')
        sp_col = f'{sparsity_frac * 100:.0f}\\%'

        if m != prev_m:
            m_cell  = rf'\multirow{{{rows_per_m}}}{{*}}{{{m}}}'
        else:
            m_cell  = ''

        if nm_ratio != prev_nm or m != prev_m:
            nm_cell = rf'\multirow{{{rows_per_nm}}}{{*}}{{{nm_col}}}'
        else:
            nm_cell = ''

        prev_m  = m
        prev_nm = nm_ratio

        lines.append(
            rf'    {m_cell} & {nm_cell} & {sp_col}'
            rf' & {as_cell(f1_sc_str)}'
            rf' & {as_cell(f1_wt_str)}'
            rf' & {as_cell(mse_sc_str)}'
            rf' & {as_cell(mse_wt_str)} \\'
        )

        # cmidrule between n/m blocks (cols 2-7, leaving col 1 open for multirow m)
        if sparsity_frac == spars[-1] and nm_ratio != nm_rats[-1]:
            lines.append(r'    \cmidrule(l){2-7}')

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

    table = build_table(df)

    RESULTS_DIR.mkdir(exist_ok=True)
    TABLE_PATH.write_text(table + '\n')
    print(f'Saved {TABLE_PATH}')
    print()
    print(table)


if __name__ == '__main__':
    main()
