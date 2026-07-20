"""Experiment 7 — LaTeX table: certificate soundness at the path anchor.

Reads results/results.pkl (from run.py) and emits
results/table_certificate_soundness.tex, a booktabs table with one row per
dataset at alpha_max (the headline false-certificate witness). The band- and
FD-step-invariance from the sensitivity sweeps is printed to the log via
sweep_summary() (the manuscript states it in prose, not in the caption).

Usage
-----
    uv run python expes_fb/expe7_certificate_soundness/table.py
    uv run python expes_fb/expe7_certificate_soundness/table.py --copy-to /path/to/paper
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / 'results'

DISPLAY = {
    'leukemia': r'\textsc{leukemia}', 'colon-cancer': r'\textsc{colon}',
    'duke breast-cancer': r'\textsc{duke}', 'madelon': r'\textsc{madelon}',
    'a9a': r'\textsc{a9a}', 'splice': r'\textsc{splice}',
    'rcv1.binary': r'\textsc{rcv1}', 'real-sim': r'\textsc{real-sim}',
    'news20.binary': r'\textsc{news20}',
}
ORDER = ['leukemia', 'colon-cancer', 'duke breast-cancer',
         'madelon', 'a9a', 'splice',
         'rcv1.binary', 'real-sim', 'news20.binary']


def _fmt(x, spec='.3f'):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return r'\text{---}'
    return f'{x:{spec}}'


def build(df):
    head = df[df.sweep == 'headline'].copy()
    amax = head[head.location == 'alpha_max'].set_index('dataset')
    n_shown = sum(1 for name in ORDER if name in amax.index)
    _WORDS = {6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
              11: 'eleven', 12: 'twelve'}
    n_word = _WORDS.get(n_shown, str(n_shown))

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \setlength{\tabcolsep}{10pt}')
    lines.append(r'  \caption{\textbf{Soundness of the stationarity certificate '
                 r'at the path anchor $\alpha_{\max}$.} For ' + n_word +
                 r' benchmark datasets: feature dimension $m$; norm of the '
                 r'support-restricted (null) hypergradient '
                 r'$\lVert g_{\mathrm{null}}\rVert$; the solver-measured '
                 r"value-function slope $\Phi'(\bar x;-e_{i^*})$ along the biactive "
                 r"coordinate's penalty-decrease direction (one-sided FD, "
                 r'$h=10^{-2}$); its sign-consistent estimate '
                 r'$-h^{\mathrm{sc}}_{i^*}$; whether the null zero-test yields a '
                 r'false certificate.}')
    lines.append(r'  \label{tab:certificate_soundness}')
    lines.append(r'  \begin{tabular}{lrcccc}')
    lines.append(r'  \toprule')
    lines.append(r'  & & \multicolumn{4}{c}{At $\alpha_{\max}$ (path anchor)} \\')
    lines.append(r'  \cmidrule(lr){3-6}')
    lines.append(r'  Dataset & $m$ & $\lVert g_{\mathrm{null}}\rVert$ '
                 r"& $\Phi'(\bar x;-e_{i^*})$ & $-h^{\mathrm{sc}}_{i^*}$ "
                 r'& false cert.? \\')
    lines.append(r'  \midrule')

    for name in ORDER:
        if name not in amax.index:
            continue
        a = amax.loc[name]
        disp = DISPLAY.get(name, name)
        m = int(a['m'])
        gnull = _fmt(float(a['g_null_norm']), '.2f')
        gt = _fmt(float(a['istar_gt']), '+.3f')
        sc = _fmt(float(a['istar_sc_pred']), '+.3f')
        fc = r'\textbf{yes}' if bool(a['false_certificate']) else 'no'
        lines.append(f'  {disp} & {m} & ${gnull}$ & ${gt}$ & ${sc}$ & {fc} \\\\')

    lines.append(r'  \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def sweep_summary(df):
    """Return a short text summary of band / FD-step invariance for the log."""
    amax = df[df.location == 'alpha_max']
    out = []
    for name in ORDER:
        d = amax[amax.dataset == name]
        if d.empty:
            continue
        bands = d[d.sweep.str.startswith('band_')]
        fc_all = bands.false_certificate.all() if len(bands) else np.nan
        fds = d[d.sweep.str.startswith('fd_')].sort_values('fd_step')
        conv = [f"{r.fd_step:.0e}:{r.istar_gt:+.3f}" for _, r in fds.iterrows()]
        out.append(f"  {name}: band-invariant false-cert={fc_all}; "
                   f"FD {' '.join(conv)}")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default=str(RESULTS_DIR / 'results.pkl'))
    ap.add_argument('--copy-to', default=None,
                    help='also write the .tex into this directory')
    args = ap.parse_args()

    df = pd.read_pickle(args.pkl)
    tex = build(df)
    out = RESULTS_DIR / 'table_certificate_soundness.tex'
    out.write_text(tex + '\n')
    print(tex)
    print('\n--- sensitivity sweeps (alpha_max) ---')
    print(sweep_summary(df))
    print(f'\nsaved -> {out}')
    if args.copy_to:
        dest = Path(args.copy_to) / 'table_certificate_soundness.tex'
        dest.write_text(tex + '\n')
        print(f'copied -> {dest}')


if __name__ == '__main__':
    main()
