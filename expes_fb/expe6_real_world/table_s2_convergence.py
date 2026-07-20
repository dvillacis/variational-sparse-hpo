"""Setting-2 table under the outer CONVERGENCE stop, capped at 60 (Option A).

Both gradient methods run with the same plateau early-stop (relative best-objective
improvement < 1e-4 over 5 iters) capped at 60 outer iterations, and are timed in one
clean run (results/setting2_cap60). A ``Stop`` column reports the termination
criterion so the reader sees which method actually converged: NTRBA reaches its
objective plateau early, while the fixed-step subgradient runs to the cap without
converging. scalar_cv is the fixed-budget reference from the paper run
(results/setting2).

Usage
-----
    python table_s2_convergence.py
    # Setting 2b (naturally biactive datasets) appended as a second block:
    python table_s2_convergence.py \
        --src-tags setting2_cap60 setting2_biactive \
        --scalar-tags setting2 setting2_biactive_scalar
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import table_common_clock as T   # reuse helpers, S2_* config

RES = T.RES
DEFAULT_SRC_TAGS = ['setting2_cap60']
DEFAULT_SCALAR_TAGS = ['setting2']

# termination_reason_ -> short table label
TERM_LABEL = {
    'obj_plateau': 'plat.',      # converged: objective plateau
    'step_tol': 'step',          # converged: trust-region step below tol
    'stationary': 'stat.',       # converged: gradient below tol
    'radius_min': 'radius',
    't_max': 'time',
    'completed': 'cap',          # hit the outer-iteration budget (did not converge)
    None: r'---',
}

# science + efficiency columns: (key, header, lower_is_better, spec, kind)
#   kind: 'msd' mean+/-std, 'mean' point value, 'term' termination label
COLS = [
    ('test_f1', r'F1 $\uparrow$', False, '.3f', 'msd'),
    ('sparsity', r'Active feat. (\%) $\downarrow$', True, '.3f', 'msd'),
    ('n_iter', r'It.', True, '.0f', 'mean'),
    ('termination', r'Stop', None, None, 'term'),
    ('t_per_iter', r't/it $\downarrow$', True, '.2f', 'mean'),
    ('elapsed', r'Total $\downarrow$', True, '.1f', 'mean'),
]
SHADE_KEYS = {'test_f1', 'sparsity', 't_per_iter', 'elapsed'}
NUM_KEYS = {'test_f1', 'sparsity', 'n_iter', 't_per_iter', 'elapsed'}


def _merged(src_tags, scalar_tags):
    frames = []
    for tag in src_tags:
        pkl = RES / tag / 'results.pkl'
        if pkl.exists():
            frames.append(pd.read_pickle(pkl))
        else:
            print(f'WARNING: {pkl} missing, skipped')
    if not frames:
        raise FileNotFoundError(f'no results.pkl under {RES} for tags {src_tags}')
    conv = pd.concat(frames, ignore_index=True)
    grad = conv[conv.method.isin(['sparseho_wl1', 'ntrba_wl1'])].copy()
    # The scalar_cv reference is the band-independent fixed-budget paper run; it is
    # optional here (expensive to reproduce, absent on a fresh checkout). Include it
    # only if present, otherwise the table is Sparse-HO vs NTRBA.
    scalars = []
    for tag in scalar_tags:
        pkl = RES / tag / 'results.pkl'
        if pkl.exists():
            s = pd.read_pickle(pkl)
            scalars.append(s[s.method == 'scalar_cv'])
    if scalars:
        return pd.concat(scalars + [grad], ignore_index=True)
    return grad


def _infer_cap(df):
    """The outer-iteration cap = the largest n_iter among runs that hit the budget
    (termination 'completed'), so the caption matches the data actually produced."""
    if 'termination' in df:
        comp = df[df.termination == 'completed']
        if len(comp):
            return int(round(comp.n_iter.max()))
    return None


def _term_label(block):
    vals = [v for v in block.termination.tolist() if v is not None] \
        if 'termination' in block else []
    if not vals:
        return r'---'
    mode, _ = Counter(vals).most_common(1)[0]
    return TERM_LABEL.get(mode, str(mode))


def build(df):
    present_ds = set(df.dataset.unique())
    std = [d for d in T.S2_DATASETS if d in present_ds]
    bio = [d for d in T.S2B_DATASETS if d in present_ds]
    datasets = std + bio
    present = set(df.method.unique())
    methods = [m for m in T.S2_METHODS if m in present]
    cap = _infer_cap(df)
    cap_tex = str(cap) if cap else 'the'
    scalar_note = (
        r'The scalar baseline (cross-validation search) is the fixed-budget '
        r'reference from the paper run; only its total time is comparable. '
        if 'scalar_cv' in methods else '')
    # TODO(setting2b): once the biactive results exist, consider a B_cv column
    # sourced from the exp6-diag scan so the table shows the diagnostic
    # prediction next to the outcome (SETTING2_BIACTIVE_EXTENSION proposal).
    bio_note = (
        r'The lower block contains the naturally biactive datasets of the '
        r'diagnostic (Table~\ref{tab:biactivity_diagnostic}), run under the '
        r'same protocol with stratified splits. '
        if bio and std else '')

    stats, terms = {}, {}
    for d in datasets:
        for m in methods:
            b = df[(df.dataset == d) & (df.method == m)]
            for k, _, _, _, kind in COLS:
                if kind == 'term':
                    terms[(d, m)] = _term_label(b) if m != 'scalar_cv' else r'---'
                else:
                    stats[(d, m, k)] = T._ms(b, k)

    best = {}
    for k, _, lo, _, kind in COLS:
        if k in SHADE_KEYS:
            shade_methods = ['sparseho_wl1', 'ntrba_wl1'] if k in ('t_per_iter', 'elapsed') \
                else methods
            best[k] = T._best(stats, datasets, shade_methods, k, lo)

    colspec = 'll' + ''.join('l' if kind == 'term' else 'r'
                             for _, _, _, _, kind in COLS)

    L = [r'\begin{table}[t]', r'  \centering',
         r'  \setlength{\tabcolsep}{3pt}', r'  \small', r'  \caption{%',
         r'    Experiment~5 Setting~2, run to convergence: real-world classification benchmarks',
         r'    (mean\,\textpm\,std over random splits). Both gradient methods use the same outer',
         r'    convergence stop---terminate when the best validation objective improves by less than',
         rf'    $10^{{-4}}$ (relative) over $5$ consecutive outer iterations---capped at ${cap_tex}$ outer',
         r'    iterations, and are timed in one clean run. \emph{It.}\ is outer iterations run,',
         r'    \emph{Stop} the termination criterion (\texttt{plat.}: objective plateau; \texttt{step}:',
         rf'    trust-region step below tolerance; \texttt{{cap}}: reached the ${cap_tex}$-iteration budget',
         r'    without converging), and \emph{t/it}/\emph{Total} are wall-clock seconds per iteration',
         r'    and in total. \texttt{NTRBA} reaches its plateau in $\sim\!20$ iterations; the',
         r'    fixed-step subgradient \texttt{Sparse-HO} runs to the cap without converging, yet',
         rf'    reaches comparable test F1. {scalar_note}{bio_note}Gray',
         r'    shading marks the best value within each dataset block.}',
         r'  \label{tab:expe5_setting2}',
         rf'  \begin{{tabular}}{{{colspec}}}', r'    \toprule',
         r'    & & \multicolumn{2}{c}{Quality} & \multicolumn{4}{c}{Efficiency (time in s)} \\',
         r'    \cmidrule(lr){3-4} \cmidrule(lr){5-8}',
         r'    Dataset & Method & ' + ' & '.join(h for _, h, _, _, _ in COLS) + r' \\',
         r'    \midrule']

    n_cols = 2 + len(COLS)
    for di, d in enumerate(datasets):
        if std and bio and d == bio[0]:
            L.append(rf'    \multicolumn{{{n_cols}}}{{l}}{{\emph{{Naturally biactive '
                     rf'datasets (cf.\ Table~\ref{{tab:biactivity_diagnostic}})}}}} \\')
            L.append(r'    \midrule')
        for mi, m in enumerate(methods):
            row = [rf'\multirow{{{len(methods)}}}{{*}}{{{T.S2_DLABEL[d]}}}' if mi == 0 else '',
                   T.S2_MLABEL[m]]
            for k, _, _, sp, kind in COLS:
                if kind == 'term':
                    lab = terms[(d, m)]
                    row.append(lab if lab == r'---' else rf'\texttt{{{lab}}}')
                    continue
                if kind == 'mean' and m == 'scalar_cv' and k in ('n_iter', 't_per_iter'):
                    row.append(r'$\text{---}$')          # grid search: no outer loop
                    continue
                mu, sd = stats[(d, m, k)]
                sh = False
                if k in best:
                    bv = best[k].get(d, np.nan)
                    sh = (not np.isnan(mu) and not np.isnan(bv)
                          and np.isclose(mu, bv, rtol=1e-6))
                cell = T._fmt(mu, sd, sp) if kind == 'msd' else T._fmt_mean(mu, sp)
                row.append(T._cell(cell, sh))
            L.append('    ' + ' & '.join(row) + r' \\')
        L.append(r'    \midrule' if di != len(datasets) - 1 else r'    \bottomrule')
    L += [r'  \end{tabular}', r'\end{table}']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src-tags', nargs='+', default=DEFAULT_SRC_TAGS,
                    help='results/<tag>/results.pkl inputs for the gradient '
                         'methods; add setting2_biactive for the 2b block.')
    ap.add_argument('--scalar-tags', nargs='*', default=DEFAULT_SCALAR_TAGS,
                    help='results/<tag>/results.pkl inputs for the scalar_cv '
                         'reference rows (optional, skipped if absent).')
    ap.add_argument('--out', default='table_s2_cap60.tex',
                    help='Output filename, written under results/<first src tag>/.')
    args = ap.parse_args()
    tex = build(_merged(args.src_tags, args.scalar_tags))
    # 8 columns (adds Stop to the common-clock block) exceed the 372pt single
    # column; go landscape, matching Table tab:expe5_setting1. sn-jnl redefines
    # sidewaystable (auto center+threeparttable); rotating is in the preamble.
    tex = tex.replace(r'\begin{table}[t]', r'\begin{sidewaystable}', 1)
    tex = tex.replace(r'\end{table}', r'\end{sidewaystable}', 1)
    out = RES / args.src_tags[0] / args.out
    out.write_text(tex + '\n')
    print(tex)
    print('\nsaved ->', out)


if __name__ == '__main__':
    main()
