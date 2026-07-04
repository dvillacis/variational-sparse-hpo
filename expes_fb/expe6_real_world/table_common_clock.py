"""Rebuild Experiment-5 Setting-1 and Setting-2 tables on a COMMON CLOCK.

Both tables previously reported different time metrics (Setting 1: total
"Runtime (s)"; Setting 2: "t/iter (s)"), which made NTRBA look faster in S1 and
slower in S2. This generator reports the SAME efficiency block in both tables:

    Outer it. | t/iter (s) | Total (s)

so the reader can see that (i) per-iteration cost is a stable ~2.5-3.5x overhead
on sparse-text designs in BOTH settings, and (ii) the total-runtime difference is
an outer-iteration-count effect (NTRBA's trust region terminates early in the
calibrated S1 but uses the full budget on the hard high-dim S2 datasets).

Sources (identical to the paper run, no re-runs):
  S1: results/setting1_scalefree/results.pkl        (incl. NTRBA-null)
  S2: NTRBA  = setting2_scalefree_matfree (+ mnist from setting2_scalefree_clean)
      scalar_cv / sparseho_wl1 = setting2/results.pkl (band-independent paper rows)

Usage
-----
    python table_common_clock.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE / 'results'


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _ms(block, col, ddof=0):
    # ddof matches each table's original generator: Setting 1 used pandas
    # .std() (ddof=1); Setting 2 used np.nanstd (ddof=0).
    v = block[col].values.astype(float)
    v = v[~np.isnan(v)]
    if not len(v):
        return np.nan, np.nan
    sd = np.std(v, ddof=ddof) if len(v) > ddof else 0.0
    return np.mean(v), sd


def _fmt(mu, sd, spec):
    if np.isnan(mu):
        return r'\text{---}'
    return rf'{mu:{spec}}\,\pm\,{sd:{spec}}'


def _fmt_mean(mu, spec):
    # point estimate (no std) -- used for the diagnostic efficiency block to
    # keep the table within \textwidth.
    if np.isnan(mu):
        return r'\text{---}'
    return rf'{mu:{spec}}'


def _cell(text, shaded):
    return rf'\cellcolor{{gray!15}} ${text}$' if shaded else f'${text}$'


def _best(stats, datasets, methods, col, lower):
    out = {}
    for d in datasets:
        mus = [stats[(d, m, col)][0] for m in methods
               if not np.isnan(stats[(d, m, col)][0])]
        if mus:
            out[d] = min(mus) if lower else max(mus)
    return out


# --------------------------------------------------------------------------- #
# Setting 1
# --------------------------------------------------------------------------- #
S1_DATASETS = ['rcv1.binary', 'real-sim', 'w8a', 'news20.binary']
S1_DLABEL = {'rcv1.binary': r'\textsc{RCV1}', 'real-sim': r'\textsc{Real-sim}',
             'w8a': r'\textsc{W8A}', 'news20.binary': r'\textsc{News20}'}
S1_METHODS = ['sparseho_scalar', 'sparseho_wl1', 'null_tr_wl1', 'ntrba_wl1']
S1_MLABEL = {'sparseho_scalar': r'Scalar $\ell_1$ (grid)',
             'sparseho_wl1': r'\texttt{Sparse-HO} ($w\ell_1$)',
             'null_tr_wl1': r'\texttt{NTRBA}-null ($w\ell_1$)',
             'ntrba_wl1': r'\texttt{NTRBA}-$w\ell_1$ (ours)'}
# science columns (key, header, lower_is_better, spec)
S1_SCI = [('hidden_recall', r'Hidden recall $\uparrow$', False, '.3f'),
          ('active_features_pct', r'Act. feat. (\%) $\downarrow$', True, '.1f'),
          ('f1', r'F1 $\uparrow$', False, '.3f'),
          ('test_logloss', r'Log-loss $\downarrow$', True, '.3f')]

# efficiency columns (key, header, lower_is_better, spec, shade, scalar_na)
# Units live in the group header "Efficiency (time in s)"; keep sub-headers
# narrow so the point-value columns do not force extra width.
EFF = [('n_iter', r'It.', True, '.0f', False, True),
       ('t_per_iter', r't/it $\downarrow$', True, '.2f', True, True),
       ('elapsed', r'Total $\downarrow$', True, '.1f', True, False)]


def build_s1():
    df = pd.read_pickle(RES / 'setting1_scalefree' / 'results.pkl')
    datasets = [d for d in S1_DATASETS if d in set(df.dataset.unique())]
    cols = S1_SCI + [(k, h, lo, sp) for (k, h, lo, sp, _, _) in EFF]
    stats = {}
    for d in datasets:
        for m in S1_METHODS:
            b = df[(df.dataset == d) & (df.method == m)]
            for k, _, _, _ in cols:
                stats[(d, m, k)] = _ms(b, k, ddof=1)  # S1: pandas .std()

    best_sci = {(k): _best(stats, datasets, S1_METHODS, k, lo)
                for k, _, lo, _ in S1_SCI}
    # efficiency: shade over the iterative methods only (exclude scalar grid)
    iter_methods = ['sparseho_wl1', 'null_tr_wl1', 'ntrba_wl1']
    best_eff = {}
    for k, _, lo, _, shade, _ in EFF:
        if shade:
            best_eff[k] = _best(stats, datasets, iter_methods, k, lo)

    L = [r'\begin{table}[t]', r'  \centering',
         r'  \setlength{\tabcolsep}{3pt}', r'  \small', r'  \caption{%',
         r'    Experiment~5 Setting~1: semi-synthetic sparse-text benchmark built from real sparse',
         r'    design matrices, reported as mean\,\textpm\,std over 5 train/val/test splits. Active',
         r'    features is the percentage of non-zero coefficients in the final inner solution; lower',
         r'    is better. \texttt{NTRBA}-null uses the support-restricted (null) selection with the',
         r'    trust-region optimizer, so the \texttt{NTRBA}-null\,$\to$\,\texttt{NTRBA} gap isolates',
         r'    the sign-consistent oracle from the optimizer. The efficiency block reports a common',
         r'    clock: outer iterations run (It.), wall-clock seconds per outer iteration (t/it), and',
         r'    total wall-clock seconds (Total); \texttt{Sparse-HO}, \texttt{NTRBA}-null and \texttt{NTRBA} share an outer budget',
         r'    of $60$, but the trust-region methods may terminate early on a stationarity/step test.',
         r'    The scalar baseline is a grid search (no outer loop), so only its total time is',
         r'    comparable. Gray shading marks the best value within each dataset block.}',
         r'  \label{tab:expe5_setting1}',
         r'  \begin{tabular}{llrrrrrrr}', r'    \toprule',
         r'    & & Recovery & Sparsity & Classification & Pred.\ loss & \multicolumn{3}{c}{Efficiency (time in s)} \\',
         r'    \cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5} \cmidrule(lr){6-6} \cmidrule(lr){7-9}',
         r'    Dataset & Method & ' + ' & '.join(h for _, h, _, _ in cols) + r' \\',
         r'    \midrule']

    for di, d in enumerate(datasets):
        for mi, m in enumerate(S1_METHODS):
            row = [rf'\multirow{{{len(S1_METHODS)}}}{{*}}{{{S1_DLABEL[d]}}}' if mi == 0 else '',
                   S1_MLABEL[m]]
            for k, _, _, sp in S1_SCI:
                mu, sd = stats[(d, m, k)]
                bv = best_sci[k].get(d, np.nan)
                sh = (not np.isnan(mu) and not np.isnan(bv) and np.isclose(mu, bv, rtol=1e-6))
                row.append(_cell(_fmt(mu, sd, sp), sh))
            for k, _, _, sp, shade, scalar_na in EFF:
                if scalar_na and m == 'sparseho_scalar':
                    row.append(r'$\text{---}$')
                    continue
                mu, sd = stats[(d, m, k)]
                sh = False
                if shade:
                    bv = best_eff[k].get(d, np.nan)
                    sh = (not np.isnan(mu) and not np.isnan(bv) and np.isclose(mu, bv, rtol=1e-6))
                row.append(_cell(_fmt_mean(mu, sp), sh))
            L.append('    ' + ' & '.join(row) + r' \\')
        L.append(r'    \midrule' if di != len(datasets) - 1 else r'    \bottomrule')
    L += [r'  \end{tabular}', r'\end{table}']
    return '\n'.join(L)


# --------------------------------------------------------------------------- #
# Setting 2
# --------------------------------------------------------------------------- #
S2_DATASETS = ['mnist', 'news20.binary', 'phishing', 'rcv1.binary', 'real-sim']
S2_DLABEL = {'mnist': r'\textsc{mnist} (0/1)', 'news20.binary': r'\textsc{news20}',
             'phishing': r'phishing', 'rcv1.binary': r'\textsc{rcv1.binary}',
             'real-sim': r'\textsc{real-sim}'}
S2_METHODS = ['scalar_cv', 'sparseho_wl1', 'ntrba_wl1']
S2_MLABEL = {'scalar_cv': r'Scalar $\ell_1$ (CV)',
             'sparseho_wl1': r'\texttt{Sparse-HO} ($w\ell_1$)',
             'ntrba_wl1': r'\texttt{NTRBA}-$w\ell_1$ (ours)'}
S2_SCI = [('test_f1', r'F1 $\uparrow$', False, '.3f'),
          ('sparsity', r'Active feat. (\%) $\downarrow$', True, '.3f')]


def _s2_merged():
    old = pd.read_pickle(RES / 'setting2' / 'results.pkl')
    mf = pd.read_pickle(RES / 'setting2_scalefree_matfree' / 'results.pkl')
    cl = pd.read_pickle(RES / 'setting2_scalefree_clean' / 'results.pkl')
    ntrba = pd.concat([mf[mf.dataset != 'mnist'], cl[cl.dataset == 'mnist']],
                      ignore_index=True)
    return pd.concat([old[old.method != 'ntrba_wl1'], ntrba], ignore_index=True)


def build_s2():
    df = _s2_merged()
    datasets = [d for d in S2_DATASETS if d in set(df.dataset.unique())]
    cols = S2_SCI + [(k, h, lo, sp) for (k, h, lo, sp, _, _) in EFF]
    stats = {}
    for d in datasets:
        for m in S2_METHODS:
            b = df[(df.dataset == d) & (df.method == m)]
            for k, _, _, _ in cols:
                stats[(d, m, k)] = _ms(b, k)

    best_sci = {k: _best(stats, datasets, S2_METHODS, k, lo) for k, _, lo, _ in S2_SCI}
    iter_methods = ['sparseho_wl1', 'ntrba_wl1']
    best_eff = {}
    for k, _, lo, _, shade, _ in EFF:
        if shade:
            best_eff[k] = _best(stats, datasets, iter_methods, k, lo)

    L = [r'\begin{table}[t]', r'  \centering',
         r'  \setlength{\tabcolsep}{3pt}', r'  \small', r'  \caption{%',
         r'    Experiment~5 Setting~2: real-world classification benchmarks',
         r'    (mean\,\textpm\,std over random splits). Active features is the fraction of features',
         r'    with non-zero weight at convergence. The efficiency block uses the same common clock',
         r'    as Table~\ref{tab:expe5_setting1}: outer iterations run (It.), wall-clock seconds per',
         r'    outer iteration (t/it), and total wall-clock seconds (Total). \texttt{Sparse-HO} and \texttt{NTRBA} share an outer',
         r'    budget of $60$; unlike the calibrated Setting~1, the \texttt{NTRBA} trust region reaches',
         r'    its stopping test early only on the dense/low-dimensional datasets and otherwise runs',
         r'    the full budget, so its per-iteration overhead is paid on every step. The scalar',
         r'    baseline is a cross-validation search (no outer loop); only its total time is',
         r'    comparable. Gray shading marks the best value within each dataset block.}',
         r'  \label{tab:expe5_setting2}',
         r'  \begin{tabular}{llrrrrr}', r'    \toprule',
         r'    & & \multicolumn{2}{c}{Quality} & \multicolumn{3}{c}{Efficiency (time in s)} \\',
         r'    \cmidrule(lr){3-4} \cmidrule(lr){5-7}',
         r'    Dataset & Method & ' + ' & '.join(h for _, h, _, _ in cols) + r' \\',
         r'    \midrule']

    for di, d in enumerate(datasets):
        for mi, m in enumerate(S2_METHODS):
            row = [rf'\multirow{{{len(S2_METHODS)}}}{{*}}{{{S2_DLABEL[d]}}}' if mi == 0 else '',
                   S2_MLABEL[m]]
            for k, _, _, sp in S2_SCI:
                mu, sd = stats[(d, m, k)]
                bv = best_sci[k].get(d, np.nan)
                sh = (not np.isnan(mu) and not np.isnan(bv) and np.isclose(mu, bv, rtol=1e-6))
                row.append(_cell(_fmt(mu, sd, sp), sh))
            for k, _, _, sp, shade, scalar_na in EFF:
                if scalar_na and m == 'scalar_cv':
                    row.append(r'$\text{---}$')
                    continue
                mu, sd = stats[(d, m, k)]
                sh = False
                if shade:
                    bv = best_eff[k].get(d, np.nan)
                    sh = (not np.isnan(mu) and not np.isnan(bv) and np.isclose(mu, bv, rtol=1e-6))
                row.append(_cell(_fmt_mean(mu, sp), sh))
            L.append('    ' + ' & '.join(row) + r' \\')
        L.append(r'    \midrule' if di != len(datasets) - 1 else r'    \bottomrule')
    L += [r'  \end{tabular}', r'\end{table}']
    return '\n'.join(L)


def main():
    t1 = build_s1()
    # S1 has 4 mean+/-std science columns plus the common-clock efficiency block;
    # at 9 columns it exceeds the 372pt single-column width, so it goes landscape.
    # sn-jnl redefines sidewaystable (auto center+threeparttable); rotating is in
    # the preamble. Landscape width = \textheight (~552pt) fits it at full \small.
    t1 = t1.replace(r'\begin{table}[t]', r'\begin{sidewaystable}', 1)
    t1 = t1.replace(r'\end{table}', r'\end{sidewaystable}', 1)
    (RES / 'setting1_scalefree' / 'table_s1_common_clock.tex').write_text(t1 + '\n')
    t2 = build_s2()
    (RES / 'setting2' / 'table_s2_common_clock.tex').write_text(t2 + '\n')
    print(t1)
    print('\n\n')
    print(t2)
    print('\n\nSaved:')
    print('  ', RES / 'setting1_scalefree' / 'table_s1_common_clock.tex')
    print('  ', RES / 'setting2' / 'table_s2_common_clock.tex')


if __name__ == '__main__':
    main()
