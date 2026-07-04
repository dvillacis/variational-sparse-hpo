"""LaTeX tables for Experiment 6 Setting 3 — natural-degeneracy counterfactual.

Produces two booktabs tables (mean +/- std over seeds), matching the style of
``table_s2.py``:

  tab:expe6_counterfactual  per-dataset warm-start counterfactual
  tab:expe6_invariance      biactive count vs the detection band eps_B (circularity)

Usage
-----
    python table_s3.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # noqa: BLE001
    wilcoxon = None

RESULTS_DIR = Path(__file__).parent / 'results' / 'setting3'
CF_PATH = RESULTS_DIR / 'results_counterfactual.pkl'
INV_PATH = RESULTS_DIR / 'results_invariance.pkl'
BR_PATH = RESULTS_DIR / 'results_branch.pkl'

DATASET_DISPLAY = {
    'mnist': r'\textsc{mnist} (0/1)', 'rcv1.binary': r'\textsc{rcv1}',
    'rcv1_train.binary': r'\textsc{rcv1}', 'real-sim': r'\textsc{real-sim}',
    'news20.binary': r'\textsc{news20}', 'phishing': r'\textsc{phishing}',
}


def _fmt(mu, sd, spec='.3f'):
    if mu is None or (isinstance(mu, float) and np.isnan(mu)):
        return r'\text{---}'
    if sd is None or (isinstance(sd, float) and np.isnan(sd)):
        return rf'{mu:{spec}}'
    return rf'{mu:{spec}}\,\pm\,{sd:{spec}}'


def _wilcoxon_greater(diffs):
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    if wilcoxon is None or d.size < 5 or np.allclose(d, 0.0):
        return np.nan
    try:
        return float(wilcoxon(d, alternative='greater').pvalue)
    except ValueError:
        return np.nan


def build_counterfactual_table(cf: pd.DataFrame, br: pd.DataFrame) -> str:
    datasets = sorted(cf.dataset.unique())
    lines = [
        r'\begin{table}[t]', r'  \centering', r'  \setlength{\tabcolsep}{4pt}',
        r'  \caption{%',
        r'    Experiment~6 Setting~3: natural-degeneracy warm-start counterfactual on the',
        r'    fully real datasets (mean\,\textpm\,std over seeds, honest band',
        r'    $\varepsilon_{\mathcal B}=10^{-3}$, standard uniform initialization).',
        r'    $B$: biactive coordinates at the \texttt{Sparse-HO} fixed point;',
        r'    $m_{\mathrm{sel}}$: those passing the sign-consistency descent test;',
        r'    $\Delta\Phi$: held-out objective decrease of a same-optimizer continuation',
        r'    under the null vs sign-consistent oracle; $\Delta$F1: test-F1 change (SC$-$null);',
        r'    branch-ok: fraction of probed biactive coords satisfying Prop.~15 branch stability.',
        r'  }',
        r'  \label{tab:expe6_counterfactual}',
        r'  \begin{tabular}{lrrrrrr}', r'    \toprule',
        r'    Dataset & $B$ & $m_{\mathrm{sel}}$ & $\Delta\Phi$ (null) & '
        r'$\Delta\Phi$ (SC) & $\Delta$F1 & branch-ok \\',
        r'    \midrule',
    ]
    for d in datasets:
        s = cf[cf.dataset == d]
        B = _fmt(s.biactive.mean(), s.biactive.std(), '.1f')
        msel = _fmt(s.selected_biactive.mean(), s.selected_biactive.std(), '.1f')
        dphi_n = _fmt(s.dPhi_null.mean(), s.dPhi_null.std(), '.2e' if s.dPhi_null.abs().max() < 1e-2 else '.4f')
        dphi_s = _fmt(s.dPhi_sc.mean(), s.dPhi_sc.std(), '.2e' if s.dPhi_sc.abs().max() < 1e-2 else '.4f')
        df1 = _fmt(s.d_test_f1.mean(), s.d_test_f1.std(), '.4f')
        if br is not None and not br.empty and (br.dataset == d).any():
            bok = f"{br[br.dataset == d].branch_ok.mean():.2f}"
        else:
            bok = r'\text{---}'
        lines.append(
            rf'    {DATASET_DISPLAY.get(d, d)} & ${B}$ & ${msel}$ & ${dphi_n}$ & '
            rf'${dphi_s}$ & ${df1}$ & {bok} \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def build_invariance_table(inv: pd.DataFrame) -> str:
    """Biactive count as a function of the band eps_B (the circularity check)."""
    eps_vals = sorted(inv.eps_B.unique())
    # average over the tightest tau (invariance already established) and seeds/datasets
    tau_min = inv.tau.min()
    sub = inv[np.isclose(inv.tau, tau_min)]
    col_spec = 'l' + 'r' * len(eps_vals)
    header = ' & '.join([r'Quantity'] + [rf'$\varepsilon_{{\mathcal B}}={e:g}$' for e in eps_vals])
    lines = [
        r'\begin{table}[t]', r'  \centering', r'  \setlength{\tabcolsep}{5pt}',
        r'  \caption{%',
        rf'    Experiment~6 Setting~3: biactive incidence vs the detection band at the tightest'
        rf' inner tolerance $\tau={tau_min:g}$ (mean over datasets and seeds). The count is'
        r'    invariant to $\tau$ (not a solver artifact) but grows with $\varepsilon_{\mathcal B}$;'
        r'    at an honest band it is negligible, confirming the effect is band-controlled.',
        r'  }',
        r'  \label{tab:expe6_invariance}',
        rf'  \begin{{tabular}}{{{col_spec}}}', r'    \toprule',
        '    ' + header + r' \\', r'    \midrule',
    ]
    for key, label in [('biactive', r'Biactive $B$'),
                       ('selected_biactive', r'SC-selected $m_{\mathrm{sel}}$')]:
        cells = [label]
        for e in eps_vals:
            v = sub[np.isclose(sub.eps_B, e)][key]
            cells.append(rf'${v.mean():.1f}$')
        lines.append('    ' + ' & '.join(cells) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    if not CF_PATH.exists():
        raise FileNotFoundError(f'{CF_PATH} missing. Run run_s3_counterfactual.py first.')
    cf = pd.read_pickle(CF_PATH)
    inv = pd.read_pickle(INV_PATH) if INV_PATH.exists() else pd.DataFrame()
    br = pd.read_pickle(BR_PATH) if BR_PATH.exists() else pd.DataFrame()

    print(f'Loaded {len(cf)} counterfactual rows | datasets={sorted(cf.dataset.unique())} '
          f'| seeds={sorted(cf.seed.unique())}')
    # per-dataset Wilcoxon on paired (dPhi_sc - dPhi_null)
    for d in sorted(cf.dataset.unique()):
        s = cf[cf.dataset == d]
        p = _wilcoxon_greater((s.dPhi_sc - s.dPhi_null).values)
        print(f'  {d:16s} B={s.biactive.mean():.1f} m_sel={s.selected_biactive.mean():.1f} '
              f'dPhi_sc-null={ (s.dPhi_sc - s.dPhi_null).mean():+.3e} '
              f'dF1={s.d_test_f1.mean():+.4f} wilcoxon_p={p}')

    t1 = build_counterfactual_table(cf, br)
    (RESULTS_DIR / 'table_s3_counterfactual.tex').write_text(t1 + '\n')
    print('\nSaved', RESULTS_DIR / 'table_s3_counterfactual.tex')
    if not inv.empty:
        t2 = build_invariance_table(inv)
        (RESULTS_DIR / 'table_s3_invariance.tex').write_text(t2 + '\n')
        print('Saved', RESULTS_DIR / 'table_s3_invariance.tex')
        print('\n' + t2)


if __name__ == '__main__':
    main()
