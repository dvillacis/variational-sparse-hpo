"""Assemble the biactivity-diagnostic tables (Panels A/B/C) into booktabs LaTeX,
matching the manuscript style (booktabs, mean values, gray shading unused here).

Reads:
  results/biactivity_scan/scan_full.pkl        (microarray incidence)
  results/biactivity_scan/scan_categorical.pkl (engineered/categorical incidence)
  results/spurious_stationarity/spurious_suite.pkl (alpha_max correctness)
  results/biactivity_scan/noharm_suite.pkl     (does-no-harm control)

Emits:
  results/biactivity_scan/table_biactivity_diagnostic.tex
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SCAN = HERE / 'results' / 'biactivity_scan'
SPUR = HERE / 'results' / 'spurious_stationarity'

MECH = {
    'leukemia': 'microarray', 'colon-cancer': 'microarray',
    'duke breast-cancer': 'microarray', 'duke-breast-cancer': 'microarray',
    'gisette': 'engineered', 'madelon': 'engineered',
    'a9a': 'categorical', 'dna': 'categorical', 'splice': 'categorical',
}
PRETTY = {
    'leukemia': r'\textsc{leukemia}', 'colon-cancer': r'\textsc{colon}',
    'duke breast-cancer': r'\textsc{duke}', 'duke-breast-cancer': r'\textsc{duke}',
    'madelon': r'\textsc{madelon}', 'gisette': r'\textsc{gisette}',
    'a9a': r'\textsc{a9a}', 'splice': r'\textsc{splice}', 'dna': r'\textsc{dna}',
    'real-sim': r'\textsc{real-sim}', 'news20.binary': r'\textsc{news20}',
    'rcv1.binary': r'\textsc{rcv1}', 'phishing': r'\textsc{phishing}',
}


def _incidence():
    scan = pd.concat([
        pd.read_pickle(SCAN / 'scan_full.pkl'),
        pd.read_pickle(SCAN / 'scan_categorical.pkl')], ignore_index=True)
    scan['is_amax'] = scan.location == 'alpha_max'
    scan['is_cv'] = scan.location.str.startswith('cv_opt')
    out = {}
    for d, g in scan.groupby('dataset'):
        amax = g[g.is_amax]
        cv = g[g.is_cv]
        out[d] = dict(
            m=int(g.m.iloc[0]),
            B_amax=int(amax.B_scalefree.max()) if len(amax) else 0,
            gap_amax=float(amax.near_kink_med.min()) if len(amax) else np.nan,
            B_cv=int(cv.B_scalefree.max()) if len(cv) else 0,
            coupling=int(g.coupling.max()))
    return out


def _correctness():
    sp = pd.read_pickle(SPUR / 'spurious_suite.pkl')
    out = {}
    for d, g in sp.groupby('dataset'):
        out[d] = dict(
            gnull=float(g.gnull_norm.mean()), gsc=float(g.gsc_norm.mean()),
            n_probe=int(len(g)), descent=int(g.descent_confirmed.sum()),
            med_fd=float(g.fd_slope.median()),
            med_pred=float(g.pred_slope_sc.median()))
    return out


def _key(name):
    return {'duke-breast-cancer': 'duke breast-cancer'}.get(name, name)


def build():
    inc = _incidence()
    cor = _correctness()

    # ---- Table A: incidence + alpha_max correctness (biactive datasets) ----
    order = ['leukemia', 'colon-cancer', 'duke breast-cancer',
             'madelon', 'a9a', 'splice']
    L = []
    L.append(r'\begin{tabular}{llrrrrrr}')
    L.append(r'\toprule')
    L.append(r'Dataset & Mechanism & $m$ & $B_{\alpha_{\max}}$ & '
             r'$\lVert g_{\mathrm{null}}\rVert$ & $\lVert g_{\mathrm{sc}}\rVert$ '
             r'& FD-desc. & $B_{\mathrm{cv}}$ ($c$) \\')
    L.append(r'\midrule')
    for d in order:
        i = inc.get(d, {})
        c = cor.get(d) or cor.get(d.replace(' ', '-'))
        if not i:
            continue
        cell_c = (f"{c['descent']}/{c['n_probe']}" if c else '--')
        gnull = (f"{c['gnull']:.3f}" if c else '--')
        gsc = (f"{c['gsc']:.3f}" if c else '--')
        cv = f"{i['B_cv']} ({i['coupling']})" if i['B_cv'] else f"-- ({i['coupling']})"
        L.append(
            rf"{PRETTY.get(d, d)} & {MECH.get(d, '')} & {i['m']} & "
            rf"{i['B_amax']} & ${gnull}$ & ${gsc}$ & {cell_c} & {cv} \\")
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    tableA = '\n'.join(L)

    # ---- Table B: does-no-harm control (sparse text) ----
    nh_path = SCAN / 'noharm_suite.pkl'
    tableB = None
    if nh_path.exists():
        nh = pd.read_pickle(nh_path)
        B = []
        B.append(r'\begin{tabular}{lrrr}')
        B.append(r'\toprule')
        B.append(r'Dataset & $m$ & $B_{\mathrm{cv}}$ & '
                 r'$\lVert g_{\mathrm{sc}}-g_{\mathrm{null}}\rVert$ \\')
        B.append(r'\midrule')
        for _, r in nh.iterrows():
            d = float(r['hgrad_discrepancy'])
            if d <= 0:
                disc = '0'
            else:
                exp = int(np.floor(np.log10(d)))
                disc = rf"{d/10**exp:.1f}\times10^{{{exp}}}"
            B.append(rf"{PRETTY.get(r['dataset'], r['dataset'])} & {int(r['m'])} & "
                     rf"{int(r['biactive'])} & ${disc}$ \\")
        B.append(r'\bottomrule')
        B.append(r'\end{tabular}')
        tableB = '\n'.join(B)

    out = SCAN / 'table_biactivity_diagnostic.tex'
    with open(out, 'w') as f:
        f.write("% Panel A+B: natural biactivity incidence & alpha_max correctness\n")
        f.write(tableA + "\n\n")
        if tableB:
            f.write("% Panel C: does-no-harm (B=0 on sparse text => SC == Sparse-HO)\n")
            f.write(tableB + "\n")
    print("=== Table A (incidence + correctness) ===")
    print(tableA)
    if tableB:
        print("\n=== Table B (does-no-harm) ===")
        print(tableB)
    print(f"\nsaved -> {out}")


if __name__ == '__main__':
    build()
