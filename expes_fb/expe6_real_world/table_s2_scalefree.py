"""Rebuild the Setting-2 (Table 5) results with the scale-free biactive band.

Only NTRBA-wl1 depends on the band; scalar_cv and sparseho_wl1 are band-independent,
so their rows are taken from the original paper run and the NTRBA rows are replaced by
the scale-free re-run (tags: scalefree = mnist, scalefree2 = sparse datasets).

Also emits a biactivity-properties table for the Setting-2 datasets (B at the operating
penalty, scale-free vs legacy) that explains why SC == Sparse-HO where B=0 and why the
mnist numbers change.

Usage
-----
    python table_s2_scalefree.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import table_s2  # reuse build_table + formatting

HERE = Path(__file__).parent
RES = HERE / 'results'
OLD = RES / 'setting2' / 'results.pkl'
# NTRBA rows are re-measured under the scale-free band in one clean, uncontended run
# (all 5 datasets, tag scalefree_clean). scalar_cv and sparseho_wl1 are band-independent
# and are kept from the paper run.
# Sparse datasets take the matrix-free-oracle run (which reduces their per-iteration
# cost); mnist's support is below the matrix-free threshold, so its computation is
# unchanged and its clean dedicated measurement is used.
MATFREE_TAG = 'setting2_scalefree_matfree'
CLEAN_TAG = 'setting2_scalefree_clean'
MNIST_FROM_CLEAN = True
OUT_TABLE = RES / 'setting2' / 'table_s2_scalefree.tex'
SCAN = RES / 'biactivity_scan'

PRETTY = table_s2.DATASET_DISPLAY


def _read_ntrba(tag):
    for fn in ('results.pkl', 'results_checkpoint.pkl'):
        p = RES / tag / fn
        if p.exists():
            df = pd.read_pickle(p)
            return df[df.method == 'ntrba_wl1'].copy()
    raise FileNotFoundError(f"NTRBA results not found for tag {tag}.")


def _load_clean_ntrba():
    mf = _read_ntrba(MATFREE_TAG)
    if MNIST_FROM_CLEAN:
        cl = _read_ntrba(CLEAN_TAG)
        mf = pd.concat([mf[mf.dataset != 'mnist'],
                        cl[cl.dataset == 'mnist']], ignore_index=True)
    return mf


def build_results_table():
    old = pd.read_pickle(OLD)
    new_ntrba = _load_clean_ntrba()

    # replace ALL ntrba_wl1 rows with the clean scale-free re-run; keep the
    # band-independent scalar_cv / sparseho_wl1 paper rows
    merged = pd.concat([old[old.method != 'ntrba_wl1'], new_ntrba], ignore_index=True)
    merged = merged.sort_values(['dataset', 'method', 'seed']).reset_index(drop=True)

    print("=== NTRBA-wl1: OLD (legacy band, paper) vs NEW (scale-free, clean) ===")
    on = old[old.method == 'ntrba_wl1']
    for d in sorted(new_ntrba.dataset.unique()):
        o = on[on.dataset == d]
        n = new_ntrba[new_ntrba.dataset == d]
        if len(o) and len(n):
            print(f"  {d:14s} sparsity {o.sparsity.mean():6.3f}->{n.sparsity.mean():6.3f}%  "
                  f"t/iter {o.t_per_iter.mean():7.2f}->{n.t_per_iter.mean():6.2f}s  "
                  f"F1 {o.test_f1.mean():.3f}->{n.test_f1.mean():.3f}")
    print()

    table = table_s2.build_table(merged)
    OUT_TABLE.write_text(table + '\n')
    print(f"saved -> {OUT_TABLE}\n")
    print(table)


def build_biactivity_props():
    """Panel: biactivity incidence at the operating penalty for Setting-2 datasets,
    scale-free vs legacy band, showing why NTRBA == Sparse-HO where B=0."""
    p = SCAN / 'scan_setting2.pkl'
    if not p.exists():
        print("(scan_setting2.pkl not ready; skipping properties table)")
        return
    scan = pd.read_pickle(p)
    # operating region for run_s2 (init 0.1*a_max, honest band eps=1e-3): report the
    # max legacy vs scale-free B over path/cv/fixed at eps=1e-3.
    band = scan[np.isclose(scan.eps_B, 1e-3)]
    rows = []
    for d, g in band.groupby('dataset'):
        rows.append(dict(dataset=d, m=int(g.m.iloc[0]),
                         B_legacy=int(g.B_legacy.max()),
                         B_scalefree=int(g.B_scalefree.max())))
    L = [r'\begin{tabular}{lrrr}', r'\toprule',
         r'Dataset & $m$ & $B$ (legacy) & $B$ (scale-free) \\', r'\midrule']
    for r in rows:
        L.append(rf"{PRETTY.get(r['dataset'], r['dataset'])} & {r['m']} & "
                 rf"{r['B_legacy']} & {r['B_scalefree']} \\")
    L += [r'\bottomrule', r'\end{tabular}']
    print("\n=== Biactivity properties (Setting-2 datasets, eps_B=1e-3) ===")
    print('\n'.join(L))
    (SCAN / 'table_setting2_biactivity_props.tex').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    build_results_table()
    build_biactivity_props()
