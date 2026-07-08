"""Setting-1 table (Experiment 5) — scale-free common-clock format.

Reuses ``table_common_clock.build_s1``: 4 methods (incl. the ``NTRBA-null``
oracle-isolation control), the common-clock efficiency block (It./t/it/Total),
and ddof=1 std over the 5 splits, reading ``results/setting1/``. Emitted as a
``sidewaystable`` so it matches Table ``tab:expe5_setting1`` in the manuscript.

(The previous 3-method, single-"Runtime (s)" renderer lived here; it produced a
different table from the manuscript. The build logic now lives in
``table_common_clock.build_s1`` and is shared with the paper table.)

Usage
-----
    python table_s1.py
"""

from table_common_clock import RES, build_s1

OUT = RES / 'setting1' / 'table_s1_metrics.tex'


def main():
    tex = build_s1()
    # 9 columns exceed the single text column -> landscape (matches the manuscript).
    tex = tex.replace(r'\begin{table}[t]', r'\begin{sidewaystable}', 1)
    tex = tex.replace(r'\end{table}', r'\end{sidewaystable}', 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex + '\n')
    print(tex)
    print('\nsaved ->', OUT)


if __name__ == '__main__':
    main()
