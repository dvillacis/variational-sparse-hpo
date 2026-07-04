"""Generate the LaTeX table for Experiment 6 Setting 2 (GD variant).

This mirrors ``table_s2.py`` but reads the NBA-vs-SparseHO-vs-scalar results
from ``results/setting2_gd``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import table_s2 as base


RESULTS_DIR = Path(__file__).parent / "results" / "setting2_gd"
RESULTS_PATH = RESULTS_DIR / "results.pkl"
TABLE_PATH = RESULTS_DIR / "table_s2_gd_results.tex"

METHODS = ["scalar_cv", "sparseho_wl1", "nba_wl1"]
METHOD_LABELS = {
    "scalar_cv": r"Scalar $\ell_1$ (CV)",
    "sparseho_wl1": r"\textsc{Sparse-HO} (w$\ell_1$)",
    "nba_wl1": r"\textsc{NBA}-w$\ell_1$",
}


def _configure_base_module():
    base.RESULTS_DIR = RESULTS_DIR
    base.RESULTS_PATH = RESULTS_PATH
    base.TABLE_PATH = TABLE_PATH
    base.METHODS = list(METHODS)
    base.METHOD_LABELS = dict(METHOD_LABELS)


_configure_base_module()


def main():
    base.main()


if __name__ == "__main__":
    main()
