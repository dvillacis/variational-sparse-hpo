"""Plot results for Experiment 6 Setting 2 (GD variant).

This mirrors ``plot_s2.py`` but reads the NBA-vs-SparseHO-vs-scalar results
from ``results/setting2_gd``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_s2 as base
from expes_fb.shared.plotting import get_method_style


RESULTS_DIR = Path(__file__).parent / "results" / "setting2_gd"
RESULTS_PATH = RESULTS_DIR / "results.pkl"

METHODS = ["scalar_cv", "sparseho_wl1", "nba_wl1"]
METHOD_LABELS = {
    "scalar_cv": "Scalar ℓ1 (CV)",
    "sparseho_wl1": "SparseHO (wℓ1)",
    "nba_wl1": "NBA-wℓ1",
}
METHOD_STYLES = {key: get_method_style(key) for key in METHODS}
METHOD_COLORS = {key: METHOD_STYLES[key]["color"] for key in METHODS}


def _configure_base_module():
    base.RESULTS_DIR = RESULTS_DIR
    base.RESULTS_PATH = RESULTS_PATH
    base.METHODS = list(METHODS)
    base.METHOD_LABELS = dict(METHOD_LABELS)
    base.METHOD_STYLES = dict(METHOD_STYLES)
    base.METHOD_COLORS = dict(METHOD_COLORS)


_configure_base_module()


def main():
    base.main()


if __name__ == "__main__":
    main()
