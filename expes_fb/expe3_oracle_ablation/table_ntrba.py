"""Compatibility wrapper for the unified Experiment 3 ablation table.

The old optimizer-only companion table has been retired. Running this script
now generates the single consolidated table from ``table.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import table as unified_table


def main():
    print(
        "table_ntrba.py is deprecated; generating the unified "
        "Experiment 3 ablation table instead."
    )
    unified_table.main()


if __name__ == "__main__":
    main()
