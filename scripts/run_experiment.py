#!/usr/bin/env python
"""Run a single experiment's run -> plot -> table triad.

Thin wrapper over run_all.py. Any options after the experiment id are forwarded
to the global runner (e.g. --smoke, --skip-run, --parallel).

Examples
--------
    python scripts/run_experiment.py exp4
    python scripts/run_experiment.py exp3 --smoke
    python scripts/run_experiment.py exp4 --skip-run
    python scripts/run_experiment.py exp6-diag --smoke
"""

import sys

from run_all import REGISTRY, main


def _usage():
    print(__doc__)
    print("experiment ids:")
    for k, v in REGISTRY.items():
        print(f"  {k:10s} {v['desc']}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _usage()
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    exp_id = sys.argv[1]
    if exp_id not in REGISTRY and exp_id != "exp6":
        print(f"unknown experiment '{exp_id}'.\n")
        _usage()
        raise SystemExit(2)
    raise SystemExit(main(["--only", exp_id, *sys.argv[2:]]))
