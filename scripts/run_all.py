#!/usr/bin/env python
"""Global experiment runner for the variational-sparse-hpo companion repo.

Runs each experiment's ``run.py -> plot.py -> table.py`` triad (the exact scripts
vary per experiment) and reports a summary. Experiments are grouped into two tiers:

  synthetic : exp1..exp4 + st_plot   (self-contained, no downloads)
  real      : exp6-s1/s2/diag        (needs the LIBSVM datasets; see download_data.py)

Examples
--------
    python scripts/run_all.py                          # synthetic tier, full config
    python scripts/run_all.py --smoke                  # synthetic tier, tiny/fast
    python scripts/run_all.py --only exp3 exp4         # just those experiments
    python scripts/run_all.py --tier real --smoke      # exp6 s3 + diagnostic (light)
    python scripts/run_all.py --tier all               # everything (heavy!)
    python scripts/run_all.py --only exp4 --skip-run   # regen figures/tables only
    python scripts/run_all.py --only exp3 --parallel   # use joblib run_parallel.py

Exit code is non-zero if any step fails.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPES = REPO_ROOT / "expes_fb"
PY = sys.executable

SMOKE_ENV = {"VSHPO_SMOKE": "1"}


def _step(kind, script, args=None, smoke_env=None, smoke_args=None):
    """One script invocation. kind in {run, plot, table}."""
    return {
        "kind": kind,
        "script": script,
        "args": list(args or []),
        "smoke_env": dict(smoke_env or {}),
        "smoke_args": list(smoke_args or []),
    }


# Registry: experiment id -> {dir, tier, desc, steps, [parallel], [smoke_skip]}.
# `parallel` names an alternate joblib driver used for the run step under --parallel.
# `smoke_skip` marks heavy experiments with no smoke knob (skipped under --smoke).
REGISTRY = {
    "exp1": {
        "dir": "expe1_illustrative",
        "tier": "synthetic",
        "desc": "FB exactness + gradient starvation (toy) -> fig_experiment1.pdf",
        "steps": [_step("run", "run.py"), _step("plot", "plot.py")],
    },
    "exp2": {
        "dir": "expe2_feature_resolution",
        "tier": "synthetic",
        "desc": "feature-wise vs scalar -> fig_penalty_profile.pdf, tab:expe2_metrics",
        "parallel": "run_parallel.py",
        "steps": [
            _step("run", "run.py", smoke_env=SMOKE_ENV),
            _step("plot", "plot.py"),
            _step("table", "table.py"),
        ],
    },
    "exp3": {
        "dir": "expe3_oracle_ablation",
        "tier": "synthetic",
        "desc": "oracle x optimizer ablation -> fig_convergence.pdf, "
                "tab:expe3_oracle_ablation",
        "parallel": "run_parallel.py",
        "steps": [
            _step("run", "run.py", smoke_env=SMOKE_ENV),
            _step("plot", "plot.py"),
            _step("table", "table.py"),
        ],
    },
    "exp4": {
        "dir": "expe4_sota_comparison",
        "tier": "synthetic",
        "desc": "gradient starvation vs baselines -> fig_convergence.pdf, tab:expe4_sota",
        "parallel": "run_parallel.py",
        "steps": [
            _step("run", "run.py", smoke_env=SMOKE_ENV),
            _step("plot", "plot.py"),
            _step("table", "table.py"),
        ],
    },
    "st_plot": {
        "dir": "st_plot",
        "tier": "synthetic",
        "desc": "soft-thresholding coderivative figure -> graph_soft_thresholding.pdf",
        "steps": [_step("plot", "plot.py")],
    },
    "exp6-s1": {
        "dir": "expe6_real_world",
        "tier": "real",
        "smoke_skip": True,
        "desc": "Exp5 Setting 1 semi-synthetic -> tab:expe5_setting1 (heavy; no smoke)",
        "steps": [
            _step("run", "run_s1.py"),
            _step("table", "table_s1.py"),
        ],
    },
    "exp6-s2": {
        "dir": "expe6_real_world",
        "tier": "real",
        "smoke_skip": True,
        "desc": "Exp5 Setting 2 real classification -> tab:expe5_setting2 (heavy; no smoke)",
        "steps": [
            _step("run", "run_s2.py"),
            _step("table", "table_s2.py"),
            _step("table", "table_dataset_stats.py"),
        ],
    },
    "exp6-diag": {
        "dir": "expe6_real_world",
        "tier": "real",
        "desc": "biactivity diagnostic -> tab:biactivity_diagnostic",
        "steps": [
            _step("run", "scan_biactivity.py", smoke_args=["--max-samples", "2000"]),
            _step("table", "table_biactivity_diagnostic.py"),
        ],
    },
}


def log(msg):
    print(msg, flush=True)


def resolve_ids(tier, only):
    if only:
        ids = []
        for token in only:
            if token in REGISTRY:
                ids.append(token)
            elif token == "exp6":
                ids.extend(k for k in REGISTRY if k.startswith("exp6"))
            else:
                raise SystemExit(
                    f"unknown experiment '{token}'. choices: {', '.join(REGISTRY)}"
                )
        return ids
    if tier == "all":
        return list(REGISTRY)
    return [k for k, v in REGISTRY.items() if v["tier"] == tier]


def run_step(exp, st, *, smoke, skip_run, parallel):
    if skip_run and st["kind"] == "run":
        log(f"    · skip {st['script']} (--skip-run)")
        return True

    script = st["script"]
    if parallel and st["kind"] == "run" and exp.get("parallel") and not smoke:
        script = exp["parallel"]

    env = dict(os.environ)
    args = list(st["args"])
    if smoke:
        env.update(st["smoke_env"])
        args += st["smoke_args"]

    cmd = [PY, script, *args]
    log(f"    $ ({exp['dir']}) {' '.join(cmd[1:])}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=EXPES / exp["dir"], env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        log(f"    ✗ {script} failed (exit {proc.returncode}, {dt:.1f}s)")
        return False
    log(f"    ✓ {script} ({dt:.1f}s)")
    return True


def run_experiment(exp_id, *, smoke=False, skip_run=False, parallel=False):
    exp = REGISTRY[exp_id]
    if smoke and exp.get("smoke_skip"):
        log(f"[{exp_id}] SKIPPED under --smoke (no smoke knob; heavy real-data run). "
            f"Run without --smoke to execute the full job.")
        return "skipped"
    log(f"[{exp_id}] {exp['desc']}")
    for st in exp["steps"]:
        if not run_step(exp, st, smoke=smoke, skip_run=skip_run, parallel=parallel):
            return "failed"
    return "ok"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=["synthetic", "real", "all"], default="synthetic",
                   help="which tier to run when --only is not given (default: synthetic)")
    p.add_argument("--only", nargs="+", metavar="ID",
                   help=f"run specific experiments. choices: {', '.join(REGISTRY)} "
                        f"(or 'exp6' for all exp6-*)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny/fast configuration for CI and spot-checks")
    p.add_argument("--skip-run", action="store_true",
                   help="skip the compute step; regenerate figures/tables from cached .pkl")
    p.add_argument("--parallel", action="store_true",
                   help="use joblib run_parallel.py drivers for exp2/3/4 (full runs only)")
    p.add_argument("--list", action="store_true", help="list experiments and exit")
    args = p.parse_args(argv)

    if args.list:
        for k, v in REGISTRY.items():
            print(f"  {k:10s} [{v['tier']:9s}] {v['desc']}")
        return 0

    ids = resolve_ids(args.tier, args.only)
    log(f"== running {len(ids)} experiment(s): {', '.join(ids)} "
        f"(smoke={args.smoke}, skip_run={args.skip_run}, parallel={args.parallel}) ==")

    results = {}
    t0 = time.time()
    for exp_id in ids:
        results[exp_id] = run_experiment(
            exp_id, smoke=args.smoke, skip_run=args.skip_run, parallel=args.parallel
        )

    log(f"\n== summary ({time.time() - t0:.1f}s total) ==")
    for exp_id, status in results.items():
        mark = {"ok": "✓", "failed": "✗", "skipped": "·"}[status]
        log(f"  {mark} {exp_id}: {status}")

    return 1 if any(s == "failed" for s in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
