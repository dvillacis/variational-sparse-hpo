"""Parallel driver for Experiment 2 (20-seed re-run, scale-free band).

Reuses the tested ``run.run_one`` config runner and the sweep grid defined in
``run.py`` (M_VALUES x NM_RATIOS x SPARSITY_FRACS x range(N_SEEDS)); only the
outer loop is parallelized with joblib. Produces the identical ``results.pkl``
schema that ``table.py`` consumes.

    python run_parallel.py            # from within this directory
"""

import sys
import time
from itertools import product
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
import run  # noqa: E402

N_JOBS = 6


def _safe_run_one(m, nm_ratio, sparsity_frac, seed):
    try:
        return run.run_one(m, nm_ratio, sparsity_frac, seed)
    except Exception as exc:
        print(f"  ERROR m={m} nm={nm_ratio} sf={sparsity_frac} seed={seed}: {exc}",
              flush=True)
        return []


def main():
    run.RESULTS_DIR.mkdir(exist_ok=True)
    grid = list(product(run.M_VALUES, run.NM_RATIOS, run.SPARSITY_FRACS))
    configs = [(m, nm, sf, seed)
               for (m, nm, sf) in grid for seed in range(run.N_SEEDS)]
    print(f"Launching {len(configs)} configs x 2 methods on {N_JOBS} workers "
          f"(N_SEEDS={run.N_SEEDS})", flush=True)

    t0 = time.time()
    batches = Parallel(n_jobs=N_JOBS, backend='loky', verbose=10)(
        delayed(_safe_run_one)(m, nm, sf, seed) for (m, nm, sf, seed) in configs
    )
    rows = [r for batch in batches for r in batch]
    df = pd.DataFrame(rows)
    df.to_pickle(run.RESULTS_PATH)
    print(f"\nSaved {len(df)} rows to {run.RESULTS_PATH} "
          f"in {time.time() - t0:.0f}s", flush=True)
    print(f"seeds: {sorted(df.seed.unique())}", flush=True)


if __name__ == '__main__':
    main()
