"""Experiment 5 — Oracle scalability: O(|S|^3) vs O(m^3).

What is being timed
-------------------
The dominant cost of one hypergradient oracle call: assembling the reduced
Hessian H_S = X_S^T X_S / n_tr  and solving H_S p_S = -z*_S via Cholesky
factorisation.  Three oracle variants are compared:

  full_dense  — working set is ALL m features (naive baseline).
                H is m × m;  cost O(n_tr m^2) assembly + O(m^3) Cholesky.

  null        — working set is the strict primal support S = {j : y*_j ≠ 0}.
                With sparsity density ρ_s, |S| = ρ_s m.
                Cost O(n_tr |S|^2) + O(|S|^3)  =  O(ρ_s^3 m^3) (dominant).

  da          — same as null but S is augmented with biactive coordinates
                selected by the descent-aligned policy.  In practice the
                augmentation adds a small fraction δ·m of features, making
                |S_da| = (ρ_s + δ) m  with δ ≪ ρ_s.  The overhead is
                ((ρ_s + δ)/ρ_s)^3 − 1, invisible on a log-log scale.

Why the inner solve is excluded
--------------------------------
The inner (lower-level) optimisation is shared by all three oracle variants —
they all need y*(x) before the adjoint solve begins.  Including it in the
timing would conflate the oracle overhead with the inner solver cost, masking
the effect we want to show.  With warm-starting, the inner solve takes
O(1) extra iterations at a fixed x and does not depend on the oracle policy.

Primary sweep
--------------
m ∈ {10^2, 10^2.5, 10^3, 10^3.5, 10^4, 10^4.5, 10^5} (7 log-spaced points).
ρ_s = 0.05 fixed.  Repeat each timing N_REPEAT times; report median.
The full-dense oracle is skipped for m > M_DENSE_MAX to avoid OOM.

Secondary sweep (sparsity sensitivity)
---------------------------------------
Fixed m = 10^4.  ρ_s ∈ {1%, 2%, 5%, 10%, 20%, 50%}.  DA oracle only.

Usage
-----
    python run.py            # from within this directory, or
    python expes_fb/expe5_scalability/run.py   # from repo root
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# m grid: integer values at each half-decade on [10^2, 10^5]
M_VALUES = [int(round(10 ** x)) for x in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]]

RHO_S = 0.05          # primary sparsity density
N_RATIO = 0.10        # n_tr = N_RATIO * m  (overparameterised regime)
ALPHA_L2 = 1e-3       # ridge penalty (small, just ensures SPD; same for all m)
N_REPEAT = 10         # timing repetitions; median is reported

# Biactive augmentation for the DA oracle:
# we add BIACTIVE_FRAC * m extra features beyond the strict support.
# At ρ_s = 5%, BIACTIVE_FRAC = 0.5% gives |S_da| / |S_null| = 1.10
# → (1.10)^3 ≈ 1.33 overhead, barely visible on log-log.
BIACTIVE_FRAC = 0.005

# Full-dense oracle only run up to this m (avoids OOM / excessively long runs)
M_DENSE_MAX = 3200    # ≈ 10^3.5

# Secondary sweep
M_SPARSITY = 10_000
RHO_S_VALUES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_PATH = RESULTS_DIR / 'results.pkl'

# --- smoke-test overrides: set VSHPO_SMOKE=1 for a fast, tiny run (CI / spot-check).
# When the variable is unset, the full configuration above is used unchanged.
import os
if os.environ.get('VSHPO_SMOKE'):
    M_VALUES     = [100, 316, 1000]
    N_REPEAT     = 1
    M_SPARSITY   = 1000
    RHO_S_VALUES = [0.05, 0.20]


# ---------------------------------------------------------------------------
# Core timing routine
# ---------------------------------------------------------------------------

def _time_reduced_solve(n_tr, n_S, n_repeat, rng):
    """Time H_S assembly and Cholesky solve for a system of size n_S.

    Generates a random X_S ∈ R^{n_tr × n_S} and z_S ∈ R^{n_S}, then times:

        H_S = X_S^T X_S / n_tr  +  ALPHA_L2 * I
        p_S = cho_solve(cho_factor(H_S), -z_S)

    This is the exact computation in Phase 3 of Algorithm 1.

    Parameters
    ----------
    n_tr : int
        Number of training samples.
    n_S : int
        Working set size (|S|).
    n_repeat : int
        Number of timing repetitions.
    rng : np.random.Generator

    Returns
    -------
    t_median : float
        Median wall-clock time in seconds.
    """
    if n_S == 0:
        return 0.0

    X_S = rng.standard_normal((n_tr, n_S))
    z_S = rng.standard_normal(n_S)

    def _solve():
        H = X_S.T @ X_S / n_tr
        H.flat[:: n_S + 1] += ALPHA_L2 + 1e-8   # ridge + numerical safety
        try:
            cho = cho_factor(H, lower=True, check_finite=False)
            return cho_solve(cho, -z_S, check_finite=False)
        except LinAlgError:
            return np.linalg.solve(H, -z_S)

    _solve()   # warmup (JIT, cache)

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _solve()
        times.append(time.perf_counter() - t0)

    return float(np.median(times))


# ---------------------------------------------------------------------------
# Instance parameter helper
# ---------------------------------------------------------------------------

def _system_sizes(m, rho_s):
    """Return (n_tr, n_S_null, n_S_da) for a given (m, rho_s) configuration."""
    n_tr = max(int(N_RATIO * m), 5)
    n_null = max(int(rho_s * m), 1)
    n_da = n_null + max(int(BIACTIVE_FRAC * m), 0)
    return n_tr, n_null, n_da


# ---------------------------------------------------------------------------
# Primary sweep: time vs m
# ---------------------------------------------------------------------------

def run_primary_sweep(rng):
    rows = []

    for m in M_VALUES:
        n_tr, n_null, n_da = _system_sizes(m, RHO_S)
        print(
            f"  m={m:7d}  n_tr={n_tr:5d}  "
            f"|S_null|={n_null:5d}  |S_da|={n_da:5d}",
            end="   ", flush=True,
        )

        # Null oracle
        t_null = _time_reduced_solve(n_tr, n_null, N_REPEAT, rng)
        rows.append(dict(
            sweep='primary', oracle='null',
            m=m, rho_s=RHO_S, n_tr=n_tr, n_S=n_null,
            t_median=t_null,
        ))

        # DA oracle (slightly larger working set)
        t_da = _time_reduced_solve(n_tr, n_da, N_REPEAT, rng)
        rows.append(dict(
            sweep='primary', oracle='sc',
            m=m, rho_s=RHO_S, n_tr=n_tr, n_S=n_da,
            t_median=t_da,
        ))

        # Full-dense oracle (only for small m)
        if m <= M_DENSE_MAX:
            t_dense = _time_reduced_solve(n_tr, m, N_REPEAT, rng)
        else:
            t_dense = np.nan
        rows.append(dict(
            sweep='primary', oracle='dense',
            m=m, rho_s=RHO_S, n_tr=n_tr, n_S=m,
            t_median=t_dense,
        ))

        print(
            f"null={t_null:.4f}s  sc={t_da:.4f}s"
            f"dense={'N/A' if np.isnan(t_dense) else f'{t_dense:.4f}s'}",
            flush=True,
        )

    return rows


# ---------------------------------------------------------------------------
# Secondary sweep: time vs ρ_s at fixed m
# ---------------------------------------------------------------------------

def run_secondary_sweep(rng):
    rows = []
    m = M_SPARSITY

    for rho_s in RHO_S_VALUES:
        n_tr, _, n_da = _system_sizes(m, rho_s)
        print(
            f"  rho_s={rho_s:.2f}  m={m}  n_tr={n_tr}  |S_da|={n_da}",
            end="   ", flush=True,
        )
        t = _time_reduced_solve(n_tr, n_da, N_REPEAT, rng)
        rows.append(dict(
            sweep='secondary', oracle='sc',
            m=m, rho_s=rho_s, n_tr=n_tr, n_S=n_da,
            t_median=t,
        ))
        print(f"t={t:.4f}s", flush=True)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    print("=== Primary sweep: wall-clock time vs m  (rho_s=5%) ===")
    rows = run_primary_sweep(rng)

    print(f"\n=== Secondary sweep: time vs rho_s  (m={M_SPARSITY}) ===")
    rows += run_secondary_sweep(rng)

    df = pd.DataFrame(rows)
    df.to_pickle(RESULTS_PATH)
    print(f"\nSaved {len(df)} rows to {RESULTS_PATH}")
    print(df.to_string())


if __name__ == '__main__':
    main()
