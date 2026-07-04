# variational-sparse-hpo

Companion code for the paper

> **A Variational Analysis Approach for Bilevel Hyperparameter Optimization with
> Sparse Regularization**
> Pedro Pérez-Aros, Emilio Vilches, David Villacís.
> *Computational Optimization and Applications* (COAP), Special Issue on Machine
> Learning and Optimization.

It reproduces every figure and table in the paper's numerical section. The core
library is a vendored, self-contained copy of `sparse_ho` (a fork of
[`QB3/sparse-ho`](https://github.com/QB3/sparse-ho), BSD-3) extended with:

- **weighted models** — `WeightedElasticNet`, `WeightedSparseLogReg` (per-feature
  regularization for the lower level);
- a **support-reduced variational hypergradient** — `ImplicitVariational` with
  sign-consistent / biactive-set selection (`select_biactive_self_consistent`);
- **nonsmooth outer optimizers** — `TrustRegion` (NTRBA) and
  `NormalizedSubgradient` (NBA).

## Requirements

- Python **3.13** (pinned in `.python-version`). The binding constraint is
  `numba` (0.64) + `celer` (0.7.4); if wheels are unavailable for your platform,
  see *Troubleshooting* below.
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

## Install

```bash
uv sync          # creates .venv, installs deps + the vendored sparse_ho package
```

## Quickstart

```bash
# fast, tiny end-to-end check of all synthetic experiments (minutes):
uv run python scripts/run_all.py --tier synthetic --smoke
# or:
make smoke
```

## Running experiments

Every experiment lives in `expes_fb/expeN_*/` and follows a
`run.py -> plot.py -> table.py` triad writing to a local `results/` directory.
Two entry points orchestrate them:

```bash
# global runner
uv run python scripts/run_all.py                    # synthetic tier, full config
uv run python scripts/run_all.py --smoke            # synthetic tier, tiny/fast
uv run python scripts/run_all.py --only exp3 exp4   # selected experiments
uv run python scripts/run_all.py --tier real        # exp6 (needs data, see below)
uv run python scripts/run_all.py --tier all         # everything (heavy)
uv run python scripts/run_all.py --only exp4 --skip-run   # regen figures/tables only
uv run python scripts/run_all.py --only exp3 --parallel   # joblib run_parallel driver
uv run python scripts/run_all.py --list             # list all experiment ids

# single experiment
uv run python scripts/run_experiment.py exp4
make exp4
```

- `--smoke` shrinks the sweeps (fewer seeds/sizes/iterations) via the `VSHPO_SMOKE`
  environment variable; default runs are unaffected.
- `--parallel` uses each experiment's joblib `run_parallel.py` driver (full runs).
- The heavy real-data runs `exp6-s1`/`exp6-s2` have no smoke knob and are skipped
  under `--smoke`.

## Experiment → paper artifact map

| Experiment id | Paper item | Figure | Table |
|---|---|---|---|
| `exp1` | Experiment 1 (FB exactness + gradient starvation, toy) | `fig_experiment1.pdf` | — |
| `exp2` | Experiment 2 (feature-wise vs scalar) | `fig_penalty_profile.pdf` | `tab:expe2_metrics` |
| `exp3` | Experiment 3 (oracle × optimizer ablation) | `fig_convergence.pdf` | `tab:expe3_oracle_ablation` |
| `exp4` | Experiment 4 (gradient starvation vs baselines) | `fig_convergence.pdf` | `tab:expe4_sota` |
| `exp6-s1` | Experiment 5, Setting 1 (semi-synthetic) | — | `tab:expe5_setting1`, `tab:expe5_dataset_stats` |
| `exp6-s2` | Experiment 5, Setting 2 (real classification) | — | `tab:expe5_setting2` |
| `exp6-diag` | Diagnostic (natural biactivity) | — | `tab:biactivity_diagnostic` |
| `st_plot` | Theory figure (soft-thresholding coderivative) | `graph_soft_thresholding.pdf` | — |

Figures/tables are written to each experiment's `results/` folder (git-ignored).
The manuscript expects experiment-suffixed filenames (e.g. `fig_convergence_expe3.pdf`);
copy/rename from `results/` when importing into the paper.

## Data for Experiment 6 (real-world)

Experiments 1–5 are fully synthetic (no downloads). Experiment 6 needs real LIBSVM
datasets (~790 MB):

```bash
uv run python scripts/download_data.py            # auto (rcv1, mnist) + manual LIBSVM
uv run python scripts/download_data.py --list     # show datasets and sizes
uv run python scripts/download_data.py --datasets phishing w8a
```

`rcv1` (scikit-learn) and `mnist` (libsvmdata) are auto-cached; the LIBSVM binary
files (`real-sim`, `news20.binary`, `phishing`, `w8a`, `rcv1_train.binary`) are
fetched into `expes_fb/expe6_real_world/data/` (git-ignored). The individual expe6
scripts accept their own arguments (`--datasets`, `--max-samples`, `--smoke`,
`--tag`) for fine-grained control.

## Tests

```bash
make test        # paper-relevant solver/optimizer tests (green: 32 passed)
make test-all    # full inherited suite (has known upstream-drift failures, below)
```

`NUMBA_DISABLE_JIT=1` (set by the make targets) makes the Numba kernels traceable.
`make test` covers the fork's core solvers (`WeightedElasticNet`,
`WeightedSparseLogReg`) and the outer optimizers — the code this paper relies on.

The **full** suite (`make test-all`) additionally runs tests inherited from upstream
`sparse-ho`. Several fail under the modern pinned dependencies; none are exercised by
the paper's experiments (which reproduce cleanly):

| Failing test | Cause | Paper-relevant? |
|---|---|---|
| `test_wlogreg_solver::test_dense_sparse_agree` | dense vs sparse `WeightedSparseLogReg` differ ~2.2e-4 > atol 1e-4 (known tolerance) | solver, but a benign tolerance edge case |
| `test_grid_search::test_cross_val_criterion` (×4) | inside **celer** (`LinearModelCV` uses `n_alphas`, renamed to `alphas` in scikit-learn 1.9) | no — celer CV dropin, unused |
| `test_multiclass::test_our_vs_sklearn` | `LogisticRegression(multi_class=…)` removed in scikit-learn | no — multiclass path, unused |
| `test_docstring_parameters` (×2) | optional `numpydoc` not installed; tab-style meta-check | no — doc/style meta-tests |

The `cvxpylayers` reference-oracle tests (`test_criterion`, `test_models`) **skip**
unless the optional extra is installed: `uv sync --extra reference`.

## Repository layout

```
sparse_ho/            vendored core library (+ tests/)
expes_fb/             this paper's experiments
  expe1..6_*/         run.py -> plot.py -> table.py triads
  shared/             calibrated data generator + matplotlib paper style
  st_plot/            soft-thresholding theory figure
scripts/
  run_all.py          global runner (registry-driven)
  run_experiment.py   single-experiment dispatcher
  download_data.py    Experiment 6 dataset downloader
docs/
  experimental_strategy.md   authoritative experiment specification
```

## Attribution & license

BSD-3-Clause. The `sparse_ho` package derives from
[`QB3/sparse-ho`](https://github.com/QB3/sparse-ho) by Q. Bertrand and
Q. Klopfenstein; see `LICENSE.txt`. The weighted models, variational hypergradient,
and nonsmooth optimizers are additions for this paper.
