# Experiment 7 — Soundness of the stationarity certificate

Replacement for the old Section 5.5 "natural incidence of biactivity" diagnostic
(the 19/19 finite-difference table). Instead of counting biactive coordinates,
it adjudicates **whether the support-restricted (null) stopping test `‖h‖=0` is a
sound stationarity certificate**, against an independent ground truth.

## The claim it demonstrates

Both the sign-consistent (SC) and support-restricted (null) selections are valid
elements of the residual-enlarged subdifferential, but only SC yields a **sound**
certificate. At the regularization-path anchor `α_max` — the deterministic warm
start of every path solver (glmnet/celer/Sparse-HO) — the support is empty, so the
null hypergradient is identically zero and its stopping test certifies global
stationarity. Yet the value-function directional derivative `Φ'(x̄;−e_{i*})`,
measured by a **model-free finite difference of the real solver**, is strictly
negative: a penalty-decrease descent direction exists. The SC component
`−h^sc_{i*}` reproduces it. The null certificate is therefore **false** at the
anchor, and sound (coincides with SC) everywhere the method actually operates.

Ground truth is *independent* of the oracle (model-free FD + a closed-form
synthetic), never "SC agrees with SC."

## Artifacts → paper

| Script | Output | Paper artifact |
|---|---|---|
| `run.py` | `results/results.pkl` | — |
| `table.py` | `results/table_certificate_soundness.tex` | `tab:certificate_soundness` |
| `plot.py` | `results/fig_certificate_soundness.pdf` | `fig_certificate_soundness` |
| `synthetic.py` | `results/synthetic.pkl` (+ stdout) | the minimal closed-form counterexample (text/inline) |

## Run

```bash
uv run python expes_fb/expe7_certificate_soundness/run.py          # 9 datasets + sweeps
uv run python expes_fb/expe7_certificate_soundness/synthetic.py    # closed-form anchor
uv run python expes_fb/expe7_certificate_soundness/table.py --copy-to <paper_dir>
uv run python expes_fb/expe7_certificate_soundness/plot.py  --copy-to <paper_dir>
```

`--quick` on `run.py` audits only `α_max` + `cv_opt`; `--no-sweep` skips the
band/FD sensitivity grids. Most datasets are cheap (pointwise, no end-to-end HPO
runs); the microarray/categorical block is a few minutes, but `news20.binary`
(1.35M features) adds ~25 min of FD re-solves, so a full 9-dataset run is ~30 min.

## What the run reports (seed 0)

- **α_max, all nine datasets:** `‖g_null‖ = 0` (null certifies stationarity),
  `Φ'(x̄;−e_{i*}) < 0` (descent exists, −0.03 to −0.36), `−h^sc` matches to ~1%
  → **false certificate** on every dataset. Band-invariant (the anchor feature is
  biactive by definition); FD gap → 0 under Richardson refinement.
- **Interior path + cv-opt:** the biactive set is empty or carries negligible
  slope (`real-sim` B=1, `news20` B=2 at cv-opt, both `Φ' ~ 1e-4`), `‖g_null‖ > 0`
  (null does not certify), so the two selections **effectively coincide** — the
  sound/parity regime.

## Design notes (referee-proofing)

- **Independent ground truth:** `Φ'` from a one-sided FD of the actual solver's
  validation loss (`certificate_audit.fd_directional`), cross-checked by the
  closed-form synthetic (`synthetic.py`). The SC oracle is compared *to* this
  ground truth, not used *as* it.
- **Not contrived:** `α_max` is a structural point (every path solver starts
  there), not a hand-tuned kink; the anchor feature is biactive by definition, so
  the headline survives any detection band.
- **Pointwise only:** we do NOT run an end-to-end "solver stuck at α_max"
  trajectory — that fixed point is knife-edge-sensitive (the practical optimizer
  steps off it, and both methods then reach the same objective). The claim is
  certificate *validity*, a pointwise property, not a performance race.
- **Negative control:** the companion terminal-solution audit
  (`../expe6_real_world/audit_terminal_certificate.py`) shows that at the solution
  the method actually returns under standard init, `B=0` and the oracles are
  bit-identical — the certificate difference has no realized consequence there.

## Dependencies

Imports only stable shared helpers from `../expe6_real_world` (dataset loaders,
train/val split, inner solver, held-out objective) and the `sparse_ho` oracles;
all audit logic is local to this folder.
