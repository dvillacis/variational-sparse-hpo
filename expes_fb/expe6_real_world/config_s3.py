"""Frozen configuration for Experiment 6 Setting 3 — natural-degeneracy counterfactual.

PRE-REGISTRATION: this block is committed BEFORE running and must not be edited
afterwards. It answers Reviewer #3's "contrived" objection by testing whether
biactive gradient starvation arises naturally on the fully real Setting-2 datasets
(standard init, no planted signal, band decoupled from any calibration slack).

See ``Revision2_23062026/EXP_NATURAL_STARVATION_SPEC.md`` for the full design and the
pre-registered hypotheses H1-H5.
"""

# --- datasets / seeds / budgets (reuse Setting 2, more seeds) -----------------
DATASETS = ['phishing', 'real-sim', 'news20.binary', 'rcv1.binary', 'mnist']
N_SEEDS = 10                 # was 3 in Setting 2; this diagnostic is load-bearing
N_OUTER_CONVERGE = 60        # Sparse-HO run to its fixed point
N_OUTER_CONT = 40            # warm-start continuation budget K
INIT_FRACTION = 0.10         # log_alpha0 = log(0.10 * alpha_max) * ones(m)

# --- frozen reference knobs ---------------------------------------------------
INNER_TOL_REF = 1e-8         # tightened from Setting-2 1e-4: near-threshold != under-solved
INNER_MAX_ITER = 10000
BIACTIVE_TOL_REL_REF = 1e-3  # honest band, decoupled from any planting slack
NBA_STEP_SIZE = 0.1          # optimizer held FIXED across all arms (deconfounding)
LIN_SYS_TOL = 1e-8           # ImplicitVariational tol_lin_sys (tighter than 1e-6 default)

# --- invariance grids (sub-experiment B) --------------------------------------
TAU_LADDER = [1e-4, 1e-6, 1e-8, 1e-10]        # inner-solver tol; B(tau) must not shrink
EPS_B_PLATEAU = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # band; 0.10 = 2*delta is the RIGHT edge
COMPUTE_DPHI_GRID = True     # also compute a 1-step decrease per grid cell

# --- branch-stability / Prop-15 check (sub-experiment C) ----------------------
BRANCH_T = [0.1, 0.5, 1.0]   # log-alpha perturbations for the +/- e_i probe
MAX_BRANCH_COORDS = 20       # cap probed biactive coords per (dataset, seed) for cost

# --- misc ---------------------------------------------------------------------
ACTIVE_THR = 1e-10           # |beta_i| > thr => active/support
DIFF_THR = 1e-12             # |g_sc - g_null| > thr => SC-selected biactive coord
