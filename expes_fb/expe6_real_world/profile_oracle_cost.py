"""Decompose the NTRBA-vs-Sparse-HO runtime gap on sparse high-dim data.

The biactive band only affects DENSE mnist (B was inflated 770). On sparse datasets
B=0, so the band is not the cause. This isolates the two real candidates:
  (1) ORACLE: ImplicitVariational (builds an explicit reduced Hessian block, O(|S|)
      Hessian-vector products) vs Implicit (matrix-free adjoint).
  (2) OPTIMIZER: TrustRegion (an extra trial inner-solve per iteration for the rho-test)
      vs NormalizedSubgradient (one solve per iteration).

We time a single hypergradient from each oracle at a fixed penalty (warm-started),
isolating (1), and report the support size.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).parent))
from data_loaders import get_dataset
from run_s2 import (_make_split, _preprocess_features_for_split, _alpha_max,
                    _HeldOutLogisticWithMaxIter)

from sparse_ho.models import WeightedSparseLogReg
from sparse_ho.algo import (Implicit, ImplicitVariational,
                            select_biactive_self_consistent)

INNER_TOL = 1e-7
INNER_MAX_ITER = 10000


def _time_oracle(algo, model, X, y, idx_tr, idx_val, x_log, warm, reps=3):
    crit = _HeldOutLogisticWithMaxIter(idx_tr, idx_val, INNER_MAX_ITER)
    # warm up (populates warm start)
    crit.get_val_grad(model, X, y, x_log, algo.compute_beta_grad, tol=INNER_TOL)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        crit.get_val_grad(model, X, y, x_log, algo.compute_beta_grad, tol=INNER_TOL)
        t.append(time.perf_counter() - t0)
    info = dict(getattr(algo, 'last_run_info', {}) or {})
    return float(np.median(t)), info


def main():
    for dname, ms in [('rcv1.binary', 8000), ('real-sim', 8000)]:
        X, y = get_dataset(dname, Path(__file__).parent / 'data')
        if issparse(X):
            X = X.tocsc()
        if X.shape[0] > ms:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], ms, replace=False)
            X, y = X[idx], y[idx]
        n, m = X.shape
        idx_tr, idx_val, idx_te = _make_split(n, 0)
        Xp, _ = _preprocess_features_for_split(dname, X, idx_tr)
        if issparse(Xp):
            Xp = Xp.tocsc()
        X_tr, y_tr = Xp[idx_tr], y[idx_tr]
        model = WeightedSparseLogReg(alpha_l2=1.0 / len(idx_tr))
        a_max = _alpha_max(X_tr, y_tr)
        x_log = np.log(0.10 * a_max) * np.ones(m)   # run_s2 operating point

        t_imp, i_imp = _time_oracle(Implicit(), model, Xp, y, idx_tr, idx_val, x_log, True)
        orc = ImplicitVariational(policy=select_biactive_self_consistent,
                                  biactive_tol_rel=1e-3, biactive_scale_floor=0.0)
        t_var, i_var = _time_oracle(orc, model, Xp, y, idx_tr, idx_val, x_log, True)
        supp = int(i_var.get('strict_active_size', 0) + i_var.get('selected_biactive_size', 0))
        B = int(i_var.get('biactive_size', 0))
        print(f"\n{dname} (n={n}, m={m}):  support|S|~{supp}  B={B}")
        print(f"  Implicit (Sparse-HO oracle)     : {t_imp*1e3:8.1f} ms / hypergrad")
        print(f"  ImplicitVariational (our oracle): {t_var*1e3:8.1f} ms / hypergrad"
              f"   -> {t_var/max(t_imp,1e-9):.1f}x")


if __name__ == '__main__':
    main()
