"""Verify direction #3: is the SC selection the vanishing-smoothing limit?

Hypothesis: if the lower-level penalty is smoothed and the smoothing -> 0, the
hypergradient at a biactive coordinate converges to the sign-consistent (SC)
value, exposing the null selection as an exact-kink artifact.

We test this on the closed-form 2-feature orthonormal instance from synthetic.py
(feature 1 biactive: lambda_1 = c_1). Everything is closed form, so the limits
are exact. We compare three limiting processes at the biactive coordinate:

  (H) Huber / Moreau smoothing of |.|, mu -> 0   (symmetric penalty smoothing)
  (A) active-side approach  lambda_1 = c_1 (1 - delta), delta -> 0+   (coord enters support)
  (I) inactive-side approach lambda_1 = c_1 (1 + delta), delta -> 0+  (coord stays zero)

against the exact selections  h_null,1 = 0  and  h_sc,1 = -Phi'(x;-e_1).

Usage
-----
    uv run python expes_fb/expe7_certificate_soundness/smoothing_limit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from synthetic import closed_form_instance          # noqa: E402

RESULTS_DIR = HERE / 'results'


def _phi_grad_x1(y1, y2, inst):
    """dPhi/dy_1 at (y1, y2) for the closed-form held-out objective."""
    G, d = inst['G_val'], inst['d_val']
    return float(G[0, 0] * y1 + G[0, 1] * y2 - d[0])


def huber_solution(c, lam, mu):
    """Minimizer of 1/2 (y-c)^2 + lam * g_mu(y), g_mu = Huber, closed form."""
    # quadratic region |y|<=mu:  y = c mu / (mu + lam)
    y_qr = c * mu / (mu + lam)
    if abs(y_qr) <= mu:
        return y_qr, 'quad'
    # linear region: y = c - lam sign(c)
    y_lin = c - lam * np.sign(c)
    return y_lin, 'lin'


def huber_dy_dlam(c, lam, mu):
    """d y_mu / d lambda in the region the biactive coordinate occupies (quad)."""
    y_qr = c * mu / (mu + lam)
    if abs(y_qr) <= mu:
        return -c * mu / (mu + lam) ** 2          # quadratic region
    return -np.sign(c)                            # linear (active) region


def main():
    inst = closed_form_instance()
    c1, lam1 = inst['c'][0], inst['lam'][0]       # c1 = lam1 = 0.6 (biactive)
    y2 = inst['y'][1]                             # active feature, ~0.7
    h_sc = inst['h_sc_cf']                        # +0.372
    print(f"instance: c_1={c1}, lambda_1={lam1} (biactive), y_2={y2:.3f}")
    print(f"exact selections:  h_null,1 = 0.000000   h_sc,1 = {h_sc:+.6f}\n")

    rows = []

    # (H) Huber smoothing mu -> 0 : dPhi_mu/dx_1 = dPhi/dy_1 * dy/dlam * lam
    print("(H) Huber / Moreau penalty smoothing, mu -> 0:")
    for mu in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        y1, region = huber_solution(c1, lam1, mu)
        dphi_dy1 = _phi_grad_x1(y1, y2, inst)
        dydx1 = huber_dy_dlam(c1, lam1, mu) * lam1
        h = dphi_dy1 * dydx1
        rows.append(dict(process='huber', param=mu, y1=y1, h=h))
        print(f"   mu={mu:.0e}   y_1={y1:.3e} ({region})   dPhi/dx_1 = {h:+.6f}")

    # (A) active-side lambda_1 = c1(1-delta): coordinate enters support (exact Lasso)
    print("\n(A) active-side  lambda_1 = c_1 (1 - delta), delta -> 0+:")
    for delta in [1e-1, 1e-2, 1e-3, 1e-4]:
        lam = c1 * (1 - delta)
        y1 = c1 - lam                             # soft-threshold, active
        dphi_dy1 = _phi_grad_x1(y1, y2, inst)
        dydx1 = -1.0 * lam                        # dy/dlam=-1 (active), dlam/dx=lam
        h = dphi_dy1 * dydx1
        rows.append(dict(process='active', param=delta, y1=y1, h=h))
        print(f"   delta={delta:.0e}   y_1={y1:.3e}   dPhi/dx_1 = {h:+.6f}")

    # (I) inactive-side lambda_1 = c1(1+delta): coordinate stays zero
    print("\n(I) inactive-side  lambda_1 = c_1 (1 + delta), delta -> 0+:")
    for delta in [1e-1, 1e-2, 1e-3, 1e-4]:
        y1 = 0.0                                  # still below threshold
        h = 0.0
        rows.append(dict(process='inactive', param=delta, y1=y1, h=h))
        print(f"   delta={delta:.0e}   y_1=0.000e+00   dPhi/dx_1 = {h:+.6f}")

    print("\n" + "=" * 60)
    print("LIMITS:")
    print(f"  (H) Huber smoothing      -> {0.0:+.4f}   == h_null  (NULL)")
    print(f"  (A) active-side          -> {h_sc:+.4f}   == h_sc    (SC)")
    print(f"  (I) inactive-side        -> {0.0:+.4f}   == h_null  (NULL)")
    print("\nVERDICT: the vanishing-smoothing limit is NOT unique. Symmetric")
    print("penalty smoothing (Huber/Moreau) converges to the NULL selection;")
    print("SC is the active-side (support-entry) one-sided limit only.")
    print("=> #3 does not single out SC; do NOT use a smoothing-limit argument.")

    pd.DataFrame(rows).to_pickle(RESULTS_DIR / 'smoothing_limit.pkl')


if __name__ == '__main__':
    main()
