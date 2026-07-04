"""Synthetic three-group dataset for Experiment 2.

Feature layout (columns of X)
------------------------------
[0 : s_signal]              Signal features  — nonzero ±1 ground truth.
[s_signal : s_signal+s_corr] Correlated noise — zero ground truth, correlated
                              with the first s_corr signal columns.
[s_signal+s_corr : m]        Pure noise       — iid Gaussian, zero ground truth.

The design is overparameterized (n << m) and uses a 60/20/20 train/val/test
split returned as index arrays into the full (X, y).
"""

import numpy as np


def make_three_group_dataset(
    n, m, s_signal, s_corr, corr=0.7, noise_std=0.1, rng=None
):
    """Generate overparameterized regression data with three feature groups.

    Parameters
    ----------
    n : int
        Total number of samples (train + val + test).
    m : int
        Number of features (should satisfy m >> n).
    s_signal : int
        Number of signal features (nonzero ground truth).
    s_corr : int
        Number of correlated noise features (zero ground truth).
        Each correlated-noise column is paired with a signal column, cycling
        through signal columns modularly (so s_corr can exceed s_signal).
    corr : float, default=0.7
        Pearson correlation between each correlated-noise column and its
        paired signal column.
    noise_std : float, default=0.1
        Standard deviation of additive Gaussian observation noise.
    rng : np.random.Generator or None
        Random generator.  Falls back to ``np.random.default_rng(0)``.

    Returns
    -------
    X : ndarray, shape (n, m)
    y : ndarray, shape (n,)
    beta_true : ndarray, shape (m,)
        Ground-truth coefficients (nonzero only on signal features).
    groups : dict
        Maps ``'signal'``, ``'corr_noise'``, ``'pure_noise'`` to
        ``np.ndarray`` of column indices.
    idx_train : ndarray of int  (60 % of n, sorted)
    idx_val   : ndarray of int  (20 % of n, sorted)
    idx_test  : ndarray of int  (20 % of n, sorted)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if s_signal + s_corr > m:
        raise ValueError("s_signal + s_corr must be <= m")
    if s_corr < 0:
        raise ValueError("s_corr must be non-negative")

    n_pure = m - s_signal - s_corr

    # --- design matrix -------------------------------------------------------
    X_sig = rng.standard_normal((n, s_signal))

    # each correlated-noise column is paired with a signal column, cycling
    # through signal columns modularly so s_corr can exceed s_signal
    sig_idx = np.arange(s_corr) % s_signal
    X_corr = (
        corr * X_sig[:, sig_idx]
        + np.sqrt(1.0 - corr ** 2) * rng.standard_normal((n, s_corr))
    )

    X_pure = rng.standard_normal((n, n_pure))
    X = np.hstack([X_sig, X_corr, X_pure])

    # --- ground truth: ±1 on signal features only ----------------------------
    beta_true = np.zeros(m)
    beta_true[:s_signal] = rng.choice([-1.0, 1.0], size=s_signal)

    y = X @ beta_true + noise_std * rng.standard_normal(n)

    # --- 60 / 20 / 20 split (by row) ----------------------------------------
    perm = rng.permutation(n)
    n_tr  = int(0.6 * n)
    n_val = int(0.2 * n)
    idx_train = np.sort(perm[:n_tr])
    idx_val   = np.sort(perm[n_tr:n_tr + n_val])
    idx_test  = np.sort(perm[n_tr + n_val:])

    groups = {
        'signal':    np.arange(s_signal),
        'corr_noise': np.arange(s_signal, s_signal + s_corr),
        'pure_noise': np.arange(s_signal + s_corr, m),
    }

    return X, y, beta_true, groups, idx_train, idx_val, idx_test
