"""Dataset loading utilities for Experiment 6.

Supported datasets
------------------
rcv1
    Loaded via ``sklearn.datasets.fetch_rcv1``.  Labels are binarised by
    selecting the most balanced single category from the 103-class multilabel
    target (typically achieves ~40-60% positive rate).  Labels in {-1, +1}.

rcv1.binary
    Binary LIBSVM file.  Must be downloaded manually:
        wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
        bzip2 -d rcv1_train.binary.bz2
    Place the decompressed file as ``<data_dir>/rcv1_train.binary``.
    The alias ``rcv1_train.binary`` is also accepted.

real-sim
    Binary LIBSVM file.  Must be downloaded manually:
        wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2
        bzip2 -d real-sim.bz2
    Place the decompressed file as ``<data_dir>/real-sim``.

phishing
    Binary LIBSVM file from the phishing websites benchmark.
    Download manually and place as ``<data_dir>/phishing``.

w8a
    Binary LIBSVM file from the Adult-family sparse benchmark.
    We use the training split file ``w8a`` (not ``w8a.t``) and then create
    our own random train/validation/test split inside Experiment 6.

news20.binary   (optional — very large)
    Binary LIBSVM file.  Must be downloaded manually:
        wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2
        bzip2 -d news20.binary.bz2
    Place the decompressed file as ``<data_dir>/news20.binary``.

mnist
    Loaded via ``libsvmdata`` and restricted to digits 0 vs 1 so that the
    Experiment 6 Setting 2 pipeline remains binary classification.

breast-cancer
    Loaded via ``libsvmdata``.

leukemia
    Loaded via ``libsvmdata``.

All loaders return (X, y) where
    X : scipy.sparse.csr_matrix  (or dense ndarray for small datasets)
    y : ndarray, shape (n_samples,), values in {-1, +1}
"""

from pathlib import Path

import numpy as np
from scipy.sparse import issparse


# Default cache directory (can be overridden)
DEFAULT_DATA_DIR = Path(__file__).parent / 'data'

_LIBSVM_URLS = {
    'phishing': (
        'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'
        'phishing'
    ),
    'rcv1.binary': (
        'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'
        'rcv1_train.binary.bz2'
    ),
    'rcv1_train.binary': (
        'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'
        'rcv1_train.binary.bz2'
    ),
    'real-sim': (
        'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'
        'real-sim.bz2'
    ),
    'w8a': (
        'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'
        'w8a'
    ),
    'news20.binary': (
        'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'
        'news20.binary.bz2'
    ),
}

_LIBSVM_FILENAMES = {
    'rcv1.binary': 'rcv1_train.binary',
    'rcv1_train.binary': 'rcv1_train.binary',
}

_LIBSVMDATA_BINARY_DATASETS = {
    'mnist',
    'breast-cancer',
    'leukemia',
    'diabetes',
    # degeneracy-prone candidates (biactivity screen, 2026-07-02):
    # correlated microarray probes, engineered redundant/probe features,
    # and one-hot categorical designs with complementary (tied-gradient) pairs.
    'colon-cancer',
    'duke breast-cancer',
    'gisette',
    'madelon',
    'a9a',
    'dna',
    'splice',
    'svmguide1',
    'sonar',
}

_MANUAL_BINARY_DATASETS = {
    'phishing',
    'rcv1.binary',
    'rcv1_train.binary',
    'real-sim',
    'w8a',
    'news20.binary',
}


def _normalize_binary_labels(y, positive_label=None):
    """Return labels as a float array in {-1.0, +1.0}.

    Parameters
    ----------
    y : array-like
        Binary labels in any numeric two-class encoding.
    positive_label : scalar, optional
        Explicit value to map to +1. The other class is mapped to -1.

    Notes
    -----
    If ``positive_label`` is omitted, the mapping is deterministic:
    the smaller unique label is mapped to -1 and the larger one to +1.
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError("Empty label array.")

    unique_labels = np.unique(y)
    if unique_labels.size != 2:
        raise ValueError(
            f"Expected exactly 2 classes, got {unique_labels.size}: "
            f"{unique_labels.tolist()}"
        )

    if positive_label is not None:
        positive_matches = unique_labels == positive_label
        if not np.any(positive_matches):
            raise ValueError(
                f"positive_label={positive_label!r} not found in classes "
                f"{unique_labels.tolist()}"
            )
        positive = unique_labels[positive_matches][0]
        negative = unique_labels[~positive_matches][0]
    else:
        negative, positive = unique_labels[0], unique_labels[1]

    y_pm1 = np.where(y == positive, 1.0, -1.0).astype(float, copy=False)
    if not np.all(np.isfinite(y_pm1)):
        raise ValueError("Normalized labels contain non-finite values.")
    return y_pm1


def _prepare_binary_dataset(X, y, name=None):
    """Validate shapes and normalize labels to {-1.0, +1.0}."""
    y = _normalize_binary_labels(y)
    if X.shape[0] != y.shape[0]:
        dataset_name = f" for dataset '{name}'" if name is not None else ""
        raise ValueError(
            f"Mismatched number of samples{dataset_name}: "
            f"X has {X.shape[0]}, y has {y.shape[0]}"
        )
    return X, y


# ---------------------------------------------------------------------------
# RCV1
# ---------------------------------------------------------------------------

def load_rcv1(data_dir=None):
    """Return (X, y) for RCV1 with binarised single-category labels.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory for sklearn's cache.  Defaults to ``data/`` next to this
        file.

    Returns
    -------
    X : csr_matrix, shape (n_samples, 47236)
    y : ndarray, shape (n_samples,), values in {-1, +1}
    """
    from sklearn.datasets import fetch_rcv1

    kw = {}
    if data_dir is not None:
        kw['data_home'] = str(data_dir)

    data = fetch_rcv1(subset='all', download_if_missing=True, **kw)
    X = data.data  # sparse (n, 47236)
    target = data.target  # sparse multilabel (n, 103)

    # Select the category with the most balanced positive rate
    counts = np.asarray(target.sum(axis=0)).ravel()
    n = X.shape[0]
    balance = np.abs(counts / n - 0.5)
    best_col = int(np.argmin(balance))
    y = np.asarray(target[:, best_col].todense()).ravel()
    return _prepare_binary_dataset(X, y, name='rcv1')


# ---------------------------------------------------------------------------
# Manual LIBSVM files (real-sim, news20)
# ---------------------------------------------------------------------------

def load_libsvm(name, data_dir=None):
    """Load a LIBSVM binary classification file.

    Parameters
    ----------
    name : str
        Dataset name: manual LIBSVM binary benchmark such as ``'phishing'``,
        ``'w8a'``, ``'rcv1.binary'``, ``'rcv1_train.binary'``,
        ``'real-sim'``, or ``'news20.binary'``.
    data_dir : str or Path, optional
        Directory containing the downloaded file.  Defaults to ``data/``
        next to this file.

    Returns
    -------
    X : csr_matrix
    y : ndarray, shape (n_samples,), values in {-1, +1}

    Raises
    ------
    FileNotFoundError
        If the dataset file is not found, with download instructions.
    """
    from sklearn.datasets import load_svmlight_file

    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    filename = _LIBSVM_FILENAMES.get(name, name)
    path = data_dir / filename
    if not path.exists():
        url = _LIBSVM_URLS.get(name, '(unknown)')
        download_lines = [f"  wget {url}"]
        filename = url.rsplit('/', 1)[-1]
        if filename.endswith('.bz2'):
            download_lines.append(f"  bzip2 -d {filename}")
            filename = filename[:-4]
        download_lines.append(f"  mv {filename} {data_dir}/")
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {path}.\n"
            f"Download with:\n"
            + "\n".join(download_lines)
        )

    X, y = load_svmlight_file(str(path))
    return _prepare_binary_dataset(X, y, name=name)


# ---------------------------------------------------------------------------
# libsvmdata datasets (mnist, breast-cancer, leukemia)
# ---------------------------------------------------------------------------

def load_libsvmdata_binary(name, data_dir=None):
    """Load a libsvmdata dataset and coerce it to binary labels in {-1, +1}.

    Notes
    -----
    ``data_dir`` is currently ignored for these datasets because
    ``libsvmdata`` manages its own cache location.

    For ``mnist``, only digits 0 and 1 are retained.
    """
    from libsvmdata.datasets import fetch_dataset

    X, y = fetch_dataset(name)
    y = np.asarray(y).ravel()

    if name == 'mnist':
        keep = np.logical_or(y == 0, y == 1)
        X = X[keep]
        y = y[keep]
        return _prepare_binary_dataset(X, y, name=name)

    return _prepare_binary_dataset(X, y, name=name)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_dataset(name, data_dir=None):
    """Load a named dataset.

    Parameters
    ----------
    name : str
        ``'rcv1'``, ``'rcv1.binary'``, ``'rcv1_train.binary'``,
        ``'real-sim'``, ``'news20.binary'``, ``'mnist'``,
        ``'breast-cancer'``, or ``'leukemia'``.
    data_dir : str or Path, optional

    Returns
    -------
    X : csr_matrix or ndarray
    y : ndarray, values in {-1, +1}
    """
    if name == 'rcv1':
        return load_rcv1(data_dir)
    elif name in _MANUAL_BINARY_DATASETS:
        return load_libsvm(name, data_dir)
    elif name in _LIBSVMDATA_BINARY_DATASETS:
        return load_libsvmdata_binary(name, data_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'.  "
                         f"Supported: 'rcv1', 'rcv1.binary', "
                         f"'rcv1_train.binary', 'phishing', 'w8a', "
                         f"'real-sim', 'news20.binary', 'mnist', "
                         f"'breast-cancer', 'leukemia', 'diabetes'.")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def top_variance_features(X, k):
    """Return indices of the k highest-variance columns of X."""
    if issparse(X):
        # Var = E[x^2] - E[x]^2 for each column
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
        mean_x  = np.asarray(X.mean(axis=0)).ravel()
        var = mean_sq - mean_x ** 2
    else:
        var = np.var(X, axis=0)
    return np.argsort(var)[::-1][:k]


def correlation_matrix(X):
    """Return Pearson correlation matrix (k × k) for the columns of X.

    X must be dense or will be converted to dense.
    """
    if issparse(X):
        X = np.asarray(X.todense())
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    norms = np.linalg.norm(X_c, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    X_n = X_c / norms
    return X_n.T @ X_n


def top_correlated_pairs(corr, n_pairs, exclude_indices=()):
    """Return the n_pairs pairs (i,j) with highest |corr[i,j]|, i<j,
    excluding any index in exclude_indices and using greedy pair selection
    (each index appears at most once)."""
    k = corr.shape[0]
    used = set(exclude_indices)
    pairs = []
    # Flatten upper triangle
    triu_idx = np.array([(i, j)
                         for i in range(k)
                         for j in range(i + 1, k)])
    scores = np.abs(corr[triu_idx[:, 0], triu_idx[:, 1]])
    order = np.argsort(scores)[::-1]

    for idx in order:
        i, j = triu_idx[idx]
        if i in used or j in used:
            continue
        pairs.append((int(i), int(j)))
        used.add(i)
        used.add(j)
        if len(pairs) == n_pairs:
            break

    return pairs
