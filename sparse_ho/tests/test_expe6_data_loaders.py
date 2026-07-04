import numpy as np
from scipy import sparse

from expes_fb.expe6_real_world.data_loaders import (
    _normalize_binary_labels,
    _prepare_binary_dataset,
    get_dataset,
    load_rcv1,
)


class _FakeRCV1:
    def __init__(self, X, target):
        self.data = X
        self.target = target


def test_load_rcv1_casts_labels_before_pm1_mapping(monkeypatch):
    import sklearn.datasets

    X = sparse.csr_matrix(np.eye(4))
    target = sparse.csr_matrix(
        np.array(
            [
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
            ],
            dtype=np.uint8,
        )
    )

    def _fake_fetch_rcv1(*args, **kwargs):
        return _FakeRCV1(X, target)

    monkeypatch.setattr(sklearn.datasets, "fetch_rcv1", _fake_fetch_rcv1)

    _, y = load_rcv1()

    np.testing.assert_array_equal(np.unique(y), np.array([-1.0, 1.0]))
    np.testing.assert_array_equal(y, np.array([-1.0, 1.0, -1.0, 1.0]))


def test_normalize_binary_labels_handles_zero_one_uint8():
    y = np.array([0, 1, 0, 1], dtype=np.uint8)
    np.testing.assert_array_equal(
        _normalize_binary_labels(y),
        np.array([-1.0, 1.0, -1.0, 1.0]),
    )


def test_normalize_binary_labels_handles_pm1_passthrough():
    y = np.array([-1, 1, -1, 1], dtype=np.int64)
    np.testing.assert_array_equal(
        _normalize_binary_labels(y),
        np.array([-1.0, 1.0, -1.0, 1.0]),
    )


def test_normalize_binary_labels_handles_nonstandard_binary_encoding():
    y = np.array([1, 2, 2, 1], dtype=np.int64)
    np.testing.assert_array_equal(
        _normalize_binary_labels(y),
        np.array([-1.0, 1.0, 1.0, -1.0]),
    )


def test_prepare_binary_dataset_rejects_nonbinary_labels():
    X = sparse.csr_matrix(np.eye(3))
    y = np.array([0, 1, 2], dtype=np.int64)
    try:
        _prepare_binary_dataset(X, y, name="toy")
    except ValueError as exc:
        assert "Expected exactly 2 classes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-binary labels")


def test_get_dataset_dispatches_manual_sparse_names(monkeypatch):
    calls = []

    def _fake_load_libsvm(name, data_dir=None):
        calls.append((name, data_dir))
        return sparse.csr_matrix(np.eye(2)), np.array([-1.0, 1.0])

    monkeypatch.setattr(
        "expes_fb.expe6_real_world.data_loaders.load_libsvm",
        _fake_load_libsvm,
    )

    for name in [
        "phishing", "w8a", "rcv1.binary", "rcv1_train.binary",
        "real-sim", "news20.binary"
    ]:
        X, y = get_dataset(name)
        assert X.shape == (2, 2)
        np.testing.assert_array_equal(y, np.array([-1.0, 1.0]))

    assert [name for name, _ in calls] == [
        "phishing", "w8a", "rcv1.binary", "rcv1_train.binary",
        "real-sim", "news20.binary"
    ]
