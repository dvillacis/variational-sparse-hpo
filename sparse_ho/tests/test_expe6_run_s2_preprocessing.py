import numpy as np

from expes_fb.expe6_real_world.run_s2 import _preprocess_features_for_split


def test_preprocess_breast_cancer_drops_id_and_scales():
    X = np.array(
        [
            [1000.0, 1.0, 10.0],
            [2000.0, 2.0, 20.0],
            [3000.0, 3.0, 30.0],
            [4000.0, 4.0, 40.0],
        ]
    )
    idx_train = np.array([0, 1])

    X_proc, notes = _preprocess_features_for_split(
        'breast-cancer', X, idx_train
    )

    assert X_proc.shape == (4, 2)
    assert "dropped column 0 (ID-like feature)" in notes
    assert "standardized dense features using train split stats" in notes

    train = X_proc[idx_train]
    np.testing.assert_allclose(train.mean(axis=0), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(train.std(axis=0), np.ones(2), atol=1e-12)


def test_preprocess_dense_uses_train_statistics_only():
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [100.0, 100.0],
        ]
    )
    idx_train = np.array([0, 1])

    X_proc, notes = _preprocess_features_for_split('toy-dense', X, idx_train)

    assert X_proc.shape == X.shape
    assert "standardized dense features using train split stats" in notes
    np.testing.assert_allclose(X_proc[idx_train].mean(axis=0), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(X_proc[idx_train].std(axis=0), np.ones(2), atol=1e-12)
    assert np.all(X_proc[2] > 90.0)
