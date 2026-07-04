import numpy as np
from scipy.sparse import csc_matrix
import pytest
import celer
from celer.datasets import make_correlated_data

from sparse_ho.utils import Monitor
from sparse_ho.models import Lasso

from sparse_ho import Forward
from sparse_ho import ImplicitForward
from sparse_ho import Implicit
from sparse_ho.criterion import HeldOutMSE
# XXX TODO test FiniteDiffMonteCarloSure crtiterion
from sparse_ho.ho import grad_search
from sparse_ho.optimizers import (
    LineSearch, GradientDescent, TrustRegion, TrustRegionBFGS)

n_samples = 100
n_features = 100
snr = 3
corr = 0.5

X, y, _ = make_correlated_data(
    n_samples, n_features, corr=corr, snr=snr, random_state=42)
sigma_star = 0.1
X_train_s = csc_matrix(X)

idx_train = np.arange(0, 50)
idx_val = np.arange(50, 100)

alpha_max = np.max(np.abs(X[idx_train, :].T @ y[idx_train])) / len(idx_train)
p_alpha = 0.7
alpha0 = p_alpha * alpha_max
# log_alpha = np.log(alpha)

log_alphas = np.log(alpha_max * np.geomspace(1, 0.1))
tol = 1e-16
max_iter = 1000

# dict_log_alpha0 = {}
# dict_log_alpha0["lasso"] = log_alpha
# tab = np.linspace(1, 1000, n_features)
# dict_log_alpha0["wlasso"] = log_alpha + np.log(tab / tab.max())

models = [
    Lasso(estimator=None),
]

estimator = celer.Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True)
models_custom = [
    Lasso(estimator=estimator),
]

Optimizers = [LineSearch, GradientDescent]


@pytest.mark.parametrize('Optimizer', Optimizers)
@pytest.mark.parametrize('model', models)
@pytest.mark.parametrize('crit', ['MSE', 'sure'])
def test_grad_search(Optimizer, model, crit):
    """check that the paths are the same in the line search"""
    n_outer = 2

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor1 = Monitor()
    algo = Forward()
    optimizer = Optimizer(n_outer=n_outer, tol=1e-16)
    grad_search(
        algo, criterion, model, optimizer, X, y, alpha0, monitor1)

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor2 = Monitor()
    algo = Implicit()
    optimizer = Optimizer(n_outer=n_outer, tol=1e-16)
    grad_search(algo, criterion, model, optimizer, X, y, alpha0, monitor2)

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor3 = Monitor()
    algo = ImplicitForward(tol_jac=1e-8, n_iter_jac=5000)
    optimizer = Optimizer(n_outer=n_outer, tol=1e-16)
    grad_search(algo, criterion, model, optimizer, X, y, alpha0, monitor3)

    np.testing.assert_allclose(
        np.array(monitor1.alphas), np.array(monitor3.alphas))
    np.testing.assert_allclose(
        np.array(monitor1.grads), np.array(monitor3.grads), rtol=1e-5)
    np.testing.assert_allclose(
        np.array(monitor1.objs), np.array(monitor3.objs))
    assert not np.allclose(
        np.array(monitor1.times), np.array(monitor3.times))


def test_trust_region_bfgs_runs():
    criterion = HeldOutMSE(idx_train, idx_val)
    monitor = Monitor()
    algo = Implicit()
    optimizer = TrustRegionBFGS(n_outer=3, tol=1e-16, radius0=0.5)

    grad_search(
        algo, criterion, models[0], optimizer, X, y, alpha0, monitor)

    assert len(monitor.objs) >= 1
    assert np.all(np.isfinite(np.array(monitor.objs)))
    assert len(optimizer.history_) >= 1
    assert optimizer.termination_reason_ in {
        'completed', 'stationary', 'step_tol', 'radius_min', 't_max'}
    first_record = optimizer.history_[0]
    assert 'step_type' in first_record
    assert 'bfgs_update' in first_record


def test_trust_region_rejected_trial_restores_state():
    state = {"point": None}

    def _get_val_grad(log_alpha, tol, monitor):
        x = float(np.asarray(log_alpha, dtype=float).reshape(-1)[0])
        state["point"] = x
        value = 0.0 if abs(x) < 1e-15 else 10.0
        grad = np.array([1.0])
        if monitor is not None:
            monitor(value, grad, alpha=np.array([np.exp(x)]))
        return value, grad

    def _save_state():
        return {"point": state["point"]}

    def _restore_state(saved_state):
        state["point"] = saved_state["point"]

    _get_val_grad.save_state = _save_state
    _get_val_grad.restore_state = _restore_state

    def _proj(log_alpha):
        return np.asarray(log_alpha, dtype=float)

    monitor = Monitor()
    optimizer = TrustRegion(n_outer=1, radius0=0.5, tol=1e-16)
    optimizer._grad_search(_get_val_grad, _proj, np.array([0.0]), monitor)

    assert optimizer.history_[0]["accepted"] == 0
    assert state["point"] == 0.0


if __name__ == '__main__':
    models = [
        Lasso(estimator=None)]
    crits = ['sure']
    # crits = ['MSE']
    for model in models:
        for crit in crits:
            test_grad_search(model, crit)
