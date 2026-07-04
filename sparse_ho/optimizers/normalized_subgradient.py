import numpy as np
from numpy.linalg import norm

from sparse_ho.optimizers.base import BaseOptimizer


class NormalizedSubgradient(BaseOptimizer):
    """Projected normalized subgradient (NBA) for the outer problem.

    Each outer step takes the form

        x_{k+1} = proj( x_k - step_size * g_k / ‖g_k‖ )

    with a *fixed* step size.  No line search, no adaptivity.  This is
    intentionally simple: the method is parameter-light and serves as the
    baseline optimizer in the oracle×optimizer ablation (Experiment 3).

    Its key property is that gradient bias (e.g., from the null biactive
    policy) propagates directly into the iterate sequence with no
    self-correction mechanism.

    Parameters
    ----------
    n_outer : int, optional (default=60)
        Maximum number of outer iterations.
    step_size : float, optional (default=1.0)
        Fixed step length in log-hyperparameter space.  Comparable to
        TrustRegion's ``radius0``; use the same value for a fair comparison.
    tol : float, optional (default=1e-5)
        Stopping tolerance on the outer gradient norm.
    verbose : bool, optional (default=False)
        Verbosity.
    t_max : float, optional (default=10_000)
        Maximum running time threshold in seconds.
    """

    def __init__(
            self, n_outer=60, step_size=1.0, tol=1e-5,
            verbose=False, t_max=10_000):
        self.n_outer = n_outer
        self.step_size = step_size
        self.tol = tol
        self.verbose = verbose
        self.t_max = t_max
        self.history_ = []
        self.termination_reason_ = None

    def _grad_search(
            self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):

        if isinstance(log_alpha0, np.ndarray):
            log_alphak = log_alpha0.copy()
        else:
            log_alphak = log_alpha0

        log_alphak = proj_hyperparam(log_alphak)
        self.history_ = []
        self.termination_reason_ = None
        value_outer, grad_outer = None, None

        for i in range(self.n_outer):
            value_outer, grad_outer = _get_val_grad(log_alphak, self.tol, monitor)
            grad_norm = norm(grad_outer)

            record = {
                'iteration': i,
                'grad_norm': float(grad_norm),
                'value': float(value_outer),
                'stop_reason': None,
            }

            if not np.isfinite(grad_norm) or grad_norm < self.tol:
                record['stop_reason'] = 'stationary'
                self.history_.append(record)
                self.termination_reason_ = 'stationary'
                break

            # fixed-step normalized subgradient
            log_alphak = proj_hyperparam(
                log_alphak - self.step_size * grad_outer / grad_norm
            )

            if self.verbose:
                print(
                    "Iteration %i/%i || " % (i + 1, self.n_outer) +
                    "Value outer criterion: %.2e || " % value_outer +
                    "norm grad %.2e" % grad_norm
                )

            self.history_.append(record)

            if len(monitor.times) > 0 and monitor.times[-1] > self.t_max:
                self.termination_reason_ = 't_max'
                break
        else:
            self.termination_reason_ = 'completed'

        return log_alphak, value_outer, grad_outer
