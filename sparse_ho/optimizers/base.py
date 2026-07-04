from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):

    @abstractmethod
    def __init__(cls):
        pass

    @abstractmethod
    def _grad_search(
            self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):
        return NotImplemented

    @staticmethod
    def _plateau_stop(best_hist, patience, rtol):
        """Outer-loop convergence test on the validation objective.

        Returns True when the best-so-far outer objective has improved by less
        than ``rtol`` (relative) over the last ``patience`` iterations, i.e. the
        outer optimization has effectively converged and further iterations only
        cost wall-clock without improving the solution.

        Applying the same test to every optimizer makes the outer budget a
        genuine convergence criterion rather than a fixed iteration count: a
        second-order method that reaches its solution early stops early, while a
        slow first-order method that is still descending runs to the cap.

        Parameters
        ----------
        best_hist : list of float
            Best (minimum) outer objective seen after each outer iteration.
        patience : int or None
            Window length. ``None`` (or <= 0) disables the test.
        rtol : float
            Relative-improvement threshold over the window.
        """
        if patience is None or patience <= 0 or len(best_hist) <= patience:
            return False
        cur = best_hist[-1]
        prev = best_hist[-1 - patience]
        if not (np.isfinite(cur) and np.isfinite(prev)):
            return False
        denom = max(abs(cur), 1e-12)
        return (prev - cur) <= rtol * denom
