import numpy as np
from numpy.linalg import LinAlgError, norm

from sparse_ho.optimizers.base import BaseOptimizer


def _dot(a, b):
    try:
        return float(np.dot(a.ravel(), b.ravel()))
    except Exception:
        return float(a * b)


def _as_vector(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.copy()


def _restore_shape(x, scalar_input):
    if scalar_input:
        return float(np.asarray(x, dtype=float).reshape(-1)[0])
    return np.asarray(x, dtype=float).copy()


def _save_eval_state(get_val_grad):
    save_state = getattr(get_val_grad, "save_state", None)
    if callable(save_state):
        return save_state()
    return None


def _restore_eval_state(get_val_grad, state):
    restore_state = getattr(get_val_grad, "restore_state", None)
    if state is not None and callable(restore_state):
        restore_state(state)


class TrustRegionBFGS(BaseOptimizer):
    """Trust-region optimizer with a damped BFGS quadratic model.

    The local model is

        m_k(s) = f_k + g_k^T s + 0.5 * s^T B_k s

    where ``B_k`` is a dense positive-definite Hessian approximation updated by
    damped BFGS after accepted steps only. The trust-region subproblem is
    approximately solved with a dogleg strategy.
    """

    def __init__(
            self, n_outer=100, radius0=1.0, radius_min=1e-12, radius_max=1e2,
            eta_accept=0.1, eta_expand=0.75, gamma_dec=0.25, gamma_inc=2.0,
            verbose=False, tol=1e-5, tol_decrease=None, t_max=10_000,
            init_scale=1.0, powell_damping=0.2, min_curvature=1e-12):
        self.n_outer = n_outer
        self.radius0 = radius0
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.eta_accept = eta_accept
        self.eta_expand = eta_expand
        self.gamma_dec = gamma_dec
        self.gamma_inc = gamma_inc
        self.verbose = verbose
        self.tol = tol
        self.tol_decrease = tol_decrease
        self.t_max = t_max
        self.step_tol = 1e-12
        self.init_scale = float(init_scale)
        self.powell_damping = float(powell_damping)
        self.min_curvature = float(min_curvature)
        self.history_ = []
        self.termination_reason_ = None
        self.final_radius_ = float(radius0)
        self.final_hessian_ = None

    def _initial_hessian(self, dim):
        return self.init_scale * np.eye(dim)

    def _newton_step(self, B_k, grad_outer):
        try:
            step = np.linalg.solve(B_k, -grad_outer)
        except LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None
        if _dot(grad_outer, step) >= -self.min_curvature:
            return None
        return step

    def _cauchy_step(self, B_k, grad_outer, radius, grad_norm):
        if grad_norm <= self.step_tol:
            return np.zeros_like(grad_outer)
        B_g = B_k @ grad_outer
        grad_quad = _dot(grad_outer, B_g)
        if (not np.isfinite(grad_quad)) or grad_quad <= self.min_curvature:
            return -(radius / grad_norm) * grad_outer
        alpha_sd = (grad_norm ** 2) / grad_quad
        step = -alpha_sd * grad_outer
        step_norm = norm(step)
        if step_norm >= radius:
            return -(radius / grad_norm) * grad_outer
        return step

    def _dogleg_step(self, B_k, grad_outer, radius):
        grad_norm = norm(grad_outer)
        cauchy_step = self._cauchy_step(B_k, grad_outer, radius, grad_norm)
        cauchy_norm = norm(cauchy_step)
        if cauchy_norm >= radius - self.step_tol:
            return cauchy_step, "cauchy_boundary"

        newton_step = self._newton_step(B_k, grad_outer)
        if newton_step is None:
            return cauchy_step, "cauchy"
        if norm(newton_step) <= radius:
            return newton_step, "newton"

        segment = newton_step - cauchy_step
        a_quad = _dot(segment, segment)
        b_quad = 2.0 * _dot(cauchy_step, segment)
        c_quad = _dot(cauchy_step, cauchy_step) - radius ** 2
        if a_quad <= self.min_curvature:
            return cauchy_step, "cauchy"
        disc = max(b_quad ** 2 - 4.0 * a_quad * c_quad, 0.0)
        tau = (-b_quad + np.sqrt(disc)) / (2.0 * a_quad)
        tau = min(max(float(tau), 0.0), 1.0)
        return cauchy_step + tau * segment, "dogleg"

    def _update_hessian(self, B_k, step, grad_diff):
        step_norm = norm(step)
        if step_norm < self.step_tol:
            return B_k, "skipped_step_tol"

        B_step = B_k @ step
        step_B_step = _dot(step, B_step)
        if (not np.isfinite(step_B_step)) or step_B_step <= self.min_curvature:
            return self._initial_hessian(len(step)), "reset_model"

        grad_curv = _dot(step, grad_diff)
        status = "updated"
        if grad_curv < self.powell_damping * step_B_step:
            denom = step_B_step - grad_curv
            if denom <= self.min_curvature:
                return self._initial_hessian(len(step)), "reset_model"
            theta = (1.0 - self.powell_damping) * step_B_step / denom
            grad_diff = theta * grad_diff + (1.0 - theta) * B_step
            status = "damped"

        step_grad = _dot(step, grad_diff)
        if (not np.isfinite(step_grad)) or step_grad <= self.min_curvature:
            return self._initial_hessian(len(step)), "reset_model"

        B_next = (
            B_k
            - np.outer(B_step, B_step) / step_B_step
            + np.outer(grad_diff, grad_diff) / step_grad
        )
        B_next = 0.5 * (B_next + B_next.T)
        return B_next, status

    def _grad_search(
            self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):
        is_scalar = not isinstance(log_alpha0, np.ndarray)
        log_alphak = _as_vector(log_alpha0)
        log_alphak = _as_vector(
            proj_hyperparam(_restore_shape(log_alphak, is_scalar)))

        if self.tol_decrease is not None:
            tols = np.geomspace(1e-2, self.tol, num=self.n_outer)
        else:
            tols = np.ones(self.n_outer) * self.tol

        radius = float(self.radius0)
        B_k = self._initial_hessian(len(log_alphak))
        self.history_ = []
        self.termination_reason_ = None
        self.final_radius_ = radius
        value_outer, grad_outer = _get_val_grad(
            _restore_shape(log_alphak, is_scalar), tols[0], monitor)
        grad_outer_vec = _as_vector(grad_outer)

        for i, tol in enumerate(tols):
            if i > 0:
                value_outer, grad_outer = _get_val_grad(
                    _restore_shape(log_alphak, is_scalar), tol, monitor)
                grad_outer_vec = _as_vector(grad_outer)

            grad_norm = norm(grad_outer_vec)
            record = {
                "iteration": i,
                "radius": float(radius),
                "radius_after": float(radius),
                "grad_norm": float(grad_norm),
                "rho": np.nan,
                "accepted": 1,
                "predicted_decrease": np.nan,
                "actual_decrease": np.nan,
                "step_norm": 0.0,
                "step_type": None,
                "bfgs_update": None,
                "stop_reason": None,
            }
            if not np.isfinite(grad_norm) or grad_norm < self.tol:
                record["stop_reason"] = "stationary"
                self.history_.append(record)
                self.termination_reason_ = "stationary"
                self.final_radius_ = float(radius)
                break

            step, step_type = self._dogleg_step(B_k, grad_outer_vec, radius)
            trial = proj_hyperparam(
                _restore_shape(log_alphak + step, is_scalar))
            trial = _as_vector(trial)
            step_eff = trial - log_alphak
            step_norm = norm(step_eff)
            record["step_norm"] = float(step_norm)
            record["step_type"] = step_type
            if step_norm < self.step_tol:
                record["stop_reason"] = "step_tol"
                self.history_.append(record)
                self.termination_reason_ = "step_tol"
                self.final_radius_ = float(radius)
                break

            pred = -_dot(grad_outer_vec, step_eff)
            pred -= 0.5 * _dot(step_eff, B_k @ step_eff)
            record["predicted_decrease"] = float(pred)
            if (not np.isfinite(pred)) or pred <= 1e-18:
                radius = max(self.radius_min, self.gamma_dec * radius)
                record["accepted"] = 0
                record["radius_after"] = float(radius)
                record["bfgs_update"] = "skipped_reject"
                if radius <= self.radius_min:
                    record["stop_reason"] = "radius_min"
                    self.history_.append(record)
                    self.termination_reason_ = "radius_min"
                    self.final_radius_ = float(radius)
                    break
                self.history_.append(record)
                self.final_radius_ = float(radius)
                if len(monitor.times) > 0 and monitor.times[-1] > self.t_max:
                    self.termination_reason_ = "t_max"
                    break
                continue

            trial_state = _save_eval_state(_get_val_grad)
            value_trial, grad_trial = _get_val_grad(
                _restore_shape(trial, is_scalar), tol, None)
            grad_trial_vec = _as_vector(grad_trial)
            ared = value_outer - value_trial
            rho = ared / pred
            record["rho"] = float(rho)
            record["actual_decrease"] = float(ared)

            if rho < self.eta_accept:
                _restore_eval_state(_get_val_grad, trial_state)
                radius = max(self.radius_min, self.gamma_dec * radius)
                record["accepted"] = 0
                record["bfgs_update"] = "skipped_reject"
                if radius <= self.radius_min:
                    record["radius_after"] = float(radius)
                    record["stop_reason"] = "radius_min"
                    self.history_.append(record)
                    self.termination_reason_ = "radius_min"
                    self.final_radius_ = float(radius)
                    break
            else:
                update_status = "skipped"
                grad_diff = grad_trial_vec - grad_outer_vec
                B_k, update_status = self._update_hessian(B_k, step_eff, grad_diff)
                record["bfgs_update"] = update_status
                log_alphak = trial
                value_outer = value_trial
                grad_outer = grad_trial
                grad_outer_vec = grad_trial_vec
                if (rho > self.eta_expand) and (step_norm >= 0.95 * radius):
                    radius = min(self.radius_max, self.gamma_inc * radius)
            record["radius_after"] = float(radius)

            if self.verbose:
                print(
                    "Iteration %i/%i || " % (i + 1, self.n_outer) +
                    "Value outer criterion: %.2e || " % value_outer +
                    "norm grad %.2e || " % grad_norm +
                    "rho %.2e || " % rho +
                    "accepted %s || " % bool(record["accepted"]) +
                    "radius %.2e || " % radius +
                    "step %s" % record["step_type"]
                )

            self.history_.append(record)
            self.final_radius_ = float(radius)
            if len(monitor.times) > 0 and monitor.times[-1] > self.t_max:
                self.termination_reason_ = "t_max"
                break
        else:
            self.termination_reason_ = "completed"

        self.final_hessian_ = B_k.copy()
        return _restore_shape(log_alphak, is_scalar), value_outer, grad_outer
