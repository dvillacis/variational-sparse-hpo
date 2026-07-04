import numpy as np
from numpy.linalg import norm

from sparse_ho.optimizers.base import BaseOptimizer


def _dot(a, b):
	try:
		return float(np.dot(a.ravel(), b.ravel()))
	except Exception:
		return float(a * b)


def _save_eval_state(get_val_grad):
	save_state = getattr(get_val_grad, "save_state", None)
	if callable(save_state):
		return save_state()
	return None


def _restore_eval_state(get_val_grad, state):
	restore_state = getattr(get_val_grad, "restore_state", None)
	if state is not None and callable(restore_state):
		restore_state(state)


class TrustRegion(BaseOptimizer):
	"""Trust-region optimizer for the outer problem.

	This method builds a first-order (linear) local model of the objective and
	proposes steps constrained in an Euclidean trust region.

	Parameters
	----------
	n_outer: int, optional (default=100)
		Maximum number of outer iterations.
	radius0: float, optional (default=1.0)
		Initial trust-region radius (in log-hyperparameter space).
	radius_min: float, optional (default=1e-12)
		Minimal trust-region radius.
	radius_max: float, optional (default=1e2)
		Maximal trust-region radius.
	eta_accept: float, optional (default=0.1)
		Acceptance threshold for the ratio of actual/predicted decrease.
	eta_expand: float, optional (default=0.75)
		Threshold above which the trust-region radius can be expanded.
	gamma_dec: float, optional (default=0.25)
		Multiplicative decrease factor for the trust-region radius.
	gamma_inc: float, optional (default=2.0)
		Multiplicative increase factor for the trust-region radius.
	verbose: bool, optional (default=False)
		Verbosity.
		tol : float, optional (default=1e-5)
			Stopping tolerance on the outer gradient norm.
	tol_decrease: bool, optional (default=None)
		If not None, uses a geometric schedule from 1e-2 to `tol`.
	t_max: float, optional (default=10_000)
		Maximum running time threshold in seconds.
	"""

	def __init__(
			self, n_outer=100, radius0=1.0, radius_min=1e-12, radius_max=1e2,
			eta_accept=0.1, eta_expand=0.75, gamma_dec=0.25, gamma_inc=2.0,
			verbose=False, tol=1e-5, tol_decrease=None, t_max=10_000):
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
		self.history_ = []
		self.termination_reason_ = None
		self.final_radius_ = float(radius0)
		self.n_fallback_tried_ = 0
		self.n_fallback_accepted_ = 0

	def _grad_search(
			self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):

		is_multiparam = isinstance(log_alpha0, np.ndarray)
		if is_multiparam:
			log_alphak = log_alpha0.copy()
		else:
			log_alphak = log_alpha0

		log_alphak = proj_hyperparam(log_alphak)

		if self.tol_decrease is not None:
			tols = np.geomspace(1e-2, self.tol, num=self.n_outer)
		else:
			tols = np.ones(self.n_outer) * self.tol

		radius = float(self.radius0)
		self.history_ = []
		self.termination_reason_ = None
		self.final_radius_ = radius
		self.n_fallback_tried_ = 0
		self.n_fallback_accepted_ = 0
		value_outer, grad_outer = _get_val_grad(log_alphak, tols[0], monitor)
		# Flag: when True the current (value_outer, grad_outer) was obtained from
		# a trial evaluation that has already been accepted and the monitor has
		# already been called, so we can skip the re-evaluation at the top of the
		# next iteration.
		_skip_reeval = False

		for i, tol in enumerate(tols):
			if i > 0:
				if _skip_reeval:
					_skip_reeval = False
				else:
					value_outer, grad_outer = _get_val_grad(log_alphak, tol, monitor)

			grad_norm = norm(grad_outer)
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
				"stop_reason": None,
				"fallback_tried": 0,
				"fallback_accepted": 0,
				"rho_fallback": np.nan,
			}
			if not np.isfinite(grad_norm) or grad_norm < self.tol:
				record["stop_reason"] = "stationary"
				self.history_.append(record)
				self.termination_reason_ = "stationary"
				self.final_radius_ = float(radius)
				break

			step = -(radius / grad_norm) * grad_outer
			trial = proj_hyperparam(log_alphak + step)
			step_eff = trial - log_alphak
			step_norm = norm(step_eff)
			record["step_norm"] = float(step_norm)
			if step_norm < self.step_tol:
				record["stop_reason"] = "step_tol"
				self.history_.append(record)
				self.termination_reason_ = "step_tol"
				self.final_radius_ = float(radius)
				break

			pred = -_dot(grad_outer, step_eff)
			record["predicted_decrease"] = float(pred)
			if (not np.isfinite(pred)) or pred <= 1e-18:
				radius = max(self.radius_min, self.gamma_dec * radius)
				record["accepted"] = 0
				record["radius_after"] = float(radius)
				accepted = False
				if self.verbose:
					print(
						"Iteration %i/%i || " % (i + 1, self.n_outer) +
						"rejected (nonpositive predicted decrease) || " +
						"radius %.2e" % radius
					)
				record["stop_reason"] = (
					"radius_min" if radius <= self.radius_min else None
				)
				self.history_.append(record)
				self.final_radius_ = float(radius)
				if radius <= self.radius_min:
					self.termination_reason_ = "radius_min"
					break
				if len(monitor.times) > 0 and monitor.times[-1] > self.t_max:
					self.termination_reason_ = "t_max"
					break
				continue

			# Trial point evaluation should not mutate warm starts when rejected.
			trial_state = _save_eval_state(_get_val_grad)
			value_trial, grad_trial = _get_val_grad(trial, tol, None)
			ared = value_outer - value_trial
			rho = ared / pred
			record["rho"] = float(rho)
			record["actual_decrease"] = float(ared)

			if rho < self.eta_accept:
				_restore_eval_state(_get_val_grad, trial_state)
				record["accepted"] = 0
				accepted = False
				# Safeguard: retry the rejected step along the fallback
				# direction at the same radius before contracting.
				(accepted, log_alphak, value_outer, grad_outer, radius,
					_skip_reeval) = self._try_fallback_step(
					_get_val_grad, proj_hyperparam, monitor, log_alphak,
					value_outer, grad_outer, radius, trial, tol, record)
				if not accepted:
					radius = max(self.radius_min, self.gamma_dec * radius)
					if radius <= self.radius_min:
						record["radius_after"] = float(radius)
						record["stop_reason"] = "radius_min"
						self.history_.append(record)
						self.termination_reason_ = "radius_min"
						self.final_radius_ = float(radius)
						break
			else:
				accepted = True
				log_alphak = trial
				value_outer = value_trial
				grad_outer = grad_trial
				# Notify monitor of the accepted point now, so the next
				# iteration can skip re-evaluating _get_val_grad at this point.
				if monitor is not None:
					monitor(value_outer, grad_outer, alpha=np.exp(log_alphak))
				_skip_reeval = True

				if (rho > self.eta_expand) and (norm(step_eff) >= 0.95 * radius):
					radius = min(self.radius_max, self.gamma_inc * radius)
			record["radius_after"] = float(radius)

			if self.verbose:
				print(
					"Iteration %i/%i || " % (i + 1, self.n_outer) +
					"Value outer criterion: %.2e || " % value_outer +
					"norm grad %.2e || " % grad_norm +
					"rho %.2e || " % rho +
					"accepted %s || " % accepted +
					"radius %.2e" % radius
				)

			self.history_.append(record)
			self.final_radius_ = float(radius)
			if len(monitor.times) > 0 and monitor.times[-1] > self.t_max:
				self.termination_reason_ = "t_max"
				break
		else:
			self.termination_reason_ = "completed"

		return log_alphak, value_outer, grad_outer

	def _try_fallback_step(
			self, _get_val_grad, proj_hyperparam, monitor, log_alphak,
			value_outer, grad_outer, radius, rejected_trial, tol, record):
		"""Retry a rejected step along the fallback direction, same radius.

		The fallback gradient is recomputed at the current iterate (warm
		started, so the inner solve is essentially free) and a trial step is
		taken along it with the current radius. The retry is skipped when the
		fallback trial coincides with the rejected one (identical gradients,
		e.g. an empty biactive set makes both oracles agree).

		Returns (accepted, log_alphak, value_outer, grad_outer, radius,
		skip_reeval).
		"""
		not_accepted = (
			False, log_alphak, value_outer, grad_outer, radius, False)
		fallback = getattr(_get_val_grad, "fallback", None)
		if fallback is None:
			return not_accepted
		self.n_fallback_tried_ += 1
		record["fallback_tried"] = 1
		_, grad_fb = fallback(log_alphak, tol, None)
		grad_fb_norm = norm(grad_fb)
		if not np.isfinite(grad_fb_norm) or grad_fb_norm == 0:
			return not_accepted
		step = -(radius / grad_fb_norm) * grad_fb
		trial = proj_hyperparam(log_alphak + step)
		step_eff = trial - log_alphak
		pred = -_dot(grad_fb, step_eff)
		if (np.allclose(trial, rejected_trial) or not np.isfinite(pred)
				or pred <= 1e-18 or norm(step_eff) < self.step_tol):
			return not_accepted
		trial_state = _save_eval_state(_get_val_grad)
		value_trial, grad_trial = _get_val_grad(trial, tol, None)
		ared = value_outer - value_trial
		rho = ared / pred
		record["rho_fallback"] = float(rho)
		if rho < self.eta_accept:
			_restore_eval_state(_get_val_grad, trial_state)
			return not_accepted
		self.n_fallback_accepted_ += 1
		record["fallback_accepted"] = 1
		record["accepted"] = 1
		if monitor is not None:
			monitor(value_trial, grad_trial, alpha=np.exp(trial))
		if (rho > self.eta_expand) and (norm(step_eff) >= 0.95 * radius):
			radius = min(self.radius_max, self.gamma_inc * radius)
		return True, trial, value_trial, grad_trial, radius, True
