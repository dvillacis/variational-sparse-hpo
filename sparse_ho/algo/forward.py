import time

import numpy as np
from scipy.sparse import issparse


class Forward():
    """Algorithm to compute the hypergradient using forward differentiation of
    proximal coordinate descent.

    The algorithm jointly and iteratively computes the regression coefficients
    and the Jacobian using forward differentiation of proximal
    coordinate descent.

    Parameters
    ----------
    use_stop_crit: bool, optional (default=True)
        Use stopping criterion in hypergradient computation. If False,
        run to maximum number of iterations.
    verbose: bool, optional (default=False)
        Verbosity of the algorithm.
    """

    def __init__(self, use_stop_crit=True, verbose=False):
        self.use_stop_crit = use_stop_crit
        self.verbose = verbose

    def compute_beta_grad(
            self, X, y, log_alpha, model, get_grad_outer, mask0=None,
            dense0=None, quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            full_jac_v=False):
        """Compute beta and hypergradient, with forward differentiation of
        proximal coordinate descent.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float or np.array, shape (n_features,)
            Logarithm of hyperparameter.
        model:  instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        get_grad_outer: callable
            Function which returns the gradient of the outer criterion.
        mask0: ndarray, shape (n_features,)
            Boolean of active feature of the previous regression coefficients
            beta for warm start.
        dense0: ndarray, shape (mask.sum(),)
            Initial value of the previous regression coefficients
            beta for warm start.
        quantity_to_warm_start: ndarray
            Previous Jacobian of the inner optimization problem.
        max_iter: int
            Maximum number of iteration for the inner solver.
        tol: float
            The tolerance for the inner optimization problem.
        full_jac_v: bool
            TODO
        """
        # jointly compute the regression coefficients beta and the Jacobian
        mask, dense, jac = compute_beta(
            X, y, log_alpha, model, mask0=mask0, dense0=dense0,
            jac0=quantity_to_warm_start, max_iter=max_iter, tol=tol,
            compute_jac=True, verbose=self.verbose,
            use_stop_crit=self.use_stop_crit)
        if jac is not None:
            jac_v = model.get_jac_v(X, y, mask, dense, jac, get_grad_outer)
            if full_jac_v:
                jac_v = model.get_full_jac_v(mask, jac_v, X.shape[1])
        else:
            jac_v = None

        return mask, dense, jac_v, jac


def compute_beta(
        X, y, log_alpha, model, mask0=None, dense0=None, jac0=None,
        max_iter=1000, tol=1e-3, compute_jac=True, return_all=False,
        save_iterates=False, verbose=False, use_stop_crit=True, gap_freq=10):
    """
    Parameters
    --------------
    X: array-like, shape (n_samples, n_features)
        Design matrix.
    y: ndarray, shape (n_samples,)
        Observation vector.
    log_alpha: float or np.array, shape (n_features,)
        Logarithm of hyperparameter.
    beta0: ndarray, shape (n_features,)
        initial value of the regression coefficients
        beta for warm start
    dbeta0: ndarray, shape (n_features,)
        initial value of the jacobian dbeta for warm start
    max_iter: int
        number of iterations of the algorithm
    tol: float
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        primal decrease for optimality and continues until it
        is smaller than ``tol``
    compute_jac: bool
        to compute or not the Jacobian along with the regression
        coefficients
    model:  instance of ``sparse_ho.base.BaseModel``
        A model that follows the sparse_ho API.
    return_all: bool
        to store the iterates or not in order to compute the Jacobian in a
        backward way
    use_stop_crit: bool
        use a stopping criterion or do all the iterations
    gap_freq : int
        After how many passes on the data the dual gap should be computed
        to stop the iterations.

    Returns
    -------
    mask : ndarray, shape (n_features,)
        The mask of non-zero coefficients in beta.
    dense : ndarray, shape (n_nonzeros,)
        The beta coefficients on the support
    jac : ndarray, shape (n_nonzeros,) or (n_nonzeros, q)
        The jacobian restricted to the support. If there are more than
        one hyperparameter then it has two dimensions.
    """
    n_samples, n_features = X.shape
    is_sparse = issparse(X)
    if not is_sparse and not np.isfortran(X):
        X = np.asfortranarray(X)
    L = model.get_L(X)
    debug_records = getattr(model, "inner_debug_records_", None)
    debug_record = None
    debug_start = None
    if debug_records is not None:
        alpha_arr = np.asarray(np.exp(log_alpha), dtype=float).ravel()
        debug_record = {
            "call_index": int(len(debug_records)),
            "context": getattr(model, "inner_debug_context_", None),
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "is_sparse": bool(is_sparse),
            "compute_jac": bool(compute_jac),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "gap_freq": int(gap_freq),
            "alpha_min": float(alpha_arr.min()) if alpha_arr.size else np.nan,
            "alpha_max": float(alpha_arr.max()) if alpha_arr.size else np.nan,
            "alpha_mean": float(alpha_arr.mean()) if alpha_arr.size else np.nan,
        }
        debug_start = time.perf_counter()

    ############################################
    alpha = np.exp(log_alpha)

    if hasattr(model, 'estimator') and model.estimator is not None:
        return model._use_estimator(X, y, alpha, tol)

    try:
        alpha.shape[0]
        alphas = alpha.copy()
    except Exception:
        alphas = np.ones(n_features) * alpha
    ############################################
    # warm start for beta
    beta, dual_var = model._init_beta_dual_var(X, y, mask0, dense0)
    ############################################
    # warm start for dbeta
    dbeta, ddual_var = model._init_dbeta_ddual_var(
        X, y, mask0=mask0, dense0=dense0, jac0=jac0, compute_jac=compute_jac)

    # store the values of the objective
    pobj0 = model._get_pobj0(dual_var, np.zeros(X.shape[1]), alphas, y)
    pobj = []
    stop_reason = "max_iter"
    gap_checks = 0
    last_dual_gap = np.nan
    last_relative_decrease = np.nan
    last_stop_crit = np.nan
    stop_crit_name = None
    use_dense_active_set = (
        not is_sparse
        and hasattr(model, "should_use_dense_active_set")
        and hasattr(model, "_update_beta_jac_bcd_restricted_impl")
        and model.should_use_dense_active_set(X)
    )
    use_sparse_active_set = (
        is_sparse
        and hasattr(model, "should_use_sparse_active_set")
        and hasattr(model, "_update_beta_jac_bcd_sparse_restricted_impl")
        and model.should_use_sparse_active_set(X)
    )
    active_idx = None
    full_passes = 0
    restricted_passes = 0
    active_size_sum = 0.0
    active_size_max = 0

    ############################################
    # store the iterates if needed
    if return_all:
        list_beta = []
    if save_iterates:
        list_beta = []
        list_jac = []

    for i in range(max_iter):
        if verbose:
            print("%i -st iteration over %i" % (i, max_iter))
        if is_sparse:
            do_full_pass = (
                not use_sparse_active_set
                or i == 0
                or i % max(int(getattr(model, "active_set_refresh", 1)), 1) == 0
            )
            if do_full_pass:
                model._update_beta_jac_bcd_sparse(
                    X.data, X.indptr, X.indices, y, n_samples, n_features, beta,
                    dbeta, dual_var, ddual_var, alphas, L,
                    compute_jac=compute_jac)
                full_passes += 1
                if use_sparse_active_set:
                    active_idx = model.get_sparse_active_set(
                        X, y, beta, dual_var, alphas)
            else:
                if active_idx is None or active_idx.size == 0:
                    active_idx = model.get_sparse_active_set(
                        X, y, beta, dual_var, alphas)
                model._update_beta_jac_bcd_sparse_restricted_impl(
                    X.data, X.indptr, X.indices, y, n_samples, n_features, beta,
                    dbeta, dual_var, ddual_var, alphas, L, active_idx,
                    compute_jac=compute_jac)
                restricted_passes += 1
                active_size = int(active_idx.size)
                active_size_sum += active_size
                active_size_max = max(active_size_max, active_size)
        else:
            do_full_pass = (
                not use_dense_active_set
                or i == 0
                or i % max(int(getattr(model, "active_set_refresh", 1)), 1) == 0
            )
            if do_full_pass:
                model._update_beta_jac_bcd(
                    X, y, beta, dbeta, dual_var, ddual_var, alphas,
                    L, compute_jac=compute_jac)
                full_passes += 1
                if use_dense_active_set:
                    active_idx = model.get_dense_active_set(
                        X, y, beta, dual_var, alphas)
            else:
                if active_idx is None or active_idx.size == 0:
                    active_idx = model.get_dense_active_set(
                        X, y, beta, dual_var, alphas)
                model._update_beta_jac_bcd_restricted_impl(
                    X, y, beta, dbeta, dual_var, ddual_var, alphas, L,
                    active_idx, compute_jac=compute_jac)
                restricted_passes += 1
                active_size = int(active_idx.size)
                active_size_sum += active_size
                active_size_max = max(active_size_max, active_size)

        pobj.append(model._get_pobj(dual_var, X, beta, alphas, y))

        if i > 1:
            if verbose:
                print("relative decrease = ", (pobj[-2] - pobj[-1]) / pobj0)

        if use_stop_crit and i % gap_freq == 0 and i > 0:
            if hasattr(model, "_get_stop_crit"):
                stop_crit = model._get_stop_crit(dual_var, X, beta, alphas, y)
                gap_checks += 1
                last_stop_crit = float(stop_crit)
                stop_crit_name = "_get_stop_crit"
                if verbose:
                    print("stop crit %.2e" % stop_crit)
                if stop_crit < tol:
                    stop_reason = "stop_crit"
                    break
            elif hasattr(model, "_get_dobj"):
                dobj = model._get_dobj(dual_var, X, beta, alpha, y)
                dual_gap = pobj[-1] - dobj
                gap_checks += 1
                last_dual_gap = float(dual_gap)
                stop_crit_name = "_get_dobj"
                if verbose:
                    print("dual gap %.2e" % dual_gap)
                if verbose:
                    print("gap %.2e" % dual_gap)
                if dual_gap < pobj0 * tol:
                    stop_reason = "dual_gap"
                    break
            else:
                relative_decrease = pobj[-2] - pobj[-1]
                gap_checks += 1
                last_relative_decrease = float(relative_decrease)
                stop_crit_name = "relative_decrease"
                if (relative_decrease <= pobj0 * tol):
                    stop_reason = "relative_decrease"
                    break
        if return_all:
            list_beta.append(beta.copy())
        if save_iterates:
            list_beta.append(beta.copy())
            list_jac.append(dbeta.copy())
    else:
        if verbose:
            print('did not converge !')

    mask = beta != 0
    dense = beta[mask]
    jac = model._get_jac(dbeta, mask)
    if debug_record is not None:
        debug_record.update({
            "elapsed": float(time.perf_counter() - debug_start),
            "n_passes": int(len(pobj)),
            "stop_reason": stop_reason,
            "support_size": int(mask.sum()),
            "objective0": float(pobj0),
            "final_objective": float(pobj[-1]) if pobj else np.nan,
            "gap_checks": int(gap_checks),
            "stop_crit_name": stop_crit_name,
            "last_stop_crit": float(last_stop_crit),
            "last_dual_gap": float(last_dual_gap),
            "last_relative_decrease": float(last_relative_decrease),
            "used_dense_active_set": bool(use_dense_active_set or use_sparse_active_set),
            "active_set_mode": (
                "sparse" if use_sparse_active_set else
                ("dense" if use_dense_active_set else "none")
            ),
            "full_passes": int(full_passes),
            "restricted_passes": int(restricted_passes),
            "mean_active_size": (
                float(active_size_sum / restricted_passes)
                if restricted_passes else np.nan
            ),
            "max_active_size": int(active_size_max),
        })
        debug_records.append(debug_record)
    if hasattr(model, 'dual'):
        model.dual_var = dual_var
        if compute_jac:
            model.ddual_var = ddual_var
    if save_iterates:
        return np.array(list_beta), np.array(list_jac)
    if return_all:
        return mask, dense, list_beta
    else:
        if compute_jac:
            return mask, dense, jac
        else:
            return mask, dense, None
