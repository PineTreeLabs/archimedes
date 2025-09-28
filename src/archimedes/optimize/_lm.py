# ruff: noqa: N806

from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, Tuple, TypeVar

import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import OptimizeResult

import archimedes as arc
from archimedes import tree

from ._common import _ravel_args

if TYPE_CHECKING:
    from archimedes.typing import Tree

    T = TypeVar("T", bound=Tree)


__all__ = [
    "lm_solve",
    "LMStatus",
    "LMResult",
]


class LMStatus(IntEnum):
    """Status codes for Levenberg-Marquardt optimization results.

    These codes follow the MINPACK convention and provide detailed information
    about the termination condition of the optimization algorithm. Understanding
    these codes is essential for interpreting optimization results and diagnosing
    potential issues.

    Attributes
    ----------
    FTOL_REACHED : int = 1
        Both actual and predicted relative reductions in the objective function
        are at most ``ftol``. This indicates convergence based on minimal
        improvement in the objective function.

    XTOL_REACHED : int = 2
        Relative error between two consecutive parameter iterates is at most
        ``xtol``. This indicates convergence based on minimal change in the
        parameters.

    BOTH_TOL_REACHED : int = 3
        Both ``ftol`` and ``xtol`` conditions are satisfied simultaneously.
        This represents the strongest convergence criterion.

    GTOL_REACHED : int = 4
        The cosine of the angle between the gradient and any column of the
        Jacobian is at most ``gtol`` in absolute value. For constrained
        problems, this applies to the projected gradient. This indicates
        convergence to a critical point.

    MAX_FEVAL : int = 5
        Maximum number of function evaluations (``max_nfev``) has been reached
        without achieving convergence. This typically indicates that either
        more iterations are needed, the tolerances are too tight, or the
        problem is ill-conditioned.

    Notes
    -----
    **Success vs. Failure**:
        Status codes 1-4 indicate successful convergence, while code 5 indicates
        failure to converge within the iteration limit. Use the ``success``
        property to check for overall success.

    **Interpretation Guide**:
        - **FTOL_REACHED**: Good for most applications, indicates objective
          function has stopped improving significantly
        - **XTOL_REACHED**: May indicate convergence to a local minimum or
          numerical precision limits
        - **BOTH_TOL_REACHED**: Strongest convergence guarantee
        - **GTOL_REACHED**: Gradient-based convergence, good indicator of
          optimality conditions
        - **MAX_FEVAL**: Check if more iterations are needed or if the problem
          requires reformulation
    """

    # Success codes (convergence achieved)
    FTOL_REACHED = 1  # Function tolerance convergence
    XTOL_REACHED = 2  # Parameter tolerance convergence
    BOTH_TOL_REACHED = 3  # Both ftol and xtol satisfied
    GTOL_REACHED = 4  # Gradient tolerance convergence

    # Failure codes
    MAX_FEVAL = 5  # Maximum function evaluations reached

    @property
    def message(self) -> str:
        """Get descriptive message for this status code."""
        messages = {
            self.FTOL_REACHED: (
                "Both actual and predicted relative reductions in the sum of squares "
                "are at most ftol"
            ),
            self.XTOL_REACHED: (
                "Relative error between two consecutive iterates is at most xtol"
            ),
            self.BOTH_TOL_REACHED: "Conditions for ftol and xtol both hold",
            self.GTOL_REACHED: (
                "The cosine of the angle between fvec and any column of the Jacobian "
                "is at most gtol in absolute value"
            ),
            self.MAX_FEVAL: "Number of function evaluations has reached max_nfev",
        }
        return messages.get(self, "Unknown status")

    @property
    def success(self) -> bool:
        """Check if this status indicates successful convergence."""
        return self in (
            self.FTOL_REACHED,
            self.XTOL_REACHED,
            self.BOTH_TOL_REACHED,
            self.GTOL_REACHED,
        )


class LMProgress:
    """Handle progress reporting for LM optimization."""

    def __init__(self, logger=None):
        self.logger = logger
        self.iteration = 0
        self.prev_cost = None
        self.header_printed = False

    def report(self, cost, grad_norm, step_norm, nfev):
        """Report progress in SciPy-style format."""
        if not self.header_printed:
            self._print_header()
            self.header_printed = True

        # Calculate cost reduction
        if self.prev_cost is not None:
            cost_reduction = self.prev_cost - cost
        else:
            cost_reduction = None

        self._print_iteration(
            self.iteration, nfev, cost, cost_reduction, step_norm, grad_norm
        )

        self.prev_cost = cost
        self.iteration += 1

    def _print_header(self):
        msg = (
            f"{'Iteration':^10} {'Total nfev':^12} {'Cost':^15} "
            f"{'Cost reduction':^15} {'Step norm':^12} {'Optimality':^12}"
        )
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)

    def _print_iteration(
        self, iter_num, nfev, cost, cost_reduction, step_norm, grad_norm
    ):
        """Print a single iteration row."""
        # Format numbers with appropriate precision
        cost_str = f"{cost:.4e}"
        grad_str = f"{grad_norm:.2e}"

        if cost_reduction is not None:
            cost_red_str = f"{cost_reduction:.2e}"
            step_str = f"{step_norm:.2e}" if step_norm is not None else ""
        else:
            cost_red_str = ""
            step_str = ""

        msg = (
            f"{iter_num:^10} {nfev:^12} {cost_str:^15} "
            f"{cost_red_str:^15} {step_str:^12} {grad_str:^12}"
        )
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)


class LMResult(OptimizeResult):
    """Result of Levenberg-Marquardt optimization.

    This class provides detailed information about the optimization process
    and results, following SciPy conventions while adding specialized fields
    for system identification applications.

    Attributes
    ----------
    x : ndarray or Tree
        Solution parameters with the same structure as the initial guess.
        For system identification, this preserves the nested parameter
        organization (e.g., ``{"mass": 1.0, "damping": {"c1": 0.1}}``).
    success : bool
        Whether optimization terminated successfully. True for status codes
        1-4 (converged), False for status code 5 (max iterations reached).
    status : LMStatus
        Detailed termination status code indicating the specific convergence
        criterion that was satisfied or reason for termination.
    message : str
        Human-readable description of the termination reason corresponding
        to the status code.
    fun : float
        Final objective function value. For least-squares problems, this
        is 0.5 times the sum of squared residuals.
    jac : ndarray
        Final gradient vector of shape ``(n,)``. For constrained problems,
        this is the full gradient (not projected).
    hess : ndarray
        Final Hessian matrix or approximation of shape ``(n, n)``. This
        can be used for uncertainty quantification and further analysis.
    nfev : int
        Number of objective function evaluations performed during optimization.
        Each evaluation computes the objective value, gradient, and Hessian.
    njev : int
        Number of Jacobian evaluations. For this implementation, this equals
        ``nfev`` since gradient and Hessian are computed simultaneously.
    nit : int
        Number of algorithm iterations. Each iteration may involve multiple
        function evaluations due to the trust region approach.
    history : List[Dict[str, Any]]
        Detailed iteration history containing convergence diagnostics:

        - ``iter`` : Iteration number
        - ``cost`` : Objective function value
        - ``grad_norm`` : Gradient norm (or projected gradient for constrained)
        - ``lambda`` : Levenberg-Marquardt damping parameter
        - ``x`` : Parameter values at this iteration
        - ``step_norm`` : Step size (when step is accepted)
        - ``actred`` : Actual reduction in objective
        - ``prered`` : Predicted reduction from quadratic model
        - ``ratio`` : Ratio of actual to predicted reduction

    See Also
    --------
    lm_solve : Function that returns this result type
    LMStatus : Detailed description of status codes
    """

    # x: np.ndarray
    # success: bool
    # status: LMStatus
    # message: str
    # fun: float
    # jac: np.ndarray
    # hess: np.ndarray
    # nfev: int
    # njev: int
    # nit: int
    # history: List[Dict[str, Any]]


def _compute_step(hess, grad, diag, lambda_val, x, bounds=None, qp_solver=None):
    """
    Compute the Levenberg-Marquardt step by solving the damped normal equations.

    Uses hybrid approach: first try unconstrained step, then QP if bounds violated.

    Parameters
    ----------
    hess : ndarray, shape (n, n)
        Hessian matrix or approximation
    grad : ndarray, shape (n,)
        Gradient vector
    diag : ndarray, shape (n,)
        Scaling factors for the variables
    lambda_val : float
        Levenberg-Marquardt damping parameter
    x : ndarray, shape (n,)
        Current parameter values
    bounds : tuple, optional
        Tuple of (lower_bounds, upper_bounds) for parameters.
    qp_solver : osqp.OSQP, optional
        OSQP solver instance for handling bounds. Must be provided if bounds are set.

    Returns
    -------
    step : ndarray, shape (n,)
        Step direction
    """
    n = len(grad)
    # Form the damped Hessian: H + λ·diag(diag)²
    H_damped = hess.copy()
    for i in range(n):
        H_damped[i, i] += lambda_val * diag[i] ** 2

    # Step 1: Compute unconstrained step
    step = None
    try:
        # Use Cholesky decomposition for better numerical stability
        # This requires the matrix to be positive definite
        L = np.linalg.cholesky(H_damped)  # H_damped = L @ L.T
        # First solve L @ y = -grad
        y = np.linalg.solve(L, -grad)
        # Then solve L.T @ step = y
        step = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        # Fallback for ill-conditioned or non-positive definite matrices
        try:
            # Try standard solver
            step = np.linalg.solve(H_damped, -grad)
        except np.linalg.LinAlgError:
            # Last resort: gradient descent direction with normalization
            step = -grad / (np.linalg.norm(grad) + 1e-8)

    # Step 2: Check if bounds are satisfied
    if bounds is None:
        return step

    lb, ub = bounds
    x_trial = x + step

    # Check if trial point violates bounds
    violates_bounds = np.any(x_trial < lb) or np.any(x_trial > ub)

    if not violates_bounds:
        return step  # Fast path: no QP needed

    # Step 3: Solve constrained QP
    if qp_solver is None:
        raise ValueError("qp_solver must be provided if bounds are set.")

    # Update QP matrices: min 0.5*p^T*H_damped*p + grad^T*p
    # subject to: lb <= x + p <= ub
    # Rearranged: (lb - x) <= p <= (ub - x)
    l_qp = lb - x  # Lower bounds for step
    u_qp = ub - x  # Upper bounds for step

    # Update the QP problem
    qp_solver.update(P=H_damped, q=grad, l=l_qp, u=u_qp)

    # Warm start with unconstrained solution (projected to bounds)
    qp_solver.warm_start(x=np.clip(step, l_qp, u_qp))

    # Solve QP
    qp_results = qp_solver.solve()

    if qp_results.info.status != "solved":
        print(f"OSQP solver status: {qp_results.info.status}")
        # Fallback: project unconstrained step to bounds
        return np.clip(step, l_qp, u_qp)

    return qp_results.x


def _compute_predicted_reduction(grad, step, hess, current_objective=None):
    """
    Compute the predicted reduction in the objective function.

    For the quadratic model q(p) = f + g^T·p + 0.5·p^T·H·p,
    the predicted reduction is: pred_red = -(g^T·p + 0.5·p^T·H·p)

    Parameters
    ----------
    grad : ndarray, shape (n,)
        Gradient at the current point
    step : ndarray, shape (n,)
        Proposed step
    hess : ndarray, shape (n, n)
        Hessian matrix or approximation
    current_objective : float, optional
        Current objective function value for scaling (if provided)

    Returns
    -------
    pred_red : float
        Predicted reduction in the objective function
    """
    # For step computed from (H + λI)p = -g, we expect:
    # pred_red = -g^T·p - 0.5·p^T·H·p
    linear_term = np.dot(grad, step)
    quadratic_term = 0.5 * np.dot(step, hess @ step)
    pred_red = -(linear_term + quadratic_term)

    # Scale by current objective to make it relative
    # (like MINPACK does with residual norm)
    if current_objective is not None and current_objective > 0:
        # Add small epsilon to prevent division by zero near optimum
        epsilon = 1e-16
        pred_red = pred_red / (current_objective + epsilon)

    return pred_red


def _project_gradient(grad, x, bounds, atol=1e-8):
    """
    Project gradient onto the tangent cone of box constraints.

    This computes the "projected gradient" which is the correct measure
    for convergence in constrained optimization. At a constrained optimum,
    the projected gradient should be zero, not the full gradient.

    Parameters
    ----------
    grad : ndarray, shape (n,)
        Current gradient vector
    x : ndarray, shape (n,)
        Current parameter values
    bounds : tuple of (lower, upper)
        Box constraints where lower and upper are arrays of shape (n,)
        Use -np.inf/np.inf for unbounded variables
    atol : float, optional
        Absolute tolerance for determining if a variable is at its bound

    Returns
    -------
    grad_proj : ndarray, shape (n,)
        Projected gradient
    active_lower : ndarray, shape (n,), dtype=bool
        True where lower bounds are active
    active_upper : ndarray, shape (n,), dtype=bool
        True where upper bounds are active

    Mathematical Notes
    -----------------
    For box constraints l ≤ x ≤ u, the projected gradient is:

    g_proj[i] = {
        0       if x[i] = l[i] and g[i] > 0  (at lower bound, gradient points out)
        0       if x[i] = u[i] and g[i] < 0  (at upper bound, gradient points out)
        g[i]    otherwise                    (interior or admissible direction)
    }

    This captures the KKT optimality conditions:
    - Interior variables: g_proj[i] = g[i] = 0 at optimum
    - Boundary variables: g_proj[i] = 0 always (projected out)
    """
    lower, upper = bounds

    # Determine which constraints are active (within tolerance)
    active_lower = (x <= lower + atol) & np.isfinite(lower)
    active_upper = (x >= upper - atol) & np.isfinite(upper)

    # Start with full gradient
    grad_proj = grad.copy()

    # Zero out gradient components that point "outward" from active constraints
    # At lower bound: zero positive gradients (can't move further left)
    grad_proj = np.where(active_lower & (grad > 0), 0.0, grad_proj)

    # At upper bound: zero negative gradients (can't move further right)
    grad_proj = np.where(active_upper & (grad < 0), 0.0, grad_proj)

    return grad_proj, active_lower, active_upper


def _check_constrained_convergence(grad, x, bounds, gtol=1e-8, atol=1e-8):
    """
    Check convergence for box-constrained optimization using projected gradient.

    Parameters
    ----------
    grad : ndarray
        Current gradient
    x : ndarray
        Current parameters
    bounds : tuple of (lower, upper)
        Box constraints
    gtol : float
        Gradient tolerance for convergence
    atol : float
        Tolerance for determining active constraints

    Returns
    -------
    converged : bool
        True if converged according to constrained optimality conditions
    grad_proj_norm : float
        Norm of projected gradient (should be small at optimum)
    active_info : dict
        Information about active constraints
    """
    grad_proj, active_lower, active_upper = _project_gradient(grad, x, bounds, atol)

    grad_proj_norm = np.linalg.norm(grad_proj, np.inf)
    converged = grad_proj_norm <= gtol

    active_info = {
        "n_active_lower": np.sum(active_lower),
        "n_active_upper": np.sum(active_upper),
        "total_active": np.sum(active_lower | active_upper),
        "grad_proj_norm": grad_proj_norm,
        "full_grad_norm": np.linalg.norm(grad, np.inf),
    }

    return converged, grad_proj_norm, active_info


# TODO:
# - support sparse matrices for Hessian and Jacobian
def lm_solve(
    func: Callable,
    x0: T,
    args: Tuple[Any, ...] = (),
    bounds: Tuple[T, T] | None = None,
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-6,
    max_nfev: int = 100,
    diag: T | None = None,
    lambda0: float = 1e-3,
    log_level: int | None = None,
) -> OptimizeResult:
    """Solve nonlinear least squares using modified Levenberg-Marquardt algorithm.

    Solves optimization problems of the form:

    .. code-block:: text

        minimize    0.5 * ||r(x)||²
        subject to  lb <= x <= ub

    where ``r(x)`` are residuals computed by the objective function.

    This implementation uses a direct Hessian approximation and Cholesky
    factorization instead of the standard QR approach.  It also supports
    box constraints by switching to a quadratic programming solver
    ([OSQP](https://osqp.org/)) when bounds are active.

    Parameters
    ----------
    func : callable
        Objective function with signature ``func(x, *args) -> r``, where ``r`` is
        a vector of residuals.
    x0 : Tree
        Initial parameter guess. Can be a flat array or a tree structure which
        will be preserved in the solution.
    args : tuple, optional
        Extra arguments passed to the objective function.
    bounds : tuple of (Tree, Tree), optional
        Box constraints specified as ``(lower_bounds, upper_bounds)``.
        Each bound array must have the same tree structure as ``x0``. Use
        ``-np.inf`` and ``np.inf`` for unbounded variables. Enables physical
        constraints like mass > 0, damping > 0, etc.
    ftol : float, default=1e-6
        Tolerance for relative reduction in objective function.
        Convergence occurs when both actual and predicted reductions
        are smaller than this value.
    xtol : float, default=1e-6
        Tolerance for relative change in parameters. Convergence occurs
        when the relative step size is smaller than this value.
    gtol : float, default=1e-6
        Tolerance for gradient norm. Convergence occurs when the infinity
        norm of the gradient (or projected gradient for constrained problems)
        falls below this threshold.
    max_nfev : int, default=100
        Maximum number of function evaluations.
    diag : Tree, optional
        Diagonal scaling factors for variables, in the form of a tree matching
        the structure of ``x0``. If None, automatic scaling is used based on the
        Hessian diagonal. Custom scaling can improve convergence for ill-conditioned
        problems.
    lambda0 : float, default=1e-3
        Initial Levenberg-Marquardt damping parameter. Larger values
        bias toward gradient descent, smaller values toward Gauss-Newton.
    log_level : int, default=0
        Print progress every ``nprint`` iterations. Set to 0 to disable
        progress output.

    Returns
    -------
    result : OptimizeResult
        Optimization result containing:

        - ``x`` : Solution parameters with same structure as ``x0``
        - ``success`` : Whether optimization succeeded
        - ``status`` : Termination status (:class:`LMStatus`)
        - ``message`` : Descriptive termination message
        - ``fun`` : Final objective value
        - ``jac`` : Final gradient vector
        - ``hess`` : Final Hessian matrix
        - ``nfev`` : Number of function evaluations
        - ``nit`` : Number of iterations
        - ``history`` : Detailed iteration history

    Notes
    -----
    The algorithm implements the damped normal equations approach:

    .. code-block:: text

        (H + λI)p = -g

    where ``H`` is the Hessian approximation, ``λ`` is the damping parameter,
    and ``p`` is the step direction. The damping parameter is adapted based
    on the ratio of actual to predicted reduction.

    For box-constrained problems, the algorithm uses projected gradients for
    convergence testing and OSQP for constrained step computation when bounds
    are violated.  When the damped normal equations yield a step that would violate
    the bounds, the algorithm switches to solving the quadratic program:

    .. code-block:: text

        minimize    0.5 * p^T * (H + λI) * p + g^T * p
        subject to  lb <= x + p <= ub

    Examples
    --------
    >>> import numpy as np
    >>> from archimedes.sysid import lm_solve
    >>> import archimedes as arc
    >>>
    >>> # Rosenbrock function as least-squares problem
    >>> def rosenbrock_func(x):
    ...     # Residuals: r1 = 10*(x[1] - x[0]²), r2 = (1 - x[0])
    ...     return np.hstack([10*(x[1] - x[0]**2), 1 - x[0]])
    >>>
    >>> # Solve unconstrained problem
    >>> result = lm_solve(rosenbrock_func, x0=np.array([-1.2, 1.0]))
    >>> print(f"Solution: {result.x}")
    Solution: [0.99999941 0.99999881
    >>> print(f"Converged: {result.success} ({result.message})")
    Converged: True (The cosine of the angle between fvec and any column...)
    >>>
    >>> # Solve with box constraints (physical parameter limits)
    >>> bounds = (np.array([0.0, 0.0]), np.array([0.8, 2.0]))
    >>> result = lm_solve(rosenbrock_func, x0=np.array([0.5, 0.5]), bounds=bounds)
    >>> print(f"Constrained solution: {result.x}")
    Constrained solution: [0.79999998 0.6391025 ]

    See Also
    --------
    archimedes.sysid.pem : Parameter estimation using prediction error minimization
    LMStatus : Convergence status codes and meanings
    scipy.optimize.least_squares : SciPy's least-squares solver
    """
    if log_level is None:
        logger = None
    else:
        logger = logging.getLogger("levenberg_marquardt")
        logger.setLevel(log_level)

    x0_flat, bounds_flat, unravel = _ravel_args(x0, bounds)  # Validate bounds structure

    progress = LMProgress(logger)  # Initialize logger

    # Constants
    MACHEP = np.finfo(float).eps  # Machine precision
    SQRT_MACHEP = np.sqrt(MACHEP)  # Square root of machine precision

    # Initialize parameters and arrays
    x = x0_flat.copy()  # Start with the flattened initial guess
    n = len(x)

    if bounds is not None:
        lb, ub = bounds_flat  # Unravel bounds to flat arrays
        # Initialize OSQP solver for box-constrained QP
        qp_solver = osqp.OSQP()
        qp_solver.setup(
            P=sparse.csc_matrix(np.ones((n, n))),  # Will be updated each iteration
            q=np.zeros(n),  # Will be updated each iteration
            A=sparse.csc_matrix(np.eye(n)),  # Identity matrix for box constraints
            l=lb - x,  # Step lower bounds (will be updated)
            u=ub - x,  # Step upper bounds (will be updated)
            verbose=False,  # Suppress OSQP output
            eps_abs=1e-8,  # Absolute tolerance
            eps_rel=1e-8,  # Relative tolerance
        )
    else:
        qp_solver = None

    # Wrap the original function to apply the unravel and compute
    # gradient and Hessian
    _func = arc.compile(func)

    def res_func(x_flat):
        x = unravel(x_flat)
        r = _func(x, *args)
        return tree.ravel(r)[0]  # Return flattened residuals

    @arc.compile
    def func_and_grads(x):
        r = res_func(x)
        J = arc.jac(res_func)(x)
        V = 0.5 * np.sum(r**2)
        g = J.T @ r
        H = J.T @ J
        return V, g, H

    # Auto-detect scaling: if diag is None, use automatic scaling
    auto_scale = diag is None
    if diag is None:
        diag: np.ndarray = np.ones(n)  # type: ignore[no-redef]
    else:
        diag: np.ndarray = tree.ravel(diag)[0]  # type: ignore[no-redef]

    # Initialize counters and status variables
    nfev = 0  # Number of function evaluations
    njev = 0  # Number of Jacobian evaluations
    iter = 0  # Iteration counter
    status = None  # Will be set to LMStatus value

    # Always collect iteration history
    history = []

    # Initial evaluation
    cost, grad, hess = func_and_grads(x)
    nfev += 1

    # Calculate gradient norm for convergence check
    g_norm = np.linalg.norm(grad, np.inf)

    # Initialize the Levenberg-Marquardt parameter
    lambda_val = lambda0  # Initial damping parameter

    # Main iteration loop
    while nfev < max_nfev:
        # Record iteration history before computing step
        history_entry = {
            "iter": iter,
            "cost": float(cost),
            "grad_norm": float(g_norm),
            "lambda": float(lambda_val),
            "x": x.copy(),  # Current parameter values
        }

        # Add constrained optimization info if bounds are present
        if bounds_flat is not None:
            _, g_proj_norm, active_info = _check_constrained_convergence(
                grad, x, bounds_flat, gtol
            )
            history_entry.update(
                {
                    "grad_proj_norm": float(g_proj_norm),
                    "n_active_lower": int(active_info["n_active_lower"]),
                    "n_active_upper": int(active_info["n_active_upper"]),
                    "total_active": int(active_info["total_active"]),
                }
            )

        history.append(history_entry)

        # Increment Jacobian evaluations counter
        njev += 1

        # Update diagonal scaling if using automatic scaling
        if iter == 0 and auto_scale:
            # Use the diagonal of the Hessian for scaling
            diag = np.sqrt(np.maximum(np.diag(hess), 1e-8))

        # Calculate scaled vector norm
        xnorm = np.linalg.norm(diag * x)

        # Check gradient convergence (constrained vs unconstrained)
        if bounds_flat is not None:
            # Constrained optimization: use projected gradient
            converged, g_proj_norm, active_info = _check_constrained_convergence(
                grad, x, bounds_flat, gtol
            )
            if converged:
                status = LMStatus.GTOL_REACHED
                break
            # Use projected gradient norm for progress reporting
            effective_grad_norm = g_proj_norm
        else:
            # Unconstrained optimization: use standard gradient norm
            if g_norm <= gtol:
                status = LMStatus.GTOL_REACHED
                break
            effective_grad_norm = g_norm

        # Inner loop - compute step and try it
        inner_loop_exit = False
        while True:
            # Compute step using damped normal equations
            step = _compute_step(
                hess, grad, diag, lambda_val, x, bounds_flat, qp_solver
            )

            # Compute trial point
            x_new = x + step
            if logger is not None:
                logger.debug(f"lambda: {lambda_val}")
                logger.debug(f"Trial point: {x_new}")

            # Compute scaled step norm
            pnorm = np.linalg.norm(diag * step)

            # Evaluate function at trial point
            cost_new, grad_new, hess_new = func_and_grads(x_new)
            nfev += 1

            if logger is not None:
                logger.debug(
                    f"Trial func eval: cost={cost_new}, grad={grad_new}, "
                    f"hess={hess_new}"
                )

            # Compute actual reduction
            actred = -1.0
            if cost_new < cost:  # Only consider actual reduction if cost decreased
                actred = 1.0 - cost_new / cost

            # Compute predicted reduction using quadratic model
            prered = _compute_predicted_reduction(grad, step, hess, cost)

            # Compute ratio of actual to predicted reduction
            ratio = 0.0
            if prered > 0.0:  # Ensure we have a positive predicted reduction
                ratio = actred / prered

            # Update lambda based on ratio (Trust region update)
            if ratio <= 0.25:
                if actred >= 0:
                    inv_scale = 0.5
                else:
                    dirder = np.dot(grad, step)
                    inv_scale = 0.5 * dirder / (dirder + 0.5 * actred)
                    inv_scale = max(inv_scale, 0.1)
            else:
                inv_scale = 2.0
            lambda_val = lambda_val / inv_scale

            # Test for successful iteration
            if ratio >= 1.0e-4:  # Step provides sufficient decrease
                # Accept the step
                x = x_new
                cost, grad, hess = cost_new, grad_new, hess_new
                g_norm = np.linalg.norm(grad, np.inf)
                xnorm = np.linalg.norm(diag * x)

                # Update effective gradient norm for progress reporting
                if bounds_flat is not None:
                    _, g_proj_norm, _ = _check_constrained_convergence(
                        grad, x, bounds_flat, gtol
                    )
                    effective_grad_norm = g_proj_norm
                else:
                    effective_grad_norm = g_norm

                # Report progress (use appropriate gradient norm)
                progress.report(cost, effective_grad_norm, pnorm, nfev)

                # Machine precision check: if step norm is tiny relative to
                # parameter norm, we're likely at numerical precision limits
                # and should terminate
                if pnorm <= SQRT_MACHEP * max(xnorm, 1.0):
                    status = LMStatus.XTOL_REACHED
                    break

                # Record detailed step information in history
                if len(history) > 0:
                    # Update current iteration's history with step details
                    history[-1].update(
                        {
                            "step_norm": float(pnorm),
                            "actred": float(actred),
                            "prered": float(prered),
                            "ratio": float(ratio),
                            "lambda_next": float(lambda_val),
                        }
                    )

                iter += 1
                break

            # If maximum function evaluations reached during inner loop
            if nfev >= max_nfev:
                status = LMStatus.MAX_FEVAL
                inner_loop_exit = True
                break

        # Check if we exited inner loop due to max function evaluations
        if inner_loop_exit:
            break

        # Test convergence conditions
        # 1. Function value convergence (ftol)
        if abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1.0:
            status = LMStatus.FTOL_REACHED

        # 2. Parameter convergence (xtol)
        # Check step size relative to parameter magnitude
        if pnorm <= xtol * xnorm:
            # Check if we also satisfied ftol for combined convergence
            if status == LMStatus.FTOL_REACHED:
                status = LMStatus.BOTH_TOL_REACHED
            else:
                status = LMStatus.XTOL_REACHED

        if status is not None:
            break

    if status is None:
        status = LMStatus.MAX_FEVAL

    if logger is None:
        print(status.message)
    else:
        logger.info(status.message)

    # Calculate final residuals
    res = res_func(x)

    # Unravel the final solution
    x = unravel(x)

    # Create and return result
    return LMResult(
        x=x,
        success=status.success,
        status=status,
        message=status.message,
        fun=res,  # Final residuals value
        jac=grad,
        hess=hess,
        nfev=nfev,
        njev=njev,
        nit=iter,
        history=history,
    )
