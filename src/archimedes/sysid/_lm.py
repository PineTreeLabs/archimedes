import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any, List, NamedTuple
from enum import IntEnum

from archimedes import tree, compile

__all__ = [
    "lm_solve",
    "LMStatus",
    "LMResult",
]


class LMStatus(IntEnum):
    """Status codes for Levenberg-Marquardt optimization results."""
    
    # Success codes (convergence achieved)
    FTOL_REACHED = 1      # Function tolerance convergence
    XTOL_REACHED = 2      # Parameter tolerance convergence  
    BOTH_TOL_REACHED = 3  # Both ftol and xtol satisfied
    GTOL_REACHED = 4      # Gradient tolerance convergence
    
    # Failure codes
    MAX_FEVAL = 5         # Maximum function evaluations reached

    @property
    def message(self) -> str:
        """Get descriptive message for this status code."""
        messages = {
            self.FTOL_REACHED: "Both actual and predicted relative reductions in the sum of squares are at most ftol",
            self.XTOL_REACHED: "Relative error between two consecutive iterates is at most xtol", 
            self.BOTH_TOL_REACHED: "Conditions for ftol and xtol both hold",
            self.GTOL_REACHED: "The cosine of the angle between fvec and any column of the Jacobian is at most gtol in absolute value",
            self.MAX_FEVAL: "Number of function evaluations has reached maxfev",
        }
        return messages.get(self, "Unknown status")
    
    @property 
    def success(self) -> bool:
        """Check if this status indicates successful convergence."""
        return self in (self.FTOL_REACHED, self.XTOL_REACHED, 
                       self.BOTH_TOL_REACHED, self.GTOL_REACHED)


class LMProgress:
    """Handle progress reporting for LM optimization."""
    
    def __init__(self, nprint=0):
        self.nprint = nprint
        self.iteration = 0
        self.prev_cost = None
        self.header_printed = False
        
    def report(self, cost, grad_norm, step_norm, nfev):
        """Report progress in SciPy-style format."""
        if self.nprint <= 0:
            return
            
        if self.iteration % self.nprint == 0:
            if not self.header_printed:
                self._print_header()
                self.header_printed = True
            
            # Calculate cost reduction
            if self.prev_cost is not None:
                cost_reduction = self.prev_cost - cost
            else:
                cost_reduction = None
            
            self._print_iteration(self.iteration, nfev, cost, cost_reduction, 
                                step_norm, grad_norm)
        
        self.prev_cost = cost
        self.iteration += 1
    
    def _print_header(self):
        print(f"{'Iteration':^10} {'Total nfev':^12} {'Cost':^15} "
              f"{'Cost reduction':^15} {'Step norm':^12} {'Optimality':^12}")
    
    def _print_iteration(self, iter_num, nfev, cost, cost_reduction, 
                        step_norm, grad_norm):
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
        
        print(f"{iter_num:^10} {nfev:^12} {cost_str:^15} "
              f"{cost_red_str:^15} {step_str:^12} {grad_str:^12}")


class LMResult(NamedTuple):
    """Result of Levenberg-Marquardt optimization.

    Attributes
    ----------
    x : ndarray
        Solution array
    success : bool
        Whether optimization succeeded
    status : LMStatus
        Status code indicating termination reason
    message : str
        Description of termination reason
    fun : float
        Final objective function value
    jac : ndarray
        Final gradient vector
    hess : ndarray
        Final Hessian matrix
    nfev : int
        Number of function evaluations
    njev : int
        Number of Jacobian evaluations
    nit : int
        Number of iterations
    history : List[Dict[str, Any]]
        Iteration history with convergence details
    """

    x: np.ndarray
    success: bool
    status: LMStatus
    message: str
    fun: float
    jac: np.ndarray
    hess: np.ndarray
    nfev: int
    njev: int
    nit: int
    history: List[Dict[str, Any]]


def _compute_step(hess, grad, diag, lambda_val, bounds):
    """
    Compute the Levenberg-Marquardt step by solving the damped normal equations.

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
    bounds : tuple, optional
        Tuple of (lower_bounds, upper_bounds) for parameters.

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

    # Solve for the step
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

    return step


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

    # Scale by current objective to make it relative (like MINPACK does with residual norm)
    if current_objective is not None and current_objective > 0:
        # Add small epsilon to prevent division by zero near optimum
        epsilon = 1e-16
        pred_red = pred_red / (current_objective + epsilon)

    return pred_red


def _check_bounds(x, bounds):
    if bounds is None:
        return
    lower, upper = bounds
    x_treedef = tree.structure(x)
    lb_treedef = tree.structure(lower)
    ub_treedef = tree.structure(upper)
    if x_treedef != lb_treedef:
        raise ValueError(
            f"Lower bounds must have the same structure as x0 but got {x_treedef}"
            f"for x0 and {lb_treedef} for lower bounds."
        )
    if x_treedef != ub_treedef:
        raise ValueError(
            f"Upper bounds must have the same structure as x0 but got {x_treedef}"
            f"for x0 and {ub_treedef} for upper bounds."
        )


def lm_solve(
    func,
    x0,
    args=(),
    bounds=None,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    maxfev=100,
    diag=None,
    lambda0=1e-3,
    nprint=0,
) -> LMResult:
    """
    Solve nonlinear least squares problems using Levenberg-Marquardt.

    Parameters
    ----------
    func : callable
        Function with signature func(params) -> (objective, gradient, hessian)
        - objective: scalar value (0.5 * sum of squared residuals)
        - gradient: gradient vector of the objective
        - hessian: Hessian matrix or approximation (e.g., J.T @ J)
    x0 : PyTree
        Initial guess to the parameters
    args : tuple, optional
        Extra arguments passed to func
    bounds : tuple, optional
        Tuple of (lower_bounds, upper_bounds) for parameters.  Default is None
        (unconstrained).  If provided, both must have the same PyTree structure as x0.
    ftol : float, optional
        Relative error desired in the sum of squares
    xtol : float, optional
        Relative error desired in the approximate solution
    gtol : float, optional
        Orthogonality desired between function vector and Jacobian columns
    maxfev : int, optional
        Maximum number of function evaluations
    diag : ndarray, optional
        N-element array of scaling factors. If None, automatic scaling used.
    lambda0 : float, optional
        Initial LM damping parameter. Default is 1e-3.
    nprint : int, optional
        Print progress every nprint iterations

    Returns
    -------
    result : LMResult
        Optimization result with solution, convergence info, and history
    """
    _check_bounds(x0, bounds)  # Validate bounds structure

    progress = LMProgress(nprint)  # Initialize logger

    # Constants
    MACHEP = np.finfo(float).eps  # Machine precision

    # By default, just flatten/unflatten the PyTree
    x0_flat, unravel = tree.ravel(x0)

    # Initialize parameters and arrays
    x = x0_flat.copy()  # Start with the flattened initial guess
    n = len(x)

    # Wrap the original function to apply the unravel
    _func = compile(func)
    def func(x_flat, *args):
        x = unravel(x_flat)
        return _func(x, *args)

    # Auto-detect scaling: if diag is None, use automatic scaling
    auto_scale = diag is None
    if diag is None:
        diag = np.ones(n)
    else:
        diag = np.asarray(diag)

    # Initialize counters and status variables
    nfev = 0  # Number of function evaluations
    njev = 0  # Number of Jacobian evaluations
    iter = 0  # Iteration counter
    status = None  # Will be set to LMStatus value

    # Always collect iteration history
    history = []

    # Initial evaluation
    cost, grad, hess = func(x, *args)
    nfev += 1

    # Calculate gradient norm for convergence check
    g_norm = np.linalg.norm(grad, np.inf)

    # Initialize the Levenberg-Marquardt parameter
    lambda_val = lambda0  # Initial damping parameter

    # Main iteration loop
    while nfev < maxfev:
        
        # Record iteration history before computing step
        history.append(
            {
                "iter": iter,
                "cost": float(cost),
                "grad_norm": float(g_norm),
                "lambda": float(lambda_val),
                "x": x.copy(),  # Current parameter values
            }
        )

        # Increment Jacobian evaluations counter
        njev += 1

        # Update diagonal scaling if using automatic scaling
        if iter == 0 and auto_scale:
            # Use the diagonal of the Hessian for scaling
            diag = np.sqrt(np.maximum(np.diag(hess), 1e-8))

        # Calculate scaled vector norm
        xnorm = np.linalg.norm(diag * x)

        # Check gradient convergence
        if g_norm <= gtol:
            status = LMStatus.GTOL_REACHED
            break

        # Inner loop - compute step and try it
        inner_loop_exit = False
        while True:
            # Compute step using damped normal equations
            step = _compute_step(hess, grad, diag, lambda_val, bounds)

            # Compute trial point
            x_new = x + step

            # Compute scaled step norm
            pnorm = np.linalg.norm(diag * step)

            # Evaluate function at trial point
            cost_new, grad_new, hess_new = func(x_new, *args)
            nfev += 1

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
            # Use more conservative updates similar to MINPACK
            if ratio < 0.25:  # Poor agreement: increase damping
                lambda_val = lambda_val * 4.0
            elif ratio > 0.75:  # Good agreement: decrease damping
                lambda_val = lambda_val * 0.5
            # For 0.25 <= ratio <= 0.75, keep lambda unchanged

            # Ensure lambda stays reasonably bounded
            lambda_val = max(lambda_val, 1e-12)
            lambda_val = min(lambda_val, 1e12)

            # Test for successful iteration
            if ratio >= 1.0e-4:  # Step provides sufficient decrease
                # Accept the step
                x = x_new
                cost, grad, hess = cost_new, grad_new, hess_new
                g_norm = np.linalg.norm(grad, np.inf)
                xnorm = np.linalg.norm(diag * x)

                # Report progress
                progress.report(cost, g_norm, pnorm, nfev)

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
            if nfev >= maxfev:
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
            if (status == LMStatus.FTOL_REACHED):
                status = LMStatus.BOTH_TOL_REACHED
            else:
                status = LMStatus.XTOL_REACHED

        if status is not None:
            break

    if status is None:
        status = LMStatus.MAX_FEVAL

    if nprint > 0:
        print(status.message)

    # Unravel the final solution
    x = unravel(x)

    # Create and return LMResult
    return LMResult(
        x=x,
        success=status.success,
        status=status,
        message=status.message,
        fun=cost,
        jac=grad,  # This is the gradient
        hess=hess,
        nfev=nfev,
        njev=njev,
        nit=iter,
        history=history,
    )
