import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any


def compute_step(hess, grad, diag, lambda_val):
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
        
    Returns
    -------
    step : ndarray, shape (n,)
        Step direction
    """
    n = len(grad)
    # Form the damped Hessian: H + λ·diag(diag)²
    H_damped = hess.copy()
    for i in range(n):
        H_damped[i, i] += lambda_val * diag[i]**2
    
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


def compute_predicted_reduction(grad, step, hess, current_objective=None):
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


def lmder(
    func, 
    x0, 
    jac=None, 
    args=(), 
    ftol=1e-8, 
    xtol=1e-8, 
    gtol=1e-8,
    maxfev=100, 
    diag=None, 
    mode=1, 
    factor=100.0,
    nprint=0,
    **options
) -> Dict[str, Any]:
    """
    Solve nonlinear least squares problems using Levenberg-Marquardt with analytical Jacobian.
    
    Parameters
    ----------
    func : callable
        Function with signature func(params) -> (objective, gradient, hessian)
        - objective: scalar value of the objective function (0.5 * sum of squared residuals)
        - gradient: gradient vector of the objective
        - hessian: Hessian matrix of the objective or approximation (e.g., J.T @ J)
    x0 : ndarray, shape (n,)
        Initial guess to the parameters
    args : tuple, optional
        Extra arguments passed to func and jac
    ftol : float, optional
        Relative error desired in the sum of squares
    xtol : float, optional
        Relative error desired in the approximate solution
    gtol : float, optional
        Orthogonality desired between the function vector and the columns of the Jacobian
    maxfev : int, optional
        Maximum number of function evaluations
    diag : ndarray, optional
        N-element array of scaling factors for the variables
    mode : int, optional
        If 1, variables are scaled internally. If 2, variables are scaled by diag
    factor : float, optional
        Initial step bound factor
    nprint : int, optional
        Print progress every nprint iterations
    **options : dict, optional
        Additional options
    
    Returns
    -------
    result : OptimizeResult
        Optimization result with fields:
        - x: solution array
        - success: boolean indicating success
        - status: integer status code
        - message: description of the termination reason
        - fun: final value of the objective function
        - jac: final Jacobian matrix
        - nfev: number of function evaluations
        - njev: number of Jacobian evaluations
    """
    # Constants
    MACHEP = np.finfo(float).eps  # Machine precision
    
    # Initialize parameters and arrays
    x = np.asarray(x0).copy()
    n = len(x)
    
    if diag is None:
        diag = np.ones(n)
    else:
        diag = np.asarray(diag)
    
    # Initialize counters and status variables
    nfev = 0  # Number of function evaluations
    njev = 0  # Number of Jacobian evaluations
    iter = 0  # Iteration counter
    info = 0  # Convergence info
    
    # Initial evaluation
    cost, grad, hess = func(x, *args)
    nfev += 1
    
    # Calculate gradient norm for convergence check
    g_norm = np.linalg.norm(grad, np.inf)
    
    # Initialize the Levenberg-Marquardt parameter
    lambda_val = 0.001  # Initial damping parameter
    
    # Main iteration loop
    while nfev < maxfev:
        # If requested, call function for printing
        if nprint > 0 and iter % nprint == 0:
            print(f"Iteration {iter}, Cost: {cost}, Grad norm: {g_norm}")
        
        # Increment Jacobian evaluations counter
        njev += 1
        
        # If first iteration or mode=1, update diagonal scaling
        if iter == 0 and mode == 1:
            # In our case, we can use the diagonal of the Hessian for scaling
            diag = np.sqrt(np.maximum(np.diag(hess), 1e-8))
        
        # Calculate scaled vector norm
        xnorm = np.linalg.norm(diag * x)
        
        # Check gradient convergence
        if g_norm <= gtol:
            info = 4
            break
        
        # Inner loop - compute step and try it
        while True:
            # Compute step using damped normal equations
            step = compute_step(hess, grad, diag, lambda_val)
            
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
            prered = compute_predicted_reduction(grad, step, hess, cost)
            
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
                iter += 1
                break
            
            # If lambda becomes too large, the step becomes too small
            if lambda_val >= 1e10:
                info = 7  # xtol is too small
                break
            
            # If maximum function evaluations reached
            if nfev >= maxfev:
                info = 5
                break
        
        # Check if we exited inner loop due to max function evaluations
        if info == 5:
            break
        
        # Test convergence conditions
        # 1. Function value convergence (ftol)
        if abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1.0:
            info = 1
        
        # 2. Parameter convergence (xtol)
        # In our case, we use step size relative to parameter magnitude
        if pnorm <= xtol * xnorm:
            info = 2
        
        # 3. Both criteria met
        if abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1.0 and info == 2:
            info = 3
        
        # 4. Gradient convergence already checked above
        
        # 5. Check if ftol is too small (relative error in the sum of squares)
        if abs(actred) <= MACHEP and prered <= MACHEP and 0.5 * ratio <= 1.0:
            info = 6
        
        # 6. Check if xtol is too small (relative error in the approximate solution)
        if pnorm <= MACHEP * xnorm:
            info = 7
        
        # 7. Check if gtol is too small (orthogonality between fvec and its derivatives)
        if g_norm <= MACHEP:
            info = 8
        
        if info != 0:
            break
    
    # Final check for max function evaluations if we exited the main loop
    if info == 0 and nfev >= maxfev:
        info = 5
    
    # Set final message based on info code
    messages = {
        0: "Improper input parameters",
        1: "Both actual and predicted relative reductions in the sum of squares are at most ftol",
        2: "Relative error between two consecutive iterates is at most xtol",
        3: "Conditions for info = 1 and info = 2 both hold",
        4: "The cosine of the angle between fvec and any column of the Jacobian is at most gtol in absolute value",
        5: "Number of function evaluations has reached maxfev",
        6: "ftol is too small: no further reduction in the sum of squares is possible",
        7: "xtol is too small: no further improvement in the approximate solution is possible",
        8: "gtol is too small: fvec is orthogonal to the columns of the Jacobian to machine precision"
    }
    
    success = info in (1, 2, 3, 4)  # Check if convergence criteria met
    message = messages.get(info, "Unknown error")
    
    # Create and return result object similar to SciPy's OptimizeResult
    result = {
        'x': x,
        'success': success,
        'status': info,
        'message': message,
        'fun': cost,
        'jac': grad,  # This is not the Jacobian but the gradient
        'hess': hess, # Added Hessian information
        'nfev': nfev,
        'njev': njev,
        'nit': iter
    }
    
    return result


