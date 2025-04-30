



def lmder(
    initial_params,
    func,
    max_iter=100,
    lambda_init=1.0,
    lambda_max=1e8,
    lambda_min=1e-8,
    tol=1e-6,
):
    """
    Efficient Levenberg-Marquardt implementation using pre-computed derivatives
    
    Args:
        initial_params: Initial parameter vector
        func: Function that returns (V, J, H) - objective, gradient, and Hessian
        max_iter: Maximum iterations
        lambda_init: Initial LM damping parameter
        lambda_max/min: Bounds for the damping parameter
        tol: Convergence tolerance
    
    Returns:
        Optimized parameters
    """
    theta = initial_params.copy()
    d = len(theta)
    lambda_val = lambda_init
    
    # Initial evaluation
    V, J, H = func(theta)

    print("Iteration    Total nfev     Cost       Optimality")
    print("----------   ----------  ----------    ----------")
    fmt = "  {:>2}             {:>2}     {:>8.4e}        {:>10.4f}"

    # cost = 0.5 * np.dot(f, f)
    # g = J.T.dot(f)
    # g_norm = norm(g, ord=np.inf)
    
    iteration = 0
    nfev = 1
    while iteration < max_iter:
        # Apply Levenberg-Marquardt regularization to Hessian approximation
        H_lm = H + lambda_val * np.eye(d)

        # Compute step direction
        try:
            delta = np.linalg.solve(H_lm, J)
        except np.linalg.LinAlgError:
            print("Matrix is singular, using fallback")
            # Fallback if matrix is singular
            delta = J / (np.linalg.norm(J) + 1e-8)

        # Compute proposed new parameters
        theta_new = theta + delta

        # Evaluate at new parameters
        V_new, J_new, H_new = func(theta_new)
        nfev += 1
        # print(lambda_val)

        # Accept/reject step and adjust lambda
        if V_new < V:
            # print("Step accepted")
            # Success - accept step
            improvement_ratio = (V - V_new) / (0.5 * np.dot(delta, J))  # Actual vs. predicted reduction

            # Update parameters and derivatives
            theta = theta_new
            V, J, H = V_new, J_new, H_new

            # Adjust lambda based on improvement
            if improvement_ratio > 0.75:
                lambda_val = max(lambda_val / 10, lambda_min)
            elif improvement_ratio < 0.25:
                lambda_val = min(lambda_val * 2, lambda_max)

            # Print status
            iteration += 1
            print(fmt.format(iteration, nfev, V, 0.0))

        else:
            # Failure - reject step and increase lambda
            lambda_val = min(lambda_val * 10, lambda_max)

        # Check convergence
        if np.linalg.norm(delta) < tol * (np.linalg.norm(theta) + tol):
            break

        # Also check gradient norm for convergence
        if np.linalg.norm(J) < tol:
            break

    return theta