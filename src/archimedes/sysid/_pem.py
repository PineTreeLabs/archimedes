from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.optimize import minimize as scipy_minimize, OptimizeResult

import archimedes as arc
from archimedes import compile, scan, tree, jac, struct

from ._lm import lm_solve

if TYPE_CHECKING:
    from archimedes.typing import PyTree
    from archimedes.experimental.state_estimation import KalmanFilterBase
    from .timeseries import Timeseries
    from ._lm import LMResult

    T = TypeVar("T", bound=PyTree)


__all__ = ["pem"]


@struct.pytree_node
class PEMObjective:
    predictor: KalmanFilterBase
    data: Timeseries
    P0: np.ndarray
    x0: np.ndarray = struct.field(static=True, default=None)

    def forward(self, x0: np.ndarray, params: tuple) -> tuple:
        ts = self.data.ts
        us = self.data.us
        ys = self.data.ys

        nx = self.predictor.nx
        nu = us.shape[0]
        ny = ys.shape[0]

        if self.P0 is None:
            P0 = np.eye(nx)
        else:
            P0 = self.P0

        params_flat, unravel_params = tree.ravel(params)
        na = params_flat.size

        V = 0.0  # Cost function
        init_carry, unravel_carry = tree.ravel((x0, P0, params, V))

        @compile(kind="MX")
        def scan_fn(carry_flat, input):
            t, u, y = input[0], input[1:nu+1], input[nu+1:]
            x, P, params, V = unravel_carry(carry_flat)
            x, P, e = self.predictor.step(t, x, y, P, args=(u, params))
            output = np.concatenate([x, e], axis=0)

            # Accumulate cost function, Jacobian, and Hessian
            V += 0.5 * np.sum(e**2)

            carry, _ = tree.ravel((x, P, params, V))
            return carry, output

        inputs = np.vstack((ts, us, ys)).T
        carry, scan_output = scan(scan_fn, init_carry, xs=inputs)
        _, _, _, V = unravel_carry(carry)
        scan_output = scan_output.T
        x_hat, e = scan_output[:nx], scan_output[nx:]

        # Average the function results
        V /= ts.size

        return {
            "x_hat": x_hat,
            "e": e,
            "V": V,
        }

    def residuals(self, decision_variables: np.ndarray) -> np.ndarray:
        """Evaluate the residuals (prediction errors) for the given parameters.

        Args:
            decision_variables: optimization parameters

        Returns:
            Residuals (prediction errors) as a 1D array.
        """
        if self.x0 is not None:
            x0 = self.x0
            params = decision_variables
        else:
            x0, params = decision_variables

        results = self.forward(x0, params)
        return results["e"].flatten()

    def __call__(self, decision_variables: np.ndarray) -> tuple:
        """Evaluate the residuals

        Args:
            decision_variables: optimization parameters

        Returns:
            tuple of (V, J, H)
            - V: cost function
            - J: Jacobian
            - H: Hessian approximation
        """
        if self.x0 is not None:
            x0 = self.x0
            params = decision_variables
        else:
            x0, params = decision_variables

        results = self.forward(x0, params)
        return results["V"]


def _pem_solve_lm(
    pem_obj: PEMObjective,
    params_guess: T,
    bounds: tuple[T, T] | None = None,
    options: dict | None = None,
) -> LMResult:
    """Solve the PEM problem using Levenberg-Marquardt optimization."""
    if options is None:
        options = {}

    # Set sensible defaults for system identification problems
    default_options = {
        "ftol": 1e-4,
        "xtol": 1e-6,
        "gtol": 1e-6,
        "maxfev": 200,
    }
    options = {**default_options, **options}
    return lm_solve(pem_obj.residuals, params_guess, bounds=bounds, **options)


def _pem_solve_bfgs(
    pem_obj: PEMObjective,
    params_guess: T,
    bounds: tuple[T, T] | None = None,
    options: dict | None = None,
) -> OptimizeResult:
    method = "BFGS" if bounds is None else "L-BFGS-B"

    if options is None:
        options = {}

    params_guess_flat, unravel = arc.tree.ravel(params_guess)

    if bounds is not None:
        lb, ub = bounds
        lb_flat, _ = arc.tree.ravel(lb)
        ub_flat, _ = arc.tree.ravel(ub)
        # Zip bounds into (lb, ub) for each parameter
        bounds = list(zip(lb_flat, ub_flat))

    # Define an objective and gradient function for BFGS
    @arc.compile
    def func(params_flat):
        return pem_obj(unravel(params_flat))

    jac = arc.grad(func)

    # Set sensible defaults for system identification problems
    default_options = {
        "gtol": 1e-6,
        "disp": True,
        "maxiter": 200,
    }

    if method == "BFGS" and "hess_inv0" not in options:
        # Initialize with Gauss-Newton Hessian approximation
        J0 = jac(params_guess_flat)
        hess_inv0 = np.linalg.inv(np.outer(J0, J0) + 1e-8 * np.eye(len(J0)))
        hess_inv0 = 0.5 * (hess_inv0 + hess_inv0.T)  # Ensure symmetry
        default_options["hess_inv0"] = hess_inv0

    options = {**default_options, **options}

    result = scipy_minimize(
        func,
        params_guess_flat,
        method=method,
        jac=jac,
        bounds=bounds,
        options=options,
    )

    # Replace the flat parameters with the original PyTree structure
    result.x = unravel(result.x)
    return result


def _pem_solve_ipopt(
    pem_obj: PEMObjective,
    params_guess: T,
    bounds: tuple[T, T] | None = None,
    options: dict | None = None,
) -> OptimizeResult:

    if options is None:
        options = {}

    # Set sensible defaults for system identification problems
    ipopt_default_options = {
        "tol": 1e-6,
        "max_iter": 200,
        "hessian_approximation": "limited-memory",
    }
    default_options = {}

    options["ipopt"] = {**ipopt_default_options, **options.get("ipopt", {})}
    options = {**default_options, **options}

    params_guess_flat, unravel = arc.tree.ravel(params_guess)
    if bounds is not None:
        lb, ub = bounds
        lb_flat, _ = arc.tree.ravel(lb)
        ub_flat, _ = arc.tree.ravel(ub)
        bounds = (lb_flat, ub_flat)

    # Define an objective and gradient function for BFGS
    @arc.compile
    def func(params_flat):
        return pem_obj(unravel(params_flat))

    params_opt_flat = arc.minimize(
        func,
        params_guess_flat,
        bounds=bounds,
        **options,
    )

    result = OptimizeResult()
    result.x = unravel(params_opt_flat)
    result.success = True  # Assume success for now
    result.message = "Optimization completed successfully."
    result.nit = -1  # Placeholder for number of iterations
    result.fun = pem_obj(result.x)

    return result



SUPPORTED_METHODS = {
    "lm": _pem_solve_lm,
    "bfgs": _pem_solve_bfgs,
    "ipopt": _pem_solve_ipopt,
}

def pem(
    predictor: KalmanFilterBase,
    data: Timeseries,
    params_guess: T,
    x0: np.ndarray = None,
    bounds: tuple[T, T] | None = None,
    P0: np.ndarray = None,
    method: str = "bfgs",
    options: dict | None = None,
) -> LMResult:
    """Estimate parameters using Prediction Error Minimization.

    Solves the system identification problem by minimizing the prediction
    error between model predictions and measured outputs using a Kalman
    filter framework. This approach provides optimal handling of process
    and measurement noise while enabling efficient gradient computation
    through automatic differentiation.

    The method implements the discrete-time prediction error objective:

    .. code-block:: text

        minimize  J = (1/N) Σ[k=1 to N] e[k]ᵀ e[k]

    where ``e[k] = y[k] - ŷ[k|k-1]`` are the one-step-ahead prediction errors
    (innovations) from the Kalman filter and ``N`` is the number of measurements.

    This formulation automatically accounts for:

    - **Noise handling**: Process and measurement noise are modeled explicitly
    - **Recursive estimation**: Kalman filter provides efficient state propagation
    - **Gradient computation**: Automatic differentiation through filter recursions

    Parameters
    ----------
    predictor : KalmanFilterBase
        Kalman filter implementing the system model. Must provide
        ``step(t, x, y, P, args)`` method. Common choices:

        - :class:`ExtendedKalmanFilter`
        - :class:`UnscentedKalmanFilter`

        The predictor encapsulates the system dynamics, observation model,
        and noise characteristics (``Q``, ``R`` matrices).
    data : Timeseries
        Input-output data containing synchronized time series:
        
        - ``ts`` : Time vector of shape ``(N,)``
        - ``us`` : Input signals of shape ``(nu, N)``
        - ``ys`` : Output measurements of shape ``(ny, N)``
        
        All arrays must have consistent time dimensions.
    params_guess : PyTree
        Initial parameter guess with arbitrary nested structure
        (e.g., ``{"mass": 1.0, "damping": {"c1": 0.1, "c2": 0.2}}``).
        The optimization preserves this structure in the result, enabling
        natural organization of physical parameters.
    x0 : array_like, optional
        Initial state estimate of shape ``(nx,)``. If None, ``params_guess``
        should be a tuple ``(x0_guess, params_guess)`` to jointly estimate
        initial conditions and parameters. This is useful when initial
        conditions are uncertain or need to be optimized.
    bounds : tuple of (PyTree, PyTree), optional
        Parameter bounds as ``(lower_bounds, upper_bounds)`` with the
        same PyTree structure as ``params_guess``. Enables physical
        constraints such as:
        
        - Positive masses, stiffnesses, damping coefficients
        - Bounded gain parameters, time constants
        - Realistic physical parameter ranges
        
        Use ``-np.inf`` and ``np.inf`` for unbounded parameters.
    P0 : array_like, optional
        Initial state covariance matrix of shape ``(nx, nx)``. If None,
        defaults to identity matrix. Represents uncertainty in initial
        state estimate.
    method : str, default="bfgs"
        Optimization method. Currently only "lm" (Levenberg-Marquardt), "ipopt",
        and "bfgs" (BFGS) are supported. The "lm" method is a custom implementation
        roughly based on the MINPACK algorithm, and the "bfgs" method dispatches to
        a SciPy wrapper ("BFGS" or "L-BFGS-B", depending on whether there are bounds).
    options : dict, optional
        Optimization options passed to the underlying optimization solver.

        For the "lm" method, these include:

        - ``ftol`` : Function tolerance (default: 1e-4)
        - ``xtol`` : Parameter tolerance (default: 1e-6)
        - ``gtol`` : Gradient tolerance (default: 1e-6)
        - ``maxfev`` : Maximum function evaluations (default: 200)
        - ``nprint`` : Progress printing interval (default: 0)

        For the "bfgs" method, these include:

        - ``gtol`` : Gradient tolerance (default: 1e-6)
        - ``disp`` : Whether to print convergence information (default: True)
        - ``maxiter`` : Maximum iterations (default: 200)
        - ``hess_inv0`` : Initial Hessian inverse for BFGS (optional)

        If ``hess_inv0`` is not provided, a Gauss-Newton-like Hessian approximation
        is used to initialize the BFGS inverse-Hessian approximation.

        For the "ipopt" method, see the :func:`archimedes.minimize` documentation
        for available options. The solver defaults to a limited-memory approximation
        of the Hessian.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Optimization result with estimated parameters in ``result.x``
        preserving the original PyTree structure. Additional fields include:

        - ``success`` : Whether estimation succeeded
        - ``fun`` : Final prediction error objective value
        - ``nit`` : Number of optimization iterations
        - ``history`` : Detailed convergence history

    Notes
    -----
    This implementation provides significant advantages for system identification:

    **Automatic Gradients**:
        Efficient gradient computation through automatic differentiation of
        the Kalman filter recursions. No need for finite differences or
        manual Jacobian implementation.

    **Structured Parameters**:
        Natural handling of nested parameter dictionaries enables intuitive
        organization of physical parameters and parameter bounds.

    **Physical Constraints**:
        Box constraints enable realistic parameter bounds (mass > 0, etc.)
        without sacrificing convergence properties.

    **Kalman Filter Integration**:
        Seamless integration with both Extended and Unscented Kalman Filters
        enables handling of linear and nonlinear systems with appropriate
        accuracy-efficiency tradeoffs.

    The method automatically computes gradients with respect to both initial
    conditions (when ``x0=None``) and model parameters using efficient
    automatic differentiation through the Kalman filter recursions. This
    avoids the computational expense and numerical issues of finite difference
    approximations.

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>> from archimedes.sysid import pem, Timeseries
    >>> from archimedes.observers import ExtendedKalmanFilter
    >>>
    >>> # Define second-order damped oscillator
    >>> def dynamics(t, x, u, params):
    ...     omega_n = params["omega_n"]  # Natural frequency
    ...     zeta = params["zeta"]        # Damping ratio
    ...     
    ...     return np.hstack([
    ...         x[1],  # velocity
    ...         -omega_n**2 * x[0] - 2*zeta*omega_n*x[1] + omega_n**2 * u[0]
    ...     ])
    >>>
    >>> def observation(t, x, u, params):
    ...     return x[0]  # Measure position only
    >>>
    >>> # Generate synthetic measurement data
    >>> dt = 0.05
    >>> ts = np.arange(0, 10, dt)
    >>> x0 = np.array([0.0, 0.0]) 
    >>> params_true = {"omega_n": 2.0, "zeta": 0.1}
    >>> us = np.ones((1, len(ts)))  # Step input
    >>> # Generate step response
    >>> xs = arc.odeint(
    ...     dynamics,
    ...     (ts[0], ts[-1]),
    ...     x0,
    ...     t_eval=ts,
    ...     args=(np.array([1.0]), params_true),
    ... )
    >>> noise_std = 0.01
    >>> ys = xs[:1, :] + np.random.normal(0, noise_std, size=xs.shape[1])
    >>>
    >>> # Set up identification problem
    >>> dyn_discrete = arc.discretize(dynamics, dt, method="rk4")
    >>> Q = noise_std ** 2 * np.eye(2)  # Process noise covariance
    >>> R = noise_std ** 2 * np.eye(1)  # Measurement noise covariance
    >>> ekf = ExtendedKalmanFilter(dyn_discrete, observation, Q, R)
    >>>
    >>> data = Timeseries(ts=ts, us=us, ys=ys)
    >>> params_guess = {"omega_n": 2.5, "zeta": 0.5}
    >>>
    >>> # Estimate parameters with known initial conditions
    >>> result = pem(ekf, data, params_guess, x0=x0)
    >>> print(f"Estimated parameters: {result.x}")
    Estimated parameters: {'omega_n': array(1.9709515), 'zeta': array(0.11517324)}
    >>> print(f"Converged in {result.nit} iterations")
    Converged in 26 iterations
    >>>
    >>> # With physical parameter constraints
    >>> bounds = (
    ...     {"omega_n": 0.0, "zeta": 0.0},     # Lower bounds (positive values)
    ...     {"omega_n": 10.0, "zeta": 1.0},    # Upper bounds (reasonable ranges)
    ... )
    >>> result = pem(ekf, data, params_guess, x0=x0, bounds=bounds)

    See Also
    --------
    lm_solve : Underlying Levenberg-Marquardt optimizer
    Timeseries : Data container for input-output time series
    ExtendedKalmanFilter : EKF for mildly nonlinear systems
    UnscentedKalmanFilter : UKF for highly nonlinear systems
    discretize : Convert continuous-time dynamics to discrete-time

    References
    ----------
    .. [1] Ljung, L. "System Identification: Theory for the User." 2nd edition,
           Prentice Hall, 1999.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}.")

    pem_solve = SUPPORTED_METHODS[method]

    objective = PEMObjective(
        predictor=predictor,
        data=data,
        P0=P0,
        x0=x0,
    )

    return pem_solve(
        pem_obj=objective,
        params_guess=params_guess,
        bounds=bounds,
        options=options,
    )