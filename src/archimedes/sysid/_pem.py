# ruff: noqa: N806, N803

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeVar

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

import archimedes as arc
from archimedes import compile, scan, tree
from archimedes.optimize import lm_solve

if TYPE_CHECKING:
    from archimedes.observers import KalmanFilterBase
    from archimedes.typing import Tree

    from ._timeseries import Timeseries

    T = TypeVar("T", bound=Tree)


__all__ = ["pem"]


@tree.struct
class PEMObjective:
    """Prediction Error Minimization objective function for system identification.

    Low-level interface for PEM optimization that provides both residual and
    cost function evaluations. This class encapsulates the Kalman filter
    forward pass and automatic differentiation for gradient computation.

    This is typically used internally by :func:`pem`. For most applications,
    use the higher-level :func:`pem` function directly. This can be useful mainly
    for constructing custom optimization workflows.

    Parameters
    ----------
    predictor : KalmanFilterBase
        Kalman filter implementing the system model with
        ``step(t, x, y, P, args)`` method.
    data : Timeseries
        Input-output data with ``ts``, ``us``, and ``ys`` arrays.
    P0 : array_like
        Initial state covariance matrix of shape ``(nx, nx)``.
    x0 : array_like, optional
        Initial state estimate of shape ``(nx,)``. If None, the optimization
        variables should include both initial state and parameters.

    Notes
    -----
    The objective implements the prediction error formulation:

    .. code-block:: text

        J = (1/N) Σ[k=1 to N] e[k]ᵀ e[k]

    where ``e[k]`` are the Kalman filter innovations (prediction errors).

    See Also
    --------
    pem : High-level parameter estimation interface
    """

    predictor: KalmanFilterBase
    data: Timeseries
    P0: np.ndarray | None
    x0: Optional[np.ndarray] = tree.field(static=True, default=None)  # type: ignore

    def forward(self, x0: np.ndarray, params: Tree) -> dict:
        """Run Kalman filter forward pass and compute prediction errors.

        Parameters
        ----------
        x0 : array_like
            Initial state estimate of shape ``(nx,)``.
        params : Tree
            Model parameters with arbitrary nested structure.

        Returns
        -------
        dict
            Results dictionary with keys:

            - ``"x_hat"`` : State estimates of shape ``(nx, N)``
            - ``"e"`` : Prediction errors of shape ``(ny, N)``
            - ``"V"`` : Average cost function value (scalar)
        """
        ts = self.data.ts
        us = self.data.us
        ys = self.data.ys

        nx = self.predictor.nx
        nu = us.shape[0]

        if self.P0 is None:
            P0 = np.eye(nx)
        else:
            P0 = self.P0

        V = 0.0  # Cost function
        init_carry, unravel_carry = tree.ravel((x0, P0, params, V))

        @compile(kind="MX")
        def scan_fn(carry_flat, input):
            t, u, y = input[0], input[1 : nu + 1], input[nu + 1 :]
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

    def residuals(
        self, decision_variables: Tree | tuple[np.ndarray, Tree]
    ) -> np.ndarray:
        """Evaluate prediction error residuals for least-squares optimization.

        Parameters
        ----------
        decision_variables : Tree or tuple[np.ndarray, Tree]
            Optimization parameters. If ``x0`` is provided during construction,
            this contains only model parameters. Otherwise, it contains both
            initial state and parameters as a flattened array.

        Returns
        -------
        residuals : ndarray
            Flattened prediction errors of shape ``(ny * N,)`` suitable for
            least-squares optimization methods like Levenberg-Marquardt.
        """
        if self.x0 is not None:
            x0 = self.x0
            _, params = decision_variables  # Dummy x0 for interface compatibility
        else:
            x0, params = decision_variables

        results = self.forward(x0, params)
        e: np.ndarray = results["e"].flatten()
        return e

    def __call__(self, decision_variables: Tree | tuple[np.ndarray, Tree]) -> float:
        """Evaluate cost function for general optimization methods.

        Parameters
        ----------
        decision_variables : array_like
            Optimization parameters. If ``x0`` is provided during construction,
            this contains only model parameters. Otherwise, it contains both
            initial state and parameters as a flattened array.

        Returns
        -------
        cost : float
            Average prediction error cost function value.

        Notes
        -----
        This method provides the scalar cost function interface needed for
        general optimization methods (BFGS, IPOPT, etc.).
        """
        if self.x0 is not None:
            x0 = self.x0
            _, params = decision_variables  # Dummy x0 for interface compatibility
        else:
            x0, params = decision_variables

        results = self.forward(x0, params)
        V: float = results["V"]
        return V


@tree.struct
class CompoundPEMObjective:
    """Compound PEM objective function for multiple sub-objectives.

    Shares the same interface as `PEMObjective` but allows
    combining multiple PEM objectives into a single cost function.
    """

    sub_objectives: list[PEMObjective]

    def _make_sub_dvs(
        self, decision_variables: tuple[np.ndarray, Tree]
    ) -> list[tuple[np.ndarray | None, Tree]]:
        """Prepare sub-decision variables for each sub-objective."""
        sub_x0, params = decision_variables
        sub_dvs: list[tuple[np.ndarray | None, Tree]] = []

        for x0, pem_obj in zip(sub_x0, self.sub_objectives):
            if pem_obj.x0 is not None:
                # If x0 is provided, use it directly
                sub_dvs.append((None, params))

            else:
                # Otherwise, include x0 in the decision variables
                sub_dvs.append((x0, params))

        return sub_dvs

    def __call__(self, decision_variables: tuple[np.ndarray, Tree]) -> float:
        """Evaluate cost function for compound PEM objective."""
        sub_dvs = self._make_sub_dvs(decision_variables)
        total_cost = 0.0
        for dvs, pem_obj in zip(sub_dvs, self.sub_objectives):
            cost = pem_obj(dvs)
            total_cost += cost

        return total_cost

    def residuals(
        self, decision_variables: Tree | tuple[np.ndarray, Tree]
    ) -> np.ndarray:
        """Evaluate residuals for compound PEM objective."""
        sub_dvs = self._make_sub_dvs(decision_variables)
        total_residuals = []
        for dvs, pem_obj in zip(sub_dvs, self.sub_objectives):
            residuals = pem_obj.residuals(dvs)
            total_residuals.append(residuals)

        if len(total_residuals) == 1:
            return total_residuals[0]

        return np.hstack(total_residuals)


def _pem_solve_lm(
    pem_obj: CompoundPEMObjective,
    p_guess: T,
    bounds: tuple[T, T] | None = None,
    options: dict | None = None,
) -> OptimizeResult:
    """Solve the PEM problem using Levenberg-Marquardt optimization."""
    if options is None:
        options = {}

    # Set reasonable defaults for system identification problems
    default_options = {
        "ftol": 1e-4,
        "xtol": 1e-6,
        "gtol": 1e-6,
        "max_nfev": 200,
    }
    options = {**default_options, **options}
    result: OptimizeResult = lm_solve(
        pem_obj.residuals, p_guess, bounds=bounds, **options
    )
    return result


def _pem_solve_bfgs(
    pem_obj: CompoundPEMObjective,
    p_guess: T,
    bounds: tuple[T, T] | None = None,
    options: dict | None = None,
) -> OptimizeResult:
    method = "BFGS" if bounds is None else "L-BFGS-B"

    if options is None:
        options = {}

    p_guess_flat, unravel = arc.tree.ravel(p_guess)

    if bounds is not None:
        lb, ub = bounds
        lb_flat, _ = arc.tree.ravel(lb)
        ub_flat, _ = arc.tree.ravel(ub)
        # Zip bounds into (lb, ub) for each parameter
        bounds = list(zip(lb_flat, ub_flat))  # type: ignore

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

    options = {**default_options, **options}

    result: OptimizeResult = scipy_minimize(
        func,
        p_guess_flat,
        method=method,
        jac=jac,
        bounds=bounds,
        options=options,
    )

    # Replace the flat parameters with the original tree structure
    result.x = unravel(result.x)
    return result


def _pem_solve_ipopt(
    pem_obj: CompoundPEMObjective,
    p_guess: T,
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
    default_options: dict[str, str | float | int] = {}

    options["ipopt"] = {**ipopt_default_options, **options.get("ipopt", {})}
    options = {**default_options, **options}

    p_guess_flat, unravel = arc.tree.ravel(p_guess)
    if bounds is not None:
        lb, ub = bounds
        lb_flat, _ = arc.tree.ravel(lb)
        ub_flat, _ = arc.tree.ravel(ub)
        bounds = (lb_flat, ub_flat)  # type: ignore

    # Define an objective and gradient function for BFGS
    @arc.compile
    def func(params_flat):
        return pem_obj(unravel(params_flat))

    result: OptimizeResult = arc.minimize(
        func,
        p_guess_flat,
        bounds=bounds,
        options=options,
    )

    result.x = unravel(result.x)
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
    data: Timeseries | tuple[Timeseries, ...],
    p_guess: T,
    x0: np.ndarray | tuple[np.ndarray, ...],
    estimate_x0: bool | tuple[bool, ...] = False,
    bounds: tuple[T, T] | None = None,
    P0: np.ndarray | None | tuple[np.ndarray | None, ...] = None,
    method: str = "lm",
    options: dict | None = None,
) -> OptimizeResult:
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

        All arrays must have consistent time dimensions.  Multiple
        :class:`Timeseries` instances can be provided as a tuple for
        multi-experiment identification, allowing joint parameter estimation
        across different datasets.

        If multiple datasets are provided, the initial state estimates
        (``x0``) should also be a tuple of initial conditions corresponding
        to each dataset.
    p_guess : Tree
        Initial parameter guess with arbitrary nested structure
        (e.g., ``{"mass": 1.0, "damping": {"c1": 0.1, "c2": 0.2}}``).
        The optimization preserves this structure in the result, enabling
        natural organization of physical parameters.
    x0 : array_like, optional
        Initial state estimate of shape ``(nx,)``.  Used as an initial guess
        if ``estimate_x0=True``.  For multiple datasets, this can be a tuple
        of initial conditions, allowing different initial states for each
        dataset.
    estimate_x0 : bool, default=False
        Whether to estimate the initial state ``x0`` along with parameters.
    bounds : tuple of (Tree, Tree), optional
        Parameter bounds as ``(lower_bounds, upper_bounds)`` with the
        same tree structure as ``p_guess``. Enables physical
        constraints such as:

        - Positive masses, stiffnesses, damping coefficients
        - Bounded gain parameters, time constants
        - Realistic physical parameter ranges

        Use ``-np.inf`` and ``np.inf`` for unbounded parameters.
    P0 : array_like, optional
        Initial state covariance matrix of shape ``(nx, nx)``. If None,
        defaults to identity matrix. Represents uncertainty in initial
        state estimate.  Can be a tuple of matrices if multiple
        datasets are provided, allowing different initial uncertainties
        for each dataset.
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
        - ``max_nfev`` : Maximum function evaluations (default: 200)
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
        preserving the original tree structure. Additional fields include:

        - ``success`` : Whether estimation succeeded
        - ``fun`` : Final prediction error objective value
        - ``nit`` : Number of optimization iterations
        - ``history`` : Detailed convergence history

    Notes
    -----
    This implementation provides:

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

    **Multi-Experiment Support**:
        Can handle multiple datasets simultaneously, allowing joint parameter
        estimation across different experiments or operating conditions.

    The method automatically computes gradients with respect to both initial
    conditions (when ``estimate_x0=True``) and model parameters using efficient
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
    >>> p_guess = {"omega_n": 2.5, "zeta": 0.5}
    >>>
    >>> # Estimate parameters with known initial conditions
    >>> result = pem(ekf, data, p_guess, x0=x0)
    >>> print(f"Estimated parameters: {result.p}")
    Estimated parameters: {'omega_n': array(1.9709515), 'zeta': array(0.11517324)}
    >>> print(f"Converged in {result.nit} iterations")
    Converged in 26 iterations
    >>>
    >>> # With physical parameter constraints
    >>> bounds = (
    ...     {"omega_n": 0.0, "zeta": 0.0},     # Lower bounds (positive values)
    ...     {"omega_n": 10.0, "zeta": 1.0},    # Upper bounds (reasonable ranges)
    ... )
    >>> result = pem(ekf, data, p_guess, x0=x0, bounds=bounds)

    See Also
    --------
    archimedes.optimize.lm_solve : Underlying Levenberg-Marquardt optimizer
    Timeseries : Data container for input-output time series
    archimedes.observers.ExtendedKalmanFilter : EKF for mildly nonlinear systems
    archimedes.observers.UnscentedKalmanFilter : UKF for highly nonlinear systems
    archimedes.discretize : Convert continuous-time dynamics to discrete-time

    References
    ----------
    .. [1] Ljung, L. "System Identification: Theory for the User." 2nd edition,
           Prentice Hall, 1999.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}.")

    pem_solve = SUPPORTED_METHODS[method]

    if not isinstance(data, (tuple, list)):
        data = (data,)

    n_expt = len(data)

    # For each argument, replicate for all experiments if needed

    if isinstance(x0, np.ndarray):
        x0 = tuple([x0] * n_expt)

    if P0 is None or isinstance(P0, np.ndarray):
        P0 = tuple([P0] * n_expt)

    if isinstance(estimate_x0, bool):
        estimate_x0 = tuple([estimate_x0] * n_expt)

    sub_objectives = []
    x0_guess: list[np.ndarray | None] = []
    for i in range(n_expt):
        if estimate_x0[i]:
            x0_guess.append(x0[i])
            sub_x0 = None
        else:
            x0_guess.append(None)
            sub_x0 = x0[i]

        sub_objectives.append(
            PEMObjective(
                predictor=predictor,
                data=data[i],
                P0=P0[i],
                x0=sub_x0,
            )
        )

    objective = CompoundPEMObjective(sub_objectives=sub_objectives)
    dvs_guess = (x0_guess, p_guess)

    if bounds is not None:
        x0_bounds = [None] * n_expt
        lb, ub = bounds
        lb = (x0_bounds, lb)  # type: ignore
        ub = (x0_bounds, ub)  # type: ignore
        bounds = (lb, ub)

    result: OptimizeResult = pem_solve(
        pem_obj=objective,
        p_guess=dvs_guess,
        bounds=bounds,
        options=options,
    )

    x0_opt, params_opt = result.x
    delattr(result, "x")
    result.p = params_opt

    if n_expt == 1:
        x0_opt = x0_opt[0]

    result.x0 = x0_opt  # Store optimized initial condition

    return result
