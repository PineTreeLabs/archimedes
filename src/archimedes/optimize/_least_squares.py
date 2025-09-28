from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares as scipy_lstsq

import archimedes as arc
from archimedes import tree

from ._common import _ravel_args
from ._lm import lm_solve

if TYPE_CHECKING:
    from ..typing import Tree

    T = TypeVar("T", bound=Tree)


__all__ = ["least_squares"]


SCIPY_METHODS = ["trf", "dogbox", "lm"]
SUPPORTED_METHODS = SCIPY_METHODS + ["hess-lm"]


def least_squares(
    func: Callable[[T, Any], T],
    x0: T,
    args: tuple[Any, ...] = (),
    method: str = "hess-lm",
    bounds: tuple[T, T] | None = None,
    options: dict | None = None,
) -> OptimizeResult:
    """
    Solve nonlinear least squares problems

    Minimize the sum of squares of residuals:

    .. code-block:: text

        minimize    0.5 * ||r(x)||²
        subject to  lb <= x <= ub

    This function provides access to both custom least-squares algorithms and standard
    SciPy methods, with full tree parameter structure support and automatic
    differentiation. The custom ``"hess-lm"`` method can provide superior performance
    for system identification and parameter estimation problems.

    Parameters
    ----------
    func : callable
        Residual function with signature ``func(x, *args) -> residuals``, where
        ``residuals`` is an array-like object. The parameter ``x`` can be any
        tree structure matching ``x0``.
    x0 : Tree
        Initial parameter guess. Can be a flat array, nested dictionary, dataclass,
        or any tree structure. The solution preserves this exact structure.

        Examples::

            # Physical system parameters
            x0 = {
                "dynamics": {"mass": 1.0, "damping": 0.1},
                "sensor": {"bias": 0.0, "scale": 1.0}
            }

            # Simple array
            x0 = np.array([1.0, 0.1, 0.0, 1.0])

    args : tuple, optional
        Extra arguments passed to the residual function.
    method : str, optional
        Least-squares method to use. Default is ``"hess-lm"``. Available methods:

        **Archimedes Methods:**
            - ``"hess-lm"`` (default): Custom Levenberg-Marquardt with direct Hessian.
              Supports box constraints and specialized convergence criteria.

        **SciPy Methods:**
            - ``"trf"``: Trust Region Reflective, robust for large problems
            - ``"dogbox"``: Dog-leg method in rectangular trust regions
            - ``"lm"``: Standard SciPy Levenberg-Marquardt (unconstrained only)

    bounds : tuple of (Tree, Tree), optional
        Box constraints specified as ``(lower_bounds, upper_bounds)`` with the same
        tree structure as ``x0``. Use ``-np.inf`` and ``np.inf`` for unbounded
        variables.

        Examples::

            # tree bounds for physical constraints
            bounds = (
                {"dynamics": {"mass": 0.1, "damping": 0.0}},  # Lower bounds
                {"dynamics": {"mass": 10.0, "damping": 1.0}}   # Upper bounds
            )

            # Array bounds
            bounds = (np.array([0.1, 0.0]), np.array([10.0, 1.0]))

    options : dict, optional
        Method-specific options. For ``"hess-lm"``, see :py:func:`lm_solve`
        documentation. For SciPy methods, see ``scipy.optimize.least_squares``
        documentation.

    Returns
    -------
    result : OptimizeResult
        Optimization result with preserved tree structure:

        - ``x`` : Solution parameters (same tree structure as ``x0``)
        - ``success`` : Whether optimization converged successfully
        - ``status`` : Termination status (LMStatus for "hess-lm")
        - ``message`` : Descriptive termination message
        - ``fun`` : Final residual vector
        - ``cost`` : Final objective value (0.5 * ||residuals||²)
        - ``nfev`` : Number of function evaluations
        - ``nit`` : Number of iterations
        - Additional method-specific fields

    Notes
    -----
    **Tree Parameter Organization:**

    Tree support enables natural organization of complex parameter structures::

        # Organized system parameters
        params = {
            "mechanical": {
                "mass": 1.0,
                "damping": {"viscous": 0.1, "coulomb": 0.05},
                "stiffness": {"linear": 100.0, "cubic": 0.001}
            },
            "electrical": {
                "resistance": 10.0,
                "inductance": 0.01,
                "capacitance": 1e-6
            },
            "initial_conditions": np.array([0.0, 0.0])
        }

        # After optimization - same structure preserved
        result = least_squares(residuals_func, params)
        optimized_params = result.x  # Maintains full nested structure

    **Custom Hessian Method** (``"hess-lm"``):

    The default method leverages specialized Hessian approximations for efficient
    parameter estimation.  This method solves the damped normal equations directly
    using a Cholesky factorization instead of the typical QR approach.  This can
    improve performance for problems such as parameter estimation where there are
    many residuals compared to the number of parameters.  This method also handles
    box constraints by switching to a quadratic programming solver
    ([OSQP](https://osqp.org/)) when bounds are active.

    Key options passed via ``options`` dict::

        # Common options for custom LM method
        lm_options = {
            "ftol": 1e-6,      # Function tolerance
            "xtol": 1e-6,      # Parameter tolerance
            "gtol": 1e-6,      # Gradient tolerance
            "max_nfev": 200,     # Maximum function evaluations
            "lambda0": 1e-3,   # Initial damping parameter
        }
        result = least_squares(func, x0, method="hess-lm", options=lm_options)

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> def rosenbrock(x):
    ...     return np.hstack([100 * (x[1] - x[0]**2), 1 - x[0]])
    >>>
    >>> x0 = np.array([0.0, 0.0])
    >>> bounds = (np.array([0.0, 0.0]), np.array([0.8, 2.0]))
    >>> result = arc.optimize.least_squares(rosenbrock, x0, bounds=bounds)
    >>> print(result.x)  # Optimized parameters
    [0.79999903 0.63998596]

    See Also
    --------
    lm_solve : Direct access to custom Levenberg-Marquardt algorithm
    minimize : General nonlinear programming with multiple solver options
    scipy.optimize.least_squares : SciPy's least-squares interface
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Method '{method}' is not supported. "
            f"Supported methods are: {', '.join(SUPPORTED_METHODS)}."
        )

    if options is None:
        options = {}

    # Custom implementation
    if method == "hess-lm":
        return lm_solve(
            func=func,
            x0=x0,
            args=args,
            bounds=bounds,
            **options,
        )

    x0_flat, flat_bounds, unravel = _ravel_args(x0, bounds)
    if flat_bounds is None:
        flat_bounds = (-np.inf, np.inf)

    # Compile the function and Jacobian
    @arc.compile
    def obj_func(x_flat):
        x = unravel(x_flat)
        r = func(x, *args)
        return tree.ravel(r)[0]  # Return flattened residuals

    # Call the scipy least_squares function
    result = scipy_lstsq(
        obj_func,
        x0_flat,
        args=args,
        jac=arc.jac(obj_func),
        method=method,
        bounds=flat_bounds,
        **options,
    )
    result.x = unravel(result.x)  # Unravel the result back to original shape

    return result
