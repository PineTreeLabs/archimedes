"""Defining and solving nonlinear problems

This interface is patterned after the `scipy.optimize` module, but with
additional functionality for solving nonlinear problems symbolically. It
also dispatches to IPOPT rather than solvers available in SciPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, TypeVar, cast

import casadi as cs
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

from archimedes import tree
from archimedes._core import (
    FunctionCache,
    SymbolicArray,
    _as_casadi_array,
    array,
    compile,
    grad,
    sym_like,
)

from ..typing import Tree
from ._common import _ravel_args

if TYPE_CHECKING:
    from ..typing import ArrayLike

__all__ = [
    "nlp_solver",
    "minimize",
]

T = TypeVar("T", bound=Tree)


class UnravelContainer(Generic[T]):
    def __init__(self):
        self._unravel: Callable[[ArrayLike], T] | None = None

    def __call__(self, x: ArrayLike) -> T:
        unravel = self._unravel
        if TYPE_CHECKING:
            unravel = cast(Callable[[ArrayLike], T], self._unravel)
        return unravel(x)


def _make_nlp_solver(
    obj: Callable,
    constr: Callable | None = None,
    static_argnames: str | Sequence[str] | None = None,
    constrain_x: bool = False,
    name: str | None = None,
    method: str = "ipopt",
    options: dict | None = None,
) -> FunctionCache:
    if not isinstance(obj, FunctionCache):
        obj = FunctionCache(obj, static_argnames=static_argnames)

    if constr is not None:
        if not isinstance(constr, FunctionCache):
            constr = FunctionCache(constr, static_argnames=static_argnames)

        # Check that arguments and static arguments are the same for both functions
        if not len(obj.arg_names) == len(constr.arg_names):
            raise ValueError(
                "Objective and constraint functions must have the same number of "
                "arguments"
            )

        if not len(obj.static_argnums) == len(constr.static_argnums):
            raise ValueError(
                "Objective and constraint functions must have the same number of "
                "static arguments"
            )

    unravel: UnravelContainer = UnravelContainer()

    # Define a function that will solve the NLP
    # This function will be evaluated with SymbolicArray objects.
    def _solve(x0, lbx, ubx, lbg, ubg, *args) -> ArrayLike:
        x0 = array(x0)
        ret_shape = x0.shape
        ret_dtype = x0.dtype

        # We have to flatten all of the symbolic user-defined arguments
        # into a single vector to pass to CasADi.  If there is static data,
        # this needs to be stripped out and passed separately.
        if static_argnames:
            # The first argument is `x`, so skip this in checking
            # for static arguments.
            static_argnums = [i - 1 for i in obj.static_argnums]
            _static_args, sym_args, _arg_types = obj.split_args(static_argnums, *args)

        else:
            # No static data - all arguments can be treated symbolically
            sym_args = args

        # Flatten the arguments into a single array `p`.  If necessary,
        # this could be unflattened using `_param_struct.unravel`
        p, _unravel = tree.ravel(sym_args)

        # Define a state variable for the optimization
        x_flat = sym_like(x0, name="x", kind="MX")
        x = unravel(x_flat)
        f = obj(x, *args)

        # For type checing
        f = cast(SymbolicArray, f)

        nlp = {
            "x": x_flat._sym,
            "f": f._sym,
        }

        if p.size != 0:
            p = cast(SymbolicArray, p)
            nlp["p"] = p._sym

        if constr is not None:
            g = constr(x, *args)
            g = cast(SymbolicArray, g)
            nlp["g"] = g._sym

        solver = cs.nlpsol("solver", method, nlp, options)

        # Before calling the CasADi solver interface, make sure everything is
        # either a CasADi symbol or a NumPy array
        p_arg = False if p is None else p
        x0, lbx, ubx, lbg, ubg, p_arg = map(
            _as_casadi_array,
            (x0, lbx, ubx, lbg, ubg, p_arg),
        )

        # The return is a dict with keys `f`, `g`, `x` (and dual variables)
        sol = solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
            p=p_arg,
        )

        # For now we only return the state variable `x`
        return SymbolicArray(
            sol["x"],
            dtype=ret_dtype,
            shape=ret_shape,
        )

    # The first arguments to the function will be the decision variables
    # and constraints, otherwise the args will be user defined
    arg_names = ["x0", "lbx", "ubx", "lbg", "ubg", *obj.arg_names[1:]]

    # Close over unneeded arguments depending on the constraint configuration
    # There are four possibilities for call signatures, depending on whether
    # there are bounds on the decision variables and constraints - the need
    # to explicitly enumerate these is a result of the way CasADi constructs
    # the callable objects
    constrain_g = constr is not None
    if not constrain_x:
        if not constrain_g:

            def _solve_explicit(x0, *args):  # type: ignore[misc]
                return _solve(x0, -np.inf, np.inf, -np.inf, np.inf, *args)

            arg_names.remove("lbg")
            arg_names.remove("ubg")

        else:

            def _solve_explicit(x0, lbg, ubg, *args):  # type: ignore[misc]
                return _solve(x0, -np.inf, np.inf, lbg, ubg, *args)

        arg_names.remove("lbx")
        arg_names.remove("ubx")

    else:
        if not constrain_g:

            def _solve_explicit(x0, lbx, ubx, *args):  # type: ignore[misc]
                return _solve(x0, lbx, ubx, -np.inf, np.inf, *args)

            arg_names.remove("lbg")
            arg_names.remove("ubg")

        else:

            def _solve_explicit(x0, lbx, ubx, lbg, ubg, *args):  # type: ignore[misc]
                return _solve(x0, lbx, ubx, lbg, ubg, *args)

    if name is None:
        name = f"{obj.name}_nlp"

    # Wrap for tree compatibility
    def _wrapped_solve(x0, *args):
        if constrain_x:
            bounds = args[:2]
            args = args[2:]
        else:
            bounds = None

        x0_flat, bounds_flat, unravel_func = _ravel_args(x0, bounds)
        unravel._unravel = unravel_func  # store unravel function

        if bounds_flat is not None:
            lbx, ubx = bounds_flat
            args = (lbx, ubx, *args)

        x_opt_flat = _solve_explicit(x0_flat, *args)
        return unravel(x_opt_flat)

    _wrapped_solve.__name__ = name

    return FunctionCache(
        _wrapped_solve,
        arg_names=tuple(arg_names),
        static_argnames=static_argnames,
        kind="MX",
    )


def nlp_solver(
    obj: Callable,
    constr: Callable | None = None,
    static_argnames: str | Sequence[str] | None = None,
    constrain_x: bool = False,
    name: str | None = None,
    method: str = "ipopt",
    **options,
) -> FunctionCache:
    """Create a reusable solver for a nonlinear optimization problem.

    This function transforms an objective function and optional constraint function
    into an efficient solver for nonlinear programming problems of the form:

    .. code-block:: text

        minimize        f(x, p)
        subject to      lbx <= x <= ubx
                        lbg <= g(x, p) <= ubg

    where ``x`` represents decision variables and ``p`` represents parameters.

    Parameters
    ----------
    obj : callable
        Objective function to minimize with signature ``obj(x, *args)``.
        Must return a scalar value.
    constr : callable, optional
        Constraint function with signature ``constr(x, *args)``.
        Must return a vector of constraint values where the constraints
        are interpreted as ``lbg <= constr(x, *args) <= ubg``.
    static_argnames : tuple of str, optional
        Names of arguments in ``obj`` and ``constr`` that should be treated
        as static parameters rather than symbolic variables. Static arguments
        are not differentiated through and the solver will be recompiled when
        their values change.
    constrain_x : bool, default=False
        If True, the solver will accept bounds on decision variables ``(lbx, ubx)``.
        If False, no bounds on ``x`` will be applied (equivalent to ``-∞ <= x <= ∞``).
    name : str, optional
        Name for the resulting solver function. If None, a name will be
        generated based on the objective function name.
    method : str, optional
        The optimization method to use. Default is "ipopt". See CasADi documentation
        for available methods.
    **options : dict
        Additional options passed to the underlying optimization solver.
        See :py:func:`minimize` and the
        [CasADi documentation](https://web.casadi.org/python-api/#nlp) for available
        options.

    Returns
    -------
    solver : FunctionCache
        A callable function that solves the nonlinear optimization problem.
        The signature of this function depends on the values of ``constrain_x``
        and whether a constraint function was provided:

        - With constraints and x bounds: ``solver(x0, lbx, ubx, lbg, ubg, *args)``

        - With constraints, no x bounds: ``solver(x0, lbg, ubg, *args)``

        - With x bounds, no constraints: ``solver(x0, lbx, ubx, *args)``

        - No constraints or x bounds: ``solver(x0, *args)``

        The returned solver can be evaluated both numerically and symbolically.

    Notes
    -----

    By default the NLP solver uses the IPOPT interior point method which is suitable
    for large-scale nonlinear problems.  See the :py:func:`minimize` documentation for
    additional solvers and configuration options.

    The function leverages automatic differentiation to compute exact derivatives of
    the objective and constraints, unless this behavior is overridden via configuration
    (e.g. by passing a custom evaluation function or using an L-BFGS approximation).

    Both ``obj` and `constr`` must accept the same arguments, and if
    ``static_argnames`` is specified, the static arguments must be the same for both
    functions.

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Define the Rosenbrock function
    >>> def f(x):
    ...     return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    >>>
    >>> # Define a constraint function
    >>> def g(x):
    ...     g1 = (x[0] - 1)**3 - x[1] + 1
    ...     g2 = x[0] + x[1] - 2
    ...     return np.array([g1, g2], like=x)
    >>>
    >>> # Create the solver
    >>> solver = arc.nlp_solver(f, constr=g)
    >>>
    >>> # Initial guess
    >>> x0 = np.array([2.0, 0.0])
    >>>
    >>> # Constraint bounds: g <= 0
    >>> lbg = -np.inf * np.ones(2)
    >>> ubg = np.zeros(2)
    >>>
    >>> # Solve the problem
    >>> x_opt = solver(x0, lbg, ubg)
    >>> print(x_opt)
    [0.99998266 1.00000688]

    See Also
    --------
    minimize : One-time solver for nonlinear optimization problems
    implicit : Create a function that solves F(x, p) = 0 for x
    """
    # TODO: Inspect function signature

    return _make_nlp_solver(
        obj,
        constr=constr,
        static_argnames=static_argnames,
        constrain_x=constrain_x,
        name=name,
        method=method,
        options=options,
    )


SCIPY_METHODS = ["BFGS", "L-BFGS-B"]
CASADI_METHODS = ["ipopt", "sqpmethod", "blocksqp", "feasiblesqpmethod"]
SUPPORTED_METHDOS = SCIPY_METHODS + CASADI_METHODS


def _minimize_with_scipy(
    obj: Callable,
    x0: T,
    args: Sequence[Any] = (),
    static_argnames: str | Sequence[str] | None = None,
    constr: Callable | None = None,
    bounds: T | None = None,
    constr_bounds: ArrayLike | None = None,
    method: str = "bfgs",
    options: dict | None = None,
) -> OptimizeResult:
    if constr is not None or constr_bounds is not None:
        raise NotImplementedError(
            "SciPy wrapper does not yet support constraints. "
            "Use the CasADi interface (e.g. IPOPT) for constrained optimization."
        )

    if options is None:
        options = {}

    x0_flat, bounds, unravel = _ravel_args(x0, bounds, zip_bounds=True)

    # Define objective, gradient, and Hessian functions
    @compile
    def func(x_flat):
        return obj(unravel(x_flat), *args)

    jac = grad(func)

    result = scipy_minimize(
        func,
        x0_flat,
        method=method,
        jac=jac,
        bounds=bounds,
        options=options,
    )

    result.x = unravel(result.x)
    return result


def minimize(
    obj: Callable,
    x0: T,
    args: Sequence[Any] = (),
    static_argnames: str | Sequence[str] | None = None,
    constr: Callable | None = None,
    bounds: tuple[T, T] | None = None,
    constr_bounds: ArrayLike | None = None,
    method: str = "ipopt",
    options: dict | None = None,
) -> OptimizeResult:
    """
    Minimize a scalar function with optional constraints and tree support.

    Solve a nonlinear programming problem of the form:

    .. code-block:: text

        minimize        f(x, p)
        subject to      lbx <= x <= ubx
                        lbg <= g(x, p) <= ubg

    This function provides a unified interface to multiple optimization methods,
    including CasADi-based solvers (IPOPT, SQP) and SciPy optimizers (BFGS, L-BFGS-B),
    with automatic differentiation and native tree parameter structure support.

    **Key Features:**
        - **Tree Support**: Parameters can be nested dictionaries, dataclasses, or
            any tree structure
        - **Automatic Differentiation**: Exact gradients and Hessians (as needed)
            computed efficiently and automatically

    Parameters
    ----------
    obj : callable
        Objective function to minimize, with signature ``obj(x, *args)``.
        Must return a scalar value. The function ``x`` parameter can be a Tree
        matching the structure of ``x0``.
    x0 : Tree
        Initial guess for the optimization. Can be a flat array, nested dictionary,
        dataclass, or any tree structure. The solution will preserve this structure.

        Examples::

            # Flat array
            x0 = np.array([1.0, 2.0])

            # Nested dictionary (preserved in solution)
            x0 = {"mass": 1.0, "damping": {"c1": 0.1, "c2": 0.2}}

            # Dataclass-like nested structure
            @archimedes.struct
            class Params:
                mass: float
                stiffness: float

            x0 = Params(mass=1.0, stiffness=100.0)

    args : tuple, optional
        Extra arguments passed to the objective and constraint functions.
    static_argnames : tuple of str, optional
        Names of arguments that should be treated as static (non-symbolic)
        parameters. Static arguments are not differentiated through and trigger
        solver recompilation when changed. Useful for discrete parameters.
    constr : callable, optional
        Constraint function with the same signature as ``obj``.
        Must return an array of constraint values where the constraints
        are interpreted as ``lbg <= constr(x, *args) <= ubg``.
        Note: SciPy methods (BFGS, L-BFGS-B) do not support general constraints.
    bounds : tuple, optional
        Bounds on the decision variables, given as a tuple ``(lb, ub)``.
        Each bound must have the same tree structure as ``x0``. Use ``-np.inf``
        and ``np.inf`` for unbounded variables.

        Examples::

            # Array bounds
            bounds = (np.array([0.0, -1.0]), np.array([10.0, 1.0]))

            # tree bounds (matching x0 structure)
            bounds = (
                {"mass": 0.1, "damping": {"c1": 0.0, "c2": 0.0}},  # lower bounds
                {"mass": 10.0, "damping": {"c1": 1.0, "c2": 1.0}}  # upper bounds
            )

    constr_bounds : tuple of (array_like, array_like), optional
        Bounds on the constraint values, given as a tuple ``(lbg, ubg)``.
        If None and constr is provided, defaults to ``(0, 0)`` for equality constraints
    method : str, optional
        The optimization method to use. Default is "ipopt". Available methods:

        **CasADi Methods (support constraints):**
            - ``"ipopt"`` (default): Interior point method, excellent for large
                constrained problems
            - ``"sqpmethod"``: Sequential quadratic programming
            - ``"blocksqp"``: Block-structured SQP for large problems
            - ``"feasiblesqpmethod"``: SQP with feasibility restoration

        **SciPy Methods (unconstrained or box-constrained only):**
            - ``"BFGS"``: Quasi-Newton method with automatic exact gradients
            - ``"L-BFGS-B"``: Limited-memory BFGS with box constraints

    options : dict, optional
        Method-specific options. See Notes section for details.

    Returns
    -------
    result : OptimizeResult
        Optimization result with tree structure preserved:

        - ``x`` : Solution parameters (same tree structure as ``x0``)
        - ``success`` : Whether optimization succeeded
        - ``message`` : Descriptive termination message
        - ``fun`` : Final objective value
        - ``nfev`` : Number of function evaluations
        - Additional method-specific fields

    Notes
    -----
    **Method Selection Guide:**

    - **Constrained problems**: Use ``"ipopt"`` (default) or SQP methods
    - **Unconstrained problems**: Use ``"BFGS"`` for fast convergence
    - **Box constraints only**: Use ``"L-BFGS-B"`` for memory efficiency
    - **Least-squares problems**: Use :py:func:`least_squares` with ``method="hess-lm"``
      for specialized Levenberg-Marquardt algorithm

    **Automatic Differentiation:**

    All methods use exact gradients computed via automatic differentiation.
    For CasADi methods, both gradients and Hessians are computed exactly.
    For SciPy methods, gradients are exact and Hessians are approximated using BFGS.

    **Tree Structure Preservation:**

    The solution ``result.x`` maintains the exact same nested structure as the
    initial guess ``x0``. This enables natural parameter organization for
    complex models::

        # Initial parameters
        params = {"dynamics": {"mass": 1.0, "damping": 0.1}, "controller": {"kp": 1.0}}

        # After optimization - same structure
        result = minimize(objective, params)
        final_params = result.x  # Has same nested structure
        print(final_params["dynamics"]["mass"])  # Optimized mass value

    **IPOPT Configuration** (method="ipopt"):

    Common IPOPT options passed via ``options={"ipopt": {...}}``::

        ipopt_opts = {
            "print_level": 0,        # Suppress output
            "max_iter": 100,         # Maximum iterations
            "tol": 1e-6,            # Convergence tolerance
            "acceptable_tol": 1e-4,  # Acceptable tolerance
        }
        result = minimize(obj, x0, method="ipopt", options={"ipopt": ipopt_opts})

    **SQP Configuration** (method="sqpmethod"):

    Common SQP options passed directly via ``options``::

        options = {
            "qpsol": "osqp",                    # QP solver
            "hessian_approximation": "exact",   # Use exact Hessian
            "max_iter": 50,                     # SQP iterations
            "tol_pr": 1e-6,                     # Primal feasibility tolerance
            "tol_du": 1e-6,                     # Dual feasibility tolerance
            }
        result = minimize(obj, x0, method="sqpmethod", options=options)

    **SciPy Integration** (method="BFGS" or "L-BFGS-B"):

    SciPy methods receive exact gradients via autodiff and support standard SciPy
    options::

        result = minimize(
            obj, x0,
            method="BFGS",
            options={"gtol": 1e-8, "maxiter": 100}
        )

    Examples
    --------
    **Basic Usage:**

    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Rosenbrock function with structured parameters
    >>> def rosenbrock(params):
    ...     x, y = params["x"], params["y"]
    ...     return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
    >>>
    >>> # Initial guess
    >>> x0 = {"x": 2.0, "y": 1.0}
    >>>
    >>> # Unconstrained optimization
    >>> result = arc.optimize.minimize(rosenbrock, x0, method="BFGS")
    >>> print(result.x)  # Preserves tree structure
    {'x': 0.9999999999999994, 'y': 0.9999999999999989}

    **Constrained Optimization:**

    >>> # Add constraints
    >>> def constraint(params):
    ...     x, y = params["x"], params["y"]
    ...     return x + y - 1.5  # x + y >= 1.5
    >>>
    >>> # Solve with inequality constraint
    >>> result = arc.optimize.minimize(
    ...     rosenbrock, x0,
    ...     constr=constraint,
    ...     constr_bounds=(0.0, np.inf),  # g(x) >= 0.0
    ...     method="ipopt"
    ... )

    **Box Constraints with structured data:**

    >>> # Tree bounds matching x0 structure
    >>> bounds = (
    ...     {"x": 0.0, "y": 0.0},      # Lower bounds
    ...     {"x": 2.0, "y": 2.0}       # Upper bounds
    ... )
    >>> result = arc.optimize.minimize(rosenbrock, x0, bounds=bounds)

    See Also
    --------
    least_squares : Specialized Levenberg-Marquardt solver for residual minimization
    nlp_solver : Create a reusable solver for repeated optimization
    root : Find roots of nonlinear equations
    scipy.optimize.minimize : SciPy's optimization interface
    """
    if method not in SUPPORTED_METHDOS:
        raise ValueError(
            f"Unsupported method: {method}. Supported methods are: {SUPPORTED_METHDOS}"
        )

    if method in SCIPY_METHODS:
        return _minimize_with_scipy(
            obj,
            x0=x0,
            args=args,
            static_argnames=static_argnames,
            constr=constr,
            bounds=bounds,
            constr_bounds=constr_bounds,
            method=method,
            options=options,
        )

    if options is None:
        options = {}

    solver = nlp_solver(
        obj,
        static_argnames=static_argnames,
        constr=constr,
        constrain_x=bounds is not None,
        method=method,
        **options,
    )

    # Construct a list of arguments to the solver. The content
    # of this will depend on the configuration of constraints.
    solver_args: dict[str, T | ArrayLike] = {"x0": x0}

    # Add bounds on the state variables
    if bounds is not None:
        lbx, ubx = bounds
        solver_args = {**solver_args, "lbx": lbx, "ubx": ubx}

    # Add bounds on the constraints
    if constr is not None:
        if constr_bounds is not None:
            lbg, ubg = constr_bounds
        else:
            lbg, ubg = 0, 0
        solver_args = {**solver_args, "lbg": lbg, "ubg": ubg}

    # Add the varargs
    arg_names = [name for name in solver.arg_names if name not in solver_args]
    solver_args = {**solver_args, **dict(zip(arg_names, args))}

    x = solver(**solver_args)
    return OptimizeResult(
        x=x,
        success=True,
        message="Optimization terminated successfully.",
    )
