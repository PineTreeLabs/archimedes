"""Defining and solving nonlinear problems

This interface is patterned after the `scipy.optimize` module, but with
additional functionality for solving nonlinear problems symbolically. It
also dispatches to IPOPT rather than solvers available in SciPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, cast

import casadi as cs
import numpy as np

from archimedes import tree
from archimedes._core import (
    FunctionCache,
    SymbolicArray,
    _as_casadi_array,
    array,
    sym_like,
)

if TYPE_CHECKING:
    from ..typing import ArrayLike

    T = TypeVar("T", bound=ArrayLike)

__all__ = [
    "nlp_solver",
    "minimize",
]


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

    # Define a function that will solve the NLP
    # This function will be evaluated with SymbolicArray objects.
    def _solve(x0, lbx, ubx, lbg, ubg, *args) -> ArrayLike:
        x0 = array(x0)
        ret_shape = x0.shape
        ret_dtype = x0.dtype

        # TODO: Shape checking for bounds
        if len(ret_shape) > 1 and ret_shape[1] > 1:
            raise ValueError(
                "Only scalar and vector decision variables are supported. "
                f"Got shape {ret_shape}"
            )

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
        x = sym_like(x0, name="x", kind="MX")
        f = obj(x, *args)

        # For type checing
        f = cast(SymbolicArray, f)

        nlp = {
            "x": x._sym,
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

    _solve_explicit.__name__ = name

    return FunctionCache(
        _solve_explicit,
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


def minimize(
    obj: Callable,
    x0: T,
    args: Sequence[Any] = (),
    static_argnames: str | Sequence[str] | None = None,
    constr: Callable | None = None,
    bounds: T | None = None,
    constr_bounds: ArrayLike | None = None,
    method: str = "ipopt",
    **options,
) -> T:
    """
    Minimize a scalar function with optional constraints.

    Solve a nonlinear programming problem of the form


    .. code-block:: text

        minimize        f(x, p)
        subject to      lbx <= x <= ubx
                        lbg <= g(x, p) <= ubg

    This function provides a simplified interface to nonlinear optimization
    solvers like IPOPT for solving a single optimization problem.

    Parameters
    ----------
    obj : callable
        Objective function to minimize, with signature ``obj(x, *args)``.
        Must return a scalar value.
    x0 : array_like
        Initial guess for the optimization.
    args : tuple, optional
        Extra arguments passed to the objective and constraint functions.
    static_argnames : tuple of str, optional
        Names of arguments that should be treated as static (non-symbolic)
        parameters. Static arguments are not differentiated through.
    constr : callable, optional
        Constraint function with the same signature as ``obj``.
        Must return an array of constraint values where the constraints
        are interpreted as ``lbg <= constr(x, *args) <= ubg``.
    bounds : tuple of (array_like, array_like), optional
        Bounds on the decision variables, given as a tuple ``(lb, ub)``.
        Each bound can be either a scalar or an array matching the shape
        of ``x0``. Use ``-np.inf`` and ``np.inf`` to specify no bound.
    constr_bounds : tuple of (array_like, array_like), optional
        Bounds on the constraint values, given as a tuple ``(lbg, ubg)``.
        Each bound can be a scalar or an array matching the shape of the
        constraint function output. If None and constr is provided,
        defaults to ``(0, 0)`` for equality constraints.
    method : str, optional
        The optimization method to use. Default is "ipopt". Other options
        may be available depending on the installed solver. See the CasADi
        documentation for available methods.
    **options : dict
        Additional options passed to the optimization solver through
        :py:func:`nlp_solver`. Available options depend on the solver method.
        See notes for examples.

    Returns
    -------
    x_opt : ndarray
        The optimal solution found by the solver, with the same shape as ``x0``.

    Notes
    -----
    This function dispatches to different optimization methods depending on the
    ``method`` argument.  By default, it uses the IPOPT interior point optimizer
    which is effective for large-scale constrained nonlinear problems. IPOPT
    requires derivatives of the objective and constraints, which are automatically
    computed using automatic differentiation.

    For IPOPT, see the
    [CasADi plugin documentation](https://web.casadi.org/python-api/#ipopt) for
    options that may be passed directly as keyword arguments.  However, most IPOPT
    configuration options documented in the
    [IPOPT manual](https://coin-or.github.io/Ipopt/OPTIONS.html) must be passed as
    an ``ipopt`` dictionary in the ``options`` argument.  For example, typical options
    for IPOPT can be passed as follows:

    .. highlight:: python
    .. code-block:: python

        ipopt_options = {
            "print_level": 0,
            "max_iter": 100,
            "tol": 1e-6,
        }
        minimize(obj, x0, ..., method="ipopt", options={"ipopt": ipopt_options})

    Another common solver is the sequential quadratic programming (SQP) method,
    available via the "sqpmethod", "blocksqp", or "feasiblesqpmethod" method names.
    SQP methods require a quadratic programming (QP) solver to solve the QP
    subproblems, specified via a ``qpsol`` keyword argument.  The default QP solver
    is "qpoases", but other options include "osqp", "proxqp", and plugins for solvers
    like "cplex" and "gurobi".

    Typical configuration options for "sqpmethod" which may be passed directly as
    keyword arguments include:

    - ``hessian_approximation``:  "exact" (default, uses automatic differentiation) or
        "limited-memory" (uses a limited-memory BFGS approximation).
    - ``max_iter``: Maximum number of SQP iterations (default is 25).
    - ``qpsol``: QP solver to use (default is "qpoases").
    - ``qpsol_options``: Options for the QP solver, passed as a dictionary.
    - ``tol_du``: Stopping tolerance for dual feasibility (default is 1e-6).
    - ``tol_pr``: Stopping tolerance for primal feasibility (default is 1e-6).

    For other configuration options, see the CasADi documentation for
    ["sqpmethod"](https://web.casadi.org/python-api/#sqpmethod),
    ["blocksqp"](https://web.casadi.org/python-api/#blocksqp), and
    ["feasiblesqpmethod"](https://web.casadi.org/python-api/#feasiblesqpmethod).
    For QP solver options, see the documentation for the specific QP solver
    (e.g., [OSQP](https://osqp.org/docs/release-0.6.3/interfaces/solver_settings.html)
    or [qpOASES](https://coin-or.github.io/qpOASES/doc/3.0/doxygen/classOptions.html)).

    Note that the "sqpmethod" solver with OSQP is the only combination that supports
    C code generation.

    For repeated optimization with different parameters, use :py:func:`nlp_solver`
    directly to avoid recompilation overhead.

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Unconstrained Rosenbrock function
    >>> def f(x):
    ...     return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    >>>
    >>> # Initial guess
    >>> x0 = np.array([-1.2, 1.0])
    >>>
    >>> # Solve unconstrained problem
    >>> x_opt = arc.minimize(f, x0)
    >>> print(x_opt)
    [1. 1.]
    >>>
    >>> # Constrained optimization
    >>> def g(x):
    ...     g1 = (x[0] - 1)**3 - x[1] + 1
    ...     g2 = x[0] + x[1] - 2
    ...     return np.array([g1, g2], like=x)
    >>>
    >>> # Solve with inequality constraints: g(x) <= 0
    >>> x_opt = arc.minimize(
    ...     f,
    ...     x0=np.array([2.0, 0.0]),
    ...     constr=g,
    ...     constr_bounds=(-np.inf, 0),
    ... )
    >>> print(x_opt)
    [0.99998266 1.00000688]
    >>>
    >>> # Optimization with variable bounds
    >>> x_opt = arc.minimize(
    ...     f,
    ...     x0=np.array([0.0, 0.0]),
    ...     bounds=(np.array([0.0, 0.0]), np.array([0.5, 1.5])),
    ... )
    >>> print(x_opt)
    array([0.50000001, 0.25000001])
    >>>
    >>> # Solving with sequential quadratic programming using OSQP
    >>> x_opt = arc.minimize(
    ...     f,
    ...     x0=np.array([0.0, 0.0]),
    ...     constr=g,
    ...     constr_bounds=(-np.inf, 0),
    ...     method="sqpmethod",
    ...     qpsol="osqp",
    ...     qpsol_options={"err_abs": 1e-6, "err_rel": 1e-6},
    ...     tol_du=1e-3,
    ...     tol_pr=1e-3,
    ...     max_iter=10,
    ...     hessian_approximation="limited-memory",
    ... )


    See Also
    --------
    nlp_solver : Create a reusable solver for nonlinear optimization
    root : Find the roots of a nonlinear function
    implicit : Create a function that solves ``F(x, p) = 0`` for ``x``
    scipy.optimize.minimize : SciPy's optimization interface
    """
    x0 = array(x0)
    # TODO: Expand docstring

    solver = nlp_solver(
        obj,
        constr=constr,
        static_argnames=static_argnames,
        constrain_x=bounds is not None,
        method=method,
        **options,
    )

    # Construct a list of arguments to the solver. The content
    # of this will depend on the configuration of constraints.
    solver_args = {"x0": x0}

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
    return solver(**solver_args)
