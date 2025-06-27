"""Defining and solving root-finding problems

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
https://web.casadi.org/docs/#nonlinear-root-finding-problems
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import casadi as cs

from archimedes import tree
from archimedes._core import (
    FunctionCache,
    SymbolicArray,
    _as_casadi_array,
    sym_like,
)
from archimedes.error import ShapeDtypeError

__all__ = ["implicit", "root"]


if TYPE_CHECKING:
    from ..typing import ArrayLike


def implicit(
    func: Callable,
    static_argnames: str | Sequence[str] | None = None,
    solver: str = "newton",
    name: str | None = None,
    **options,
) -> FunctionCache:
    """
    Construct an explicit function from an implicit relation.

    Given a relation ``F(x, p) = 0`` that implicitly defines ``x`` as a function of
    ``p``, this function creates a solver that computes ``x`` given ``p`` and an
    initial guess. This effectively transforms the equation ``F(x, p) = 0`` into an
    explicit function ``x = G(x0, p)`` that returns the solution ``x`` for parameters
    ``p`` starting from initial guess ``x0``.

    Parameters
    ----------
    func : callable
        The implicit function, with signature ``func(x, *args)``. Must return a
        residual with the same shape and dtype as the input ``x``.
    static_argnames : tuple of str, optional
        Names of arguments that should be treated as static (non-symbolic)
        parameters. Static arguments are not differentiated through and
        the solver will be recompiled when their values change.
    solver : str, default="newton"
        The root-finding method to use. Options are:

        - ``"newton"`` : Newton's method (default), best for general problems

        - ``"fast_newton"`` : Simple Newton iterations with no line search

        - ``"kinsol"`` : KINSOL solver from SUNDIALS, robust for large systems

    name : str, optional
        Name for the returned function. If ``None``, a name will be generated
        based on the input function name.
    tol : float, optional
        Absolute for convergence. If ``None``, the default tolerance for the
        chosen method will be used.
    **options : dict, optional
        Common additional options specific to the chosen method:

        For ``"newton"`` and ``"fast_newton"``:

        - max_iter : int, maximum iterations (default: 100)

        For ``"kinsol"``:

        - max_iter : int, maximum iterations
    
        - strategy : str, globalization strategy (``"none"``, ``"linesearch"``,\
        ``"picard"``, ``"fp"``)

        See the `CasADi documentation <https://web.casadi.org/python-api/#rootfinding/>`_
        for more details on the available options for each method.

    Returns
    -------
    solver : FunctionCache
        A function that, when called with signature ``solver(x0, *args)``,
        returns the solution ``x`` to ``F(x, *args) = 0`` starting from the initial
        guess ``x0``. This function can be evaluated both numerically and symbolically.

    Notes
    -----
    When to use this function:

    - When you have an equation ``F(x, p) = 0`` that you need to solve repeatedly
      for different values of parameters ``p``
    - For implementing equations of motion for constrained mechanical systems
    - For implicit numerical schemes in simulation
    - For embedding root-finding operations within larger computational graphs

    The solver automatically computes the Jacobian of the residual function
    using automatic differentiation.

    For simple one-off root-finding problems where parameters don't change,
    consider using :py:func:`root` instead which provides a simpler interface.

    The function generates a callable that behaves as a pure function, making
    it suitable for embedding in larger computational graphs or differentiating
    through.

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Example 1: Simple nonlinear equation x^2 = p
    >>> def f(x, p):
    ...     return x**2 - p
    >>>
    >>> # Create a solver for x given p
    >>> sqrt = arc.implicit(f)
    >>>
    >>> # Solve for sqrt(2) starting from initial guess 1.0
    >>> x = sqrt(1.0, 2.0)
    >>> print(f"Solution: {x:.10f}")  # Should be close to 1.4142135624
    Solution: 1.4142135624
    >>>
    >>> # Example 2: Implicit equation with vector input and parameters
    >>> def nonlinear_system(x, a, b):
    ...     # System: a*x[0]^2 + b*x[1] = 1, x[0] + x[1]^2 = 4
    ...     return np.array([
    ...         a * x[0]**2 + b * x[1] - 1,
    ...         x[0] + x[1]**2 - 4
    ...     ], like=x)
    >>>
    >>> # Create a solver with static parameter 'a'
    >>> solver = arc.implicit(nonlinear_system, static_argnames=('a',))
    >>>
    >>> # Solve the system for a=2, b=3
    >>> x0 = np.array([0.0, 0.0])  # Initial guess
    >>> solution = solver(x0, 2.0, 3.0)
    >>> print(solution)  # Should converge to a valid solution
    >>>
    >>> # Example 3: Using a different solver and options
    >>> def kepler(x, e):
    ...     # Kepler's equation: x - e*sin(x) = M where M is fixed at 0.5
    ...     return x - e * np.sin(x) - 0.5
    >>>
    >>> # Create solver with more iterations and higher accuracy
    >>> kepler_solver = arc.implicit(
    ...     kepler,
    ...     solver="newton",
    ...     max_iter=50,
    ...     tol=1e-12
    ... )
    >>>
    >>> # Solve for eccentric anomaly with eccentricity e=0.8
    >>> anomaly = kepler_solver(0.5, 0.8)
    >>> print(f"Eccentric anomaly: {anomaly:.10f}")

    See Also
    --------
    root : Simpler interface for one-off root-finding
    minimize : Find the minimum of a scalar function
    nlp_solver : Create a reusable solver for nonlinear optimization
    """
    # TODO: Inspect function signature to check for consistency
    # TODO: Support constraints on the unknowns (supported via options in CasADi)

    if not isinstance(func, FunctionCache):
        func = FunctionCache(func, static_argnames=static_argnames)

    # Define a function that will solve the root-finding problem
    # This function will be evaluated with SymbolicArray objects.
    def _solve(x0: ArrayLike, *args) -> ArrayLike:
        ret_shape = x0.shape
        ret_dtype = x0.dtype

        # TODO: Shape checking for bounds
        if len(ret_shape) > 1 and ret_shape[1] > 1:
            raise ValueError(
                "Only scalar and vector decision variables are supported. "
                f"Got shape {ret_shape}"
            )

        # Flatten the symbolic arguments into a single vector `z` to pass to CasADi.
        # If there is static data this needs to be stripped out before
        # flattening the symbolic data.
        if static_argnames:
            # The first argument is `x`, which is not accounted for in the
            # indexing of static arguments - shift by one
            static_argnums = [i - 1 for i in func.static_argnums]
            _static_args, sym_args, _arg_struct = func.split_args(static_argnums, *args)

        else:
            # No static data - all arguments can be treated symbolically
            sym_args = args

        # The root-finding problem in CasADi takes the form:
        # "Find x such that F(x, z) = 0". To define the residual
        # function, we'll flatten all the symbolic args into a single array
        # `z` and then create a CasADi Function object that evaluates
        # the residual.
        # TODO: Shouldn't something get unraveled here?
        z, _unravel = tree.ravel(sym_args)

        has_aux = z.size != 0  # Does the function have additional inputs?

        # Define a state variable for the optimization
        x = sym_like(x0, name="x", kind="MX")
        g = func(x, *args)  # Evaluate the residual symbolically

        # For type checking
        g = cast(SymbolicArray, g)
        z = cast(SymbolicArray, z)

        if g.shape != ret_shape or g.dtype != ret_dtype:
            raise ShapeDtypeError(
                f"Expected shape {ret_shape} and dtype {ret_dtype}, "
                f"got shape {g.shape} and dtype {g.dtype}.  The shape and "
                "dtype of the residual must match those of the input variable."
            )

        # Note that the return of _this_ function is actually the residual,
        # but since the `rootfinder` will have the same signature, we'll name
        # the output of the residual `x` in anticipation that we will be
        # enclosing it in the `rootfinder`.
        sym_args = [x._sym]
        arg_names = ["x"]
        if has_aux:
            sym_args.append(z._sym)
            arg_names.append("z")

        # Note: as of casadi 3.7.0, the root-finder will append a trailing "0"
        # to the first argument, so ["x"] becomes ["x0"].  The output will be
        # under the original key "x" regardless of what the specified return name
        # is in the Function object.  Here we use "res" as this return name.
        cs_func = cs.Function("F", sym_args, [g._sym], arg_names, ["res"])
        root_solver = cs.rootfinder("solver", solver, cs_func, options)

        # Before calling the CasADi rootfinder, we have to make sure
        # the input data is either a CasADi symbol or a NumPy array
        z_arg = False if z is None else z
        x0, z_arg = map(_as_casadi_array, (x0, z_arg))  # type: ignore[assignment]

        # The return is a dict with keys for the outputs of the residual
        # function.  The key "x" will contain the root of the function.
        if has_aux:
            sol = root_solver(x0=x0, z=z_arg)
        else:
            sol = root_solver(x0=x0)

        return SymbolicArray(
            sol["x"],
            dtype=ret_dtype,
            shape=ret_shape,
        )

    # The first arg name for the input function is the variable, which
    # gets replaced by the initial guess in the new function.  Otherwise
    # the arguments are user-defined
    arg_names = ("x0", *func.arg_names[1:])

    if name is None:
        name = f"{func.name}_root"

    _solve.__name__ = name

    return FunctionCache(
        _solve,
        arg_names=arg_names,
        static_argnames=static_argnames,
        kind="MX",
    )


def root(
    func: Callable,
    x0: ArrayLike,
    args: Sequence[Any] = (),
    static_argnames: str | Sequence[str] | None = None,
    method: str = "newton",
    tol: float | None = None,
    **options,
) -> ArrayLike:
    """Find a root of a nonlinear function.

    Solves the equation ``f(x) = 0`` for ``x``, where ``f`` is a vector function of
    vector ``x``.

    Parameters
    ----------
    func : callable
        The function whose root to find, with signature ``func(x, *args)``.
        The function should return an array of the same shape as `x``.
        For systems of equations, ``func`` should return a vector of residuals.
    x0 : array_like
        Initial guess for the solution. The shape of this array determines
        the dimensionality of the problem to be solved.
    args : tuple, optional
        Extra arguments passed to the function.
    static_argnames : tuple of str, optional
        Names of arguments that should be treated as static (non-symbolic)
        parameters. Static arguments are not differentiated through and
        the solver will be recompiled when their values change.
    method : str, default="newton"
        The root-finding method to use. Options are:

        - ``"newton"`` : Newton's method (default), best for general problems

        - ``"fast_newton"`` : Simple Newton iterations with no line search

        - ``"kinsol"`` : KINSOL solver from SUNDIALS, robust for large systems

    tol : float, optional
        Absolute for convergence. If ``None``, the default tolerance for the
        chosen method will be used.
    **options : dict, optional
        Common additional options specific to the chosen method:

        For ``"newton"`` and ``"fast_newton"``:

        - max_iter : int, maximum iterations (default: 100)

        For ``"kinsol"``:

        - max_iter : int, maximum iterations
    
        - strategy : str, globalization strategy (``"none"``, ``"linesearch"``,\
        ``"picard"``, ``"fp"``)

    Returns
    -------
    x : array_like
        The solution found, with the same shape as the initial guess ``x0``.
        If the algorithm fails to converge, the best estimate is returned.

    Notes
    -----
    This function leverages Archimedes' automatic differentiation to compute
    the Jacobian matrix required by most root-finding methods. For repeated
    solving with different parameters, use :py:func:`implicit` directly to create
    a reusable solver function.

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Simple scalar equation: x^2 = 2
    >>> def f1(x):
    ...     return x**2 - 2
    >>>
    >>> sol = arc.root(f1, x0=1.0)
    >>> print(f"Solution: {sol:.10f}")  # Should be close to sqrt(2)
    Solution: 1.4142135624
    >>>
    >>> # System of nonlinear equations
    >>> def f2(x):
    ...     return np.array([
    ...         x[0] + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...         0.5 * (x[1] - x[0])**3 + x[1],
    ...     ], like=x)
    >>>
    >>> sol = arc.root(f2, x0=np.array([0.0, 0.0]))
    >>> print(sol)  # Should be close to [0.8411639, 0.1588361]
    [0.8411639 0.1588361]
    >>>
    >>> # Using a different method with options
    >>> def f3(x):
    ...     return np.exp(x) - 2
    >>>
    >>> sol = arc.root(f3, x0=1.0, method='kinsol', max_iter=20, tol=1e-10)
    >>> print(f"Solution: {sol:.10f}")  # Should be close to ln(2)
    Solution: 0.6931471806
    >>>
    >>> # With additional parameters
    >>> def f4(x, a, b):
    ...     return x**2 - a*x + b
    >>>
    >>> sol = arc.root(f4, x0=2.5, args=(3, 2))
    >>> print(f"Solution: {sol:.10f}")  # Should be close to 2
    Solution: 2.0000000000

    See Also
    --------
    implicit : Create a function that solves ``F(x, p) = 0`` for ``x`` given ``p``
    minimize : Find the minimum of a scalar function
    jac : Compute the Jacobian of a function using automatic differentiation
    scipy.optimize.root : SciPy's interface to root-finding solvers
    """
    # TODO: Better documentation of common options
    if tol is not None:
        options["abstol"] = tol

    g = implicit(
        func,
        solver=method,
        static_argnames=static_argnames,
        **options,
    )
    x: ArrayLike = g(x0, *args)

    return x
