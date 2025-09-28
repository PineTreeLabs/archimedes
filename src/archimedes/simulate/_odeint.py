"""Interface for solving ordinary differential equations.

This module has two main functions: `integrator` and `odeint`.  The `integrator`
function is a transformation that creates a "forward map" for the given function.
This forward map is a new function that integrates the ODE defined by the original
function.  The `odeint` function is a simple wrapper around the `integrator` function
that calls the generated forward map with the given initial state and time span.
`odeint` has a similar interface to `scipy.integrate.solve_ivp`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import casadi as cs

from archimedes import tree
from archimedes._core import (
    FunctionCache,
    SymbolicArray,
    _as_casadi_array,
    sym,
    sym_like,
)

__all__ = ["integrator", "odeint"]

if TYPE_CHECKING:
    from archimedes.typing import ArrayLike, NDArray


def integrator(
    func: Callable,
    method: str = "cvodes",
    atol: float = 1e-6,
    rtol: float = 1e-3,
    static_argnames: str | Sequence[str] = (),
    t_eval: NDArray | None = None,
    name: str | None = None,
    **options,
) -> FunctionCache:
    """Create an ODE solver function from a dynamics function.

    Transforms a function defining an ODE into a function that propagates the ODE
    system forward in time. The resulting function can be used repeatedly with
    different initial conditions and parameters, and supports automatic differentiation
    through the solution.

    Parameters
    ----------
    func : callable
        The function defining the ODE dynamics. Must have the signature
        ``func(t, x, *args) -> x_t``, where ``t`` is the current time,
        ``x`` is the current state, and ``x_t`` is the derivative of
        the state with respect to time.
    method : str, optional
        Integration method to use. Default is ``"cvodes"``, which uses the
        SUNDIALS CVODES solver (suitable for both stiff and non-stiff systems).
        See CasADi documentation for other available methods.
    atol : float, optional
        Absolute tolerance for the solver. Default is ``1e-6``.
    rtol : float, optional
        Relative tolerance for the solver. Default is ``1e-3``.
    static_argnames : tuple of str, optional
        Names of arguments to ``func`` that should be treated as static
        (not traced symbolically).
    t_eval : array_like, optional
        Times at which to evaluate the solution. If provided, the integrator
        will return states at these times. If None, the integrator will
        only return the final state.
    name : str, optional
        Name for the created integrator function. If None, a name is
        automatically generated based on the dynamics function's name.
    **options : dict
        Additional options to pass to the CasADi integrator.

    Returns
    -------
    callable
        If ``t_eval`` is None, a function with signature
        ``solver(x0, t_span, *args) -> xf``, where ``x0`` is the initial state,
        ``t_span`` is a tuple ``(t0, tf)`` giving the integration time span, and ``xf``
        is the final state.

        If ``t_eval`` is provided, a function with signature
        ``solver(x0, *args) -> xs``, where ``xs`` is an array of states at the times in
        ``t_eval``.

    Notes
    -----
    When to use this function:

    - For repeated ODE solves with different initial conditions or parameters
    - When embedding ODE solutions in optimization problems or larger models
    - When sensitivity analysis requires derivatives of the solution
    - For improved performance over loop-based ODE integration

    The ODE solver is specified by the ``method`` argument. The default is ``"cvodes"``,
    which uses the SUNDIALS CVODES solver and is suitable for both stiff and non-stiff
    systems. This is the recommended choice for most applications.

    See the [CasADi](https://web.casadi.org/python-api/#integrator) documentation for a
    complete list of available configuration options to pass to the `options` argument.

    Conceptual model:
    This function constructs a computational graph representation of the entire
    ODE solve operation. The resulting function is more efficient than repeatedly
    calling :py:func:``odeint`` because:

    - It pre-compiles the solver for a specific state dimension and parameter structure
    - It leverages CasADi's compiled C++ implementation for high-performance integration
    - It enables automatic differentiation through the entire solution

    Limitations:

    - Only scalar and vector states are supported (shapes like ``(n,)`` or ``(n,1)``)
    - If ``t_eval`` is provided, the function caches the evaluation times internally,
      meaning new evaluation times require recompiling the function

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Example 1: Simple pendulum
    >>> def pendulum(t, x):
    >>>     # Simple pendulum dynamics.
    >>>     theta, omega = x
    >>>     return np.array([omega, -9.81 * np.sin(theta)], like=x)
    >>>
    >>> # Create an integrator
    >>> solver = arc.integrator(pendulum)
    >>>
    >>> # Solve for multiple initial conditions
    >>> x0_1 = np.array([0.1, 0.0])  # Small angle
    >>> x0_2 = np.array([1.5, 0.0])  # Large angle
    >>> t_span = (0.0, 10.0)
    >>>
    >>> x1_final = solver(x0_1, t_span)
    >>> x2_final = solver(x0_2, t_span)
    >>>
    >>> # Example 2: Parameterized system
    >>> def lotka_volterra(t, x, params):
    >>>     # Lotka-Volterra predator-prey model with parameters.
    >>>     a, b, c, d = params
    >>>     prey, predator = x
    >>>     return np.array([
    >>>         a * prey - b * prey * predator,      # prey growth rate
    >>>         c * prey * predator - d * predator   # predator growth rate
    >>>     ], like=x)
    >>>
    >>> # Create an integrator with evaluation times
    >>> t_eval = np.linspace(0, 20, 100)
    >>> solver = arc.integrator(lotka_volterra, t_eval=t_eval, method="cvodes")
    >>>
    >>> # Solve with different parameters
    >>> x0 = np.array([10.0, 5.0])  # Initial population
    >>> params1 = np.array([1.5, 0.3, 0.2, 0.8])
    >>> params2 = np.array([1.2, 0.2, 0.3, 1.0])
    >>>
    >>> solution1 = solver(x0, params1)  # Shape: (2, 100)
    >>> solution2 = solver(x0, params2)
    >>>
    >>> # Example 3: Differentiating through ODE solutions
    >>> target = solution1
    >>> @arc.compile(kind="MX")
    >>> def cost(x0, params):
    >>>     # Cost function based on final state after integration.
    >>>     final_state = solver(x0, params)
    >>>     return np.sum((final_state - target)**2)
    >>>
    >>> # Compute gradient of cost with respect to parameters
    >>> dJ_dp = arc.grad(cost, argnums=1)
    >>> print(dJ_dp(x0, params1))
    [0. 0. 0. 0.]
    >>> print(dJ_dp(x0, params2))
    [-11757.72435691 -32985.03384074  43301.00963694 -27677.60203032]

    See Also
    --------
    odeint : One-time ODE solution for a specific initial value problem
    """

    if not isinstance(func, FunctionCache):
        func = FunctionCache(func, static_argnames=static_argnames)

    options = {
        **options,
        "abstol": atol,
        "reltol": rtol,
    }

    # Function to determine the shape of the output of the ODE solver
    def _shape_inference(x0):
        # If `t_eval` is None, then the ODE solution will be a single state vector.
        if t_eval is None:
            return x0.shape

        # Otherwise the ODE solution will be an array of states at the evaluation
        # times. If `x0.shape == (n, 1)` then the output shape will be
        # `(n, len(t_eval))`, flattening over the empty dimension.
        return (*x0.shape[:1], len(t_eval))

    # Define a function that will integrate the ODE through the time span.
    # This function will be evaluated with SymbolicArray objects.
    def _forward_map(x0, t_span, *args) -> ArrayLike:
        if len(x0.shape) > 1 and x0.shape[1] > 1:
            raise ValueError(
                f"Only scalar and vector states are supported. Got shape {x0.shape}"
            )

        ret_shape = _shape_inference(x0)
        ret_dtype = x0.dtype

        # We have to flatten all of the symbolic user-defined arguments
        # into a single vector to pass to CasADi.  If there is static data,
        # this needs to be stripped out and passed separately.
        if static_argnames or func.static_argnums:
            # The first two arguments are (t, x), so skip these in checking
            # for static arguments.
            static_argnums = [i - 2 for i in func.static_argnums]
            static_args, sym_args, _arg_types = func.split_args(static_argnums, *args)

        else:
            # No static data - all arguments can be treated symbolically
            static_argnums = []
            sym_args = args

        # We cannot use a concatenation of various symbolic variables as an input to
        # the integrator, so we need to create a fresh symbolic variable for this
        p_orig, unravel = tree.ravel(sym_args)
        p = sym("p", kind="MX", shape=p_orig.shape, dtype=p_orig.dtype)

        # Create a new set of "args" that includes the fresh symbolic parameters
        # for use in evaluating the ODE function symbolically.
        args: list[ArrayLike] = list(unravel(p))  # type: ignore[no-redef]
        # If the function has static arguments, we need to interleave them back in the
        # original order.
        for i in static_argnums:
            args.insert(i, static_args[i])  # type: ignore[attr-defined]

        # Define consistent time and state variables
        t, x = sym("t", kind="MX"), sym_like(x0, name="x0", kind="MX")
        xdot = func(t, x, *args)

        # For type checking
        p = cast(SymbolicArray, p)
        xdot = cast(SymbolicArray, xdot)

        ode = {
            "t": t._sym,
            "x": x._sym,
            "ode": xdot._sym,
        }

        if p.size != 0:
            ode["p"] = p._sym

        solver = cs.integrator("solver", method, ode, *t_span, options)

        # Before calling the CasADi solver interface, make sure everything is
        # either a CasADi symbol or a NumPy array
        p_arg = False if p_orig is None else p_orig
        x0, p_arg = map(_as_casadi_array, (x0, p_arg))  # type: ignore[assignment]

        # The return is a dictionary with the final state of the system
        # and other information.  We will only return the final state.
        sol = solver(x0=x0, p=p_arg)

        return SymbolicArray(
            sol["xf"],
            dtype=ret_dtype,
            shape=ret_shape,
        )

    if t_eval is None:
        # The first two names to the function are time and state,
        # otherwise they will be user-defined
        arg_names = ("x0", "t_span", *func.arg_names[2:])
        static_argnames = ("t_span", *static_argnames)
        _wrapped_forward_map = _forward_map

    else:
        # The CasADi interface expects `t0` to be a scalar, but allows
        # `tf` to be an array of evaluation times.  Here we will close
        # over the evaluation times because this cannot be used as a
        # hashable key for the compiled function cache (i.e. a static arg).
        t0, tf = t_eval[0], t_eval

        def _wrapped_forward_map(x0, *args) -> ArrayLike:  # type: ignore
            return _forward_map(x0, (t0, tf), *args)

        arg_names = ("x0", *func.arg_names[2:])

    if name is None:
        name = f"{func.name}_odeint"

    _wrapped_forward_map.__name__ = name

    return FunctionCache(
        _wrapped_forward_map,
        arg_names=arg_names,
        static_argnames=static_argnames,
        kind="MX",
    )


def odeint(
    func: Callable,
    t_span: tuple[float, float],
    x0: ArrayLike,
    method: str = "cvodes",
    t_eval: NDArray | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    args: Sequence[Any] | None = None,
    static_argnames: str | Sequence[str] = (),
    **options,
) -> ArrayLike:
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem defined by the ODE system:

    .. math::
        \\dot{x} = f(t, x, \\theta)

    from ``t=t_span[0]`` to ``t=t_span[1]`` with initial conditions ``x0``.

    Parameters
    ----------
    func : callable
        Right-hand side of the ODE system. The calling signature is
        ``func(t, x, *args)``, where ``t`` is a scalar and ``x`` is an ndarray with
        shape ``(n,)``. func must return an array with shape ``(n,)``.
    t_span : tuple of float
        Interval of integration ``(t0, tf)``.
    x0 : array_like
        Initial state.
    method : str, optional
        Integration method to use. Default is ``"cvodes"``, which uses the
        SUNDIALS CVODES solver (suitable for both stiff and non-stiff systems).
    t_eval : array_like or None, optional
        Times at which to store the computed solution. If ``None``, the solver
        only returns the final state.
    atol : float, optional
        Absolute tolerance for the solver. Default is ``1e-6``.
    rtol : float, optional
        Relative tolerance for the solver. Default is ``1e-3``.
    args : tuple, optional
        Additional arguments to pass to the ODE function.
    static_argnames : tuple of str, optional
        Names of arguments to `func` that should be treated as static
        (not traced symbolically).
    **options : dict
        Additional options to pass to the CasADi integrator.

    Returns
    -------
    array_like
        If ``t_eval`` is None, the state at the final time ``t_span[1]``.
        If ``t_eval`` is provided, an array containing the solution evaluated at
        ``t_eval`` with shape ``(n, len(t_eval))``.

    Notes
    -----

    The underlying SUNDIALS solvers use adaptive step size control to balance
    accuracy and computational efficiency.  However, unlike SciPy's ``solve_ivp``
    function and many other ODE solver interfaces, it is not possible to output
    the solution at the times actually used by the solver (or to create a "dense"
    output using an interpolant). This is because of the requirement that all input
    and output arrays be a fixed size in order to construct the symbolic graph. The
    solution at output times specified by ``t_eval`` is computed by interpolating the
    solution at the times used by the solver.

    The ODE solver is specified by the ``method`` argument. The default is ``"cvodes"``,
    which uses the SUNDIALS CVODES solver and is suitable for both stiff and non-stiff
    systems. This is the recommended choice for most applications.

    See the `CasADi <https://web.casadi.org/python-api/#integrator/>`_ documentation
    for a complete list of available configuration options to pass to the ``options``
    argument.

    Conceptual model:
    This function is a wrapper around the :py:func:``integrator`` function that creates
    and immediately calls an ODE solver for the given initial conditions and
    parameters. It provides a simple interface similar to SciPy's ``solve_ivp``
    but leverages CasADi's efficient symbolic implementation and interface to
    SUNDIALS solvers.

    Unlike :py:func:`integrator`, which creates a reusable solver function,
    ``odeint`` performs a single solve operation. If you need to solve the same ODE
    system repeatedly with different initial conditions or parameters, it's more
    efficient to create an ``integrator`` function once and reuse it.

    Limitations:

    - Only scalar and vector states are supported (shapes like ``(n,)`` or ``(n,1)``)
    - For tree-structured states, currently the user must define a wrapper function
      to flatten and reconstruct the tree structure

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Example 1: Simple harmonic oscillator
    >>> def f(t, y):
    >>>     return np.array([y[1], -y[0]], like=y)  # y'' = -y
    >>>
    >>> t_span = (0, 10)
    >>> y0 = np.array([1.0, 0.0])  # Initial displacement and velocity
    >>> ts = np.linspace(*t_span, 100)
    >>>
    >>> # Solve ODE
    >>> ys = arc.odeint(f, t_span, y0, t_eval=ts)
    >>>
    >>> plt.figure()
    >>> plt.plot(ts, ys[0, :], label='position')
    >>> plt.plot(ts, ys[1, :], label='velocity')
    >>> plt.legend()
    >>> plt.show()
    >>>
    >>> # Example 2: Parametrized dynamics
    >>> def pendulum(t, y, g, L):
    >>>     theta, omega = y
    >>>     return np.array([omega, -(g/L) * np.sin(theta)])
    >>>
    >>> y0 = np.array([np.pi/6, 0.0])  # 30 degrees initial angle
    >>> t_span = (0, 5)
    >>> ys = arc.odeint(pendulum, t_span, y0, args=(9.81, 1.0))

    See Also
    --------
    integrator : Create a reusable ODE solver function
    scipy.integrate.solve_ivp : SciPy's ODE solver interface
    """

    solver = integrator(
        func,
        method=method,
        atol=atol,
        rtol=rtol,
        t_eval=t_eval,
        static_argnames=static_argnames,
        **options,
    )
    if args is None:
        args = ()

    if t_eval is not None:
        xs: ArrayLike = solver(x0, *args)

    else:
        xs: ArrayLike = solver(x0, t_span, *args)  # type: ignore[no-redef]

    return xs
