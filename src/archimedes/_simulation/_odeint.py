"""Interface for solving ordinary differential equations.

This module has two main functions: `integrator` and `odeint`.  The `integrator`
function is a transformation that creates a "forward map" for the given function.
This forward map is a new function that integrates the ODE defined by the original
function.  The `odeint` function is a simple wrapper around the `integrator` function
that calls the generated forward map with the given initial state and time span.
`odeint` has a similar interface to `scipy.integrate.solve_ivp`.
"""

import casadi as cs

from archimedes import tree
from archimedes._core import (
    sym,
    sym_like,
    _as_casadi_array,
    SymbolicArray,
    FunctionCache,
)


__all__ = ["integrator", "odeint"]


def integrator(
    func,
    method="cvodes",
    atol=1e-6,
    rtol=1e-3,
    static_argnames=(),
    t_eval=None,
    name=None,
    options=None,
):
    """Create a "forward map" for the given function.

    Transform the function `func` into a function that integrates the ODE
    defined by `func`. The resulting function will return the state of the
    system at the end of the integration.

    The function should have the signature `func(t, x, *args) -> x_dot`, where
    `t` is the current time, `x` is the current state, and `x_dot` is the
    derivative of the state with respect to time.

    The returned function will have the signature
    `forward_map(x0, t_span, *args) -> xf`.

    If evaluation times are specified with `t_eval`, then the ODE solve function
    will close over these times, so that the returned function will instead have the
    signature `forward_map(x0, *args) -> xs`, where `xs` is an array of states at the
    evaluation times.  In this case the function must be fully re-compiled for each set
    of evaluation times (because the evaluation times are not hashable).
    """
    if options is None:
        options = {}

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

        # Otherwise the ODE solution will be an array of states at the evaluation times.
        # If `x0.shape == (n, 1)` then the output shape will be `(n, len(t_eval))`, flattening
        # over the empty dimension.
        return (*x0.shape[:1], len(t_eval))

    # Define a function that will integrate the ODE through the time span.
    # This function will be evaluated with SymbolicArray objects.
    def _forward_map(x0, t_span, *args):
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
            _static_args, sym_args, _arg_types = func.split_args(static_argnums, *args)

        else:
            # No static data - all arguments can be treated symbolically
            sym_args = args

        p, _unravel = tree.ravel(sym_args)

        # Define consistent time and state variables
        t, x = sym("t", kind="MX"), sym_like(x0, name="x0", kind="MX")
        xdot = func(t, x, *args)

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
        p_arg = False if p is None else p
        x0, p_arg = map(_as_casadi_array, (x0, p_arg))

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
        def _wrapped_forward_map(x0, *args):
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
    func,
    t_span,
    x0,
    method="cvodes",
    t_eval=None,
    atol=1e-6,
    rtol=1e-3,
    args=None,
    static_argnames=(),
    options=None,
):
    """Integrate the ODE defined by `func` with initial state `x0`.

    This is a simple wrapper around the `integrator` function that calls the generated
    forward map with the given initial state and time span.
    """
    # TODO: Expand docstring

    solver = integrator(
        func,
        method=method,
        atol=atol,
        rtol=rtol,
        t_eval=t_eval,
        static_argnames=static_argnames,
        options=options,
    )
    if args is None:
        args = ()

    if t_eval is not None:
        return solver(x0, *args)

    return solver(x0, t_span, *args)
