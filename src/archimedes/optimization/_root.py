"""Defining and solving root-finding problems

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
https://web.casadi.org/docs/#nonlinear-root-finding-problems
"""
import casadi as cs

from archimedes import tree
from archimedes.core import (
    sym_like,
    casadi_array,
    SymbolicArray,
    SymbolicFunction,
)

from archimedes.error import ShapeDtypeError

__all__ = ["implicit", "root"]


def implicit(
    func, static_argnames=None, solver="newton", name=None, **options
):
    """Construct an explicit function from an implicit function.

    Given a function `F(x, *args) = 0`, which can be viewed as defining
    x as an implicit function of the other arguments, this function constructs
    a new function `x = f(x0, *args)` that returns the root of the implicit function.
    The returned function accepts the same arguments as `F`, except that instead
    of `x` it accepts an initial guess `x0`.

    The function `F` must return a residual that has the same shape and dtype as
    the input `x`.  The solver will attempt to find a zero of this residual.

    While this can be used to define a root-finding problem `F(x) = 0`, this is
    typically more naturally accomplished by calling `root` directly, which uses
    the same underlying solver but has a more intuitive interface for this case.

    Parameters
    ----------
    func : callable
        The function whose root to find. It must be a function of the form
        `func(x, *args)`, where `x` is the input and `args` are additional
        arguments.
    static_argnames : list, optional
        The names of the static arguments to the function. These will not be
        evaluated symbolically and will be passed directly to the implicit function.
    solver : str, optional
        The solver to use. One of "newton", "kinsol", or "nlpsol".
    name : str, optional
        The name of the returned function.
    options : dict, optional
        Additional options to pass to the root-finding solver.

    Returns
    -------
    g : callable
        The explicit function that returns the root of the implicit function.
        Will have the signature `x = g(x0, *args)`.

    Examples
    --------
    >>> def f(x):
    ...     return x**2 - 1
    >>> g = implicit(f)
    >>> g(x0=2.0)
    1.0
    """
    # TODO: Inspect function signature to check for consistency
    # TODO: Support constraints on the unknowns (supported via options in CasADi)

    if not isinstance(func, SymbolicFunction):
        func = SymbolicFunction(func, static_argnames=static_argnames)

    # Define a function that will solve the root-finding problem
    # This function will be evaluated with SymbolicArray objects.
    def _solve(x0, *args):
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
        args = [x._sym]
        arg_names = ["x0"]
        if has_aux:
            args.append(z._sym)
            arg_names.append("z")

        F = cs.Function("F", args, [g._sym], arg_names, ["x"])
        root_solver = cs.rootfinder("solver", solver, F, options)

        # Before calling the CasADi rootfinder, we have to make sure
        # the input data is either a CasADi symbol or a NumPy array
        z_arg = False if z is None else z
        x0, z_arg = map(casadi_array, (x0, z_arg))

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

    return SymbolicFunction(
        _solve,
        arg_names=arg_names,
        static_argnames=static_argnames,
        kind="MX",
    )


def root(
    func,
    x0,
    args=(),
    static_argnames=None,
    method="newton",
    **options,
):
    """Find a root of a function.

    Parameters
    ----------
    func : callable
        The function whose root to find. It must be a function of the form
        `func(x, *args)`, where `x` is the input and `args` are additional
        arguments.
    x0 : array_like
        The initial guess for the root.
    args : tuple, optional
        Additional arguments to pass to the function.
    static_argnames : list, optional
        The names of the static arguments to the function. These will not be
        evaluated symbolically and will be passed directly to the implicit function.
        All such arguments must be hashable.
    method : str, optional
        The root-finding method to use. One of "newton", "kinsol", or "nlpsol".
    options : dict, optional
        Additional options to pass to the root-finding solver. See CasADi documentation
        for more information.

    Returns
    -------
    x : array_like
        The root of the function.

    Examples
    --------
    >>> def f(x):
    ...     return x**2 - 1
    >>> root(f, x0=2.0)
    1.0
    """
    # TODO: Better documentation of common options

    g = implicit(
        func,
        solver=method,
        static_argnames=static_argnames,
        **options,
    )
    return g(x0, *args)
