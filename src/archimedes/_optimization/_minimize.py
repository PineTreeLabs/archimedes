"""Defining and solving nonlinear problems

This interface is patterned after the `scipy.optimize` module, but with
additional functionality for solving nonlinear problems symbolically. It
also dispatches to IPOPT rather than the L-BFGS-B algorithm used in SciPy.
"""
import numpy as np
import casadi as cs

from archimedes import tree
from archimedes._core import (
    array,
    sym_like,
    _as_casadi_array,
    SymbolicArray,
    FunctionCache,
)

__all__ = [
    "nlp_solver",
    "minimize",
]


def nlp_solver(
    obj,
    constr=None,
    static_argnames=None,
    constrain_x=False,
    name=None,
    print_level=5,
    **options,
):
    """Create a reusable solver for a nonlinear problem.

    This transforms the function `func` into a function that solves
    the nonlinear problem:
    ```
    minimize f(x, p)
    subject to lbx <= x <= ubx, lbg <= g(x, p) <= ubg
    ```

    Here `obj(x, *args)` is the objective function `f` and `constr(x, *args)`
    is the constraint function `g`. The `static_argnames` arguments can be used
    to specify which arguments to `obj` and `constr` are static.  Non-static
    arguments are flattened into a single array `p`.  Both `obj` and `constr` must
    accept the same arguments, and the static arguments must be the same for both
    functions.

    In general the returned function will have the signature
    ```
    solver(x0, lbx, ubx, lbg, ubg, *args).
    ```

    If `constrain_x` is False, then the lower and upper bounds on `x` will be
    infinite and the arguments will not be available.  Likewise, if no constraint
    function is provided, then the lower and upper bounds on `g` will be infinite.

    This function can be evaluated numerically or symbolically, and
    will only be re-compiled if the static arguments change.
    """
    # TODO: Support other "plugins" ("knitro", "snopt", etc.)
    # TODO: Inspect function signature

    options["ipopt.print_level"] = print_level

    if not isinstance(obj, FunctionCache):
        obj = FunctionCache(obj, static_argnames=static_argnames)

    if constr is not None:
        if not isinstance(constr, FunctionCache):
            constr = FunctionCache(constr, static_argnames=static_argnames)

        # Check that arguments and static arguments are the same for both functions
        if not len(obj.arg_names) == len(constr.arg_names):
            raise ValueError(
                "Objective and constraint functions must have the same number of arguments"
            )

        if not len(obj.static_argnums) == len(constr.static_argnums):
            raise ValueError(
                "Objective and constraint functions must have the same number of static arguments"
            )

    # Define a function that will solve the NLP
    # This function will be evaluated with SymbolicArray objects.
    def _solve(x0, lbx, ubx, lbg, ubg, *args):
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

        nlp = {
            "x": x._sym,
            "f": f._sym,
        }

        if p.size != 0:
            nlp["p"] = p._sym

        if constr is not None:
            g = constr(x, *args)
            nlp["g"] = g._sym

        solver = cs.nlpsol("solver", "ipopt", nlp, options)

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
            def _solve_explicit(x0, *args):
                return _solve(x0, -np.inf, np.inf, -np.inf, np.inf, *args)
            
            arg_names.remove("lbg")
            arg_names.remove("ubg")

        else:
            def _solve_explicit(x0, lbg, ubg, *args):
                return _solve(x0, -np.inf, np.inf, lbg, ubg, *args)

        arg_names.remove("lbx")
        arg_names.remove("ubx")

    else:
        if not constrain_g:
            def _solve_explicit(x0, lbx, ubx, *args):
                return _solve(x0, lbx, ubx, -np.inf, np.inf, *args)

            arg_names.remove("lbg")
            arg_names.remove("ubg")

        else:
            def _solve_explicit(x0, lbx, ubx, lbg, ubg, *args):
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


def minimize(
    obj,
    x0,
    args=(),
    static_argnames=None,
    constr=None,
    bounds=None,
    constr_bounds=None,
    **options,
):
    """Minimize a function using IPOPT.

    This function is a wrapper around `nlp_solver` that sets the objective
    function `obj` as the function to minimize.  The function `obj` should
    return a scalar value that is minimized.  The function signature should
    be `obj(x, *args)`.

    Optionally, a constraint function may also be provided.  The constraint
    function should return a vector of constraint values and must have the
    same signature as `obj`.

    Both the decision variables `x` and the constraint values can be
    bounded by lower and upper bounds.  By default, the bounds on the
    decision variables are infinite and the bounds on the constraints are
    zero (corresponding to equality constraints).
    """
    x0 = array(x0)
    # TODO: Expand docstring

    solver = nlp_solver(
        obj,
        constr=constr,
        static_argnames=static_argnames,
        constrain_x=bounds is not None,
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

