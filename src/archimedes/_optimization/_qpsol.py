"""Solving quadratic programming problems."""


from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence, cast, NamedTuple
import inspect

import casadi as cs
import numpy as np

from archimedes import tree, struct
from archimedes._core import (
    FunctionCache,
    SymbolicArray,
    _as_casadi_array,
    array,
    sym_like,
)

if TYPE_CHECKING:
    from ..typing import ArrayLike

__all__ = [
    "qpsol",
]


@struct.pytree_node
class QPSolution:
    x: ArrayLike
    lam_x: ArrayLike
    lam_a: ArrayLike


def qpsol(
    obj: Callable,
    constr: Callable,
    x0: ArrayLike,
    lba: ArrayLike | None = None,
    uba: ArrayLike | None = None,
    lam_x0: ArrayLike | None = None,
    lam_a0: ArrayLike | None = None,
    args: Sequence = (),
    static_argnames: str | Sequence[str] | None = None,
    verbose: bool = False,
    name: str | None = None,
    warm_start: bool = True,
    **options,
) -> FunctionCache:
    """Create a reusable solver for a quadratic programming problem
    
    This function solves a quadratic problem of the form:

    .. code-block:: text

        minimize        (1/2) x^T Q(p) x + c(p)^T x
        subject to      lba <= A(p)x <= uba

    where ``x`` represents decision variables and ``p`` represents parameters.
    The arrays ``Q``, ``c``, and ``A`` that define the convex quadratic program
    are determined by automatically differentiating the provided objective and
    constraint functions.  That is, if the objective and constraint functions do
    not define a convex quadratic program, the solver will solve the convex
    quadratic approximation to the provided problem, determined by linearization
    about the initial guess.

    Parameters
    ----------
    obj : callable
        Objective function to minimize with signature ``obj(x, *args)``.
        Must return a scalar value.
    constr : callable
        Constraint function with signature ``constr(x, *args)``.
        Must return a vector of constraint values where the constraints
        are interpreted as ``lba <= constr(x, *args) <= uba``.
    x0 : array-like
        Initial guess for the decision variables. This is used to determine
        the linearization point for the convex approximation and to warm-start
        the QP solver. If None, the initial guess will be set to zero.
    lba : array-like, optional
        Lower bounds for the constraints. If None, the lower bounds will be
        set to negative infinity.
    uba : array-like, optional
        Upper bounds for the constraints. If None, the upper bounds will be
        set to positive infinity.
    lam_x0 : array-like, optional
        Initial guess for the dual variables associated with the decision
        variables. This is used to warm-start the QP solver.
    lam_a0 : array-like, optional
        Initial guess for the dual variables associated with the constraints.
        This is used to warm-start the QP solver.
    static_argnames : tuple of str, optional
        Names of arguments in ``obj`` and ``constr`` that should be treated
        as static parameters rather than symbolic variables. Static arguments
        are not differentiated through and the solver will be recompiled when
        their values change.
    name : str, optional
        Name for the resulting solver function. If None, a name will be
        generated based on the objective function name.
    verbose : bool, default=False
        Print output from the solver, including number of iterations and convergence.
    **options : dict
        Additional options passed to the underlying QP solver (OSQP).

    Returns
    -------
    solution : QPSolution
        A named tuple containing the solution to the QP problem, including
        the optimal decision variables ``x``, the dual variables associated
        with the decision variables ``lam_x``, and the dual variables
        associated with the constraints ``lam_a``.

    Notes
    -----
    The solution to the quadratic program is unique, so the initial guess is less
    important than for more general nonlinear programming.  The exception is when
    the QP is the convex approximation to a nonlinear program, in which case the
    initial guess is used as the linearization point.

    This function supports code generation, but requires linking the OSQP C library
    to the generated code. (TODO: Add details and instructions)

    TODO: Finish docstring
    
    """

    options = {
        "warm_start_primal": warm_start,
        "warm_start_dual": warm_start,
        "osqp": {
            "verbose": verbose,
            **options,
        }
    }

    if not isinstance(obj, FunctionCache):
        obj = FunctionCache(obj, static_argnames=static_argnames)

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
    g = constr(x, *args)

    # For type checking
    f = cast(SymbolicArray, f)
    g = cast(SymbolicArray, g)

    qp = {
        "x": x._sym,
        "f": f._sym,
        "g": g._sym,
    }

    if p.size != 0:
        p = cast(SymbolicArray, p)
        qp["p"] = p._sym

    solver = cs.qpsol("qp_solver", "osqp", qp, options)

    # Setup for evaluating the QP solver
    if lba is None:
        lba = -np.inf * np.ones(g.shape)
    
    if uba is None:
        uba = np.inf * np.ones(g.shape)

    # Before calling the CasADi solver interface, make sure everything is
    # either a CasADi symbol or a NumPy array
    p_arg = False if p is None else p
    x0, lba, uba, p_arg = map(
        _as_casadi_array, (x0, lba, uba, p_arg),
    )

    kwargs = {
        "x0": x0,
        "lbg": lba,
        "ubg": uba,
    }

    # Add dual variables if provided
    if lam_x0 is not None:
        kwargs["lam_x0"] = _as_casadi_array(lam_x0)
    if lam_a0 is not None:
        kwargs["lam_g0"] = _as_casadi_array(lam_a0)

    # The return is a dict with keys `f`, `g`, `x` (and dual variables)
    sol = solver(**kwargs)

    # Return the solution and dual variables
    x = SymbolicArray(sol["x"], dtype=ret_dtype, shape=ret_shape)
    lam_x = SymbolicArray(sol["lam_x"], dtype=ret_dtype, shape=ret_shape)
    lam_a = SymbolicArray(sol["lam_g"], dtype=ret_dtype, shape=g.shape)

    return QPSolution(
        x=x,
        lam_x=lam_x,
        lam_a=lam_a,
    )
