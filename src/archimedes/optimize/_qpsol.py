"""Solving quadratic programming problems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence, cast

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

__all__ = [
    "qpsol",
]


@tree.struct
class QPSolution:
    x: ArrayLike
    lam_a: ArrayLike


def qpsol(
    obj: Callable,
    constr: Callable,
    x0: ArrayLike,
    lba: ArrayLike | None = None,
    uba: ArrayLike | None = None,
    lam_a0: ArrayLike | None = None,
    args: Sequence = (),
    verbose: bool = False,
    name: str | None = None,
    warm_start: bool = True,
    **options,
) -> QPSolution:
    """Solve a quadratic programming problem

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
    lam_a0 : array-like, optional
        Initial guess for the dual variables associated with the constraints.
        This is used to warm-start the QP solver.
    verbose : bool, default=False
        Print output from the solver, including number of iterations and convergence.
    name : str, optional
        Name for the resulting solver function. If None, a name will be
        generated based on the objective function name.
    warm_start : bool, default=True
        Whether to enable warm starting. Default is True.
    options : dict
        Additional options passed to the underlying QP solver (OSQP).

    Returns
    -------
    solution : QPSolution

        A named tuple containing the solution to the QP problem, including:
            - ``x``: The optimal decision variables.
            - ``lam_a``: The dual variables associated with the constraints.

    Notes
    -----
    When to use this function:
    - For solving convex quadratic optimization problems efficiently
    - For embedding QP solvers in larger computational graphs
    - As part of model predictive control (MPC) or trajectory optimization

    The solution to the quadratic program is unique, so the initial guess is less
    important than for more general nonlinear programming.  The exception is when
    the QP is the convex approximation to a nonlinear program, in which case the
    initial guess is used as the linearization point.

    This function supports code generation, but requires linking the OSQP C library
    to the generated code.

    Edge cases:
    - If the objective function is not convex (i.e., the Hessian is not positive
    semidefinite), OSQP may fail to find a solution.
    - For problems with equality constraints, set the same value for both the
    lower and upper bounds.
    - Currently only supports scalar and vector decision variables, not matrices.

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Define a simple QP: minimize x^2 + y^2 subject to x + y >= 1
    >>> def obj(z):
    ...     return np.dot(z, z)
    ...
    >>> def constr(z):
    ...     return z[0] + z[1]
    ...
    >>> # Create initial guess and constraint bounds
    >>> z0 = np.array([0.0, 0.0])
    >>> lba = 1.0  # Lower bound for x + y >= 1
    >>>
    >>> # Create and solve the QP
    >>> sol = arc.qpsol(obj, constr, z0, lba=lba)
    >>> print(f"Optimal solution: x = {sol.x}")
    >>> print(f"Dual variables: λ = {sol.lam_a}")

    See Also
    --------
    minimize : More general interface for nonlinear optimization problems
    root : Function for solving systems of nonlinear equations
    """

    options = {
        "warm_start_primal": warm_start,
        "warm_start_dual": warm_start,
        "osqp": {
            "verbose": verbose,
            **options,
        },
    }

    if not isinstance(obj, FunctionCache):
        obj = FunctionCache(obj)

    if not isinstance(constr, FunctionCache):
        constr = FunctionCache(constr)

    # Check that arguments and static arguments are the same for both functions
    if not len(obj.arg_names) == len(constr.arg_names):
        raise ValueError(
            "Objective and constraint functions must have the same number of arguments"
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

    # Flatten the arguments into a single array `p`.  If necessary,
    # this could be unflattened using `_param_struct.unravel`
    p_flat, unravel = tree.ravel(args)
    p_flat_sym = sym_like(p_flat, name="p", kind="MX")
    sym_args = unravel(p_flat_sym)

    # Define a state variable for the optimization
    x = sym_like(x0, name="x", kind="MX")
    f = obj(x, *sym_args)
    g = constr(x, *sym_args)

    # For type checking
    f = cast(SymbolicArray, f)
    g = cast(SymbolicArray, g)

    qp = {
        "x": x._sym,
        "f": f._sym,
        "g": g._sym,
    }

    if p_flat.size != 0:
        qp["p"] = p_flat_sym._sym

    solver = cs.qpsol("qp_solver", "osqp", qp, options)

    # Setup for evaluating the QP solver
    if lba is None:
        lba = -np.inf * np.ones(g.shape)

    if uba is None:
        uba = np.inf * np.ones(g.shape)

    # Before calling the CasADi solver interface, make sure everything is
    # either a CasADi symbol or a NumPy array
    x0, lba, uba = map(
        _as_casadi_array,
        (x0, lba, uba),
    )

    kwargs = {
        "x0": x0,
        "lbg": lba,
        "ubg": uba,
    }

    if p_flat.size != 0:
        kwargs["p"] = _as_casadi_array(p_flat)

    # Add dual variables if provided
    if lam_a0 is not None:
        kwargs["lam_g0"] = _as_casadi_array(lam_a0)

    # The return is a dict with keys `f`, `g`, `x` (and dual variables)
    sol = solver(**kwargs)

    # Return the solution and dual variables
    x = array(sol["x"], dtype=ret_dtype).reshape(ret_shape)  # type: ignore[assignment]
    lam_a = array(sol["lam_g"], dtype=ret_dtype).reshape(g.shape)

    return QPSolution(x=x, lam_a=lam_a)
