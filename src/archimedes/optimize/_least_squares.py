
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, cast

import numpy as np
from scipy.optimize import OptimizeResult, least_squares as scipy_lstsq

import archimedes as arc
from archimedes import tree
from archimedes._core import (
    compile,
    grad,
)
from ._lm import lm_solve

if TYPE_CHECKING:
    from ..typing import PyTree
    T = TypeVar("T", bound=PyTree)


__all__ = ["least_squares"]


SCIPY_METHODS = ["trf", "dogbox", "lm"]
SUPPORTED_METHODS = SCIPY_METHODS + ["hess-lm"]


def least_squares(
    func: Callable[[T, Any], T],
    x0: T,
    args: Sequence[Any] = (),
    method: str = "hess-lm",
    bounds: tuple[T, T] | None = None,
    options: dict | None = None,
) -> OptimizeResult:
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

    if bounds is not None:
        lb, ub = bounds
        lb_flat, _ = arc.tree.ravel(lb)
        ub_flat, _ = arc.tree.ravel(ub)
        # Zip bounds into (lb, ub) for each parameter
        bounds = list(zip(lb_flat, ub_flat))

    x0_flat, unravel = arc.tree.ravel(params_guess)

    # Compile the function and Jacobian
    @arc.compile
    def obj_func(x_flat):
        x = unravel(x_flat)
        r = func(x, *args)
        return tree.ravel(r)[0]  # Return flattened residuals

    # Call the scipy least_squares function
    result = scipy_lstsq(
        fun_compiled,
        x0_flat,
        args=args,
        jac=arc.jac(obj_func),
        method=method,
        bounds=bounds,
        **options,
    )
    result.x = unravel(result.x)  # Unravel the result back to original shape

    return result