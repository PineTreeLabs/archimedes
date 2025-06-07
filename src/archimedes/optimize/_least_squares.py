
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
from ._common import _ravel_args

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

    x0_flat, bounds, unravel = _ravel_args(x0, bounds)
    if bounds is None:
        bounds = (-np.inf, np.inf)

    print(f"Using method: {method}")
    print(f"bounds: {bounds}")

    # Compile the function and Jacobian
    @arc.compile
    def obj_func(x_flat):
        x = unravel(x_flat)
        r = func(x, *args)
        return tree.ravel(r)[0]  # Return flattened residuals

    # Call the scipy least_squares function
    result = scipy_lstsq(
        obj_func,
        x0_flat,
        args=args,
        jac=arc.jac(obj_func),
        method=method,
        bounds=bounds,
        **options,
    )
    result.x = unravel(result.x)  # Unravel the result back to original shape

    return result