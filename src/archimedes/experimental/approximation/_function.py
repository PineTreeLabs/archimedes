"""Element of a FunctionSpace: a fixed set of basis-expansion coefficients."""

from __future__ import annotations

import numpy as np

from archimedes import tree

from ._function_space import FunctionSpace

__all__ = ["Function"]


@tree.struct
class Function:
    """A specific element of a :class:`FunctionSpace`:

    .. math::
        f(x) = \\sum_{i=1}^n c_i \\, \\phi_i(x)

    Only evaluation and the operations that are exact and stay in the same
    space are supported: addition and scalar multiplication with another
    ``Function`` on the same ``space``. A general product of two
    ``Function``s is not yet supported, since the product of two finite basis
    expansions isn't generally representable in the same finite space.
    """

    coefficients: np.ndarray
    space: FunctionSpace = tree.field(static=True)

    def __call__(self, x, deriv: int = 0):
        """Evaluate this function (or its ``deriv``-th derivative) at ``x``."""
        return self.space.evaluate(self.coefficients, x, deriv=deriv)

    def __add__(self, other: Function) -> Function:
        if self.space != other.space:
            raise ValueError("Can only add Functions defined on the same FunctionSpace")
        return Function(self.coefficients + other.coefficients, self.space)

    def __mul__(self, scalar) -> Function:
        return Function(scalar * self.coefficients, self.space)

    __rmul__ = __mul__
