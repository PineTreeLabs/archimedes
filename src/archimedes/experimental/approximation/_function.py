"""Element of a FunctionSpace: a fixed set of basis-expansion coefficients."""

from __future__ import annotations

import numpy as np

from archimedes import tree
from archimedes.quadrature import QuadratureRule

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

    ``Function`` is a struct with two fields: ``coefficients`` (the expansion
    coefficients) and ``space`` (the :class:`FunctionSpace` that defines the
    basis and domain). Since ``FunctionSpace`` is also a struct, the entire
    Function can be symbolically traced, flattened, used in optimization problems, etc.
    """

    coefficients: np.ndarray
    space: FunctionSpace

    def __call__(self, x, deriv: int = 0):
        """Evaluate this function (or its ``deriv``-th derivative) at ``x``."""
        return self.space.evaluate(self.coefficients, x, deriv=deriv)

    def __add__(self, other: Function) -> Function:
        if not self.space.is_compatible_with(other.space):
            raise ValueError("Can only add Functions defined on the same FunctionSpace")
        return Function(self.coefficients + other.coefficients, self.space)

    def __mul__(self, scalar) -> Function:
        return Function(scalar * self.coefficients, self.space)

    __rmul__ = __mul__

    def dot(self, other: Function, quad_rule: QuadratureRule | None = None):
        """Inner product :math:`\\langle f, g \\rangle` with another
        ``Function`` on the same ``space``; see
        ``FunctionSpace.inner_product``. Unlike ``__mul__``, this is safe
        for any pair of same-space ``Function``s -- the result is a
        scalar, not another element of the space.
        """
        if not self.space.is_compatible_with(other.space):
            raise ValueError(
                "Can only take the inner product of Functions on the same FunctionSpace"
            )
        return self.space.inner_product(
            self.coefficients, other.coefficients, quad_rule
        )

    def norm(self, quad_rule: QuadratureRule | None = None):
        """:math:`\\lVert f \\rVert = \\sqrt{\\langle f, f \\rangle}`."""
        return np.sqrt(self.dot(self, quad_rule))
