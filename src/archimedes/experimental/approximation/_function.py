"""Element of a FunctionSpace: a fixed set of basis-expansion coefficients."""

from __future__ import annotations

import numpy as np

from archimedes import tree
from archimedes.quadrature import QuadratureRule

from ._function_space import FunctionSpace

__all__ = ["Function"]


def _pointwise(u, v):
    """Elementwise product of two evaluated functions, promoting a
    scalar-valued one against a vector-valued one."""
    if np.ndim(u) == np.ndim(v):
        return u * v
    return u[:, None] * v if np.ndim(u) == 1 else u * v[:, None]


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

    **Vector-valued functions.** ``coefficients`` may have shape
    ``(n_basis,)`` for a scalar-valued function or ``(n_basis, m)`` for one
    mapping a scalar to an ``m``-vector -- a trajectory ``[x(t), v(t)]``,
    say. Vector-valuedness lives entirely in the coefficients: a ``Basis``
    evaluates to ``(npts, n_basis)`` regardless, so the ``space`` is
    unchanged and the vector space :math:`V^m` is implied by the trailing
    axis. Evaluating gives ``(npts,)`` or ``(npts, m)`` correspondingly.
    """

    coefficients: np.ndarray
    space: FunctionSpace

    def __call__(self, x, deriv: int = 0):
        """Evaluate this function (or its ``deriv``-th derivative) at ``x``.

        Returns shape ``(npts,)`` or ``(npts, m)``, matching
        ``coefficients``.
        """
        return self.space.evaluate(self.coefficients, x, deriv=deriv)

    def __add__(self, other: Function) -> Function:
        if not self.space._is_compatible_with(other.space):
            raise ValueError("Can only add Functions defined on the same FunctionSpace")
        return Function(self.coefficients + other.coefficients, self.space)

    def __mul__(self, other) -> Function:
        """Scalar multiple, or the pointwise product of two ``Function``s.

        A scalar multiple stays in the same space. A ``Function`` product
        does not -- see :meth:`multiply`.
        """
        if isinstance(other, Function):
            return self.multiply(other)
        return Function(other * self.coefficients, self.space)

    __rmul__ = __mul__

    def multiply(self, other: Function, space: FunctionSpace | None = None) -> Function:
        """Pointwise product :math:`(fg)(x) = f(x) \\, g(x)`.

        The product of two basis expansions does not lie in either operand's
        space, so the result is returned in a *larger* one, which for
        polynomial families has ``n_1 + n_2 - 1`` degrees of freedom and
        represents the product **exactly** (to quadrature roundoff), not as an
        approximation.

        The degree therefore grows with each product. That is deliberate --
        an exact operation should not silently lose information -- so
        reducing back down requires an explicit :meth:`FunctionSpace.project`
        rather than an automatic truncation. Repeated products without
        projecting will grow the space quickly.

        Parameters
        ----------
        other : Function
            The other factor. Must be on a compatible space; see
            :meth:`FunctionSpace._product_space`.
        space : FunctionSpace, optional
            Result space, overriding the automatic one. If it is too small
            to represent the product, the result is the projection of the
            product onto it -- a well-defined approximation, but no longer
            exact.

        Returns
        -------
        Function
            The product, in the result space.
        """
        result_space = (
            space if space is not None else self.space._product_space(other.space)
        )
        return result_space.project(
            lambda x: _pointwise(self(x), other(x)),
            # `project`'s guard against an under-resolved rule does not apply
            # here: the product space's default rule is exact for this
            # integrand by construction.
        )

    def dot(self, other: Function, quad_rule: QuadratureRule | None = None):
        """Inner product :math:`\\langle f, g \\rangle` with another
        ``Function`` on the same ``space``; see
        ``FunctionSpace.inner_product``. Unlike ``__mul__``, this is safe
        for any pair of same-space ``Function``s -- the result is a
        scalar, not another element of the space.

        For vector-valued coefficients the integrand is contracted over
        components, so the result is a scalar.
        """
        if not self.space._is_compatible_with(other.space):
            raise ValueError(
                "Can only take the inner product of Functions on the same FunctionSpace"
            )
        return self.space.inner_product(
            self.coefficients, other.coefficients, quad_rule
        )

    def norm(self, quad_rule: QuadratureRule | None = None):
        """:math:`\\lVert f \\rVert = \\sqrt{\\langle f, f \\rangle}`."""
        return np.sqrt(self.dot(self, quad_rule))
