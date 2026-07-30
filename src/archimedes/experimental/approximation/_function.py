"""Element of a FunctionSpace: a fixed set of basis-expansion coefficients."""

from __future__ import annotations

import numpy as np

from archimedes import tree
from archimedes.quadrature import QuadratureRule

from ._basis import RIGHT
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

    def __call__(self, x, deriv: int = 0, side: str = RIGHT):
        """Evaluate this function (or its ``deriv``-th derivative) at ``x``.

        Returns shape ``(npts,)`` or ``(npts, m)``, matching
        ``coefficients``.

        ``side="left"``/``"right"`` selects which one-sided limit to take at
        a point where this function is two-valued -- on a breakpoint of a
        piecewise space, giving the values a discontinuous-Galerkin flux or
        a gradient-jump error indicator is built from. It is accepted for
        every space and irrelevant where the basis is smooth, since the two
        limits then coincide. A :class:`TensorBasis` takes one entry per
        dimension, with a bare string broadcasting. See
        :meth:`Basis.evaluate`.
        """
        return self.space._evaluate(self.coefficients, x, deriv=deriv, side=side)

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

    def derivative(self, deriv=1, space: FunctionSpace | None = None) -> Function:
        """The ``deriv``-th derivative :math:`f^{(k)}`, as a ``Function``.

        Exact, not an approximation: the result is returned in the smallest
        space that represents it, which for a polynomial family is *smaller*
        than this one (differentiating lowers the degree). That mirrors
        :meth:`multiply`, which returns the smallest space that is exact in
        the other direction -- every closed operation here gives the
        tightest exact space.

        Since the target is smaller, ``f + f.derivative()`` will not
        typecheck as-is; project one onto the other's space first, which is
        exact in either direction. For the derivative sampled at points
        rather than as a ``Function``, ``f(x, deriv=k)`` is more direct.

        Parameters
        ----------
        deriv : int or tuple of int, optional
            Derivative order; a multi-index (one order per dimension) for a
            function of several variables, a plain order otherwise -- the
            same convention as :meth:`__call__`. Default 1.
        space : FunctionSpace, optional
            Result space, overriding the automatic one. A space too small to
            hold the derivative gives its projection rather than an error,
            as in :meth:`multiply`.

        Returns
        -------
        Function
            The derivative, in the result space, with coefficients of shape
            ``(n_basis,)`` or ``(n_basis, m)`` matching this function.
        """
        target = space if space is not None else self.space._derivative_space(deriv)
        return Function(
            self.space._diff_matrix(deriv, space=target) @ self.coefficients, target
        )

    def dot(self, other: Function, quad_rule: QuadratureRule | None = None):
        """Inner product :math:`\\langle f, g \\rangle` with another
        ``Function`` on the same ``space``. Unlike ``__mul__``, this is safe
        for any pair of same-space ``Function``s -- the result is a
        scalar, not another element of the space.

        For vector-valued coefficients the integrand is contracted over
        components, so the result is a scalar.
        """
        if not self.space._is_compatible_with(other.space):
            raise ValueError(
                "Can only take the inner product of Functions on the same FunctionSpace"
            )
        return self.space._inner_product(
            self.coefficients, other.coefficients, quad_rule
        )

    def norm(self, quad_rule: QuadratureRule | None = None):
        """:math:`\\lVert f \\rVert = \\sqrt{\\langle f, f \\rangle}`."""
        return np.sqrt(self.dot(self, quad_rule))
