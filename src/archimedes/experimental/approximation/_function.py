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
    r"""A specific element of a :class:`FunctionSpace`:

    .. math::
        f(x) = \sum_{i=1}^n c_i \, \phi_i(x)

    Only evaluation and the operations that are exact and stay in the same
    space are supported: addition, subtraction, and scalar multiplication
    or division with another ``Function`` on the same ``space``. Combining
    ``Function`` objects on different spaces raises rather than silently
    projecting one onto the other -- see :meth:`FunctionSpace.project` to do
    that explicitly. A general product of two ``Function`` objects is not
    directly supported (see :meth:`multiply`), since the product of two
    finite basis expansions isn't generally representable in the same
    finite space; nor is dividing by one, since the quotient of two basis
    expansions is generally not a finite expansion at all.

    Parameters
    ----------
    coefficients : ndarray
        Expansion coefficients, shape ``(n_basis,)`` for a scalar-valued
        function or ``(n_basis, m)`` for one mapping to an ``m``-vector
        (e.g. a trajectory ``[x(t), v(t)]``). Vector-valuedness lives
        entirely here: a ``Basis`` evaluates to ``(npts, n_basis)``
        regardless, so evaluating a ``Function`` gives ``(npts,)`` or
        ``(npts, m)`` to match.
    space : FunctionSpace
        The space this function belongs to, defining its basis and domain.

    Notes
    -----
    Since ``FunctionSpace`` is also a ``@struct``, a ``Function`` can be
    symbolically traced, flattened, and used in optimization problems like
    any other pytree.
    """

    coefficients: np.ndarray
    space: FunctionSpace

    def __call__(self, x, deriv: int = 0, side: str = RIGHT):
        """Evaluate this function (or its ``deriv``-th derivative) at ``x``.

        Parameters
        ----------
        x : array_like
            Evaluation points, shape ``(npts,)``.
        deriv : int, optional
            Derivative order; a multi-index (one order per dimension) for a
            multivariate space, a plain order otherwise. Default 0.
        side : {"right", "left"}, optional
            Which one-sided limit to take where this function is
            two-valued -- e.g. at a breakpoint of a piecewise space.
            Irrelevant where the basis is smooth, since the two limits then
            coincide. A multivariate space takes one entry per dimension,
            with a bare string broadcasting. Default ``"right"``.

        Returns
        -------
        ndarray
            Shape ``(npts,)`` or ``(npts, m)``, matching ``coefficients``.
        """
        return self.space._evaluate(self.coefficients, x, deriv=deriv, side=side)

    def __add__(self, other: Function) -> Function:
        if not self.space._is_compatible_with(other.space):
            raise ValueError(
                "Can only add Functions defined on the same FunctionSpace; "
                "project one onto the other's space first, e.g. "
                "self + self.space.project(other)"
            )
        return Function(self.coefficients + other.coefficients, self.space)

    def __sub__(self, other: Function) -> Function:
        if not self.space._is_compatible_with(other.space):
            raise ValueError(
                "Can only subtract Functions defined on the same FunctionSpace; "
                "project one onto the other's space first, e.g. "
                "self - self.space.project(other)"
            )
        return Function(self.coefficients - other.coefficients, self.space)

    def __neg__(self) -> Function:
        return Function(-self.coefficients, self.space)

    def __mul__(self, other) -> Function:
        """Scalar multiple, or the pointwise product of two ``Function`` objects.

        A scalar multiple stays in the same space. A ``Function`` product
        does not -- see :meth:`multiply`.
        """
        if isinstance(other, Function):
            return self.multiply(other)
        return Function(other * self.coefficients, self.space)

    def __truediv__(self, other) -> Function:
        """Scalar division; see :meth:`__mul__`."""
        if isinstance(other, Function):
            raise ValueError(
                "Cannot divide by a Function: the quotient of two basis "
                "expansions is generally not a finite expansion at all, so "
                "there is no space to return it in. Approximate it "
                "explicitly instead, e.g. "
                "self.space.project(lambda x: self(x) / other(x))"
            )
        return self * (1 / other)

    __rmul__ = __mul__

    def multiply(self, other: Function, space: FunctionSpace | None = None) -> Function:
        r"""Pointwise product :math:`(fg)(x) = f(x) \, g(x)`.

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
            The other factor. Must be on a compatible space (see
            :class:`FunctionSpace`).
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
        than this one (differentiating lowers the degree).

        Since the target is smaller, ``f + f.derivative()`` will not
        typecheck as-is; project one onto the other's space first, which is
        exact in either direction. For the derivative sampled at points
        rather than as a ``Function``, ``f(x, deriv=k)`` is more direct.

        Parameters
        ----------
        deriv : int or tuple of int, optional
            Derivative order; a multi-index (one order per dimension) for a
            function of several variables, a plain order otherwise -- the
            same convention as calling the function directly. Default 1.
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

    def antiderivative(
        self,
        order: int = 1,
        boundary: str = "left",
        space: FunctionSpace | None = None,
    ) -> Function:
        r"""The ``order``-th antiderivative :math:`F^{(-\mathrm{order})}`.

        Numerically exact, and the dual of :meth:`derivative`: the result is returned
        in the smallest space that represents it, which for a polynomial family is
        *larger* than this one (integrating raises the degree). This is needed to
        make the antiderivative well-defined: an indefinite integral is only unique
        up to an additive constant (per order), and ``boundary`` pins it by requiring
        :math:`F` (and, for ``order > 1``, its derivatives through order ``order - 1``)
        to vanish at that endpoint of the domain:

            - ``"left"`` (the default) gives :math:`F(x) = \int_a^x f(t)\,dt`,
                so :math:`F(a) = 0`
            - ``"right"`` gives :math:`F(x) = \int_b^x f(t)\,dt = -\int_x^b f(t)\,dt`,
                so :math:`F(b) = 0`

        The two differ by the whole-domain definite integral: with
        ``boundary="right"``, ``f.antiderivative()`` is ``total`` less than
        it is with ``boundary="left"``.

        Parameters
        ----------
        order : int, optional
            Number of times to integrate. Default 1.
        boundary : {"left", "right"}, optional
            Domain endpoint at which the antiderivative (and its lower
            derivatives, for ``order > 1``) vanishes. Default ``"left"``.
        space : FunctionSpace, optional
            Result space, overriding the automatic one. Must have exactly
            ``self.space.n_basis + order`` basis functions.

        Returns
        -------
        Function
            The antiderivative, in the result space, with coefficients of
            shape ``(n_basis,)`` or ``(n_basis, m)`` matching this function.

        Raises
        ------
        NotImplementedError
            If this function's space has no integral-space construction --
            e.g. :class:`PiecewiseBasis`.
        ValueError
            If the domain has no finite endpoints to anchor at (Hermite,
            Laguerre), or if an explicit ``space`` is the wrong size.
        """
        target = space if space is not None else self.space._integral_space(order)
        return Function(
            self.space._integral_matrix(order, boundary=boundary, space=target)
            @ self.coefficients,
            target,
        )

    def integrate(self, a: float | None = None, b: float | None = None):
        r"""Definite integral :math:`\int_a^b f(x)\,dx`.

        Wherever :meth:`antiderivative` is defined for this space, this is built
        directly on it: :math:`F(b) - F(a)`, exact to quadrature roundoff, for
        *any* ``a``, ``b`` within the domain.

        Where :meth:`antiderivative` isn't defined the *whole-domain* integral
        (``a=None, b=None``) is still available, computed directly from this space's
        own quadrature rule.

        Parameters
        ----------
        a, b : float, optional
            Integration bounds. Default the domain's own endpoints (the
            whole-domain integral).

        Returns
        -------
        ndarray or float
            Shape ``()`` for a scalar-valued function, ``(m,)`` for a
            vector-valued one.

        Raises
        ------
        NotImplementedError
            If this function's space has no integral-space construction
            and ``a``/``b`` narrow the bounds below the whole domain.
        ValueError
            If the domain has no finite endpoints to integrate over
            (Hermite, Laguerre).
        """
        try:
            antideriv = self.antiderivative()
        except NotImplementedError:
            if a is not None or b is not None:
                raise
            x, w = self.space.quadrature()
            return w @ self(x)
        domain = self.space.domain
        lo = domain.a if a is None else a
        hi = domain.b if b is None else b
        return antideriv(np.array([hi]))[0] - antideriv(np.array([lo]))[0]

    def dot(self, other: Function, quad_rule: QuadratureRule | None = None):
        r"""Inner product :math:`\langle f, g \rangle` with another
        ``Function`` on the same ``space``.

        Unlike ``__mul__``, this is well-defined for any pair of same-space
        ``Function`` objects: the result is a scalar, not another element of
        the space, so there is no larger result space to resolve.

        Parameters
        ----------
        other : Function
            The other operand, on the same space as this function.
        quad_rule : QuadratureRule, optional
            Quadrature rule to approximate the integral with. Default
            this space's own quadrature.

        Returns
        -------
        float
            The inner product. For vector-valued coefficients the
            integrand is contracted over components, so the result is
            always a scalar.
        """
        if not self.space._is_compatible_with(other.space):
            raise ValueError(
                "Can only take the inner product of Functions on the same "
                "FunctionSpace; project one onto the other's space first, "
                "e.g. self.dot(self.space.project(other))"
            )
        return self.space._inner_product(
            self.coefficients, other.coefficients, quad_rule
        )

    def norm(self, quad_rule: QuadratureRule | None = None):
        r""":math:`\lVert f \rVert = \sqrt{\langle f, f \rangle}`; see :meth:`dot`."""
        return np.sqrt(self.dot(self, quad_rule))
