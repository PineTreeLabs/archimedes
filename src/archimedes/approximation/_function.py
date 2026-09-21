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

    A :class:`Function` is a callable object that can be evaluated with typical
    function-call syntax, e.g., ``f(x)``. Evaluation supports both numeric
    and symbolic evaluation for any function space.

    A limited set of arithmetic operations is supported:

    - Addition of two :class:`Function` objects on the same :attr:`space`.
    - Subtraction of two :class:`Function` objects on the same :attr:`space`.
    - Scalar multiplication and division of a :class:`Function` object.
    - Negation of a :class:`Function` object.

    Supported operations are exact and closed on the function space.
    That is, the result of any of the above is exactly representable in
    the original function space.

    Multiplication between two :class:`Function` objects is generally not
    closed on the function space, and so is not supported via operator
    overloading ("dunder" `__mul__` methods). However, since the function
    space in which a pointwise product is representable can be determined
    exactly, the operation is supported via the dedicated :meth:`multiply`
    method.

    Similarly, derivative and anti-derivative (indefinite integral) operations
    are supported and generally return ``Function`` objects in the **minimal**
    containing function space. For example, the derivative of a polynomial of
    degree ``n`` is a polynomial of degree ``n-1``. The returned space can be
    overridden by manually specifying a different target space.

    The dot product of two :class:`Function` objects is implemented in terms of
    numerical quadrature, exactly integrating the product of the two functions.
    The dot product of vector-valued functions also contracts over the vector
    components and unconditionally returns a scalar value.

    Other math operations can be implemented via L2-projection of the function.
    For example, pointwise division of two functions can be approximated by
    ``f.space.project(lambda x: f(x) / g(x))``. Note that this is an approximation,
    not an exact representation in the original function space.

    A :class:`Function` is implemented as a [struct](#archimedes.struct), making
    it compatible with flattening, unflattening, mapping, and other tree
    operations. The "children" of the struct are the coefficients and the space
    and the space itself. Note that function spaces are themselves structs, so
    **flattening the function may include boundary endpoint data**.
    
    This is useful for variable-endpoint problems, but not desirable when the
    domain endpoints are fixed. In this case, a typical approach is to work
    directly with the coefficient array, reconstructing or replacing the full
    ``Function`` object as needed.

    Parameters
    ----------
    coefficients : ndarray
        Expansion coefficients, shape ``(n_basis,)`` for a scalar-valued
        function or ``(n_basis, m)`` for one mapping to an ``m``-vector.
    space : FunctionSpace
        The space this function belongs to, defining its basis and domain.
    """

    coefficients: np.ndarray
    space: FunctionSpace

    def __call__(self, x, deriv: int = 0, side: str = RIGHT):
        """Evaluate this function (or its ``deriv``-th derivative) at ``x``.

        Supports both numerical and symbolic evaluation.

        Parameters
        ----------
        x : array_like
            Evaluation points, shape ``(npts,)``.
        deriv : int, optional
            Derivative order; a multi-index (one order per dimension) for a
            multivariate space, a plain order otherwise. Default 0.
        side : {"right", "left"}, optional
            Which one-sided limit to take if this function is two-valued.
            Irrelevant where the basis is smooth. A multivariate space takes
            one entry per dimension or broadcasts a string. Default ``"right"``.

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
                "f + f.space.project(g)"
            )
        return Function(self.coefficients + other.coefficients, self.space)

    def __sub__(self, other: Function) -> Function:
        if not self.space._is_compatible_with(other.space):
            raise ValueError(
                "Can only subtract Functions defined on the same FunctionSpace; "
                "project one onto the other's space first, e.g. "
                "f - f.space.project(g)"
            )
        return Function(self.coefficients - other.coefficients, self.space)

    def __neg__(self) -> Function:
        return Function(-self.coefficients, self.space)

    def __mul__(self, other) -> Function:
        """Scalar multiple, or the pointwise product of two ``Function`` objects. """
        if isinstance(other, Function):
            return self.multiply(other)
        return Function(other * self.coefficients, self.space)

    def __truediv__(self, other) -> Function:
        """Scalar division; see :meth:`__mul__`."""
        if isinstance(other, Function):
            raise ValueError(
                "Cannot divide by a Function. Approximate it "
                "explicitly instead, e.g. "
                "f.space.project(lambda x: f(x) / g(x))"
            )
        return self * (1 / other)

    __rmul__ = __mul__

    def multiply(self, other: Function, space: FunctionSpace | None = None) -> Function:
        r"""Pointwise product :math:`(fg)(x) = f(x) g(x)`.

        In general, the product of two basis expansions does not lie in either
        operand's space, so the result is returned in a *larger* one. For polynomial
        families, the product space has ``n_1 + n_2 - 1`` degrees of freedom and
        represents the product exactly.

        This behavior can be overridden by manually passing a specific result space,
        in which case the exact product is projected onto the specified space.

        Parameters
        ----------
        other : Function
            The other factor. Must be on a compatible space (see
            :class:`FunctionSpace`).
        space : FunctionSpace, optional
            Result space, overriding the automatic one.

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

        The result is returned in the smallest space that represents it exactly.
        For a polynomial family, this result space is smaller than the original,
        since differentiating lowers the degree.

        This behavior can be overridden by manually passing a specific result space,
        in which case the exact derivative is projected onto the specified space.

        For the derivative sampled at points rather than as a ``Function``,
        prefer using ``f(x, deriv=k)`` to ``f.derivative(k)(x)``.

        Parameters
        ----------
        deriv : int or tuple of int, optional
            Derivative order; a multi-index (one order per dimension) for a
            function of several variables, a plain order otherwise. Default 1.
        space : FunctionSpace, optional
            Result space, overriding the automatic one.

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

        The antiderivative is numerically exact and is the dual of
        :meth:`derivative` (i.e. an indefinite integral). The result is returned
        in the smallest space that represents it. For a polynomial family, that
        space is *larger* than this one, since integrating raises the degree.

        This behavior can be overridden by manually passing a specific result space,
        in which case the exact antiderivative is projected onto the specified space.

        An indefinite integral is unique only up to an additive constant (per order),
        so the antiderivative must be "pinned" at a domain boundary. The pinned
        value is determined by the ``boundary`` argument by defining the antiderivative
        to vanish at the specified boundary:

            - ``"left"`` (default) gives :math:`F(x) = \int_a^x f(t)\,dt`.
            - ``"right"`` gives :math:`F(x) = \int_b^x f(t)\,dt =
              -\int_x^b f(t)\,dt`.

        Parameters
        ----------
        order : int, optional
            Number of times to integrate. Default 1.
        boundary : {"left", "right"}, optional
            Domain endpoint at which the antiderivative (and its lower
            derivatives, for ``order > 1``) vanishes. Default ``"left"``.
        space : FunctionSpace, optional
            Result space, overriding the automatic one.

        Returns
        -------
        Function
            The antiderivative, in the result space, with coefficients of
            shape ``(n_basis,)`` or ``(n_basis, m)`` matching this function.

        Raises
        ------
        NotImplementedError
            If this function's space has no integral-space construction.
        ValueError
            If the domain has no finite endpoints to anchor at, or if an explicit
            ``space`` is the wrong size.
        """
        target = space if space is not None else self.space._integral_space(order)
        return Function(
            self.space._integral_matrix(order, boundary=boundary, space=target)
            @ self.coefficients,
            target,
        )

    def integrate(self, a: float | None = None, b: float | None = None):
        r"""Definite integral :math:`\int_a^b f(x)\,dx`.

        Wherever :meth:`antiderivative` is defined for this space, the integral
        evaluates :math:`F(b) - F(a)` (exact to quadrature roundoff) for any
        :math:`a` and :math:`b` within the domain.

        If :meth:`antiderivative` isn't defined, the whole-domain integral
        (``a=None, b=None``) is still available, computed directly from the
        quadrature rule of this function's :class:`FunctionSpace`.

        Parameters
        ----------
        a, b : float, optional
            Integration bounds. Default: the domain's endpoints.

        Returns
        -------
        ndarray or float
            Shape ``()`` for a scalar-valued function, ``(m,)`` for a
            vector-valued one.

        Raises
        ------
        NotImplementedError
            If this function's space has no integral-space construction
            and ``a`` and/or ``b`` are not ``None``.
        ValueError
            If the domain has no finite endpoints to integrate over.
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
        r"""Inner product with another ``Function`` on the same ``space``.

        Computes :math:`\langle f, g \rangle` for functions :math:`f` and
        :math:`g` on the same space.

        Unlike the multiplication operator, this is well-defined for any pair of
        ``Function`` objects on the same space, since the result is always a scalar.
        For vector-valued functions, the inner product contracts over components.

        Parameters
        ----------
        other : Function
            The other operand, on the same space as this function.
        quad_rule : QuadratureRule, optional
            Quadrature rule to approximate the integral with. Defaults to
            this space's quadrature.

        Returns
        -------
        float
            The inner product.
        """
        if not self.space._is_compatible_with(other.space):
            raise ValueError(
                "Can only take the inner product of Functions on the same "
                "FunctionSpace; project one onto the other's space first, "
                "e.g. f.dot(f.space.project(other))"
            )
        return self.space._inner_product(
            self.coefficients, other.coefficients, quad_rule
        )

    def norm(self, quad_rule: QuadratureRule | None = None):
        r"""Compute the norm :math:`\lVert f \rVert = \sqrt{\langle f, f \rangle}`.

        See :meth:`dot`.
        """
        return np.sqrt(self.dot(self, quad_rule))
