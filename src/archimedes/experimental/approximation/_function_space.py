"""Finite-dimensional linear span of a Basis, tied to a target domain."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from archimedes import tree
from archimedes.quadrature import QuadratureRule

from ._basis import Basis

if TYPE_CHECKING:
    from ._function import Function

__all__ = ["FunctionSpace"]


@tree.struct
class FunctionSpace:
    """The linear span of a :class:`Basis` on a fixed target domain.

    Pairs a ``Basis`` (family + size) with a target domain and a "natural"
    quadrature rule, and provides the operations that act on the *space*
    rather than on any one element of it: evaluation, the mass and
    stiffness matrices, and (Galerkin) projection.

    See :class:`Function` for a specific element of the space (a
    coefficient vector).

    This is a ``@struct`` rather than a plain dataclass so that ``domain``
    is a pytree leaf: the domain parameters can be symbolically traced,
    and so optimized over (moving the endpoints of an element, say) jointly
    with a ``Function``'s coefficients. ``basis`` and ``quad_rule`` are
    static -- they carry structure, not numbers.

    Parameters
    ----------
    basis : Basis
        Basis family and size. Static.
    domain : basis.Parameters
        Target-domain parameters, validated and typed per ``basis`` -- an
        instance of ``basis.Parameters`` (e.g.
        ``UnitInterval.Parameters(a=..., b=...)`` for a basis on an
        interval, ``RealLine.Parameters(mean=..., std=...)`` for a
        Hermite-derived one). A pytree leaf, so it may be traced.
    quad_rule : QuadratureRule
        The space's natural quadrature rule, used unconditionally by
        ``mass_matrix``/``stiffness_matrix`` (their integrands' required
        accuracy is fully determined by ``basis``, so there's no reason to
        let it drift from this default) and as the default for ``project``
        (which accepts an explicit override, since the right accuracy
        for a given target function isn't knowable from the space alone).
        Static.
    """

    basis: Basis = tree.field(static=True)
    domain: Any
    quad_rule: QuadratureRule = tree.field(static=True)

    def __post_init__(self):
        if not isinstance(self.domain, self.basis.Parameters):
            raise TypeError(
                f"domain must be a {self.basis.Parameters.__qualname__} "
                f"instance for this basis, got {type(self.domain).__name__}"
            )

    @property
    def n_basis(self) -> int:
        """Number of basis functions; forwarded from ``basis``."""
        return self.basis.n_basis

    def is_compatible_with(self, other: FunctionSpace) -> bool:
        """Whether ``other`` denotes the same space, as far as is decidable.

        Compares ``basis`` and ``quad_rule`` by value, and ``domain`` only
        *structurally* (via its treedef) -- deliberately not by value.

        Domain values can't be compared once traced: two symbolic
        parametrizations raise ``TypeError`` under ``bool()`` unless they
        happen to be the same objects, and independently-traced ``Function``
        arguments get distinct symbols even when their source domains were
        numerically identical. So value comparison would be both
        undecidable and prone to false rejection.

        This means a caller can add two ``Function``s whose domains differ
        *numerically* -- e.g. ``(a=0, b=1)`` and ``(a=0, b=2)`` -- without
        an error. **Callers are responsible for ensuring the domains agree
        numerically**; only the structure is enforced here.
        """
        return (
            self.basis == other.basis
            and self.quad_rule == other.quad_rule
            and tree.structure(self.domain) == tree.structure(other.domain)
        )

    def _domain_kwargs(self) -> dict:
        return {f.name: getattr(self.domain, f.name) for f in tree.fields(self.domain)}

    def _basis_eval(self, x, deriv: int = 0):
        return self.basis.evaluate(x, deriv=deriv, **self._domain_kwargs())

    def _quad_points_weights(self, quad_rule: QuadratureRule | None = None):
        rule = quad_rule if quad_rule is not None else self.quad_rule
        # phi is (npts, n_basis), so phi.T @ diag(w) @ phi has rank at most
        # min(npts, n_basis) -- below n_basis points the mass matrix is
        # exactly (not just poorly) singular, and np.linalg.solve in
        # `project` blows up rather than failing cleanly.
        if len(rule) < self.n_basis:
            raise ValueError(
                f"quad_rule has {len(rule)} points, fewer than n_basis="
                f"{self.n_basis}; the mass matrix would be singular"
            )
        domain_kwargs = self._domain_kwargs()
        return rule.scaled_points(**domain_kwargs), rule.scaled_weights(**domain_kwargs)

    def evaluate(self, coefficients: np.ndarray, x, deriv: int = 0):
        """Evaluate :math:`\\sum_i c_i \\, \\phi_i(x)` (or its ``deriv``-th
        derivative) at ``x``, for coefficients ``c = coefficients``.

        Parameters
        ----------
        coefficients : ndarray
            Shape ``(n_basis,)`` for a scalar-valued function, or
            ``(n_basis, m)`` for an ``m``-component vector-valued one (see
            :class:`Function`).
        x : array_like
            Evaluation points, shape ``(npts,)``.
        deriv : int, optional
            Derivative order. Default 0.

        Returns
        -------
        ndarray
            Shape ``(npts,)`` or ``(npts, m)``, matching ``coefficients``.
        """
        phi = self._basis_eval(x, deriv=deriv)  # (npts, n_basis)
        return phi @ coefficients

    def inner_product(
        self,
        c1: np.ndarray,
        c2: np.ndarray,
        quad_rule: QuadratureRule | None = None,
    ):
        """Inner product :math:`\\langle f, g \\rangle = \\int f(x) \\, g(x)
        \\, w(x) \\, dx` for ``f``, ``g`` in this space with coefficients
        ``c1``, ``c2``, approximated via ``quad_rule`` (default
        ``self.quad_rule``).

        Unlike a product of two ``Function``s (deliberately unsupported --
        see :class:`Function`), an inner product returns a scalar rather
        than another element of the space, so there's no aliasing/closure
        question to resolve: it's computed by evaluating both functions at
        the quadrature nodes and integrating the pointwise product, which
        is exact whenever ``quad_rule`` is accurate enough for that
        product -- equivalently ``c1 @ mass_matrix() @ c2``, but computed
        directly without forming the full ``(n_basis, n_basis)`` matrix.

        For vector-valued coefficients (shape ``(n_basis, m)``) the
        integrand is contracted over components, :math:`\\langle f, g
        \\rangle = \\int f \\cdot g \\, w \\, dx`, so the result is a
        scalar in that case too and :meth:`Function.norm` is the
        :math:`L^2` norm of the whole vector-valued function rather than
        an array of per-component norms. Both coefficient arrays must have
        the same shape; for a per-component inner product, slice the
        coefficients and call this once per component.
        """
        x, w = self._quad_points_weights(quad_rule)
        phi = self._basis_eval(x)  # (npts, n_basis)
        integrand = (phi @ c1) * (phi @ c2)  # (npts,) or (npts, m)
        if integrand.ndim > 1:
            # `ndim` is a static (trace-time) property, so this branches on
            # shape rather than on a value and is safe under `@arc.compile`.
            # `axis=1` rather than the idiomatic `axis=-1`: the symbolic
            # `np.sum` does not normalize a negative axis.
            integrand = np.sum(integrand, axis=1)
        return np.dot(w, integrand)

    def mass_matrix(self) -> np.ndarray:
        """Mass matrix :math:`M_{ij} = \\int \\phi_i \\, \\phi_j \\, w \\, dx`,
        approximated via ``self.quad_rule``.
        """
        x, w = self._quad_points_weights()
        phi = self._basis_eval(x)  # (npts, n_basis)
        return phi.T @ (w[:, None] * phi)

    def stiffness_matrix(self) -> np.ndarray:
        """Stiffness matrix :math:`K_{ij} = \\int \\phi_i' \\, \\phi_j' \\,
        w \\, dx`, approximated via ``self.quad_rule``.
        """
        x, w = self._quad_points_weights()
        dphi = self._basis_eval(x, deriv=1)
        return dphi.T @ (w[:, None] * dphi)

    def project(self, f: Callable, quad_rule: QuadratureRule | None = None) -> Function:
        """Galerkin projection of ``f`` onto this space.

        Solves ``M @ c = b`` for the coefficients ``c``, where ``M`` is the
        mass matrix and ``b_i = int f(x) phi_i(x) w(x) dx``, both
        approximated via ``quad_rule`` (default ``self.quad_rule``).
        ``quad_rule`` must be accurate enough for the product of ``f`` and
        the basis, which is generally a higher-order requirement than
        exactness for the basis alone; pass an explicit ``quad_rule`` to
        use something other than the space's natural default.

        ``f`` may be vector-valued: if ``f(x)`` has shape ``(npts, m)``,
        each component is projected onto the same space and the result has
        coefficients of shape ``(n_basis, m)``. The mass matrix is shared
        across components, so this costs one basis evaluation rather than
        ``m`` of them.

        Parameters
        ----------
        f : callable
            Target function, called once as ``f(x)`` on the full node
            array from the quadrature rule. Must return an array of shape
            ``(npts,)`` (scalar-valued) or ``(npts, m)`` (vector-valued).
        quad_rule : QuadratureRule, optional
            Quadrature rule to use instead of ``self.quad_rule``.

        Returns
        -------
        Function
            The projected function, in this space, with coefficients of
            shape ``(n_basis,)`` or ``(n_basis, m)`` to match ``f``.
        """
        from ._function import Function  # avoid a circular import

        x, w = self._quad_points_weights(quad_rule)
        phi = self._basis_eval(x)  # (npts, n_basis)
        M = phi.T @ (w[:, None] * phi)
        fx = f(x)

        # `ndim` is a static (trace-time) property, so this branches on shape
        # rather than on a value and is safe under `@arc.compile`.
        if fx.ndim == 1:
            return Function(np.linalg.solve(M, phi.T @ (w * fx)), self)

        # Vector-valued: one coefficient column per component. Solved column
        # by column because the symbolic `np.linalg.solve` accepts only a
        # vector right-hand side; `m` is static, so the loop unrolls at trace
        # time. NumPy alone would take the whole (n_basis, m) right-hand side
        # in a single call.
        rhs = phi.T @ (w[:, None] * fx)  # (n_basis, m)
        columns = [np.linalg.solve(M, rhs[:, k]) for k in range(fx.shape[1])]
        return Function(np.stack(columns, axis=-1), self)
