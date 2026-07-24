"""Finite-dimensional linear span of a Basis, tied to a target domain."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from archimedes.quadrature import QuadratureRule

from ._basis import Basis

if TYPE_CHECKING:
    from ._function import Function

__all__ = ["FunctionSpace"]


@dataclasses.dataclass(frozen=True)
class FunctionSpace:
    """The linear span of a :class:`Basis` on a fixed target domain.

    Pairs a ``Basis`` (family + size) with a target domain and a "natural"
    quadrature rule, and provides the operations that act on the *space*
    rather than on any one element of it: evaluation, the mass and
    stiffness matrices, and (Galerkin) projection.

    See :class:`Function` for a specific element of the space (a
    coefficient vector).

    Parameters
    ----------
    basis : Basis
        Basis family and size.
    domain : basis.Parameters
        Target-domain parameters, validated and typed per ``basis`` --
        an instance of ``basis.Parameters`` (e.g.
        ``LegendreMeasure.Parameters(a=..., b=...)`` for a Legendre-derived
        basis, ``HermiteMeasure.Parameters(mean=..., std=...)`` for a
        Hermite-derived one).
    quad_rule : QuadratureRule
        The space's natural quadrature rule, used unconditionally by
        ``mass_matrix``/``stiffness_matrix`` (their integrands' required
        accuracy is fully determined by ``basis``, so there's no reason to
        let it drift from this default) and as the default for ``project``
        (which accepts an explicit override, since the right accuracy
        for a given target function isn't knowable from the space alone).
    """

    basis: Basis
    domain: Any
    quad_rule: QuadratureRule

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

    def _domain_kwargs(self) -> dict:
        return dataclasses.asdict(self.domain)

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
        """
        phi = self._basis_eval(x, deriv=deriv)  # (npts, n_basis)
        return phi @ coefficients

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

        Parameters
        ----------
        f : callable
            Target function, called once as ``f(x)`` on the full node
            array from the quadrature rule.
        quad_rule : QuadratureRule, optional
            Quadrature rule to use instead of ``self.quad_rule``.

        Returns
        -------
        Function
            The projected function, in this space.
        """
        from ._function import Function  # avoid a circular import

        x, w = self._quad_points_weights(quad_rule)
        phi = self._basis_eval(x)  # (npts, n_basis)
        M = phi.T @ (w[:, None] * phi)
        rhs = phi.T @ (w * f(x))
        coefficients = np.linalg.solve(M, rhs)
        return Function(coefficients, self)
