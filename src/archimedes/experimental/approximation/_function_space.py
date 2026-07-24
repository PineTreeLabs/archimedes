"""Finite-dimensional linear span of a Basis, tied to a target domain."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

import numpy as np

from archimedes.quadrature import QuadratureRule

from ._basis import Basis

if TYPE_CHECKING:
    from ._function import Function

__all__ = ["FunctionSpace"]


@dataclasses.dataclass(frozen=True)
class FunctionSpace:
    """The linear span of a :class:`Basis` on a fixed target domain.

    Pairs a ``Basis`` (family + size) with a target domain, and provides
    the operations that act on the *space* rather than on any one element
    of it: evaluation, the mass and stiffness matrices, and (Galerkin)
    projection. All of the latter are approximated via a caller-supplied
    quadrature rule.

    See :class:`Function` for a specific element of the space (a
    coefficient vector).

    Parameters
    ----------
    basis : Basis
        Basis family and size.
    domain : tuple of float
        Target domain ``(a, b)``, forwarded to ``basis`` as ``a``/``b``
        keyword arguments for the affine domain mapping. See the specific
        ``Basis`` subclass for how (or whether) this is used.
    """

    basis: Basis
    domain: tuple[float, float]

    @property
    def n_basis(self) -> int:
        """Number of basis functions; forwarded from ``basis``."""
        return self.basis.n_basis

    def _basis_eval(self, x, deriv: int = 0):
        a, b = self.domain
        return self.basis.evaluate(x, deriv=deriv, a=a, b=b)

    def evaluate(self, coefficients: np.ndarray, x, deriv: int = 0):
        """Evaluate :math:`\\sum_i c_i \\, \\phi_i(x)` (or its ``deriv``-th
        derivative) at ``x``, for coefficients ``c = coefficients``.
        """
        phi = self._basis_eval(x, deriv=deriv)  # (npts, n_basis)
        return phi @ coefficients

    def mass_matrix(self, quad_rule: QuadratureRule) -> np.ndarray:
        """Mass matrix :math:`M_{ij} = \\int \\phi_i \\, \\phi_j \\, w \\, dx`,
        approximated via ``quad_rule``.

        ``quad_rule`` must be accurate enough to integrate products of
        basis functions exactly (or nearly so) -- e.g. at least
        ``n_basis`` points for a polynomial basis of degree ``n_basis - 1``,
        since the integrand is degree ``2 * (n_basis - 1)``.
        """
        a, b = self.domain
        phi = self._basis_eval(quad_rule.scaled_points(a, b))  # (npts, n_basis)
        w = quad_rule.scaled_weights(a, b)  # (npts,)
        return phi.T @ (w[:, None] * phi)

    def stiffness_matrix(self, quad_rule: QuadratureRule) -> np.ndarray:
        """Stiffness matrix :math:`K_{ij} = \\int \\phi_i' \\, \\phi_j' \\,
        w \\, dx`, approximated via ``quad_rule``. See ``mass_matrix`` for
        the accuracy requirement on ``quad_rule``.
        """
        a, b = self.domain
        dphi = self._basis_eval(quad_rule.scaled_points(a, b), deriv=1)
        w = quad_rule.scaled_weights(a, b)
        return dphi.T @ (w[:, None] * dphi)

    def project(self, f: Callable, quad_rule: QuadratureRule) -> Function:
        """Galerkin projection of ``f`` onto this space, via ``quad_rule``.

        Solves ``M @ c = b`` for the coefficients ``c``, where ``M`` is the
        mass matrix and ``b_i = int f(x) phi_i(x) w(x) dx``, both
        approximated via ``quad_rule``. ``quad_rule`` must be accurate
        enough for the product of ``f`` and the basis, which is generally a
        higher-order requirement than exactness for the basis alone -- the
        caller is responsible for choosing an adequate order.

        Parameters
        ----------
        f : callable
            Target function, called once as ``f(x)`` on the full node
            array from ``quad_rule``.
        quad_rule : QuadratureRule
            Quadrature rule used to approximate the mass matrix and the
            right-hand side.

        Returns
        -------
        Function
            The projected function, in this space.
        """
        from ._function import Function  # avoid a circular import

        a, b = self.domain
        x = quad_rule.scaled_points(a, b)
        w = quad_rule.scaled_weights(a, b)
        phi = self._basis_eval(x)  # (npts, n_basis)
        M = phi.T @ (w[:, None] * phi)
        rhs = phi.T @ (w * f(x))
        coefficients = np.linalg.solve(M, rhs)
        return Function(coefficients, self)
