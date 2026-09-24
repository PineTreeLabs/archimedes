from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from ._base import RIGHT, Basis, _check_side

if TYPE_CHECKING:
    from archimedes.quadrature import Quadrature

__all__ = ["ConcatBasis"]


@dataclasses.dataclass(frozen=True)
class ConcatBasis(Basis):
    r"""A basis made of combining other bases.

    The functions in this basis are the set of functions from one or more
    bases:

    .. math::
        \Phi(x) = \big[\, \Phi_1(x) ~ \Phi_2(x) ~ \cdots \,\big]

    This is a direct sum of the spaces, not a linear combination.

    Parameters
    ----------
    pieces : tuple of Basis
        The bases to concatenate, in order. Must all report the same
        :attr:`Parameters` type (i.e. live on the same kind of domain:
        interval, half-line, or real-line), but may otherwise be unrelated
        families.
    quad_rule : Quadrature
        This basis's default quadrature rule. There is no "natural rule"
        for an arbitrarily concatenated basis, so this is required.

    Examples
    --------
    Augment a quadratic polynomial basis with four sine modes:

    >>> import numpy as np
    >>> from archimedes.approximation import ConcatBasis, FourierBasis, MonomialBasis
    >>> from archimedes.quadrature import gauss_legendre
    >>> poly = MonomialBasis(3)
    >>> sines = FourierBasis(4, kind="sine")
    >>> basis = ConcatBasis((poly, sines), quad_rule=gauss_legendre(16))
    >>> basis.n_basis
    7
    >>> basis.evaluate(np.array([0.0, 0.5])).shape
    (2, 7)

    A spectral-element "vertex + bubble" basis pairs two linear Lagrange
    vertex functions (which carry the boundary values) with interior
    Legendre modes that vanish at both endpoints:

    >>> from archimedes.approximation import (
    ...     ConstrainedBasis,
    ...     LagrangeBasis,
    ...     OrthogonalPolynomialBasis,
    ... )
    >>> from archimedes.measure import LegendreMeasure
    >>> vertex = LagrangeBasis(reference_nodes=np.array([-1.0, 1.0]))
    >>> legendre = OrthogonalPolynomialBasis(LegendreMeasure(), 6)
    >>> bubble = ConstrainedBasis.dirichlet(legendre)
    >>> sem = ConcatBasis((vertex, bubble), quad_rule=gauss_legendre(8))
    >>> sem.n_basis
    6
    >>> sem.boundary_dofs()
    (0, 1)
    """

    pieces: tuple[Basis, ...]
    quad_rule: Quadrature

    def __post_init__(self):
        if len(self.pieces) == 0:
            raise ValueError("pieces must be non-empty")
        parameters = self.pieces[0].Parameters
        for piece in self.pieces[1:]:
            if piece.Parameters is not parameters:
                raise TypeError(
                    f"all pieces must share one Parameters type to be "
                    f"evaluable at the same target domain; got "
                    f"{parameters.__qualname__} and "
                    f"{piece.Parameters.__qualname__}"
                )

    @property
    def n_basis(self) -> int:
        """Total number of functions across all pieces."""
        return sum(piece.n_basis for piece in self.pieces)

    @property
    def Parameters(self) -> type:  # noqa: N802
        return self.pieces[0].Parameters

    @property
    def _measures(self):
        return (None,)

    @property
    def _required_breakpoints(self):
        per_piece = [piece._required_breakpoints for piece in self.pieces]
        present = [bp for bp in per_piece if bp is not None]
        if not present:
            return None
        return np.unique(np.concatenate(present))

    @property
    def _dof_order(self) -> np.ndarray:
        return np.concatenate([piece._dof_order for piece in self.pieces])

    def _default_quadrature(self):
        return self.quad_rule

    def boundary_dofs(self, order: int = 0) -> tuple[int | None, int | None]:
        offset = 0
        left = right = None
        for piece in self.pieces:
            piece_left, piece_right = piece.boundary_dofs(order)
            if piece_left is not None:
                if left is not None:
                    raise ValueError(
                        f"more than one piece claims the left boundary DOF "
                        f"at order={order}"
                    )
                left = offset + piece_left
            if piece_right is not None:
                if right is not None:
                    raise ValueError(
                        f"more than one piece claims the right boundary DOF "
                        f"at order={order}"
                    )
                right = offset + piece_right
            offset += piece.n_basis
        return left, right

    def evaluate(self, x, deriv: int = 0, side: str = RIGHT, **domain_kwargs):
        _check_side(side)
        return np.concatenate(
            [
                piece.evaluate(x, deriv=deriv, side=side, **domain_kwargs)
                for piece in self.pieces
            ],
            axis=-1,
        )
