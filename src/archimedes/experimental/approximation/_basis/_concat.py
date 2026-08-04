"""Direct sum of heterogeneous bases -- functions stacked side by side."""

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
    r"""A basis whose functions are the concatenation of several pieces'.

    Where :class:`ConstrainedBasis` recombines *one* basis's functions by
    a matrix (a change of basis), this stacks functions from *several*,
    possibly unrelated, bases side by side:

    .. math::
        \Phi(x) = \big[\, \Phi_1(x) \;\; \Phi_2(x) \;\; \cdots \,\big]

    This is a direct sum, not a linear combination. The two compose: e.g. a
    spectral-element "vertex + bubble" basis is ``ConcatBasis((vertex,
    ConstrainedBasis.dirichlet(base)))``, pairing two plain Lagrange
    vertex functions (carrying the boundary values) with the homogeneous
    interior modes from ``base``.

    Parameters
    ----------
    pieces : tuple of Basis
        The bases to concatenate, in order. Must all report the same
        :attr:`Parameters` type (they need to be evaluable at the same
        target-domain kwargs), but may otherwise be unrelated families.
    quad_rule : Quadrature
        This basis's default quadrature rule, **required** rather than
        derived. Unlike a single family, a concatenation's pieces have no
        one shared notion of "enough points" or "the natural weight": pass
        something adequate (e.g. the finest of the pieces' own
        ``default_quadrature()``, or a plain Gauss-Legendre rule with
        enough points for the combined degree).
    """

    pieces: tuple[Basis, ...]
    quad_rule: "Quadrature"

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
        return sum(piece.n_basis for piece in self.pieces)

    @property
    def Parameters(self) -> type:  # noqa: N802
        return self.pieces[0].Parameters

    @property
    def measures(self):
        """Always ``(None,)``; see the class docstring."""
        return (None,)

    @property
    def required_breakpoints(self):
        """Union of every piece's own, or ``None`` if none has any."""
        per_piece = [piece.required_breakpoints for piece in self.pieces]
        present = [bp for bp in per_piece if bp is not None]
        if not present:
            return None
        return np.unique(np.concatenate(present))

    @property
    def _dof_order(self) -> np.ndarray:
        return np.concatenate([piece._dof_order for piece in self.pieces])

    def default_quadrature(self):
        """The ``quad_rule`` given at construction; see the class
        docstring for why this is required rather than derived."""
        return self.quad_rule

    def boundary_dofs(self, order: int = 0) -> tuple[int | None, int | None]:
        """Union of each piece's own ``boundary_dofs``, offset into the
        concatenated index space.

        Raises
        ------
        ValueError
            If more than one piece claims the same side at this ``order``
        """
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
