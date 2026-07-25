"""Piecewise basis: a local Basis tiled across elements, with a continuity rule."""

from __future__ import annotations

import dataclasses

import numpy as np

from archimedes.measure import UnitInterval

from ._basis import Basis

__all__ = ["PiecewiseBasis"]

DISCONTINUOUS = -1
"""``continuity`` value for independent per-element DOFs (a "DG" basis)."""

C0 = 0
"""``continuity`` value for value-continuity at element boundaries ("CG")."""


@dataclasses.dataclass(frozen=True)
class PiecewiseBasis(Basis):
    """A local :class:`Basis` tiled across elements of the reference interval.

    ``breakpoints`` partition the reference domain :math:`[-1, 1]`,
    ``element_basis`` is affinely mapped onto each subinterval, and the
    result is itself a ``Basis`` on :math:`[-1, 1]` that can be mapped
    onto any target :math:`[a, b]` by :class:`~archimedes.measure.UnitInterval`
    parameters. Breakpoints are *structural* data (they determine ``n_basis``),
    so they live here rather than in ``Parameters``.

    **Continuity.** Tiling alone produces a "broken" (discontinuous) basis
    with ``n_elements * element_basis.n_basis`` degrees of freedom.
    Continuity is then imposed as a linear *assembly* map applied to that
    broken basis,

    .. math::
        \\Phi_{\\mathrm{global}}(x) = \\Phi_{\\mathrm{broken}}(x) \\, T,

    where ``T`` (``assembly_matrix``) has shape ``(n_broken, n_basis)``:

    - ``continuity=-1`` -- ``T`` is the identity; element DOFs are
      independent and the basis jumps at interior breakpoints.
    - ``continuity=0`` -- ``T`` merges each element's right-endpoint DOF
      with the next element's left-endpoint DOF, giving ``n_broken -
      (n_elements - 1)`` continuous DOFs. The merged ("vertex") DOF is a
      single basis function supported on *both* adjacent elements -- the
      standard "hat" for finite elements.

    Because continuity is just a right-multiplication, everything built on
    ``Basis`` (mass/stiffness matrices, projection, inner products) works
    through it unchanged.

    Higher continuity (:math:`C^1` and up) is not yet supported: it would
    need *derivative* degrees of freedom to identify, requiring Hermite
    elements or similar (not yet implemented).

    Note that under ``continuity=0`` the *derivative* is still
    discontinuous at breakpoints, so ``evaluate(..., deriv=1)`` returns the
    one-sided value belonging to whichever element owns the point (see
    below).

    **Element ownership** is half-open, ``[lo, hi)``, with the last element
    closed on the right. This matters: with closed intervals both elements
    adjacent to an interior breakpoint would claim it and the assembled
    basis would double-count there (partition of unity would give 2 at the
    breakpoint).

    Parameters
    ----------
    element_basis : Basis
        Local basis, defined on the reference interval ``[-1, 1]``, mapped
        onto each element. For ``continuity=0`` it must expose endpoint
        DOFs via ``boundary_dofs`` (e.g. a
        :class:`LagrangeBasis` whose nodes include both endpoints).
    breakpoints : array_like
        Element boundaries on the reference domain, shape ``(k + 1,)`` for
        ``k`` elements. Must be strictly increasing and span ``[-1, 1]``
        exactly.
    continuity : int, optional
        ``-1`` for a discontinuous basis, ``0`` for value continuity.
        Default ``0``.
    """

    element_basis: Basis
    breakpoints: np.ndarray
    continuity: int = C0

    def __post_init__(self):
        bp = np.asarray(self.breakpoints, dtype=float)
        if bp.ndim != 1 or len(bp) < 2:
            raise ValueError(
                f"breakpoints must be 1-D with at least 2 entries, got shape {bp.shape}"
            )
        if np.any(np.diff(bp) <= 0):
            raise ValueError("breakpoints must be strictly increasing")
        if bp[0] != -1.0 or bp[-1] != 1.0:
            raise ValueError(
                f"breakpoints must span the reference domain (-1.0, 1.0), got "
                f"({bp[0]}, {bp[-1]})"
            )
        if self.continuity not in (DISCONTINUOUS, C0):
            raise ValueError(
                f"continuity must be -1 (discontinuous) or 0 (C0), got "
                f"{self.continuity}"
            )
        if self.continuity == C0:
            left, right = self.element_basis.boundary_dofs()
            if left is None or right is None:
                raise ValueError(
                    f"continuity=0 requires an element basis with degrees of "
                    f"freedom at both endpoints, but "
                    f"{type(self.element_basis).__name__}.boundary_dofs() "
                    f"returned {(left, right)}"
                )
        object.__setattr__(self, "breakpoints", bp)
        object.__setattr__(self, "_assembly", self._build_assembly())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PiecewiseBasis):
            return NotImplemented
        return (
            self.element_basis == other.element_basis
            and self.continuity == other.continuity
            and np.array_equal(self.breakpoints, other.breakpoints)
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self.element_basis,
                self.continuity,
                self.breakpoints.tobytes(),
            )
        )

    @property
    def n_elements(self) -> int:
        """Number of elements."""
        return len(self.breakpoints) - 1

    @property
    def n_broken(self) -> int:
        """Degrees of freedom before continuity is imposed."""
        return self.n_elements * self.element_basis.n_basis

    @property
    def n_basis(self) -> int:
        """Degrees of freedom after continuity is imposed."""
        return self._assembly.shape[1]

    @property
    def assembly_matrix(self) -> np.ndarray:
        """The ``(n_broken, n_basis)`` map from broken to global DOFs."""
        return self._assembly

    @property
    def Parameters(self) -> type:  # noqa: N802
        """The outer affine map of the whole tiled pattern onto ``[a, b]``;
        the breakpoints themselves are structural, not parameters."""
        return UnitInterval.Parameters

    def _build_assembly(self) -> np.ndarray:
        n_loc = self.element_basis.n_basis
        if self.continuity == DISCONTINUOUS:
            return np.eye(self.n_broken)

        left, right = self.element_basis.boundary_dofs()
        n_global = self.n_broken - (self.n_elements - 1)
        assembly = np.zeros((self.n_broken, n_global))

        # Global index of each element's first *unshared* DOF.
        offset = 0
        prev_right_global = None
        for e in range(self.n_elements):
            local_to_global = {}
            for i in range(n_loc):
                if e > 0 and i == left:
                    local_to_global[i] = prev_right_global
                else:
                    local_to_global[i] = offset
                    offset += 1
            for i, g in local_to_global.items():
                assembly[e * n_loc + i, g] = 1.0
            prev_right_global = local_to_global[right]

        return assembly

    def evaluate(self, x, deriv: int = 0, a=None, b=None):
        scale, shift = UnitInterval().affine_params(a, b)
        knots = scale * self.breakpoints + shift

        blocks = []
        for e in range(self.n_elements):
            lo, hi = knots[e], knots[e + 1]
            # Half-open ownership; the final element also owns its right end.
            if e < self.n_elements - 1:
                inside = (x >= lo) & (x < hi)
            else:
                inside = (x >= lo) & (x <= hi)
            block = self.element_basis.evaluate(x, deriv=deriv, a=lo, b=hi)
            blocks.append(np.where(inside[:, None], block, np.zeros_like(block)))

        broken = np.concatenate(blocks, axis=-1)  # (npts, n_broken)
        return broken @ self._assembly
