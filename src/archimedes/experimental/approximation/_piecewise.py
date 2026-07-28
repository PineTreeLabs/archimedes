"""Piecewise basis: a local Basis tiled across elements, with a continuity rule."""

from __future__ import annotations

import dataclasses

import casadi as cs
import numpy as np

from archimedes._core._array_impl import SymbolicArray, _unwrap_sym_array
from archimedes.measure import UnitInterval

from ._basis import Basis

__all__ = ["PiecewiseBasis"]


def _as_mx(value):
    """Unwrap to a CasADi MX. ``cs.low`` and symbolic-index gathers are
    MX-only, and a constant must be promoted from DM before it can be
    indexed by a symbolic expression."""
    if isinstance(value, SymbolicArray):
        return _unwrap_sym_array(value)
    return cs.MX(cs.DM(np.asarray(value, dtype=float)))


def _locate(knots, x, symbolic: bool):
    """Index of the element owning each point of ``x``.

    Half-open ``[lo, hi)``, with both ends clamped into range, matching
    :meth:`PiecewiseBasis.evaluate`'s masking convention. ``cs.low`` is
    CasADi's ``std::lower_bound`` and already has exactly these semantics;
    ``searchsorted(..., "right") - 1`` is its NumPy equivalent.
    """
    n_elements = len(knots) - 1
    if symbolic:
        return SymbolicArray(
            cs.low(_as_mx(knots), _as_mx(x)), shape=np.shape(x), dtype=int
        )
    index = np.searchsorted(np.asarray(knots), np.asarray(x), side="right") - 1
    return np.clip(index, 0, n_elements - 1)


def _gather(values, index, symbolic: bool, npts: int):
    """``values[index]`` (rows, if ``values`` is 2-D) for a possibly
    symbolic integer ``index``."""
    if not symbolic:
        return values[index]
    shape = (npts,) if np.ndim(values) == 1 else (npts, np.shape(values)[1])
    gathered = _as_mx(values)[_as_mx(index), :]
    return SymbolicArray(gathered, shape=shape, dtype=float)


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
        # Tiling is along a single reference interval, so a multivariate
        # element basis has no meaning here; a structured multi-dimensional
        # mesh is a `TensorBasis` *of* `PiecewiseBasis` factors, not the
        # other way around.
        if self.element_basis.ndim != 1:
            raise ValueError(
                f"element_basis must be univariate, got "
                f"{self.element_basis.ndim}-dimensional "
                f"{type(self.element_basis).__name__}; for a structured mesh, "
                f"tensor together one PiecewiseBasis per dimension"
            )
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

    @property
    def measures(self):
        """The element basis's weight: tiling rescales the reference weight
        onto each element but does not change which family it is."""
        return self.element_basis.measures

    @property
    def required_breakpoints(self) -> np.ndarray:
        """This basis is only piecewise smooth: it kinks (or, for
        ``continuity=-1``, jumps) at every interior breakpoint."""
        return self.breakpoints

    def default_quadrature(self):
        """The element basis's own rule, tiled across the same breakpoints.

        Tiling the element rule is what makes the result exact: every
        subinterval then lies inside a single element, where the integrand
        is a polynomial of the degree the element rule was chosen for.
        """
        from archimedes.quadrature import composite

        return composite(self.element_basis.default_quadrature(), self.breakpoints)

    def _product_basis(self, other):
        """Same breakpoints, product element basis, weaker continuity.

        The breakpoints must match exactly: a product across two different
        partitions kinks at the union of both, which neither operand's
        partition can represent.

        Continuity is the *minimum* of the two. A product is only as smooth
        as its least smooth factor -- continuous times discontinuous is
        discontinuous -- so taking the maximum would claim a smoothness the
        result does not have.
        """
        if not isinstance(other, PiecewiseBasis):
            raise ValueError(
                f"cannot form a product basis between "
                f"{type(self).__name__} and {type(other).__name__}"
            )
        if not np.array_equal(self.breakpoints, other.breakpoints):
            raise ValueError(
                f"product requires identical breakpoints, got "
                f"{self.breakpoints} and {other.breakpoints}"
            )
        return PiecewiseBasis(
            self.element_basis._product_basis(other.element_basis),
            self.breakpoints,
            continuity=min(self.continuity, other.continuity),
        )

    def _derivative_basis(self, deriv=1):
        """Same breakpoints, derivative element basis, **discontinuous**.

        This is the one family where the derivative genuinely leaves the
        original space rather than landing in a subspace of it: a
        :math:`C^0` function has a derivative that jumps at every interior
        breakpoint, so the result is a ``continuity=-1`` basis regardless of
        what this one was. Within each element the derivative is still a
        polynomial of degree ``n_loc - 1 - deriv``, so the element basis
        shrinks in the usual way and the representation stays exact.

        .. note::
            The derivative is genuinely two-valued at an interior
            breakpoint, so a quadrature rule with a node sitting exactly on
            one (a *composite Lobatto* rule, say) samples whichever element
            owns it under the half-open convention. The default rules put
            their nodes strictly inside elements, so this only arises for an
            explicitly supplied rule.
        """
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")
        if deriv == 0:
            return self
        return PiecewiseBasis(
            self.element_basis._derivative_basis(deriv),
            self.breakpoints,
            continuity=DISCONTINUOUS,
        )

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

    def evaluate_expansion(self, coefficients, x, deriv: int = 0, a=None, b=None):
        """Locate each point's element and gather only that element's
        coefficients, instead of building the full ``(npts, n_basis)``
        matrix.

        :meth:`evaluate` must fill every column, so it evaluates the element
        basis once per element and masks, hence the cost grows with
        ``n_elements``. Here each point is instead mapped into the reference
        coordinate of *its own* element, so the element basis is evaluated
        exactly once regardless of how many elements there are, and only
        ``element_basis.n_basis`` coefficients are read per point.

        Mapping into the element's reference coordinate (rather than
        evaluating the element basis on ``[lo, hi]``) is what keeps this to
        a single call: ``lo``/``hi`` differ per point, and the element bases
        take scalar domain parameters.
        """
        symbolic = isinstance(x, SymbolicArray) or isinstance(
            coefficients, SymbolicArray
        )
        scale, shift = UnitInterval().affine_params(a, b)
        knots = scale * self.breakpoints + shift
        npts = np.shape(x)[0]

        element = _locate(knots, x, symbolic)
        lo = _gather(knots, element, symbolic, npts)
        hi = _gather(knots, element + 1, symbolic, npts)

        # Reference coordinate within the owning element, t in [-1, 1]
        width = hi - lo
        t = 2.0 * (x - lo) / width - 1.0
        phi = self.element_basis.evaluate(t, deriv=deriv)  # (npts, n_loc)

        # Expanding to per-element coefficients once makes each element's
        # block contiguous, so the gather is a fixed offset from `element`.
        # This is independent of `npts`, so it does not affect the per-point
        # cost that motivates this path.
        broken = self._assembly @ coefficients  # (n_broken,) or (n_broken, m)
        vector_valued = np.ndim(coefficients) > 1

        n_loc = self.element_basis.n_basis
        total = None
        for k in range(n_loc):
            c_k = _gather(broken, element * n_loc + k, symbolic, npts)
            phi_k = phi[:, k]
            term = phi_k[:, None] * c_k if vector_valued else phi_k * c_k
            total = term if total is None else total + term

        # Chain rule for the map into the reference coordinate, one factor
        # of dt/dx = 2/width per derivative order.
        jacobian = (2.0 / width) ** deriv

        # `evaluate` masks every element, so a point outside the domain
        # contributes nothing; `low`/`searchsorted` instead clamp to the end
        # element, which would extrapolate. Mask to keep the two paths equal.
        inside = (x >= knots[0]) & (x <= knots[-1])
        if vector_valued:
            return np.where(inside[:, None], jacobian[:, None] * total, 0.0)
        return np.where(inside, jacobian * total, 0.0)
