from __future__ import annotations

import dataclasses

import numpy as np

from archimedes._core._array_impl import SymbolicArray
from archimedes.measure import UnitInterval

from ._base import RIGHT, Basis, _check_side
from ._utils import _gather, _locate

__all__ = ["PiecewiseBasis"]


DISCONTINUOUS = -1
"""``continuity`` value for independent per-element DOFs (a "DG" basis)."""

C0 = 0
"""``continuity`` value for value-continuity at element boundaries ("CG")."""

C1 = 1
"""``continuity`` value for value and derivative continuity at element boundaries."""


@dataclasses.dataclass(frozen=True)
class PiecewiseBasis(Basis):
    """A :class:`Basis` constructed by tiling a local basis across elements.

    This constructs a finite (or spectral) element basis by partitioning the
    domain into subintervals, each of which has a basis of local support.

    The domain partitioning is determined by ``breakpoints``, defined on
    the reference domain :math:`[-1, 1]`.

    **Continuity.** Tiling alone produces a discontinuous (:math:`C^{-1}`) basis.
    A :math:`C^0` basis is continuous at element boundaries, i.e. neighboring
    elements share endpoint DOFs. A :math:`C^1` basis additionally has
    continuous first derivatives at element boundaries. These choices correspond
    to ``continuity={-1, 0, 1}``.

    **Element ownership at a breakpoint.** When the basis functions are two-valued
    at interior breakpoints (e.g. the value of a :math:`C^{-1}` function or the
    derivative of a :math:`C^0` function), the ``side`` argument
    (``"left"`` or ``"right"``) of evaluation determines which value is returned.
    This is useful for instance with discontinuous Galerkin (DG) numerical flux
    construction, which uses :math:`u^-` and :math:`u^+` at each interface.

    This class is typically not used directly; instead, it is constructed as part
    of the :meth:`FunctionSpace.piecewise` constructor. However, it can be used
    for customized piecewise bases not supported by that high-level constructor.

    See Also
    --------
    FunctionSpace.piecewise : Convenience constructor for a :class:`FunctionSpace`
        using common piecewise bases (e.g. "lagrange", "legendre", "hermite").
    """

    element_basis: Basis | tuple[Basis, ...]
    """Local basis, defined on the reference interval ``[-1, 1]``."""

    breakpoints: np.ndarray
    """Element boundaries on the reference domain

    Shape ``(k + 1,)`` for ``k`` elements. Must be strictly increasing
    and span ``[-1, 1]`` exactly.
    """

    continuity: int = C0
    """Continuity at element boundaries.
    
    ``-1`` for discontinuous, ``0`` for value continuity, ``1`` for
    first-derivative continuity.
    """

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
        n_elements = len(bp) - 1

        # A single shared basis (the common case) tiles to one entry per
        # element; a sequence must already have exactly that many.
        if isinstance(self.element_basis, Basis):
            element_bases = (self.element_basis,) * n_elements
        else:
            element_bases = tuple(self.element_basis)
            if len(element_bases) != n_elements:
                raise ValueError(
                    f"breakpoints describe {n_elements} elements, so "
                    f"element_basis must supply exactly {n_elements} bases, "
                    f"got {len(element_bases)}"
                )

        # Tiling is along a single reference interval, so a multivariate
        # element basis has no meaning here; a structured multi-dimensional
        # mesh is a `TensorBasis` *of* `PiecewiseBasis` factors, not the
        # other way around.
        for eb in element_bases:
            if eb.ndim != 1:
                raise ValueError(
                    f"element_basis must be univariate, got "
                    f"{eb.ndim}-dimensional {type(eb).__name__}; for a "
                    f"structured mesh, tensor together one PiecewiseBasis "
                    f"per dimension"
                )

        # One coherent per-dimension weight for the whole assembled basis --
        # otherwise `measures` (and the quadrature-rule checks built on it)
        # would have to pick one element's weight arbitrarily.
        measures = element_bases[0]._measures
        if any(eb._measures != measures for eb in element_bases):
            raise ValueError(
                "every element's basis must share the same measures so the "
                "assembled piecewise basis has one coherent per-dimension "
                "weight"
            )

        if self.continuity < DISCONTINUOUS:
            raise ValueError(
                f"continuity must be >= -1 (discontinuous), got {self.continuity}"
            )
        if self.continuity >= C0:
            for eb in element_bases:
                for order in range(self.continuity + 1):
                    left, right = eb.boundary_dofs(order)
                    if left is None or right is None:
                        raise ValueError(
                            f"continuity={self.continuity} requires every "
                            f"element basis to have degrees of freedom at "
                            f"both endpoints for every order 0.."
                            f"{self.continuity}, but "
                            f"{type(eb).__name__}.boundary_dofs({order}) "
                            f"returned {(left, right)}"
                        )

        object.__setattr__(self, "element_basis", element_bases)
        object.__setattr__(self, "breakpoints", bp)
        # Whether every element shares the same basis.
        # Lets `_evaluate_expansion` keep its fused O(1)-in-
        # `n_elements` fast path in the common case; see that method.
        object.__setattr__(
            self, "_uniform", all(eb == element_bases[0] for eb in element_bases)
        )
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
    def _n_broken(self) -> int:
        """Degrees of freedom before continuity is imposed."""
        return sum(eb.n_basis for eb in self.element_basis)

    @property
    def n_basis(self) -> int:
        return self._assembly.shape[1]

    @property
    def Parameters(self) -> type:  # noqa: N802
        return UnitInterval.Parameters

    @property
    def _measures(self):
        return self.element_basis[0]._measures

    @property
    def _required_breakpoints(self) -> np.ndarray:
        return self.breakpoints

    def boundary_dofs(self, order: int = 0) -> tuple[int | None, int | None]:
        if self.continuity == DISCONTINUOUS:
            return (None, None)
        left, _ = self.element_basis[0].boundary_dofs(order)
        _, right = self.element_basis[-1].boundary_dofs(order)
        if left is None or right is None:
            return (None, None)
        last_offset = self._n_broken - self.element_basis[-1].n_basis
        return (
            int(np.argmax(self._assembly[left])),
            int(np.argmax(self._assembly[last_offset + right])),
        )

    def _default_quadrature(self):
        from archimedes.quadrature import composite_quad

        return composite_quad(
            [eb._default_quadrature() for eb in self.element_basis], self.breakpoints
        )

    def _product_basis(self, other):
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
            tuple(
                e1._product_basis(e2)
                for e1, e2 in zip(self.element_basis, other.element_basis)
            ),
            self.breakpoints,
            continuity=min(self.continuity, other.continuity),
        )

    def _derivative_basis(self, deriv=1):
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")
        if deriv == 0:
            return self
        return PiecewiseBasis(
            tuple(eb._derivative_basis(deriv) for eb in self.element_basis),
            self.breakpoints,
            continuity=max(self.continuity - deriv, DISCONTINUOUS),
        )

    def _integral_basis(self, order=1):
        raise NotImplementedError(
            f"{type(self).__name__} does not define an integral basis."
        )

    def _build_assembly(self) -> np.ndarray:
        r"""Build the assembly map ``T``, shape ``(n_broken, n_basis)``, such
        that :math:`\Phi_{\mathrm{global}}(x) = \Phi_{\mathrm{broken}}(x) T`.
        """
        element_bases = self.element_basis
        if self.continuity == DISCONTINUOUS:
            return np.eye(self._n_broken)

        n_orders = self.continuity + 1
        n_global = self._n_broken - (self.n_elements - 1) * n_orders
        assembly = np.zeros((self._n_broken, n_global))

        # Global index of each element's first *unshared* DOF.
        offset = 0
        row_offset = 0
        prev_global_by_order: list[int | None] = [None] * n_orders
        for e, eb in enumerate(element_bases):
            n_loc = eb.n_basis
            left_by_order = [eb.boundary_dofs(o)[0] for o in range(n_orders)]
            right_by_order = [eb.boundary_dofs(o)[1] for o in range(n_orders)]
            # Local index -> which order it's the "left" DOF for, so a plain
            # membership test below can dispatch to the right previous-global
            # tracker regardless of how many orders are being merged.
            left_local_to_order = {left: o for o, left in enumerate(left_by_order)}

            local_to_global = {}
            for i in range(n_loc):
                if e > 0 and i in left_local_to_order:
                    order = left_local_to_order[i]
                    local_to_global[i] = prev_global_by_order[order]
                else:
                    local_to_global[i] = offset
                    offset += 1
            for i, g in local_to_global.items():
                assembly[row_offset + i, g] = 1.0
            prev_global_by_order = [
                local_to_global[right_by_order[o]] for o in range(n_orders)
            ]
            row_offset += n_loc

        return assembly

    def _blocks(self, x, masks, knots, deriv):
        """Per-element evaluations, masked and concatenated to the broken
        basis, then assembled."""
        blocks = []
        for e in range(self.n_elements):
            block = self.element_basis[e].evaluate(
                x, deriv=deriv, a=knots[e], b=knots[e + 1]
            )
            blocks.append(np.where(masks[e][:, None], block, np.zeros_like(block)))
        broken = np.concatenate(blocks, axis=-1)  # (npts, n_broken)
        return broken @ self._assembly

    def evaluate(self, x, deriv: int = 0, a=None, b=None, side: str = RIGHT):
        r"""Evaluate at arbitrary points, resolving breakpoints by ``side``.

        Parameters
        ----------
        x, deriv : array_like, int, optional
            As for :meth:`Basis.evaluate`.
        a, b : float, optional
            Target-domain endpoints; default the reference domain ``[-1, 1]``.
        side : {"right", "left"}, optional
            Which one-sided limit to take at a point lying exactly on an
            interior breakpoint, where this basis is two-valued. ``"right"``
            (default) makes ownership half-open ``[lo, hi)``; ``"left"``
            makes it ``(lo, hi]``. Irrelevant away from breakpoints and for
            ``deriv=0`` on a :math:`C^0` basis. See the class docstring.
        """
        _check_side(side)
        scale, shift = UnitInterval().affine_params(a, b)
        knots = scale * self.breakpoints + shift

        masks = []
        for e in range(self.n_elements):
            lo, hi = knots[e], knots[e + 1]
            # Exactly one element claims each breakpoint, so the assembled
            # basis stays a partition of unity; `side` chooses which. The
            # element at the far end also owns the domain's outer endpoint.
            if side == RIGHT:
                closed_end = e == self.n_elements - 1
                masks.append(
                    (x >= lo) & (x <= hi) if closed_end else (x >= lo) & (x < hi)
                )
            else:
                closed_end = e == 0
                masks.append(
                    (x >= lo) & (x <= hi) if closed_end else (x > lo) & (x <= hi)
                )

        return self._blocks(x, masks, knots, deriv)

    def _evaluate_at_nodes(self, rule, deriv=0, a=None, b=None):
        """Evaluate at a quadrature rule's nodes, using the rule's recorded
        element ownership instead of the coordinate convention."""
        x = rule.nodes
        if rule.elements is None:
            # No element structure to draw on (a plain, non-composite rule).
            # `FunctionSpace` rejects such a rule for this basis, so this is
            # only reachable via a direct call.
            return self.evaluate(x, deriv=deriv, a=a, b=b)

        scale, shift = UnitInterval().affine_params(a, b)
        knots = scale * self.breakpoints + shift

        # Each rule element lies inside exactly one basis element, since the
        # rule's breakpoints are a superset of this basis's.
        rule_bp = np.asarray(rule.breakpoints)
        owner = np.clip(
            np.searchsorted(self.breakpoints, rule_bp[:-1], side="right") - 1,
            0,
            self.n_elements - 1,
        )[rule.elements]

        masks = [owner == e for e in range(self.n_elements)]
        return self._blocks(x, masks, knots, deriv)

    def _evaluate_expansion(
        self, coefficients, x, deriv: int = 0, a=None, b=None, side: str = RIGHT
    ):
        r"""Evaluate the coefficient expansion :math:`\sum_i c_i \,
        \phi_i(x)` (or its ``deriv``-th derivative), for coefficients ``c``
        = ``coefficients``.

        Equivalent to ``evaluate(x, deriv) @ coefficients``, but cheaper
        when every element shares one basis (the common case): the cost is
        then independent of the number of elements, unlike ``evaluate()``,
        which builds the full ``(npts, n_basis)`` design matrix and so
        costs one evaluation per element. If ``element_basis`` was given
        per-element (a different order or family per element) that
        advantage doesn't apply, and this costs the same as the dense
        ``evaluate(x) @ coefficients`` path.
        """
        if not self._uniform:
            return super()._evaluate_expansion(
                coefficients, x, deriv=deriv, side=side, a=a, b=b
            )

        _check_side(side)
        symbolic = isinstance(x, SymbolicArray) or isinstance(
            coefficients, SymbolicArray
        )
        scale, shift = UnitInterval().affine_params(a, b)
        knots = scale * self.breakpoints + shift
        npts = np.shape(x)[0]

        element = _locate(knots, x, symbolic, side)
        lo = _gather(knots, element, symbolic, npts)
        hi = _gather(knots, element + 1, symbolic, npts)

        # Reference coordinate within the owning element, t in [-1, 1]
        width = hi - lo
        t = 2.0 * (x - lo) / width - 1.0
        shared = self.element_basis[0]
        phi = shared.evaluate(t, deriv=deriv)  # (npts, n_loc)

        # Expanding to per-element coefficients once makes each element's
        # block contiguous, so the gather is a fixed offset from `element`.
        # This is independent of `npts`, so it does not affect the per-point
        # cost that motivates this path.
        broken = self._assembly @ coefficients  # (n_broken,) or (n_broken, m)
        vector_valued = np.ndim(coefficients) > 1

        n_loc = shared.n_basis
        dof_order = shared._dof_order  # (n_loc,); all zeros for a homogeneous family
        # Extra *uniform* power of scale a family like `OrthogonalPolynomialBasis`
        # needs on top of the per-column chain rule, since its own normalization
        # is domain-size-dependent; zero for a family (Lagrange, Hermite) whose
        # reference-domain shape functions are already the physical ones up to
        # the chain rule alone. See `Basis._reference_scale_exponent`.
        scale_exponent = shared._reference_scale_exponent
        total = None
        for k in range(n_loc):
            c_k = _gather(broken, element * n_loc + k, symbolic, npts)
            # Chain rule for the map into the reference coordinate: dt/dx =
            # 2/width per derivative order requested, offset by this column's
            # own intrinsic DOF order (see `Basis._dof_order`) -- e.g. a
            # Hermite slope-type column (order 1) needs one fewer power of
            # 2/width than a value-type column (order 0) at the same `deriv`,
            # since its coefficient is already a physical derivative. A
            # homogeneous family has `_dof_order` all zeros, so this reduces
            # to the single scalar factor every column used to share.
            col_scale = (2.0 / width) ** (deriv - dof_order[k] + scale_exponent)
            phi_k = phi[:, k] * col_scale
            term = phi_k[:, None] * c_k if vector_valued else phi_k * c_k
            total = term if total is None else total + term

        # `evaluate` masks every element, so a point outside the domain
        # contributes nothing; `low`/`searchsorted` instead clamp to the end
        # element, which would extrapolate. Mask to keep the two paths equal.
        inside = (x >= knots[0]) & (x <= knots[-1])
        if vector_valued:
            return np.where(inside[:, None], total, 0.0)
        return np.where(inside, total, 0.0)
