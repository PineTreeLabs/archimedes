"""Piecewise basis: a local Basis tiled across elements, with a continuity rule."""

from __future__ import annotations

import dataclasses

import casadi as cs
import numpy as np

from archimedes._core._array_impl import SymbolicArray, _unwrap_sym_array
from archimedes.measure import UnitInterval

from ._base import RIGHT, Basis, _check_side

__all__ = ["PiecewiseBasis"]


def _as_mx(value):
    """Unwrap to a CasADi MX. ``cs.low`` and symbolic-index gathers are
    MX-only, and a constant must be promoted from DM before it can be
    indexed by a symbolic expression."""
    if isinstance(value, SymbolicArray):
        return _unwrap_sym_array(value)
    return cs.MX(cs.DM(np.asarray(value, dtype=float)))


def _locate(knots, x, symbolic: bool, side: str = RIGHT):
    """Index of the element owning each point of ``x``, by coordinate.

    With ``side="right"`` (the default) ownership is half-open ``[lo, hi)``,
    so a point on a breakpoint belongs to the element above it -- the limit
    from the right. ``side="left"`` gives ``(lo, hi]`` and the limit from
    the left. Both clamp out-of-range points into the end elements.

    ``cs.low`` is CasADi's ``std::lower_bound`` and has exactly the
    right-sided semantics; ``searchsorted(..., "right") - 1`` is its NumPy
    equivalent. The left-sided variant steps back one element at points that
    land exactly on a knot, which costs one comparison against a value the
    caller is gathering anyway.

    This is the coordinate-only path, used for user-supplied points. Where
    provenance exists -- quadrature nodes, which know the element they were
    generated for -- :meth:`PiecewiseBasis._evaluate_at_nodes` uses it
    instead and no convention is needed.
    """
    _check_side(side)
    n_elements = len(knots) - 1
    if symbolic:
        index = SymbolicArray(
            cs.low(_as_mx(knots), _as_mx(x)), shape=np.shape(x), dtype=int
        )
    else:
        index = np.clip(
            np.searchsorted(np.asarray(knots), np.asarray(x), side="right") - 1,
            0,
            n_elements - 1,
        )
    if side == RIGHT:
        return index
    # On a knot, back up one element; `np.where` keeps this branch-free so it
    # traces, and the max() guards the first element's left end.
    on_knot = _gather(knots, index, symbolic, np.shape(x)[0]) == x
    return np.maximum(index - on_knot, 0)


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

C1 = 1
"""``continuity`` value additionally matching first derivatives at element
boundaries -- what a cubic Hermite element needs for a 4th-order
(Euler-Bernoulli-type) weak form."""


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
    ``continuity=-1`` keeps it that way; a nonnegative ``continuity=q``
    merges each pair of adjacent elements' DOFs of every order ``0``
    through ``q`` into one shared DOF -- values at ``q=0`` (the usual
    "C0"/"CG" finite-element basis), additionally first derivatives at
    ``q=1`` ("C1", what a cubic Hermite element needs to represent a
    4th-order, Euler-Bernoulli-type weak form), and so on. The merge is a
    linear *assembly* map, a plain right-multiplication on the broken
    basis's design matrix, so everything built on ``Basis``
    (mass/stiffness matrices, projection, inner products) works through it
    unchanged. :class:`CubicHermiteBasis` is the kind of element basis a
    ``q=1`` continuity generalizes to.

    **Element ownership at a breakpoint.** This basis can be two-valued at its
    interior breakpoints (always for the derivatives of :math:`C^0` functions,
    and also the value when ``continuity=-1``) so evaluating exactly *on* one
    requires choosing a side. **Evaluation** follows the ``side`` argument: ``"right"``
    (the default) makes ownership half-open ``[lo, hi)``, giving the limit
    from above, and ``"left"`` gives ``(lo, hi]`` and the limit from
    below.

    Two-sided access is what discontinuous methods need: a DG numerical
    flux is built from :math:`u^-` and :math:`u^+` at each interface, and
    a gradient-jump error indicator for a :math:`C^0` space needs the same
    of ``deriv=1``. In one dimension an interface *is* a point, so these
    are ordinary evaluations with different ``side`` arguments.

    Parameters
    ----------
    element_basis : Basis or tuple of Basis
        Local basis, defined on the reference interval ``[-1, 1]``, mapped
        onto each element. A single ``Basis`` is shared across every
        element (the common case); a ``tuple`` gives one basis per element
        -- possibly of different order, or even a different family -- for
        per-element ("p-refined") accuracy. A tuple must have exactly one
        entry per element and every entry must agree on ``measures`` (so the
        assembled basis has one coherent per-dimension weight). For
        ``continuity=q`` every element's basis must expose endpoint DOFs of
        every order ``0`` through ``q`` via ``boundary_dofs`` (e.g. a
        :class:`LagrangeBasis` whose nodes include both endpoints, for
        ``q=0``; a :class:`CubicHermiteBasis` for ``q=1``). Always stored
        (and compared/hashed) as a per-element tuple, regardless of which
        form was passed in.
    breakpoints : array_like
        Element boundaries on the reference domain, shape ``(k + 1,)`` for
        ``k`` elements. Must be strictly increasing and span ``[-1, 1]``
        exactly.
    continuity : int, optional
        ``-1`` for a discontinuous basis; a nonnegative ``q`` for
        continuity through order ``q`` (``0`` for value continuity, ``1``
        additionally for first-derivative continuity, and so on -- see
        the class docstring). Default ``0``.
    """

    element_basis: Basis | tuple[Basis, ...]
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
        measures = element_bases[0].measures
        if any(eb.measures != measures for eb in element_bases):
            raise ValueError(
                "every element's basis must share the same measures, so the "
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
        # Lets `evaluate_expansion` keep its fused O(1)-in-
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
        """Degrees of freedom after continuity is imposed."""
        return self._assembly.shape[1]

    @property
    def Parameters(self) -> type:  # noqa: N802
        """The outer affine map of the whole tiled pattern onto ``[a, b]``;
        the breakpoints themselves are structural, not parameters."""
        return UnitInterval.Parameters

    @property
    def measures(self):
        """The element bases' shared weight

        Verified at construction to be the same across elements.
        Tiling rescales the reference weight onto each element but
        does not change which family it is.
        """
        return self.element_basis[0].measures

    @property
    def required_breakpoints(self) -> np.ndarray:
        """This basis is only piecewise smooth: it kinks (or, for
        ``continuity=-1``, jumps) at every interior breakpoint."""
        return self.breakpoints

    def boundary_dofs(self, order: int = 0) -> tuple[int | None, int | None]:
        """Global indices of the DOFs at the two ends of the tiled domain.

        ``None`` where the element basis has no such DOF.

        The left end belongs entirely to element 0 and the right end to the
        last element, so this maps each end element's own
        ``boundary_dofs(order)``. Under ``continuity=-1`` there is no shared/global
        endpoint identity (every element's DOFs are independent), so this returns
        ``(None, None)`` regardless of the element basis.
        """
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

    def default_quadrature(self):
        """Each element's own rule, tiled across the same breakpoints."""
        from archimedes.quadrature import composite_quad

        return composite_quad(
            [eb.default_quadrature() for eb in self.element_basis], self.breakpoints
        )

    def _product_basis(self, other):
        """Same breakpoints, per-element product basis, weaker continuity.

        The breakpoints must match exactly: a product across two different
        partitions kinks at the union of both, which neither operand's
        partition can represent. Products are formed element by element.

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
            tuple(
                e1._product_basis(e2)
                for e1, e2 in zip(self.element_basis, other.element_basis)
            ),
            self.breakpoints,
            continuity=min(self.continuity, other.continuity),
        )

    def _derivative_basis(self, deriv=1):
        """Same breakpoints, per-element derivative basis, weaker continuity.

        The derivative can in general leave the original space rather than being
        contained in a subspace of it, so differentiating ``deriv`` times can only
        be relied on for continuity down to ``q - deriv``.

        For example, a :math:`C^0` (``q=0``) function's derivative jumps at every
        breakpoint; a :math:`C^1` (``q=1``, e.g. cubic Hermite) function's *first*
        derivative is still continuous (``max(1 - 1, -1) = 0``), while its *second*
        derivative need not be (``max(1 - 2, -1) = -1``). Within each element the
        derivative is still a polynomial of degree ``n_loc - 1 - deriv``, so that
        element's basis shrinks in the usual way and the representation stays exact,
        whether or not the elements share an order.
        """
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
        """Not implemented: unlike :meth:`_derivative_basis`, growing each
        element's basis is not enough on its own.

        A derivative is exact element-by-element with no cross-element
        bookkeeping, since differentiating cannot lower continuity below
        ``-1`` and the standard DOF-merge assembly (:meth:`_build_assembly`)
        already handles whatever continuity remains. An antiderivative
        instead needs continuity to go *up*, which the assembly cannot
        produce by itself: it merges DOFs that are already the same shared
        quantity in both elements, but a modal family like Legendre has no
        boundary DOF to merge in the first place (:meth:`boundary_dofs`
        returns ``(None, None)`` regardless of order), even though the
        antiderivative of a Legendre element genuinely must be continuous
        with its neighbor. What's needed instead is a running constant
        carried from each element into the next -- the piecewise analogue
        of :meth:`Function.integral`'s single boundary pin, but
        applied once per element rather than once globally. That
        construction doesn't exist yet.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define an integral basis: an "
            f"exact piecewise antiderivative needs a running constant "
            f"carried across elements to stay continuous, not just a "
            f"larger per-element basis, and that construction isn't "
            f"implemented. Project onto a global (non-piecewise) space "
            f"first if you need an exact `.integral()`."
        )

    def _build_assembly(self) -> np.ndarray:
        r"""Build the assembly map ``T``, shape ``(n_broken, n_basis)``, such
        that :math:`\Phi_{\mathrm{global}}(x) = \Phi_{\mathrm{broken}}(x) \,
        T`. Because continuity is just this right-multiplication, everything
        built on ``Basis`` (mass/stiffness matrices, projection, inner
        products) works through it unchanged.

        - ``continuity=-1`` -- ``T`` is the identity; element DOFs are
          independent and the basis jumps at interior breakpoints.
        - ``continuity=q >= 0`` -- ``T`` merges each element's ``order``-th
          boundary DOF (see :meth:`Basis.boundary_dofs`) with the next
          element's same-``order`` DOF, for every ``order`` from ``0``
          through ``q``, giving ``_n_broken - (n_elements - 1) * (q + 1)``
          continuous DOFs. Each merged DOF is a single basis function
          supported on *both* adjacent elements -- the standard "hat" for
          finite elements, generalized to slope-matching and beyond.

        The merge itself is always a bare identity, including for a
        derivative-type DOF (``order >= 1``): any rescaling needed to make
        it comparable across elements of different width is the element
        basis's own responsibility (see :attr:`Basis._dof_order`), not this
        assembly's. So generalizing from one merged order (``C0``) to
        several is purely bookkeeping -- track one "previous element's
        global index" per order instead of one overall, keyed by which
        local index each order's ``boundary_dofs`` reports.
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
        basis, then assembled. ``masks[e]`` selects the points element ``e``
        owns; where that ownership comes from is the callers' business."""
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
            Target-domain endpoints; default the reference domain
            ``[-1, 1]``.
        side : {"right", "left"}, optional
            Which one-sided limit to take at a point lying exactly on an
            interior breakpoint, where this basis is two-valued. ``"right"``
            (default) makes ownership half-open ``[lo, hi)``; ``"left"``
            makes it ``(lo, hi]``. Immaterial away from breakpoints, and for
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
        element ownership instead of the coordinate convention.

        A composite rule may place nodes exactly on interior breakpoints --
        a Lobatto sub-rule places one there from *each* side, so the
        breakpoint appears twice in ``nodes`` with identical coordinates but
        different provenance. Locating by coordinate assigns both copies to
        the same element, which drops one element's endpoint contribution
        and silently mis-integrates a discontinuous basis. Using
        ``rule.elements`` distinguishes them exactly.

        The rule's breakpoints need only be a *superset* of this basis's
        (see :meth:`required_breakpoints`), so a rule element is mapped to
        the basis element containing it. That map is static, which also
        makes the resulting masks static -- one fewer runtime comparison in
        the traced graph, and one fewer branch point for autodiff.
        """
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

    def evaluate_expansion(
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

        Parameters
        ----------
        coefficients, x, deriv : array_like, array_like, int, optional
            As for :meth:`evaluate_expansion <Basis.evaluate_expansion>`.
        a, b : float, optional
            Target-domain endpoints; see :meth:`evaluate`.
        side : {"right", "left"}, optional
            Which one-sided limit to take at a point lying exactly on an
            interior breakpoint, where this basis is two-valued -- see
            :meth:`evaluate`. Default ``"right"``.
        """
        if not self._uniform:
            return super().evaluate_expansion(
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
