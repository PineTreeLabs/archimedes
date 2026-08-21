"""Finite-dimensional linear span of a Basis, tied to a target domain."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from archimedes import tree
from archimedes.measure import (
    HalfLine,
    JacobiMeasure,
    LaguerreMeasure,
    LegendreMeasure,
    PhysicistsHermiteMeasure,
    ProbabilistsHermiteMeasure,
    RealLine,
    UnitInterval,
)
from archimedes.quadrature import Quadrature

from ._basis import (
    RIGHT,
    Basis,
    BasisMatrix,
    BSplineBasis,
    CubicHermiteBasis,
    FourierBasis,
    LagrangeBasis,
    MonomialBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
    ProductParameters,
    TensorBasis,
)

if TYPE_CHECKING:
    from ._function import Function

__all__ = ["FunctionSpace"]


def _normalize_breakpoints(breakpoints) -> tuple[float, float, np.ndarray]:
    """Physical-domain ``breakpoints`` (spanning ``[a, b]``) to ``(a, b,
    ref)``, ``ref`` being the same partition on the reference domain
    ``[-1, 1]`` that :class:`PiecewiseBasis` itself expects.

    ``a``/``b`` are read directly from the array's own endpoints -- the
    caller never states them separately, so there is no second coordinate
    system to keep in sync by hand.
    """
    bp = np.asarray(breakpoints, dtype=float)
    if bp.ndim != 1 or len(bp) < 2:
        raise ValueError(
            f"breakpoints must be 1-D with at least 2 entries, got shape {bp.shape}"
        )
    if np.any(np.diff(bp) <= 0):
        raise ValueError("breakpoints must be strictly increasing")

    a, b = float(bp[0]), float(bp[-1])
    scale, shift = UnitInterval().affine_params(a, b)
    ref = (bp - shift) / scale
    # Pin the endpoints exactly: PiecewiseBasis checks `bp[0] != -1.0` by
    # equality, not tolerance, and the affine round-trip is not guaranteed
    # to land there in floating point.
    ref[0], ref[-1] = -1.0, 1.0
    return a, b, ref


_NODE_FAMILIES = {
    "lobatto": LagrangeBasis.gauss_lobatto,
    "legendre": LagrangeBasis.gauss_legendre,
    "radau_left": functools.partial(LagrangeBasis.gauss_radau, endpoint="left"),
    "radau_right": functools.partial(LagrangeBasis.gauss_radau, endpoint="right"),
    "equispaced": LagrangeBasis.equispaced,
}


def _resolve_lagrange_element(n: int, nodes) -> LagrangeBasis:
    """One order-``n`` :class:`LagrangeBasis` element for
    :meth:`FunctionSpace.piecewise`'s ``nodes`` argument -- a node-family
    name, an explicit callable/array, or (``None``) the family default.

    Dispatches on ``type(nodes)`` rather than comparing ``nodes`` against
    each family name with ``==`` directly: once an array-like reaches this
    function, ``array == "lobatto"`` is itself an elementwise comparison
    (not a clean ``False``), so a string check must run first.
    """
    if nodes is None:
        return LagrangeBasis.gauss_lobatto(n)
    if isinstance(nodes, str):
        if nodes not in _NODE_FAMILIES:
            raise ValueError(
                f"unknown nodes family {nodes!r}; expected one of "
                f"{sorted(_NODE_FAMILIES)}, a callable, or an array of "
                f"reference nodes"
            )
        return _NODE_FAMILIES[nodes](n)
    if callable(nodes):
        return LagrangeBasis(reference_nodes=nodes(n), node_family=nodes)
    arr = np.asarray(nodes, dtype=float)
    if len(arr) != n:
        raise ValueError(f"nodes has {len(arr)} points but order={n}")
    return LagrangeBasis(reference_nodes=arr)


def _resolve_element_basis(kind: str, order, nodes) -> Basis | tuple[Basis, ...]:
    """The (possibly per-element) local ``Basis`` for
    :meth:`FunctionSpace.piecewise`'s ``kind``/``order``/``nodes``.

    Returns a single ``Basis`` for a scalar ``order`` (the common, uniform
    case -- letting :class:`PiecewiseBasis` tile it, which also keeps its
    ``_uniform`` fast path), or a tuple for a per-element ``order`` tuple.
    """
    orders = order if isinstance(order, tuple) else (order,)
    if kind == "legendre":
        if nodes is not None:
            raise ValueError("nodes is only meaningful for kind='lagrange'")
        bases = tuple(OrthogonalPolynomialBasis(LegendreMeasure(), n) for n in orders)
    elif kind == "lagrange":
        bases = tuple(_resolve_lagrange_element(n, nodes) for n in orders)
    elif kind == "hermite":
        if nodes is not None:
            raise ValueError("nodes is only meaningful for kind='lagrange'")
        hermite = CubicHermiteBasis()
        for n in orders:
            if n != hermite.n_basis:
                raise ValueError(
                    f"kind='hermite' is a fixed cubic element (value + slope "
                    f"at each end, {hermite.n_basis} DOFs); order must be "
                    f"{hermite.n_basis}, got {n}"
                )
        bases = tuple(CubicHermiteBasis() for _ in orders)
    else:
        raise ValueError(
            f"kind must be 'lagrange', 'legendre', or 'hermite', got {kind!r}"
        )
    return bases if isinstance(order, tuple) else bases[0]


def _orthogonal_space(
    cls, measure, n_basis: int, domain, density: bool, quad_rule
) -> FunctionSpace:
    """Shared body for the ``OrthogonalPolynomialBasis``-backed classmethod
    constructors (:meth:`FunctionSpace.legendre`, ``.chebyshev``, ``.jacobi``,
    ``.hermite``, ``.laguerre``): only the measure and the domain-parameter
    type differ between them.
    """
    basis = OrthogonalPolynomialBasis(measure, n_basis, density=density)
    return cls(basis, domain, quad_rule=quad_rule)


def _is_superset(have: np.ndarray, required: np.ndarray, tol: float = 1e-12) -> bool:
    """Whether every point of ``required`` appears in ``have``.

    A superset, not equality: refining an element is harmless, since each
    sub-element still lies inside one element of the basis. Conversely a
    finer rule that is *not* aligned is still wrong, so comparing node
    counts would prove nothing.
    """
    return bool(np.all([np.any(np.abs(have - point) <= tol) for point in required]))


@tree.struct
class FunctionSpace:
    """The linear span of a :class:`Basis` on a fixed target domain.

    Pairs a ``Basis`` (family + size) with a target domain and a "natural"
    quadrature rule, and provides the operations that act on the *space*
    rather than on any one element of it: evaluation, (Galerkin) projection,
    and the quadrature nodes/weights lower-level assembly is built from.

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
        interval, ``RealLine.Parameters(loc=..., scale=...)`` for a
        Hermite-derived one). A pytree leaf, so it may be traced.
    quad_rule : QuadratureRule, optional
        The space's natural quadrature rule, used unconditionally wherever
        the required accuracy is fully determined by ``basis`` (its own
        differentiation matrix, say, backing ``Function.derivative``), and
        as the default for ``project`` (which accepts an explicit override,
        since the right accuracy for a given target function isn't knowable
        from the space alone). Static.

        Defaults to ``basis.default_quadrature()``, which is exact for
        those integrands by construction. An explicit rule is checked for
        compatibility with the basis and rejected if it cannot integrate it
        exactly -- see ``Basis.required_breakpoints``.
    """

    basis: Basis = tree.field(static=True)
    domain: Any
    quad_rule: Quadrature | None = tree.field(static=True, default=None)

    def __post_init__(self):
        if not isinstance(self.domain, self.basis.Parameters):
            raise TypeError(
                f"domain must be a {self.basis.Parameters.__qualname__} "
                f"instance for this basis, got {type(self.domain).__name__}"
            )

        if self.quad_rule is None:
            object.__setattr__(self, "quad_rule", self.basis.default_quadrature())
        else:
            self._validate_quad_rule(self.quad_rule)

    def _validate_quad_rule(self, rule: Quadrature) -> None:
        """Reject a rule that cannot integrate this basis correctly.

        Three structural requirements:

        - **Dimension.** A rule of the wrong ``ndim`` presents points of the
          wrong shape.
        - **Weight.** The rule's ``weights`` apply its *own* measure
          implicitly, so a rule built on a different measure computes a
          different inner product than the basis is orthogonal under.
        - **Alignment.** Where the basis is not smooth, the rule's
          subintervals must not straddle the kinks.

        Degree is *not* checked; an under-resolved rule is inaccurate but
        not categorically wrong, and ``project`` legitimately varies it.
        """
        ndim = self.basis.ndim
        if rule.ndim != ndim:
            raise ValueError(
                f"{type(self.basis).__name__} is {ndim}-dimensional but the "
                f"quadrature rule is {rule.ndim}-dimensional"
            )

        # Both sides are per-dimension tuples of length `ndim`; a basis with
        # no weight of its own (nodal, piecewise) reports None and imposes
        # no constraint.
        for d, (basis_measure, rule_measure) in enumerate(
            zip(self.basis.measures, rule.measures)
        ):
            if basis_measure is not None and basis_measure != rule_measure:
                where = "" if ndim == 1 else f" in dimension {d}"
                raise ValueError(
                    f"quadrature weight does not match the basis{where}: the "
                    f"basis is orthogonal under "
                    f"{type(basis_measure).__name__} but the rule integrates "
                    f"against {type(rule_measure).__name__}, so the two "
                    f"describe different inner products"
                )

        required = self.basis.required_breakpoints
        if required is None:
            return

        # `required_breakpoints`/`breakpoints` are per-dimension tuples for a
        # tensor basis/rule and bare values otherwise.
        per_dim_required = required if ndim > 1 else (required,)
        per_dim_have = rule.breakpoints if ndim > 1 else (rule.breakpoints,)
        for d, (req, have) in enumerate(zip(per_dim_required, per_dim_have)):
            if req is None:
                continue
            if have is None or not _is_superset(have, req):
                where = "" if ndim == 1 else f" in dimension {d}"
                raise ValueError(
                    f"{type(self.basis).__name__} is only piecewise smooth"
                    f"{where}, with breakpoints {np.asarray(req)}, so "
                    f"quadrature elements must not straddle them; got a rule "
                    f"with breakpoints {have}. Use `composite_quad(rule, "
                    f"breakpoints)` over a superset of the basis breakpoints, "
                    f"or omit `quad_rule` to use `basis.default_quadrature()`."
                )

    # --- constructors ---

    @classmethod
    def legendre(
        cls,
        n_basis: int,
        a: float = -1.0,
        b: float = 1.0,
        density: bool = False,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """Global Legendre polynomial space on ``[a, b]``.

        Sugar for ``FunctionSpace(OrthogonalPolynomialBasis(LegendreMeasure(),
        n_basis, density=density), UnitInterval.Parameters(a=a, b=b),
        quad_rule=quad_rule)``.
        """
        return _orthogonal_space(
            cls,
            LegendreMeasure(),
            n_basis,
            UnitInterval.Parameters(a=a, b=b),
            density,
            quad_rule,
        )

    @classmethod
    def chebyshev(
        cls,
        n_basis: int,
        a: float = -1.0,
        b: float = 1.0,
        second_kind: bool = False,
        density: bool = False,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        r"""Global Chebyshev polynomial space on ``[a, b]``.

        Chebyshev polynomials are the special case of
        :class:`~archimedes.measure.JacobiMeasure` with
        :math:`\alpha = \beta = -1/2` (first kind, the default) or
        :math:`\alpha = \beta = 1/2` (``second_kind=True``); see
        :class:`~archimedes.measure.JacobiMeasure`.
        """
        exponent = 0.5 if second_kind else -0.5
        return _orthogonal_space(
            cls,
            JacobiMeasure(exponent, exponent),
            n_basis,
            UnitInterval.Parameters(a=a, b=b),
            density,
            quad_rule,
        )

    @classmethod
    def jacobi(
        cls,
        alpha: float,
        beta: float,
        n_basis: int,
        a: float = -1.0,
        b: float = 1.0,
        density: bool = False,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """Global Jacobi polynomial space on ``[a, b]``; see
        :class:`~archimedes.measure.JacobiMeasure` for ``alpha``/``beta``."""
        return _orthogonal_space(
            cls,
            JacobiMeasure(alpha, beta),
            n_basis,
            UnitInterval.Parameters(a=a, b=b),
            density,
            quad_rule,
        )

    @classmethod
    def hermite(
        cls,
        n_basis: int,
        loc: float = 0.0,
        scale: float = 1.0,
        kind: Literal["phys", "prob"] = "prob",
        density: bool = False,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """Global Hermite polynomial space on the real line.

        ``kind`` selects which classical Hermite convention:

        - ``"prob"`` (default): the *probabilists'* convention, reference
          weight :math:`e^{-x^2/2}`
          (:class:`~archimedes.measure.ProbabilistsHermiteMeasure`), the
          *un-normalized* standard normal density.
        - ``"phys"``: the *physicists'* convention, reference weight
          :math:`e^{-x^2}` (:class:`~archimedes.measure.PhysicistsHermiteMeasure`).

        Neither weight integrates to 1 on its own for either ``kind``; for normalized
        (e.g. probability density models, polynomial chaos expansions), set
        ``density=True`` to normalize for either convention.

        Parameters
        ----------
        n_basis : int
            Number of basis functions.
        loc, scale : float, optional
            Location/scale of the target domain; see
            :class:`~archimedes.measure.RealLine`. Default ``0``, ``1``.
        kind : {"phys", "prob"}, optional
            Classical Hermite convention, as above. Default ``"prob"``.
        density : bool, optional
            Normalize against the probability density rather than the raw
            weight; see :attr:`Basis.density`. Default ``False``.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.

        Raises
        ------
        ValueError
            If ``kind`` is not ``"phys"`` or ``"prob"``.
        """
        if kind == "prob":
            measure = ProbabilistsHermiteMeasure()
        elif kind == "phys":
            measure = PhysicistsHermiteMeasure()
        else:
            raise ValueError(f"Hermite kind must be 'phys' or 'prob', got {kind!r}")
        return _orthogonal_space(
            cls,
            measure,
            n_basis,
            RealLine.Parameters(loc=loc, scale=scale),
            density,
            quad_rule,
        )

    @classmethod
    def fourier(
        cls,
        n_basis: int,
        a: float = -1.0,
        b: float = 1.0,
        kind: Literal["full", "cosine", "sine"] = "full",
        density: bool = False,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """Global (periodic) Fourier space on ``[a, b]``

        Treated as one periodic domain: ``a`` and ``b`` are the same point.

        Parameters
        ----------
        n_basis : int
            Number of basis functions; see :class:`FourierBasis` for the
            ``kind``-dependent convention (``kind="full"`` requires an odd
            value).
        a, b : float, optional
            Bounds of the target period. Default ``-1``, ``1``.
        kind : {"full", "cosine", "sine"}, optional
            Which trigonometric family; see :class:`FourierBasis`. Default
            ``"full"``.
        density : bool, optional
            Normalize against the probability density rather than the raw
            weight; see :attr:`Basis.density`. Default ``False``.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.

        Raises
        ------
        ValueError
            If ``kind`` is not ``"full"``, ``"cosine"``, or ``"sine"``, or
            if ``n_basis`` is invalid for the chosen ``kind`` (see
            :class:`FourierBasis`).
        """
        basis = FourierBasis(n_basis, kind=kind, density=density)
        return cls(basis, UnitInterval.Parameters(a=a, b=b), quad_rule=quad_rule)

    @classmethod
    def monomial(
        cls,
        n_basis: int,
        a: float = -1.0,
        b: float = 1.0,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """Global monomial (power series) space on ``[a, b]``.

        Sugar for ``FunctionSpace(MonomialBasis(n_basis),
        UnitInterval.Parameters(a=a, b=b), quad_rule=quad_rule)``. See
        :class:`MonomialBasis` for the reference-mapping convention and
        conditioning; prefer :meth:`legendre`/:meth:`chebyshev` for
        higher degree or numerically sensitive work.

        Parameters
        ----------
        n_basis : int
            Number of basis functions (monomial degrees ``0`` through
            ``n_basis - 1``).
        a, b : float, optional
            Bounds of the target interval. Default ``-1``, ``1``.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.
        """
        basis = MonomialBasis(n_basis)
        return cls(basis, UnitInterval.Parameters(a=a, b=b), quad_rule=quad_rule)

    @classmethod
    def laguerre(
        cls,
        n_basis: int,
        rate: float = 1.0,
        start: float = 0.0,
        density: bool = False,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """Global Laguerre polynomial space on ``[start, inf)``; see
        :class:`~archimedes.measure.LaguerreMeasure`."""
        return _orthogonal_space(
            cls,
            LaguerreMeasure(),
            n_basis,
            HalfLine.Parameters(rate=rate, start=start),
            density,
            quad_rule,
        )

    @classmethod
    def piecewise(
        cls,
        kind: str,
        order: int | tuple[int, ...],
        breakpoints,
        *,
        nodes: str | np.ndarray | Callable[[int], np.ndarray] | None = None,
        continuity: int = 0,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """A piecewise ``FunctionSpace``: a local basis tiled across
        ``breakpoints``, on the domain those breakpoints themselves span.

        Sugar over :class:`PiecewiseBasis` -- it builds the local
        ``element_basis`` and the target-domain ``UnitInterval.Parameters``
        for you, so neither ``LagrangeBasis``/``OrthogonalPolynomialBasis``
        nor ``UnitInterval`` need to be named directly for the common cases
        below. For anything else (a heterogeneous per-element family, a
        symbolic/traced domain independent of the mesh, an explicit
        reference-domain ``quad_rule``), build a ``PiecewiseBasis`` directly
        and pass it to the general ``FunctionSpace(basis, domain)``
        constructor -- nothing here is reachable only through this method.

        Parameters
        ----------
        kind : {"lagrange", "legendre", "hermite"}
            Local basis family.

            - ``"lagrange"`` is nodal (point-value degrees of freedom);
              node placement is chosen by ``nodes``.
            - ``"legendre"`` is modal (:class:`OrthogonalPolynomialBasis`
              on :class:`~archimedes.measure.LegendreMeasure`), which has
              no boundary degrees of freedom and so only supports
              ``continuity=-1``.
            - ``"hermite"`` is :class:`CubicHermiteBasis`: value *and*
              slope degrees of freedom at each end, fixed at ``order=4``
              (a cubic). Needed for ``continuity=1`` (:math:`C^1`); also
              buildable at ``continuity=0`` or ``-1``, merging (or not)
              only the value DOF.

        order : int or tuple of int
            Number of local degrees of freedom per element. A bare ``int``
            is shared by every element; a tuple gives one order per
            element (p-refinement) and must have one entry per element.
            Fixed at ``4`` for ``kind="hermite"``.
        breakpoints : array_like
            Element boundaries **on the physical (target) domain**, shape
            ``(n_elements + 1,)``, strictly increasing. Unlike
            ``PiecewiseBasis.breakpoints`` (which lives on the reference
            domain ``[-1, 1]``), this spans the whole target domain --
            ``breakpoints[0]``/``breakpoints[-1]`` *are* ``a``/``b``, read
            directly from the array rather than given separately.
        nodes : str, callable, or array_like, optional
            Node placement for a ``kind="lagrange"`` element, one of:

                - One of the family names ``"lobatto"``, ``"legendre"``,
                    ``"radau_left"``, ``"radau_right"``, ``"equispaced"``
                - An explicit ``n -> nodes`` callable
                - An explicit array of reference nodes.

            Default (``None``) is Gauss-Lobatto, which keeps ``continuity=0``,
            since Lobatto includes both endpoints. Meaningless (and rejected)
            for non-Lagrange bases.
        continuity : int, optional
            ``0`` (the default) for value-continuity (:math:`C^0`), ``-1``
            for a discontinuous (broken) basis. Forwarded to
            :class:`PiecewiseBasis` unchanged; see there for what each
            value requires of the element basis.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.
            Default ``basis.default_quadrature()``.

        Returns
        -------
        FunctionSpace
        """
        a, b, ref_breakpoints = _normalize_breakpoints(breakpoints)
        n_elements = len(ref_breakpoints) - 1
        if isinstance(order, tuple) and len(order) != n_elements:
            raise ValueError(
                f"order has {len(order)} entries but breakpoints describe "
                f"{n_elements} elements"
            )

        element_basis = _resolve_element_basis(kind, order, nodes)
        basis = PiecewiseBasis(element_basis, ref_breakpoints, continuity=continuity)
        return cls(basis, UnitInterval.Parameters(a=a, b=b), quad_rule=quad_rule)

    @classmethod
    def bspline(
        cls,
        degree: int,
        breakpoints,
        *,
        knots: np.ndarray | None = None,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """A B-spline ``FunctionSpace`` of the given ``degree``.

        Sugar over :class:`BSplineBasis` -- builds a *clamped* knot vector
        from physical ``breakpoints`` (multiplicity ``degree + 1`` at both
        ends, simple interior knots -- the common, Bezier-endpoint case) and
        derives ``Parameters`` from it automatically, so neither
        ``BSplineBasis`` nor ``UnitInterval`` need to be named directly for
        that case. For anything else -- an open (non-clamped) knot vector,
        non-simple interior multiplicity, ... -- pass ``knots`` directly;
        nothing here is reachable only through this method.

        Parameters
        ----------
        degree : int
            Polynomial degree of each piece.
        breakpoints : array_like
            Element boundaries **on the physical (target) domain**, shape
            ``(n_elements + 1,)``, strictly increasing.
            ``breakpoints[0]``/``breakpoints[-1]`` become the two
            ``degree + 1``-times-repeated end knots. Ignored if ``knots``
            is given.
        knots : array_like, optional
            An explicit, general knot vector (physical units), overriding
            the default clamped construction from ``breakpoints``. See
            :class:`BSplineBasis` for what makes a knot vector valid.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.
            Default ``basis.default_quadrature()``.

        Returns
        -------
        FunctionSpace
        """
        if knots is None:
            bp = np.asarray(breakpoints, dtype=float)
            if bp.ndim != 1 or len(bp) < 2:
                raise ValueError(
                    f"breakpoints must be 1-D with at least 2 entries, got "
                    f"shape {bp.shape}"
                )
            if np.any(np.diff(bp) <= 0):
                raise ValueError("breakpoints must be strictly increasing")
            knots = np.concatenate(
                [np.full(degree, bp[0]), bp, np.full(degree, bp[-1])]
            )

        basis = BSplineBasis(degree, knots)
        a = basis.knots[degree]
        b = basis.knots[len(basis.knots) - 1 - degree]
        return cls(basis, UnitInterval.Parameters(a=a, b=b), quad_rule=quad_rule)

    @classmethod
    def tensor(
        cls, *spaces: FunctionSpace, quad_rule: Quadrature | None = None
    ) -> FunctionSpace:
        """The tensor-product space of independent univariate ``spaces``.

        Parameters
        ----------
        *spaces : FunctionSpace
            One univariate space per dimension, in order.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.
            Default is the tensor product of the factors' own default
            rules; see :meth:`TensorBasis.default_quadrature`.

        Raises
        ------
        ValueError
            If fewer than one space is given, or any space is not
            univariate (``basis.ndim != 1``).
        """
        if len(spaces) < 1:
            raise ValueError("FunctionSpace.tensor needs at least one space")
        for i, space in enumerate(spaces):
            if space.basis.ndim != 1:
                raise ValueError(
                    f"spaces[{i}] is {space.basis.ndim}-dimensional; tensor "
                    f"factors must be univariate (tensor products are "
                    f"associative, so flatten rather than nest)"
                )
        basis = TensorBasis(tuple(space.basis for space in spaces))
        domain = ProductParameters(dims=tuple(space.domain for space in spaces))
        return cls(basis, domain, quad_rule=quad_rule)

    # --- implementation ---

    @property
    def n_basis(self) -> int:
        """Number of basis functions; forwarded from ``basis``."""
        return self.basis.n_basis

    def _is_compatible_with(self, other: FunctionSpace) -> bool:
        """Whether ``other`` denotes the same space, as far as is decidable.

        Compares ``basis`` and ``quad_rule`` by value, and ``domain`` only
        *structurally* (via its treedef) -- deliberately not by value.

        Domain values can't be compared once traced: two symbolic
        parametrizations raise ``TypeError`` under ``bool()`` unless they
        happen to be the same objects, and independently-traced ``Function``
        arguments get distinct symbols even when their source domains were
        numerically identical. So value comparison would be both
        undecidable and prone to false rejection.

        This means a caller can add two ``Function`` objects whose domains
        differ *numerically* -- e.g. ``(a=0, b=1)`` and ``(a=0, b=2)`` -- without
        an error. **Callers are responsible for ensuring the domains agree
        numerically**; only the structure is enforced here.
        """
        return (
            self.basis == other.basis
            and self.quad_rule == other.quad_rule
            and tree.structure(self.domain) == tree.structure(other.domain)
        )

    def _product_space(self, other: FunctionSpace) -> FunctionSpace:
        """The space that represents products of elements of ``self`` and
        ``other`` exactly.

        Uses ``basis._product_basis`` for the enlarged basis and this space's
        ``domain``; the quadrature rule is the product basis's own default,
        which is automatically exact for the product. With ``n_1 + n_2 - 1``
        Gauss points that rule is exact through degree
        ``2(n_1 + n_2) - 3``, and both the mass matrix and the load vector
        of the projection have degree ``2(n_1 + n_2 - 2)``.

        The domains are checked structurally only, for the same reason as
        :meth:`_is_compatible_with`: they may be traced.
        """
        if tree.structure(self.domain) != tree.structure(other.domain):
            raise ValueError(
                "product requires Functions on structurally identical domains"
            )
        return FunctionSpace(self.basis._product_basis(other.basis), domain=self.domain)

    def _derivative_space(self, deriv=1) -> FunctionSpace:
        """The smallest space that represents ``deriv``-th derivatives of
        this space's elements exactly.

        Uses the domain and derivative basis from this space's ``domain`` and
        ``basis``, with the derivative basis's own default quadrature. Smaller
        than this space for a polynomial family (differentiating lowers the degree);
        see :meth:`Basis._derivative_basis` for why the minimal space rather
        than this one.
        """
        return FunctionSpace(self.basis._derivative_basis(deriv), domain=self.domain)

    def _integral_space(self, order=1) -> FunctionSpace:
        """The smallest space whose elements' ``order``-th derivatives span
        this space's elements exactly -- the dual of :meth:`_derivative_space`.

        Larger than this space for a polynomial family (integrating raises
        the degree); see :meth:`Basis._integral_basis`.
        """
        return FunctionSpace(self.basis._integral_basis(order), domain=self.domain)

    def _integral_matrix(
        self,
        order: int = 1,
        boundary: str = "left",
        space: FunctionSpace | None = None,
    ) -> np.ndarray:
        r"""Matrix mapping this space's coefficients to those of the
        ``order``-th antiderivative, pinned to vanish (along with its first
        ``order - 1`` derivatives) at the domain's ``boundary`` endpoint.

        Built one order at a time. For a single order, the target space
        :math:`W` (this space's :meth:`_integral_space`) is exactly one
        derivative-order larger, so its own differentiation matrix ``D =
        W._diff_matrix(1, space=self)`` (shape ``(n, n + 1)``) is *surjective*
        onto this space with a 1-D nullspace (the constants). Appending one
        row pinning :math:`F(\mathrm{boundary}) = 0` -- evaluating :math:`W`'s
        basis at the chosen endpoint -- makes the system square and
        determines the unique antiderivative that both differentiates back to
        the input and vanishes there:

        .. math::
            \begin{bmatrix} D \\ \phi_W(x_{\mathrm{boundary}})^\top \end{bmatrix}
            F = \begin{bmatrix} I \\ 0 \end{bmatrix}

        Composing this step ``order`` times gives the standard iterated
        indefinite integral, vanishing together with its first ``order - 1``
        derivatives at the anchor (the usual Cauchy-formula convention for
        repeated integration) -- rather than solving one larger system with
        ``order`` boundary rows, which would need deciding what those extra
        rows should be instead of reusing this same one-condition step.

        Requires a domain with finite, literal endpoints to evaluate at:
        :class:`~archimedes.measure.RealLine`/:class:`~archimedes.measure.HalfLine`-
        parametrized bases (Hermite, Laguerre) have no boundary to anchor at
        and are rejected.

        Parameters
        ----------
        order : int, optional
            Number of times to integrate. Default 1.
        boundary : {"left", "right"}, optional
            Domain endpoint at which the antiderivative (and its lower
            derivatives, for ``order > 1``) vanishes. Default ``"left"``.
        space : FunctionSpace, optional
            Target space, overriding the default minimal
            :meth:`_integral_space`. Must have exactly ``self.n_basis +
            order`` basis functions -- unlike :meth:`_diff_matrix`, a
            target of the "wrong" size has no well-defined exact (or
            least-squares) antiderivative to fall back on here.

        Returns
        -------
        ndarray
            Shape ``(space.n_basis, self.n_basis)``.
        """
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")

        if order == 0:
            target = space if space is not None else self._integral_space(0)
            if target.n_basis != self.n_basis:
                raise ValueError(
                    f"space has {target.n_basis} basis functions, expected "
                    f"{self.n_basis} (this space's n_basis) for order=0"
                )
            return np.eye(self.n_basis)

        if boundary not in ("left", "right"):
            raise ValueError(f"boundary must be 'left' or 'right', got {boundary!r}")
        if not (hasattr(self.domain, "a") and hasattr(self.domain, "b")):
            raise ValueError(
                f"integral() needs a domain with finite endpoints to anchor "
                f"the constant of integration; {type(self.basis).__name__} "
                f"is defined on {type(self.domain).__name__}, which has none"
            )
        x_bnd = np.array([self.domain.a if boundary == "left" else self.domain.b])

        matrix = np.eye(self.n_basis)
        current = self
        for _ in range(order):
            step_target = current._integral_space(1)
            rule = step_target._resolve_rule()
            step_target._check_quad_rule_size(rule)
            # (current.n_basis, step_target.n_basis): D @ F recovers the
            # derivative of F's coefficients in this (smaller) space.
            D = step_target._diff_matrix(1, space=current)
            phi_bnd = step_target._basis_eval(x_bnd)[0]  # (step_target.n_basis,)
            system = np.concatenate([D, phi_bnd[None, :]], axis=0)
            rhs = np.concatenate(
                [np.eye(current.n_basis), np.zeros((1, current.n_basis))], axis=0
            )
            step = np.linalg.solve(
                system, rhs
            )  # (step_target.n_basis, current.n_basis)
            matrix = step @ matrix
            current = step_target

        target = space if space is not None else current
        if target.n_basis != current.n_basis:
            raise ValueError(
                f"space has {target.n_basis} basis functions, expected "
                f"{current.n_basis} (this space's n_basis + order) for an "
                f"exact antiderivative"
            )
        return matrix

    def _diff_matrix(self, deriv=1, space: FunctionSpace | None = None) -> np.ndarray:
        r"""Matrix mapping this space's coefficients to those of the
        ``deriv``-th derivative.

        .. math::
            D = M^{-1} \, \Phi_t^\top W \, \Phi^{(k)},

        the Galerkin projection of :math:`\phi_i^{(k)}` onto the target
        space, with :math:`M` and the quadrature taken from that target.
        This is exact whenever the target space contains the derivative,
        which it does by construction for both defaults below.

        Parameters
        ----------
        deriv : int or tuple of int, optional
            Derivative order; a multi-index for a
            :class:`TensorBasis`, a plain order otherwise. Default 1.
        space : FunctionSpace, optional
            Target space, overriding the default. By default the target is
            **this** space, giving the square ``(n_basis, n_basis)``
            differentiation matrix -- the form collocation and operator
            assembly want, since it keeps the coefficients' meaning (nodal
            values, modal amplitudes) unchanged. Use :meth:`Function.derivative`
            for the minimal target.

            A target too small to hold the derivative gives its projection,
            which is a well-defined approximation but no longer exact.

        Returns
        -------
        ndarray
            Shape ``(space.n_basis, self.n_basis)``.
        """
        target = self if space is None else space
        rule = target._resolve_rule()
        target._check_quad_rule_size(rule)
        phi = target.basis_matrix(quad_rule=rule)  # (npts, n_target)
        dphi = self.basis_matrix(deriv=deriv, quad_rule=rule)  # (npts, n_basis)
        M = phi.T @ phi
        return np.linalg.solve(M, phi.T @ dphi)  # type: ignore[no-any-return]

    def _domain_kwargs(self) -> dict:
        return {f.name: getattr(self.domain, f.name) for f in tree.fields(self.domain)}

    def _basis_eval(self, x, deriv: int = 0):
        return self.basis.evaluate(x, deriv=deriv, **self._domain_kwargs())

    def _resolve_rule(self, quad_rule: Quadrature | None = None) -> Quadrature:
        """The rule to use for this call: an explicit override, used
        exactly as given (no further domain mapping -- see
        :attr:`quad_rule`'s docstring), or the space's own default rule
        mapped onto ``self.domain`` via ``QuadratureRule.map_to``."""
        if quad_rule is not None:
            return quad_rule
        return self.quad_rule.map_to(**self._domain_kwargs())

    def _check_quad_rule_size(self, rule: Quadrature) -> None:
        # phi is (npts, n_basis), so phi.T @ diag(w) @ phi has rank at most
        # min(npts, n_basis) -- below n_basis points the mass matrix is
        # exactly (not just poorly) singular, and np.linalg.solve blows up
        # rather than failing cleanly. Only `_diff_matrix`/`project` solve
        # such a system, so only they call this -- `quadrature` itself makes
        # no assumption about what its caller will do with the result.
        if len(rule) < self.n_basis:
            raise ValueError(
                f"quad_rule has {len(rule)} points, fewer than n_basis="
                f"{self.n_basis}; the mass matrix would be singular"
            )

    def quadrature(
        self, quad_rule: Quadrature | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature nodes and weights mapped onto this space's domain.

        This, together with :meth:`basis_matrix`, is what an assembly like
        :meth:`project` is built from.

        Parameters
        ----------
        quad_rule : QuadratureRule, optional
            Quadrature rule to use instead of ``self.quad_rule``.

        Returns
        -------
        nodes, weights : ndarray
            Nodes and weights on this space's target domain.
        """
        rule = self._resolve_rule(quad_rule)
        # `density=self.basis.density` keeps the quadrature weights consistent
        # with the basis's own normalization (see `Basis.density`): a basis
        # orthonormal w.r.t. a probability measure needs weights that
        # integrate that same probability measure, not the raw weight.
        weights = rule.weights
        if self.basis.density:
            weights = weights / np.sum(weights)
        return rule.nodes, weights

    def basis_matrix(
        self, deriv: int = 0, quad_rule: Quadrature | None = None
    ) -> BasisMatrix:
        """This space's basis evaluated at its (or ``quad_rule``'s)
        quadrature nodes, as a :class:`BasisMatrix`.

        Uses the quadrature rule's element ownership, so it is exact for a
        piecewise basis even when discontinuous.

        The returned :class:`BasisMatrix` carries its own nodes
        (``Phi.nodes``), so a custom (Petrov-)Galerkin residual or
        projection needs only this method: evaluate pointwise expressions at
        ``Phi.nodes`` and test them against ``Phi.T``. Call
        :meth:`quadrature` directly only when weights are wanted without a
        basis matrix (a plain integral). See :meth:`project` for a worked
        example.

        Parameters
        ----------
        deriv : int or tuple of int, optional
            Derivative order; a multi-index for a :class:`TensorBasis`, a
            plain order otherwise. Default 0.
        quad_rule : QuadratureRule, optional
            Quadrature rule to use instead of ``self.quad_rule``.

        Returns
        -------
        BasisMatrix
            The design matrix, bundled with its nodes and weights.
        """
        rule = self._resolve_rule(quad_rule)
        matrix = self.basis._evaluate_at_nodes(
            rule, deriv=deriv, **self._domain_kwargs()
        )
        nodes, weights = self.quadrature(rule)
        return BasisMatrix(matrix, weights, nodes)

    def _evaluate(self, coefficients: np.ndarray, x, deriv: int = 0, side: str = RIGHT):
        r"""Evaluate :math:`\sum_i c_i \, \phi_i(x)` (or its ``deriv``-th
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
        side : str, optional
            Which one-sided limit to take where the basis is two-valued; see
            :meth:`Basis.evaluate`. Accepted for every basis and irrelevant
            for the smooth ones. Default ``"right"``.

        Returns
        -------
        ndarray
            Shape ``(npts,)`` or ``(npts, m)``, matching ``coefficients``.
        """
        # `evaluate_expansion` rather than `_basis_eval(...) @ coefficients`:
        # a locally-supported basis can fuse the two and avoid materializing
        # the full (npts, n_basis) matrix. The default implementation is
        # exactly that matrix product.
        return self.basis.evaluate_expansion(
            coefficients, x, deriv=deriv, side=side, **self._domain_kwargs()
        )

    def _inner_product(
        self,
        c1: np.ndarray,
        c2: np.ndarray,
        quad_rule: Quadrature | None = None,
    ):
        r"""Inner product :math:`\langle f, g \rangle = \int f(x) \, g(x)
        \, w(x) \, dx` for ``f``, ``g`` in this space with coefficients
        ``c1``, ``c2``, approximated via ``quad_rule`` (default
        ``self.quad_rule``).

        Unlike a product of two ``Function`` objects (deliberately unsupported
        -- see :class:`Function`), an inner product returns a scalar rather
        than another element of the space, so there's no aliasing/closure
        question to resolve: it's computed by evaluating both functions at
        the quadrature nodes and integrating the pointwise product, which
        is exact whenever ``quad_rule`` is accurate enough for that
        product -- equivalently ``c1 @ phi.T @ phi @ c2`` for the
        basis matrix ``phi`` (see :meth:`basis_matrix`), but computed
        directly without forming that full ``(n_basis, n_basis)`` matrix.

        For vector-valued coefficients (shape ``(n_basis, m)``) the
        integrand is contracted over components, :math:`\langle f, g
        \rangle = \int f \cdot g \, w \, dx`, so the result is a
        scalar in that case too and :meth:`Function.norm` is the
        :math:`L^2` norm of the whole vector-valued function rather than
        an array of per-component norms. Both coefficient arrays must have
        the same shape; for a per-component inner product, slice the
        coefficients and call this once per component.
        """
        rule = self._resolve_rule(quad_rule)
        phi = self.basis_matrix(quad_rule=rule)  # (npts, n_basis)
        integrand = (phi @ c1) * (phi @ c2)  # (npts,) or (npts, m)
        if integrand.ndim > 1:
            # `ndim` is a static (trace-time) property, so this branches on
            # shape rather than on a value and is safe under `@arc.compile`.
            integrand = np.sum(integrand, axis=-1)
        return rule.sum(integrand, density=self.basis.density)

    def project(
        self,
        f: Callable,
        quad_rule: Quadrature | None = None,
        test_space: "FunctionSpace | None" = None,
    ) -> Function:
        """Galerkin (or Petrov-Galerkin) projection of ``f`` onto this space.

        Solves ``M @ c = b`` for the coefficients ``c``, where, for this
        (trial) space's basis matrix ``Phi`` and ``test_space``'s basis
        matrix ``Psi`` (see :meth:`basis_matrix`), ``M = Psi.T @ Phi`` is
        the Gram matrix and ``b = Psi.T @ f(x)`` is the load vector -- both
        approximated via ``quad_rule`` (default ``self.quad_rule``).
        ``quad_rule`` must be accurate enough for the product of ``f`` and
        both bases, which is generally a higher-order requirement than
        exactness for either basis alone; pass an explicit ``quad_rule`` to
        use something other than the space's natural default.

        ``test_space`` defaults to this space (standard Galerkin, ``M`` the
        mass matrix). Passing a different space performs Petrov-Galerkin
        projection: the residual ``f - Phi @ c`` is made orthogonal to
        ``test_space`` rather than to this space. ``test_space`` must have
        the same ``n_basis`` as this space (so ``M`` is square) and must
        denote the same physical domain (checked structurally only, since
        domains may be traced). The result is still returned **in this
        (trial) space** -- ``test_space``
        only supplies the orthogonality condition used to solve for ``c``,
        not how ``c`` is interpreted, since ``c`` are always coefficients of
        *this* space's basis functions.

        ``f`` may be vector-valued: if ``f(x)`` has shape ``(npts, m)``,
        each component is projected and the result has coefficients of
        shape ``(n_basis, m)``. ``M`` is shared across components, so this
        costs one basis evaluation rather than ``m`` of them.

        Parameters
        ----------
        f : callable
            Target function, called once as ``f(x)`` on the full node
            array from the quadrature rule. Must return an array of shape
            ``(npts,)`` (scalar-valued) or ``(npts, m)`` (vector-valued).
        quad_rule : QuadratureRule, optional
            Quadrature rule to use instead of ``self.quad_rule``.
        test_space : FunctionSpace, optional
            Test space for a Petrov-Galerkin projection. Default this
            (trial) space, giving standard Galerkin projection.

        Returns
        -------
        Function
            The projected function, in this (trial) space, with
            coefficients of shape ``(n_basis,)`` or ``(n_basis, m)`` to
            match ``f``.

        Examples
        --------
        >>> import numpy as np
        >>> from archimedes.experimental.approximation import FunctionSpace
        >>> space = FunctionSpace.legendre(n_basis=8)
        >>> f = space.project(lambda x: np.exp(x))
        >>> np.round(f(np.array([-0.5, 0.0, 0.5])), 4)
        array([0.6065, 1.    , 1.6487])
        """
        from ._function import Function  # avoid a circular import

        test = self if test_space is None else test_space
        if test_space is not None:
            if test.n_basis != self.n_basis:
                raise ValueError(
                    f"test_space must have the same n_basis as this (trial) "
                    f"space for a square Petrov-Galerkin system; got "
                    f"test_space.n_basis={test.n_basis} vs "
                    f"n_basis={self.n_basis}"
                )
            if tree.structure(test.domain) != tree.structure(self.domain):
                raise ValueError(
                    "test_space must denote the same domain as this (trial) space"
                )

        rule = self._resolve_rule(quad_rule)
        self._check_quad_rule_size(rule)
        x, _ = self.quadrature(rule)
        phi = self.basis_matrix(quad_rule=rule)  # (npts, n_basis) trial
        psi = test.basis_matrix(quad_rule=rule)  # (npts, n_basis) test
        M = psi.T @ phi
        # `f` is an ordinary function of position, so it needs the
        # coordinates and has no breakpoint ambiguity of its own to resolve.
        # In the vector-valued case the right-hand side is the (n_basis, m)
        # matrix of stacked component loads, which `solve` handles with a
        # single factorization of the shared Gram matrix.
        rhs = psi.T @ f(x)
        return Function(np.linalg.solve(M, rhs), self)

    def function(self, coefficients: np.ndarray | None = None) -> Function:
        """A :class:`Function` on this space with known ``coefficients``.

        Sugar for ``Function(coefficients, self)`` -- the natural
        constructor when the coefficients are already in hand (solved for
        directly, deserialized, ...), as opposed to :meth:`project`, which
        computes them from a target function.

        Parameters
        ----------
        coefficients : array_like or None, optional
            Expansion coefficients, shape ``(n_basis,)`` for a scalar-valued
            function or ``(n_basis, m)`` for one mapping to an ``m``-vector.
            If ``None``, the expansion is initialized to zero. Default ``None``.
        """
        from ._function import Function  # avoid a circular import

        if coefficients is None:
            coefficients = np.zeros(self.n_basis)

        return Function(coefficients, self)
