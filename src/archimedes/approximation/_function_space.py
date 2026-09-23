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
from ._basis._utils import _reference_breakpoints

if TYPE_CHECKING:
    from ._function import Function

__all__ = ["FunctionSpace"]


def _normalize_breakpoints(breakpoints) -> tuple[float, float, np.ndarray]:
    """Physical-domain ``breakpoints`` (spanning ``[a, b]``) to ``(a, b,
    ref)``, ``ref`` being the same partition on the reference domain
    ``[-1, 1]`` that :class:`PiecewiseBasis` itself expects.

    ``a``/``b`` are read directly from the array's own endpoints. Validates
    ``breakpoints``, then defers the affine map (and its endpoint pinning) to
    ``_reference_breakpoints``, shared with :attr:`BSplineBasis._required_breakpoints`.
    """
    bp = np.asarray(breakpoints, dtype=float)
    if bp.ndim != 1 or len(bp) < 2:
        raise ValueError(
            f"breakpoints must be 1-D with at least 2 entries, got shape {bp.shape}"
        )
    if np.any(np.diff(bp) <= 0):
        raise ValueError("breakpoints must be strictly increasing")

    return _reference_breakpoints(bp)


_NODE_FAMILIES = {
    "lobatto": LagrangeBasis.gauss_lobatto,
    "legendre": LagrangeBasis.gauss_legendre,
    "radau_left": functools.partial(LagrangeBasis.gauss_radau, endpoint="left"),
    "radau_right": functools.partial(LagrangeBasis.gauss_radau, endpoint="right"),
    "equispaced": LagrangeBasis.equispaced,
}


def _resolve_lagrange_element(n_basis: int, nodes) -> LagrangeBasis:
    """One :class:`LagrangeBasis` element with ``n_basis`` nodes

    Used for the :meth:`FunctionSpace.piecewise` constructor. `nodes` can be
    a node-family name, an explicit callable/array, or (``None``) for the
    family default.
    """
    if nodes is None:
        return LagrangeBasis.gauss_lobatto(n_basis)
    if isinstance(nodes, str):
        if nodes not in _NODE_FAMILIES:
            raise ValueError(
                f"unknown nodes family {nodes!r}; expected one of "
                f"{sorted(_NODE_FAMILIES)}, a callable, or an array of "
                f"reference nodes"
            )
        return _NODE_FAMILIES[nodes](n_basis)
    if callable(nodes):
        return LagrangeBasis(reference_nodes=nodes(n_basis), node_family=nodes)
    arr = np.asarray(nodes, dtype=float)
    if len(arr) != n_basis:
        raise ValueError(
            f"nodes has {len(arr)} points but degree={n_basis - 1} needs {n_basis}"
        )
    return LagrangeBasis(reference_nodes=arr)


def _resolve_element_basis(kind: str, degree, nodes) -> Basis | tuple[Basis, ...]:
    """Resolve the local basis for a piecewise function space.

    Computes the (possibly per-element) local ``Basis`` for
    :meth:`FunctionSpace.piecewise`'s ``kind``/``degree``/``nodes``.

    Returns a single ``Basis`` for a scalar ``degree`` (the common, uniform
    case), or a tuple for a per-element ``degree`` tuple.
    """
    degrees = degree if isinstance(degree, tuple) else (degree,)
    if kind == "legendre":
        if nodes is not None:
            raise ValueError("nodes is only meaningful for kind='lagrange'")
        bases = tuple(
            OrthogonalPolynomialBasis(LegendreMeasure(), d + 1) for d in degrees
        )
    elif kind == "lagrange":
        bases = tuple(_resolve_lagrange_element(d + 1, nodes) for d in degrees)
    elif kind == "hermite":
        if nodes is not None:
            raise ValueError("nodes is only meaningful for kind='lagrange'")
        hermite = CubicHermiteBasis()
        hermite_degree = hermite.n_basis - 1
        for d in degrees:
            if d != hermite_degree:
                raise ValueError(
                    f"kind='hermite' is a fixed cubic element (value + slope "
                    f"at each end); degree must be {hermite_degree}, got {d}"
                )
        bases = tuple(CubicHermiteBasis() for _ in degrees)
    else:
        raise ValueError(
            f"kind must be 'lagrange', 'legendre', or 'hermite', got {kind!r}"
        )
    return bases if isinstance(degree, tuple) else bases[0]


def _orthogonal_space(
    cls, measure, n_basis: int, domain, density: bool, quad_rule
) -> FunctionSpace:
    """Shared body for the orthogonal polynomial constructors"""
    basis = OrthogonalPolynomialBasis(measure, n_basis, density=density)
    return cls(basis, domain, reference_quad_rule=quad_rule)


def _is_superset(have: np.ndarray, required: np.ndarray, tol: float = 1e-12) -> bool:
    """Whether every point of ``required`` appears in ``have``"""
    return bool(np.all([np.any(np.abs(have - point) <= tol) for point in required]))


@tree.struct
class FunctionSpace:
    """The linear span of a :class:`Basis` on a fixed target domain.

    Combines a ``Basis`` (family + size) with a target domain and a "natural"
    quadrature rule, and provides the operations that act on the space, e.g.
    evaluation, (Galerkin) projection, and quadrature nodes/weights.

    See :class:`Function` for a specific element of the space (a
    coefficient vector).

    ``FunctionSpace`` is a :func:`~archimedes.struct` so that the domain
    parameters can be symbolically traced and solved/optimized over jointly
    with the coefficients of a ``Function``. The ``basis`` and default quadrature
    rule are static (do not carry symbolic information).

    Parameters
    ----------
    basis : Basis
        Basis family and size. Static.
    domain : Basis.Parameters
        Target-domain parameters. An instance of ``basis.Parameters``, for example:
            - ``UnitInterval.Parameters(a=..., b=...)`` for a basis on an interval
            - ``RealLine.Parameters(loc=..., scale=...)`` for the entire real line

    reference_quad_rule : Quadrature | None, optional
        The space's natural quadrature rule, defined on the basis's **reference**
        domain. Used unconditionally wherever the required accuracy is fully determined
        by :attr:`basis` and as the default for :meth:`project`. Defaults to
        a rule that is exact by for that basis.

        Should typically be accessed via the :attr:`quad_rule` property, which maps
        from the reference domain (e.g. :math:`[0, 1]`) onto the actual ``domain``
        parameters.
    """

    basis: Basis = tree.field(static=True)
    domain: Any
    reference_quad_rule: Quadrature | None = tree.field(static=True, default=None)

    def __post_init__(self):
        if not isinstance(self.domain, self.basis.Parameters):
            raise TypeError(
                f"domain must be a {self.basis.Parameters.__qualname__} "
                f"instance for this basis, got {type(self.domain).__name__}"
            )

        if self.reference_quad_rule is None:
            object.__setattr__(
                self, "reference_quad_rule", self.basis._default_quadrature()
            )
        else:
            self._validate_quad_rule(self.reference_quad_rule)

    @property
    def quad_rule(self) -> Quadrature:
        """The space's quadrature rule mapped onto its ``domain``.

        This property does not apply density-normalization, regardless
        of whether the space has ``density=True``. Call :meth:`quadrature`
        to get optionally normalized quadrature weights and points, e.g. when
        working with probability densities.
        """
        return self.reference_quad_rule.map_to(**self._domain_kwargs())

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
            zip(self.basis._measures, rule.measures)
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

        required = self.basis._required_breakpoints
        if required is None:
            return

        # `_required_breakpoints`/`breakpoints` are per-dimension tuples for a
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
                    f"or omit `quad_rule` to use the default."
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
        """Global Legendre polynomial space on ``[a, b]``; see
        :class:`~archimedes.measure.LegendreMeasure` for the defining
        weight and recurrence.

        Parameters
        ----------
        n_basis : int
            Number of basis functions.
        a, b : float, optional
            Bounds of the target interval. Default ``-1``, ``1``.
        density : bool, optional
            Normalize against the probability density; see :attr:`Basis.density`.
            Default ``False``.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.

        Returns
        -------
        FunctionSpace
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

        Chebyshev polynomials of the first kind satisfy

        .. math::

            T_n(x) = \cos(n \cos^{-1} x)

        and are orthogonal on :math:`[-1, 1]` with respect to the weight
        :math:`(1 - x^2)^{-1/2}`.

        Chebyshev polynomials of the second kind satisfy

        .. math::

            U_n(x) = \sin((n+1) \cos^{-1} x) / \sin(\cos^{-1} x)

        and are orthogonal with respect to :math:`(1 - x^2)^{1/2}`.

        Both are special cases of :class:`~archimedes.measure.JacobiMeasure`,
        with :math:`\alpha = \beta = -1/2` and :math:`\alpha = \beta = 1/2`,
        respectively.

        Parameters
        ----------
        n_basis : int
            Number of basis functions.
        a, b : float, optional
            Bounds of the target interval. Default ``-1``, ``1``.
        second_kind : bool, optional
            Construct second-kind Chebyshev polynomials. Default ``False``.
        density : bool, optional
            Normalize against the probability density rather than the raw
            weight; see :attr:`Basis.density`. Default ``False``.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.

        Returns
        -------
        FunctionSpace
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
        r"""Global Jacobi polynomial space on ``[a, b]``.

        Defined on an interval :math:`[a, b]` with weight function
        :math:`(1 - x)^\alpha (1 + x)^\beta`. See
        :class:`~archimedes.measure.JacobiMeasure` for details and
        restrictions on ``alpha``/``beta``.

        Parameters
        ----------
        alpha, beta : float
            Jacobi measure parameters; see
            :class:`~archimedes.measure.JacobiMeasure`.
        n_basis : int
            Number of basis functions.
        a, b : float, optional
            Bounds of the target interval. Default ``-1``, ``1``.
        density : bool, optional
            Normalize against the probability density rather than the raw
            weight; see :attr:`Basis.density`. Default ``False``.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.

        Returns
        -------
        FunctionSpace
        """
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

        - ``"prob"`` (default): the probabilists' convention, reference
          weight :math:`e^{-x^2/2}`
          (:class:`~archimedes.measure.ProbabilistsHermiteMeasure`), the
          un-normalized standard normal density.
        - ``"phys"``: the physicists' convention, reference weight
          :math:`e^{-x^2}` (:class:`~archimedes.measure.PhysicistsHermiteMeasure`).

        Neither weight integrates to 1 on its own for either ``kind``; for normalized
        (e.g. probability density models, polynomial chaos expansions), set
        ``density=True``.

        Parameters
        ----------
        n_basis : int
            Number of basis functions.
        loc, scale : float, optional
            Location/scale of the Hermite weight function; see
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
            Number of basis functions; see :class:`FourierBasis`. ``kind="full"``
            requires an odd value.
        a, b : float, optional
            Bounds of the target period. Default ``-1``, ``1``.
        kind : {"full", "cosine", "sine"}, optional
            Which trigonometric family. Default ``"full"``.
        density : bool, optional
            Normalize against the probability density rather than the raw
            weight; see :attr:`Basis.density`. Default ``False``.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.

        Raises
        ------
        ValueError
            If ``kind`` is not ``"full"``, ``"cosine"``, or ``"sine"``, or
            if ``n_basis`` is invalid for the chosen ``kind``.
        """
        basis = FourierBasis(n_basis, kind=kind, density=density)
        return cls(
            basis, UnitInterval.Parameters(a=a, b=b), reference_quad_rule=quad_rule
        )

    @classmethod
    def monomial(
        cls,
        n_basis: int,
        a: float = -1.0,
        b: float = 1.0,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """Global monomial (power series) space on ``[a, b]``.

        See :class:`MonomialBasis` for details.

        Parameters
        ----------
        n_basis : int
            Number of basis functions (degrees ``0`` through ``n_basis - 1``).
        a, b : float, optional
            Bounds of the target interval. Default ``-1``, ``1``.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.
        """
        basis = MonomialBasis(n_basis)
        return cls(
            basis, UnitInterval.Parameters(a=a, b=b), reference_quad_rule=quad_rule
        )

    @classmethod
    def laguerre(
        cls,
        n_basis: int,
        rate: float = 1.0,
        start: float = 0.0,
        density: bool = False,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """Global Laguerre polynomial space on ``[start, inf)``.
        
        See :class:`~archimedes.measure.LaguerreMeasure` for details.

        Parameters
        ----------
        n_basis : int
            Number of basis functions.
        rate : float, optional
            Rate parameter for the exponential weight function; see
            :class:`~archimedes.measure.HalfLine`. Default ``1``.
        start : float, optional
            Left endpoint of the exponential weight function; see
            :class:`~archimedes.measure.HalfLine`. Default ``0``.
        density : bool, optional
            Normalize against the probability density rather than the raw
            weight; see :attr:`Basis.density`. Default ``False``.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.

        Returns
        -------
        FunctionSpace
        """
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
        degree: int | tuple[int, ...],
        breakpoints,
        *,
        nodes: str | np.ndarray | Callable[[int], np.ndarray] | None = None,
        continuity: int = 0,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """A piecewise function space created by tiling a local basis.

        Constructs a ``FunctionSpace`` by tiling a local basis across
        ``breakpoints``. The domain of the function space is determined by
        the range ``[breakpoints[0], breakpoints[-1]]``.

        This method is a convenience constructor for a :class:`PiecewiseBasis`
        for some common cases. For more fine-grained control, construct a
        ``PiecewiseBasis`` directly and pass it to the general
        ``FunctionSpace(basis, domain)`` constructor.

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
            - ``"hermite"`` is piecewise cubic, with value and slope degrees
              of freedom at each end, requiring ``degree=3``. This is the only
              option that supports ``continuity=1`` (:math:`C^1`), though also
              supports ``continuity=0`` or ``-1``.

        degree : int or tuple of int
            Polynomial degree of each element. An ``int`` is shared by every
            element; a tuple gives one degree per element and must length match
            the number of elements. Must be ``3`` for ``kind="hermite"``.
        breakpoints : array_like
            Element boundaries on the physical (not reference) domain, shape
            ``(n_elements + 1,)``, strictly increasing.
        nodes : str, callable, or array_like, optional
            Node placement for a ``kind="lagrange"`` element, one of:

                - One of the family names ``"lobatto"``, ``"legendre"``,
                    ``"radau_left"``, ``"radau_right"``, ``"equispaced"``
                - An explicit ``n -> nodes`` callable
                - An explicit array of reference nodes.

            Default (``None``) is Gauss-Lobatto, which keeps ``continuity=0``,
            since Lobatto includes both endpoints. Meaningless for
            non-Lagrange bases.
        continuity : int, optional
            ``0`` (the default) for value-continuity (:math:`C^0`), ``-1``
            for a discontinuous basis. ``kind="hermite"`` additionally supports
            ``continuity=1`` (:math:`C^1`).
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.
            Defaults to a rule that is exact for the basis.

        Returns
        -------
        FunctionSpace
        """
        a, b, ref_breakpoints = _normalize_breakpoints(breakpoints)
        n_elements = len(ref_breakpoints) - 1
        if isinstance(degree, tuple) and len(degree) != n_elements:
            raise ValueError(
                f"degree has {len(degree)} entries but breakpoints describe "
                f"{n_elements} elements"
            )

        element_basis = _resolve_element_basis(kind, degree, nodes)
        basis = PiecewiseBasis(element_basis, ref_breakpoints, continuity=continuity)
        return cls(
            basis, UnitInterval.Parameters(a=a, b=b), reference_quad_rule=quad_rule
        )

    @classmethod
    def bspline(
        cls,
        degree: int,
        knots,
        *,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """A B-spline function space built from a general, explicit knot vector.

        Constructs a :class:`BSplineBasis` with ``domain`` derived automatically
        from ``knots``. For the common case of a clamped knot vector built from
        physical element boundaries use :meth:`clamped_bspline` instead.

        Parameters
        ----------
        degree : int
            Polynomial degree of each piece.
        knots : array_like
            Nondecreasing knot vector, physical units. See
            :class:`BSplineBasis` for what makes a knot vector valid.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.
            Defaults to a rule that is exact for the basis.

        Returns
        -------
        FunctionSpace
        """
        basis = BSplineBasis(degree, knots)
        a = basis.knots[degree]
        b = basis.knots[len(basis.knots) - 1 - degree]
        return cls(
            basis, UnitInterval.Parameters(a=a, b=b), reference_quad_rule=quad_rule
        )

    @classmethod
    def clamped_bspline(
        cls,
        degree: int,
        breakpoints,
        *,
        quad_rule: Quadrature | None = None,
    ) -> FunctionSpace:
        """A B-spline space with a clamped knot vector built from element boundaries.

        The domain of the function space is determined by the range
        ``[breakpoints[0], breakpoints[-1]]``.

        Builds a knot vector with multiplicity ``degree + 1`` at both ends, so the
        first/last coefficients are the endpoint values. Interior knots are simple,
        preserving smoothest possible continuity (:math:`C^{degree - 1}`).

        Use :meth:`bspline` with an explicit knot vector for more fine-grained control.

        Parameters
        ----------
        degree : int
            Polynomial degree of each piece.
        breakpoints : array_like
            Element boundaries on the physical (target) domain, shape
            ``(n_elements + 1,)``, strictly increasing.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.
            Defaults to a rule that is exact for the basis.

        Returns
        -------
        FunctionSpace
        """
        bp = np.asarray(breakpoints, dtype=float)
        if bp.ndim != 1 or len(bp) < 2:
            raise ValueError(
                f"breakpoints must be 1-D with at least 2 entries, got shape {bp.shape}"
            )
        if np.any(np.diff(bp) <= 0):
            raise ValueError("breakpoints must be strictly increasing")
        knots = np.concatenate([np.full(degree, bp[0]), bp, np.full(degree, bp[-1])])
        return cls.bspline(degree, knots, quad_rule=quad_rule)

    @classmethod
    def tensor(
        cls, *spaces: FunctionSpace, quad_rule: Quadrature | None = None
    ) -> FunctionSpace:
        """The tensor-product space of independent univariate ``spaces``.

        Constructs a :class:`TensorBasis` from the given univariate spaces.

        Parameters
        ----------
        *spaces : FunctionSpace
            One univariate space per dimension, in order.
        quad_rule : QuadratureRule, optional
            Forwarded to the underlying ``FunctionSpace`` constructor.
            Default is the tensor product of the factors' default rules,
            a :class:`~archimedes.quadrature.TensorQuadratureRule`.

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
        return cls(basis, domain, reference_quad_rule=quad_rule)

    # --- implementation ---

    @property
    def n_basis(self) -> int:
        """Number of basis functions; forwarded from ``basis``."""
        return self.basis.n_basis

    def _is_compatible_with(self, other: FunctionSpace) -> bool:
        """Whether ``other`` denotes the same space, as far as is decidable.

        Compares ``basis`` and ``quad_rule`` by value, and ``domain`` only
        *structurally* (via its treedef).

        Domain values can't be compared once traced: two symbolic
        parametrizations raise ``TypeError`` under ``bool()`` unless they
        happen to be the same objects, and independently-traced ``Function``
        arguments get distinct symbols even when their source domains were
        numerically identical. So value comparison would be both
        undecidable and prone to false rejection.

        As a result, a caller can add two ``Function`` objects whose domains
        differ *numerically* without an error. **Callers are responsible for
        ensuring the domains agree numerically**; only the structure can be
        enforced here.
        """
        return (
            self.basis == other.basis
            and self.reference_quad_rule == other.reference_quad_rule
            and tree.structure(self.domain) == tree.structure(other.domain)
        )

    def _product_space(self, other: FunctionSpace) -> FunctionSpace:
        """The space that represents products of elements of ``self`` and
        ``other`` exactly.
        """
        if tree.structure(self.domain) != tree.structure(other.domain):
            raise ValueError(
                "product requires Functions on structurally identical domains"
            )
        return FunctionSpace(self.basis._product_basis(other.basis), domain=self.domain)

    def _derivative_space(self, deriv=1) -> FunctionSpace:
        """The smallest space that represents ``deriv``-th derivatives of
        this space's elements exactly.
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

        .. math::
            \begin{bmatrix} D \\ \phi_W(x_{\mathrm{boundary}})^\top \end{bmatrix}
            F = \begin{bmatrix} I \\ 0 \end{bmatrix}

        Composing this step ``order`` times gives an iterated indefinite integral,
        vanishing along with its first ``order - 1`` derivatives at the anchor.

        Requires a domain with finite, literal endpoints to evaluate at.

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
            order`` basis functions.

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
            rule = step_target.quad_rule
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
        This differentiation matrix is exact whenever the target space
        contains the derivative, which it does by construction for both
        defaults below.

        Parameters
        ----------
        deriv : int or tuple of int, optional
            Derivative order; a multi-index for a
            :class:`TensorBasis`, a plain order otherwise. Default 1.
        space : FunctionSpace, optional
            Target space, overriding the default. By default the target is
            **this** space, giving the square ``(n_basis, n_basis)``
            differentiation matrix. Use :meth:`Function.derivative`
            for the minimal target.

            A target too small to hold the derivative gives its projection,
            which is a well-defined approximation but no longer exact.

        Returns
        -------
        ndarray
            Shape ``(space.n_basis, self.n_basis)``.
        """
        target = self if space is None else space
        rule = target.quad_rule
        target._check_quad_rule_size(rule)
        phi = target.basis_matrix(quad_rule=rule)  # (npts, n_target)
        dphi = self.basis_matrix(deriv=deriv, quad_rule=rule)  # (npts, n_basis)
        M = phi.T @ phi
        return np.linalg.solve(M, phi.T @ dphi)  # type: ignore[no-any-return]

    def _domain_kwargs(self) -> dict:
        return {f.name: getattr(self.domain, f.name) for f in tree.fields(self.domain)}

    def _basis_eval(self, x, deriv: int = 0):
        return self.basis.evaluate(x, deriv=deriv, **self._domain_kwargs())

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

        Unlike :attr:`quad_rule`, the returned weights are normalized
        if the space has ``density=True``, in which case the returned
        weights sum to 1. Otherwise, weights sum to the integral of the weight
        function over the domain.

        Parameters
        ----------
        quad_rule : QuadratureRule, optional
            Quadrature rule to use instead of this space's own default.

        Returns
        -------
        nodes, weights : ndarray
            Nodes and weights on this space's target domain.

        See Also
        --------
        :meth:`basis_matrix` : Get the basis matrix evaluated at the quadrature nodes.
        :meth:`project` : L2 projection of a function onto this space.
        :attr:`quad_rule` : The default quadrature rule for this space.
        """
        rule = quad_rule if quad_rule is not None else self.quad_rule
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
        """This space's basis evaluated at its quadrature nodes.

        The basis matrix is a generalized Vandermonde matrix with shape
        ``(n_nodes, n_basis)``. The returned :class:`BasisMatrix` object
        also carries its quadrature nodes and weights.

        Parameters
        ----------
        deriv : int or tuple of int, optional
            Derivative order; a multi-index for a :class:`TensorBasis`, an
            ``int`` otherwise. Default 0.
        quad_rule : QuadratureRule, optional
            Quadrature rule to use instead of the default for this space.

        Returns
        -------
        BasisMatrix
            The basis matrix together with its quadrature nodes and weights.

        See Also
        --------
        :meth:`quadrature` : Get the quadrature nodes and weights for this space
        :attr:`quad_rule` : The default quadrature rule for this space.
        :meth:`project` : L2 projection of a function onto this space.
        """
        rule = quad_rule if quad_rule is not None else self.quad_rule
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
        # `_evaluate_expansion` rather than `_basis_eval(...) @ coefficients`:
        # a locally-supported basis can fuse the two and avoid materializing
        # the full (npts, n_basis) matrix. The default implementation is
        # exactly that matrix product.
        return self.basis._evaluate_expansion(
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
        ``c1``, ``c2``.

        For vector-valued coefficients (shape ``(n_basis, m)``) the
        integrand is contracted over components.
        """
        rule = quad_rule if quad_rule is not None else self.quad_rule
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

        See the :doc:`function approximation guide </handbook/approximation>`
        for mathematical details on the projection operation and how it is
        implemented numerically.

        Solves ``M @ c = b`` for the coefficients ``c``, using this (trial)
        space's basis matrix ``Phi`` and ``test_space``'s basis matrix
        ``Psi`` (see :meth:`basis_matrix`). ``M = Psi.T @ Phi`` is the Gram
        matrix and ``b = Psi.T @ f(x)`` is the load vector, both
        approximated via ``quad_rule`` (default this space's own quadrature).
        ``quad_rule`` must be accurate enough for the product of ``f`` and
        both bases, which is generally a higher-order requirement than
        exactness for either basis alone. Pass an explicit ``quad_rule`` to
        use something other than the space's natural default.

        ``test_space`` defaults to this space, performing a standard Galerkin
        projection. Passing a different space performs Petrov-Galerkin
        projection. In either case the residual ``f - Phi @ c`` is orthogonal to
        ``test_space``. ``test_space`` must have the same ``n_basis`` as this
        space and must denote the same physical domain. The resulting
        :class:`Function` is an element of this (trial) space, not the test space.

        ``f`` may be vector-valued. If ``f(x)`` has shape ``(npts, m)``, then
        each component is projected and the result has coefficients of
        shape ``(n_basis, m)``. The mass matrix is shared across components, so
        this vector-valued projection still only requires one basis evaluation.

        Parameters
        ----------
        f : callable
            Target function, called once as ``f(x)`` on the full node
            array from the quadrature rule. Must return an array of shape
            ``(npts,)`` (scalar-valued) or ``(npts, m)`` (vector-valued).
        quad_rule : QuadratureRule, optional
            Quadrature rule to use instead of this space's own default.
        test_space : FunctionSpace, optional
            Test space for a Petrov-Galerkin projection. Defaults to this
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
        >>> from archimedes.approximation import FunctionSpace
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

        rule = quad_rule if quad_rule is not None else self.quad_rule
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

        Convenience method for ``Function(coefficients, self)``. This is the
        natural constructor when the coefficients are already known, as opposed
        to :meth:`project`, which computes them from a target function.

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
