"""Fixed-node Gauss quadrature rules for classical orthogonal polynomial weights.

Gauss quadrature approximates a weighted integral

.. math::
    \\int_\\mathcal{D} f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n w_i f(x_i)

exactly for every polynomial ``f`` of degree :math:`\\leq 2n - 1`, where the
nodes :math:`x_i` are the roots of the degree-``n`` polynomial orthogonal with
respect to the weight ``w`` on the reference domain ``D``. Each classical
a "weight/domain pair" (Legendre, Jacobi, Laguerre, Hermite) is represented by
a "measure"; :class:`QuadratureRule` pairs a measure with a
fixed set of nodes and weights on its reference domain.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Protocol, Sequence, cast

import numpy as np

from archimedes import tree
from archimedes.measure import Measure, ReferenceDomain

__all__ = [
    "Quadrature",
    "QuadratureReferenceData",
    "QuadratureRule",
    "composite_quad",
]


class Quadrature(Protocol):
    """The interface every quadrature rule provides, whatever its dimension.

    Deliberately minimal: this is only what's needed by
    :class:`~archimedes.experimental.approximation.FunctionSpace` to define
    an inner product.
    """

    @property
    def ndim(self) -> int:
        """Number of dimensions of the domain integrated over."""

    @property
    def breakpoints(self) -> Any:
        """Element boundaries, for a rule assembled from sub-elements.

        ``np.ndarray | None`` for a one-dimensional rule; a per-dimension
        tuple of those for a multi-dimensional one. Consumers that care
        (see ``Basis.required_breakpoints``) dispatch on ``ndim``.
        """

    @property
    def elements(self) -> Any:
        """Which sub-element each node came from, or ``None`` for a rule with
        no element structure.

        Shape ``(n,)`` for a one-dimensional rule and ``(n, ndim)`` for a
        multi-dimensional one, mirroring ``nodes``. Present exactly when
        ``breakpoints`` is.

        This is *provenance* that coordinates cannot recover. A composite
        rule places nodes on its element boundaries and a basis that is
        discontinuous there needs to know which element each copy belongs to.
        """

    @property
    def measures(self) -> tuple[Measure, ...]:
        """The measure integrated against in each dimension, always a tuple
        of length ``ndim``."""

    def __len__(self) -> int:
        """Total number of quadrature nodes."""

    @property
    def nodes(self) -> np.ndarray:
        """Nodes on the currently-mapped target domain (see ``map_to``),
        shape ``(n,)`` for a one-dimensional rule or ``(n, ndim)``
        otherwise."""

    @property
    def weights(self) -> np.ndarray:
        """Weights including the Jacobian of the currently-mapped target
        domain (see ``map_to``), shape ``(n,)``."""

    def map_to(self, *params: Any, **kwparams: Any) -> Quadrature:
        """A new rule mapped onto the target domain."""


def _weighted_sum(
    weights: np.ndarray, values: np.ndarray, axis: int, n: int
) -> np.ndarray:
    """Contract ``values`` against quadrature ``weights`` along ``axis``."""
    if values.ndim > 2:
        raise ValueError(f"expected a 0-D, 1-D, or 2-D array, got {values.ndim}-D")
    if values.shape[axis] != n:
        raise ValueError(
            f"values.shape[{axis}] is {values.shape[axis]}, expected "
            f"{n} to match the quadrature nodes"
        )

    if values.ndim == 1 or axis == 0:
        return np.dot(weights, values)  # type: ignore[no-any-return]
    return np.dot(values, weights)  # type: ignore[no-any-return]


def _breakpoints_equal(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    """Compare optional breakpoint arrays, treating ``None`` as distinct
    from any array (a composite rule is not the same rule as a plain one,
    even if the nodes happened to coincide)."""
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(a, b)


def _check_affine_invariant(measure: Measure, params: tuple, kwparams: dict) -> None:
    """Raise if `params`/`kwparams` describe a non-reference-domain mapping
    and `measure.affine_invariant` is False.

    Syntactic (were *any* arguments given), not semantic (is the resulting
    map the identity) -- deliberately, so this stays correct under symbolic
    tracing, where `scale == 1.0` isn't a decidable Python bool. A bare
    reference-domain call is always allowed.
    """
    if (params or kwparams) and not measure.affine_invariant:
        raise ValueError(
            f"{type(measure).__name__}.affine_invariant is False: its "
            f"recurrence_coeffs relies on the generic Stieltjes-based "
            f"fallback, so mapping a rule built for it onto a different "
            f"domain is not verified to give the same rule you'd get by "
            f"building on the target domain directly. Call with no "
            f"arguments for the reference domain, or set "
            f"`affine_invariant = True` on a subclass whose closed-form "
            f"recurrence you have verified is affine-invariant."
        )


def _params_equal(
    a: "ReferenceDomain.Parameters | None", b: "ReferenceDomain.Parameters | None"
) -> bool:
    """Compare two domain ``Parameters`` (or ``None``), the same way
    :class:`QuadratureReferenceData`'s ``__eq__`` compares arrays --
    ``np.array_equal``, not ``==``, since fields can hold symbolic values."""
    if a is None or b is None:
        return a is None and b is None
    if type(a) is not type(b):
        return False
    return all(
        np.array_equal(getattr(a, f.name), getattr(b, f.name))
        for f in tree.fields(cast(Any, a))
    )


@dataclasses.dataclass(frozen=True)
class QuadratureReferenceData:
    """Reference-domain payload of a :class:`QuadratureRule` nodes/weights

    Validated once at construction; a rule's ``map_to`` reuses the same
    instance unchanged across every mapping, so this validation never
    re-runs just because the mapping changes. Never itself symbolic, so
    this stays a plain dataclass rather than a :func:`~archimedes.tree.struct`.

    Parameters
    ----------
    nodes : array_like
        Quadrature nodes :math:`x_i`, shape ``(n,)``, on
        ``measure.support``.
    weights : array_like
        Quadrature weights :math:`w_i`, shape ``(n,)``.
    measure : Measure
        Weight function and reference domain the rule is defined on.
    breakpoints : array_like, optional
        For a composite rule (see :func:`composite_quad`), the element
        boundaries it was tiled across, on ``measure.support``; ``None`` for
        a plain rule.
    elements : array_like, optional
        For a composite rule, the index of the element each node came from,
        shape ``(n,)``. Required whenever ``breakpoints`` is given and
        forbidden otherwise -- a rule that claims element structure but
        cannot say which element a node belongs to is exactly the state that
        makes a discontinuous basis mis-integrate. See
        :attr:`Quadrature.elements`.

    Raises
    ------
    ValueError
        If ``nodes`` and ``weights`` do not have the same shape, or if
        ``breakpoints`` and ``elements`` are not both given or both omitted.
    """

    nodes: np.ndarray
    weights: np.ndarray
    measure: Measure
    breakpoints: np.ndarray | None = None
    elements: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "nodes", np.asarray(self.nodes, dtype=float))
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float))
        if self.breakpoints is not None:
            object.__setattr__(
                self, "breakpoints", np.asarray(self.breakpoints, dtype=float)
            )
        if self.elements is not None:
            object.__setattr__(self, "elements", np.asarray(self.elements, dtype=int))
        if self.nodes.shape != self.weights.shape:
            raise ValueError(
                f"nodes {self.nodes.shape} and weights {self.weights.shape} "
                "must have the same shape"
            )
        if (self.breakpoints is None) != (self.elements is None):
            raise ValueError(
                "`breakpoints` and `elements` must be given together: a rule "
                "with element structure must say which element each node "
                "belongs to, since a discontinuous basis cannot recover that "
                "from the coordinates alone"
            )
        if self.elements is not None:
            if self.elements.shape != self.nodes.shape:
                raise ValueError(
                    f"elements {self.elements.shape} must have the same shape "
                    f"as nodes {self.nodes.shape}"
                )
            n_elements = len(self.breakpoints) - 1  # type: ignore[arg-type]
            if self.elements.min() < 0 or self.elements.max() >= n_elements:
                raise ValueError(
                    f"elements must index the {n_elements} intervals between "
                    f"`breakpoints`, got values in "
                    f"[{self.elements.min()}, {self.elements.max()}]"
                )

    def __eq__(self, other: object) -> bool:
        """Compare by value, elementwise on ``nodes``/``weights``.

        Defined explicitly because the ``@dataclass``-generated ``__eq__``
        would compare the array fields with ``==``, yielding an array and
        raising "truth value of an array is ambiguous" for any rule with
        more than one node. (``@dataclass`` leaves an explicitly-defined
        ``__eq__`` alone.)
        """
        if not isinstance(other, QuadratureReferenceData):
            return NotImplemented
        return (
            self.measure == other.measure
            and np.array_equal(self.nodes, other.nodes)
            and np.array_equal(self.weights, other.weights)
            and _breakpoints_equal(self.breakpoints, other.breakpoints)
            and _breakpoints_equal(self.elements, other.elements)
        )

    def __hash__(self) -> int:
        # Cheap, consistent with __eq__: equal instances agree on all of these.
        return hash((type(self), self.measure, len(self.nodes)))


@tree.struct
class QuadratureRule:
    """Fixed-node Gauss quadrature rule, optionally mapped onto a target
    domain.

    Approximates the weighted integral

    .. math::
        \\int_\\mathcal{D} f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n w_i f(x_i)

    where :math:`w` and :math:`\\mathcal{D}` are the weight function and reference
    domain of ``measure``, and ``nodes``/``weights`` are the :math:`x_i`/:math:`w_i`
    above.

    ``reference`` holds the rule's static, always-reference-domain payload
    (nodes, weights, measure, breakpoints, elements) and never changes;
    ``nodes``/``weights`` are properties that apply whatever mapping
    ``params`` currently holds (see ``map_to``).

    Parameters
    ----------
    reference : QuadratureReferenceData
        The rule's static, always-reference-domain data. Use
        :meth:`from_arrays` for the more convenient raw-array constructor.
    name : str
        Name identifying the rule.
    params : ReferenceDomain.Parameters, optional
        The currently-set mapping onto a target domain, as set by
        :meth:`map_to`. ``None`` (the default) means the reference domain.
    """

    reference: QuadratureReferenceData = tree.field(static=True)  # type: ignore[assignment]
    name: str = tree.field(static=True)  # type: ignore[assignment]
    params: ReferenceDomain.Parameters | None = None

    @classmethod
    def from_arrays(
        cls,
        nodes: np.ndarray,
        weights: np.ndarray,
        *,
        name: str,
        measure: Measure,
        breakpoints: np.ndarray | None = None,
        elements: np.ndarray | None = None,
    ) -> "QuadratureRule":
        """Build a rule from raw nodes/weights on the reference domain."""
        return cls(
            QuadratureReferenceData(nodes, weights, measure, breakpoints, elements),
            name,
        )

    # -- forwarding to the reference payload --

    @property
    def measure(self) -> Measure:
        """Weight function and reference domain the rule is defined on."""
        return self.reference.measure

    @property
    def breakpoints(self) -> np.ndarray | None:
        """Element boundaries, on the reference domain.
        
        *Always* reference domain, not affected by ``map_to``.
        """
        return self.reference.breakpoints

    @property
    def elements(self) -> np.ndarray | None:
        """Owning element per node."""
        return self.reference.elements

    def __len__(self) -> int:
        return len(self.reference.nodes)

    @property
    def ndim(self) -> int:
        """Dimension of the domain integrated over: always 1.

        Note this is the dimension of the *domain*, not of the ``nodes``
        array (which is 1-D here and N-D for a
        :class:`TensorQuadratureRule`).
        """
        return 1

    @property
    def measures(self) -> tuple[Measure, ...]:
        """This rule's single ``measure``, as a length-1 tuple."""
        return (self.measure,)

    def __eq__(self, other: object) -> bool:
        """Compare by value: same reference payload, name, and mapping."""
        if not isinstance(other, QuadratureRule):
            return NotImplemented
        return (
            self.reference == other.reference
            and self.name == other.name
            and _params_equal(self.params, other.params)
        )

    def __hash__(self) -> int:
        # Cheap, consistent with __eq__: equal rules agree on all of these.
        return hash((type(self), self.name, self.reference))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"measure={type(self.measure).__name__}, n={len(self)})"
        )

    # -- domain mapping --

    def map_to(self, *params, **kwparams) -> "QuadratureRule":
        """A new rule mapped onto the target domain.

        Given ``target = measure.domain.resolve_params(*params, **kwparams)``
        and ``(scale, shift) = measure.affine_params`` for those same
        arguments, ``nodes``/``weights`` on the returned rule become

        .. math::
            x_i = \\mathrm{scale} \\cdot t_i + \\mathrm{shift}, \\qquad
            \\tilde{w}_i = \\mathrm{scale} \\cdot w_i

        for reference node/weight :math:`t_i`/:math:`w_i`. Called with no
        arguments, maps onto the reference domain (the identity).

        The meaning of ``params``/``kwparams`` is specific to ``measure``:

        - Legendre/Jacobi: ``(a, b)`` bounds of the target interval.
        - Laguerre: ``rate`` (and optional ``start``) of the target
          exponential weight.
        - Hermite: ``loc``, ``scale`` of the target Gaussian-shaped weight.

        See the measure's ``affine_params`` docstring for details.

        Never composes with a prior ``map_to``; each call resolves fresh
        from ``(*params, **kwparams)`` against the reference domain, so
        ``rule.map_to(0, 1).map_to(2, 3)`` is exactly ``rule.map_to(2, 3)``,
        not a further mapping of ``[0, 1]``.

        Raises
        ------
        ValueError
            If called with any argument and ``measure.affine_invariant`` is
            ``False``; see ``Measure.affine_invariant``.
        """
        _check_affine_invariant(self.measure, params, kwparams)
        if not (params or kwparams):
            # `None` is the one canonical "identity" representation for
            # `params` -- not, say, `UnitInterval.Parameters(None, None)`,
            # which `resolve_params()` would otherwise return here and
            # which `_affine_params` couldn't cheaply distinguish from a
            # genuine (if degenerate) mapping without inspecting field
            # values, which isn't safe under symbolic tracing.
            return dataclasses.replace(self, params=None)
        new_params = self.measure.domain.resolve_params(*params, **kwparams)
        return dataclasses.replace(self, params=new_params)

    def _affine_params(self) -> tuple[float, float]:
        """``(scale, shift)`` for the currently-set domain."""
        if self.params is None:
            return 1.0, 0.0
        kwargs = {
            f.name: getattr(self.params, f.name)
            for f in tree.fields(cast(Any, self.params))
        }
        _check_affine_invariant(self.measure, (), kwargs)
        return self.measure.affine_params(**kwargs)

    @property
    def nodes(self) -> np.ndarray:
        """Nodes on the target domain."""
        scale, shift = self._affine_params()
        return scale * self.reference.nodes + shift

    @property
    def weights(self) -> np.ndarray:
        """Weights including the Jacobian of the domain mapping."""
        scale, _ = self._affine_params()
        return scale * self.reference.weights

    # -- integration --

    def integrate(
        self,
        f: Callable[..., np.ndarray],
        *,
        axis: int = -1,
        args: Sequence[Any] | None = None,
        density: bool = False,
    ) -> np.ndarray:
        """Approximate the weighted integral of ``f`` over the domain.

        .. math::
            \\int f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n \\tilde{w}_i
                f(x_i)

        where :math:`x_i` = ``nodes`` and :math:`\\tilde{w}_i` =
        ``weights``; see ``map_to`` to set the target domain first.

        Parameters
        ----------
        f : callable
            Integrand, called once as ``f(x, *args)`` on the full node array.
        axis : int, optional
            Axis holding the nodes in the output of ``f``. Default -1.
        args : tuple, optional
            Extra arguments passed to ``f`` after the node array.
        density : bool, optional
            If ``True``, normalize by the target measure's total mass, so
            the result approximates :math:`\\int f(x) \\, w(x) \\, dx /
            \\int w(x) \\, dx` -- e.g. an expectation under the
            corresponding probability density. See ``sum``. Default
            ``False``.

        Returns
        -------
        integral : ndarray
            Approximated integral. Shape (m,) for vector-valued
            integrands, or () for scalar integrands.
        """
        if args is None:
            args = ()
        fp = f(self.nodes, *args)
        return self.sum(fp, axis=axis, density=density)

    def sum(
        self,
        values: np.ndarray,
        *,
        axis: int = -1,
        density: bool = False,
    ) -> np.ndarray:
        """Quadrature applied to values already sampled at the nodes.

        .. math::
            \\sum_{i=1}^n \\tilde{w}_i \\, \\mathrm{values}_i

        where :math:`\\tilde{w}_i` = ``weights`` (see ``map_to`` to set the
        target domain first).

        Parameters
        ----------
        values : array_like
            Sampled values, with the quadrature nodes along ``axis``.
            Shape (n,) for scalar integrands or (m, n) for vector-valued
            integrands under the default ``axis=-1``.
        axis : int, optional
            Axis holding the nodes. Default -1 (nodes last), matching the
            natural output of a vectorized ``f``. Use ``axis=0`` for
            nodes-first data.
        density : bool, optional
            If ``True``, normalize the weights so they sum to 1 --
            equivalently divide by the target measure's total mass, since
            an ``n>=1``-point Gauss rule is exact for the constant
            integrand. Default ``False``.

        Returns
        -------
        integral : ndarray
            Approximated integral of the sampled values, with the
            quadrature nodes integrated out along ``axis``. Shape (m,) for
            vector-valued integrands, or () for scalar integrands.

        Raises
        ------
        ValueError
            If ``values`` has more than 2 dimensions, or if
            ``values.shape[axis]`` does not match the number of quadrature
            nodes.
        """
        w = self.weights
        if density:
            w = w / np.sum(w)
        return _weighted_sum(w, values, axis, len(self))


def composite_quad(
    base: QuadratureRule | Sequence[QuadratureRule], breakpoints: np.ndarray
) -> QuadratureRule:
    """Tile ``base`` across elements of its reference domain.

    Partitions ``base.measure.support`` at ``breakpoints`` and applies
    ``base``, affinely rescaled, to each element, concatenating the resulting
    nodes and weights. The result is itself a ``QuadratureRule`` on the same
    reference domain -- its nodes are just clustered at the element
    boundaries rather than spread uniformly -- so it can be mapped onto a
    target domain/measure via ``map_to``/``integrate`` exactly like any
    other rule of ``base.measure``. This works because
    ``measure.affine_params`` maps affinely, and affine maps commute with
    subdivision: rescaling the whole composite pattern onto ``[a, b]`` is
    the same as building the elements directly on the rescaled sub-intervals
    of ``[a, b]``.

    ``base`` may instead be a sequence of rules, one per element, letting
    each element carry its own order (or even its own family) -- e.g.
    ``composite_quad([gauss_legendre(2), gauss_legendre(4)], breakpoints)``
    for two elements of different degree. A single ``base`` rule is exactly
    equivalent to passing that same rule ``len(breakpoints) - 1`` times.

    Only defined for families whose reference weight is uniform (see
    ``Measure.uniform_weight``) -- otherwise each interior element
    boundary would pick up a spurious copy of the weight's shape, which is
    only meaningful at the true endpoints of the reference domain.

    Parameters
    ----------
    base : QuadratureRule or sequence of QuadratureRule
        Rule (or per-element rules) to tile across elements. Every rule's
        ``measure.uniform_weight`` must be ``True``, and (since the result
        has a single ``measure`` field) every rule must share the same
        ``measure``. A sequence must have exactly ``len(breakpoints) - 1``
        entries, one per element.
    breakpoints : array_like
        Element boundaries, shape ``(k + 1,)`` for ``k`` elements. Must be
        strictly increasing and span ``base.measure.support``
        exactly (first/last entries equal to its endpoints).

    Returns
    -------
    rule : QuadratureRule
        Composite rule on the same reference domain as ``base``, with
        ``sum(len(r) for r in rules)`` nodes (``k * len(base)`` when ``base``
        is a single rule). ``name`` is the common rule name if every
        per-element rule shares one, else the generic ``"composite"``.

    Raises
    ------
    ValueError
        If any rule's ``measure.uniform_weight`` is ``False``, if the rules
        do not all share the same ``measure``, if a sequence of rules does
        not have one entry per element, if ``breakpoints`` has fewer than 2
        entries or is not strictly increasing, or if it does not span
        ``base.measure.support`` exactly.
    """
    breakpoints = np.asarray(breakpoints, dtype=float)
    if breakpoints.ndim != 1 or len(breakpoints) < 2:
        raise ValueError(
            f"breakpoints must be 1-D with at least 2 entries, got shape "
            f"{breakpoints.shape}"
        )
    n_elements = len(breakpoints) - 1

    if isinstance(base, QuadratureRule):
        rules = [base] * n_elements
    else:
        rules = list(base)
        if len(rules) != n_elements:
            raise ValueError(
                f"breakpoints describe {n_elements} elements, so `base` must "
                f"supply exactly {n_elements} rules, got {len(rules)}"
            )

    measure = rules[0].measure
    if not measure.uniform_weight:
        raise ValueError(
            f"composite quadrature requires a measure with a uniform "
            f"reference weight, got {type(measure).__name__}"
        )
    if any(rule.measure != measure for rule in rules):
        raise ValueError(
            "composite quadrature requires every per-element rule to share "
            "the same measure, so the result has one well-defined weight"
        )

    if np.any(np.diff(breakpoints) <= 0):
        raise ValueError("breakpoints must be strictly increasing")
    lo, hi = measure.support
    if breakpoints[0] != lo or breakpoints[-1] != hi:
        raise ValueError(
            f"breakpoints must span the reference domain {(lo, hi)}, got "
            f"({breakpoints[0]}, {breakpoints[-1]})"
        )

    nodes = []
    weights = []
    elements = []
    for e, (rule, t0, t1) in enumerate(zip(rules, breakpoints[:-1], breakpoints[1:])):
        mapped = rule.map_to(t0, t1)
        nodes.append(mapped.nodes)
        weights.append(mapped.weights)
        # A node on an element boundary belongs to the element it was
        # generated for, which its coordinate alone cannot say.
        elements.append(np.full(len(rule), e, dtype=int))

    names = {rule.name for rule in rules}
    name = names.pop() if len(names) == 1 else "composite"

    return QuadratureRule.from_arrays(
        np.concatenate(nodes),
        np.concatenate(weights),
        measure=measure,
        name=name,
        breakpoints=breakpoints,
        elements=np.concatenate(elements),
    )
