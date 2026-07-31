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
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from archimedes.measure import Measure

__all__ = [
    "Quadrature",
    "QuadratureRule",
    "composite",
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
        rule places nodes on its element boundaries -- a Lobatto sub-rule
        puts one there from each side, so the boundary appears twice in
        ``nodes`` -- and a basis that is discontinuous there needs to know
        which element each copy belongs to. Locating by coordinate instead
        assigns both copies to the same element and silently mis-integrates.
        """

    @property
    def measures(self) -> tuple[Measure, ...]:
        """The measure integrated against in each dimension, always a tuple
        of length ``ndim``."""

    def __len__(self) -> int:
        """Total number of quadrature nodes."""

    def scaled_points(self, *params: Any, **kwparams: Any) -> np.ndarray:
        """Nodes mapped onto the target domain, shape ``(n,)`` for a
        one-dimensional rule or ``(n, ndim)`` otherwise. The meaning of
        ``params``/``kwparams`` is specific to the implementation."""

    def scaled_weights(
        self, *params: Any, density: bool = False, **kwparams: Any
    ) -> np.ndarray:
        """Weights including the Jacobian of the map onto the target domain,
        shape ``(n,)``. Normalized to unit total mass if ``density``."""


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


# Note: dataclass, not struct, because all the data is static
@dataclasses.dataclass(frozen=True)
class QuadratureRule:
    """Fixed-node Gauss quadrature rule on a reference domain.

    Approximates the weighted integral

    .. math::
        \\int_\\mathcal{D} f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n w_i f(x_i)

    where :math:`w` and :math:`\\mathcal{D}` are the weight function and reference
    domain of ``measure``, and ``nodes``/``weights`` are the :math:`x_i`/:math:`w_i`
    above.

    Nodes and weights are always static (NumPy) arrays. Mapping onto a
    target domain/measure is an affine transform of the reference nodes,
    whose parameters are specific to ``measure`` -- see ``scaled_points``.

    Parameters
    ----------
    nodes : array_like
        Quadrature nodes :math:`x_i`, shape ``(n,)``, on
        ``measure.support``.
    weights : array_like
        Quadrature weights :math:`w_i`, shape ``(n,)``.
    name : str
        Name identifying the rule.
    measure : Measure
        Weight function and reference domain the rule is defined on.
    breakpoints : array_like, optional
        For a composite rule (see :func:`composite`), the element boundaries
        it was tiled across, on ``measure.support``; ``None`` for a plain
        rule.
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

    nodes: np.ndarray  # shape (n,), on `measure.support`
    weights: np.ndarray  # shape (n,)
    name: str  # name for the rule
    measure: Measure
    breakpoints: np.ndarray | None = None  # element boundaries, if composite
    elements: np.ndarray | None = None  # owning element per node, if composite

    def __post_init__(self):
        # Static data, safe to unconditionally convert to NumPy arrays
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

    def __len__(self) -> int:
        return len(self.nodes)

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
        """Compare by value, elementwise on ``nodes``/``weights``.

        Defined explicitly because the ``@dataclass``-generated ``__eq__``
        would compare the array fields with ``==``, yielding an array and
        raising "truth value of an array is ambiguous" for any rule with
        more than one node. (``@dataclass`` leaves an explicitly-defined
        ``__eq__`` alone.)
        """
        if not isinstance(other, QuadratureRule):
            return NotImplemented
        return (
            self.name == other.name
            and self.measure == other.measure
            and np.array_equal(self.nodes, other.nodes)
            and np.array_equal(self.weights, other.weights)
            and _breakpoints_equal(self.breakpoints, other.breakpoints)
            and _breakpoints_equal(self.elements, other.elements)
        )

    def __hash__(self) -> int:
        # Cheap, consistent with __eq__: equal rules agree on all of these.
        return hash((type(self), self.name, self.measure, len(self)))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"measure={type(self.measure).__name__}, n={len(self)})"
        )

    # -- domain mapping --

    def scaled_points(self, *params, **kwparams):
        """Nodes mapped by ``measure``'s affine parameters.

        Given ``(scale, shift) = measure.affine_params(*params, **kwparams)``,
        the mapped nodes are

        .. math::
            x_i = \\mathrm{scale} \\cdot t_i + \\mathrm{shift}

        for reference node :math:`t_i`. Called with no arguments, returns
        the reference ``nodes`` unchanged.

        The meaning of ``params``/``kwparams`` is specific to ``measure``:

        - Legendre/Jacobi: ``(a, b)`` bounds of the target interval.
        - Laguerre: ``rate`` (and optional ``start``) of the target
          exponential weight.
        - Hermite: ``mean``, ``std`` of the target Gaussian-shaped weight.

        See the measure's ``affine_params`` docstring for details. Symbolic
        if any parameter is symbolic; the underlying nodes are static.
        """
        scale, shift = self.measure.affine_params(*params, **kwparams)
        return scale * self.nodes + shift

    def scaled_weights(self, *params, density: bool = False, **kwparams):
        """Weights including the Jacobian factor for the target
        domain/measure.

        .. math::
            \\tilde{w}_i = \\mathrm{scale} \\cdot w_i

        where :math:`\\mathrm{scale}` is the same affine scale used by
        ``scaled_points``. See ``scaled_points`` for the meaning of
        ``params``/``kwparams``.

        Parameters
        ----------
        density : bool, optional
            If ``True``, additionally divide by the target measure's total
            mass (``measure.mass(*params, **kwparams)``), so the returned
            weights sum to 1 -- i.e. they act as quadrature weights for the
            *normalized* density rather than the raw weight function.
            Default ``False``.
        """
        scale, _ = self.measure.affine_params(*params, **kwparams)
        w = scale * self.weights
        if density:
            w = w / self.measure.mass(*params, **kwparams)
        return w

    # -- integration --

    def integrate(
        self,
        f: Callable[..., np.ndarray],
        *params,
        axis: int = -1,
        args: Sequence[Any] | None = None,
        density: bool = False,
        **kwparams,
    ) -> np.ndarray:
        """Approximate the weighted integral of ``f``.

        .. math::
            \\int f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n \\tilde{w}_i
                f(x_i)

        where :math:`x_i` = ``scaled_points(*params, **kwparams)`` and
        :math:`\\tilde{w}_i` = ``scaled_weights(*params, **kwparams)``.

        Parameters
        ----------
        f : callable
            Integrand, called once as ``f(x, *args)`` on the full node
            array. Must be vectorized, returning values with the nodes
            along ``axis``. If any of ``params``/``kwparams`` is symbolic, ``f``
            must be symbolically traceable.
        *params, **kwparams
            Target domain/measure parameters; see ``scaled_points`` for
            their meaning.
        axis : int, optional
            Axis holding the nodes in the output of ``f``. Default -1.
        args : tuple, optional
            Extra arguments passed to ``f`` after the node array.
        density : bool, optional
            If ``True``, normalize by the target measure's total mass, so
            the result approximates :math:`\\int f(x) \\, w(x) \\, dx /
            \\int w(x) \\, dx` -- e.g. an expectation under the
            corresponding probability density. See ``scaled_weights``.
            Default ``False``.

        Returns
        -------
        integral : ndarray
            Approximated integral. Shape (m,) for vector-valued
            integrands, or () for scalar integrands.
        """
        if args is None:
            args = ()
        fp = f(self.scaled_points(*params, **kwparams), *args)
        return self.sum(fp, *params, axis=axis, density=density, **kwparams)

    def sum(
        self,
        values: np.ndarray,
        *params,
        axis: int = -1,
        density: bool = False,
        **kwparams,
    ) -> np.ndarray:
        """Quadrature applied to values already sampled at the nodes.

        .. math::
            \\sum_{i=1}^n \\tilde{w}_i \\, \\mathrm{values}_i

        where :math:`\\tilde{w}_i` = ``scaled_weights(*params, **kwparams)``.

        Parameters
        ----------
        values : array_like
            Sampled values, with the quadrature nodes along ``axis``.
            Shape (n,) for scalar integrands or (m, n) for vector-valued
            integrands under the default ``axis=-1``.
        *params, **kwparams
            Target domain/measure parameters, forwarded to
            ``measure.affine_params``; see ``scaled_points`` for their meaning.
        axis : int, optional
            Axis holding the nodes. Default -1 (nodes last), matching the
            natural output of a vectorized ``f``. Use ``axis=0`` for
            nodes-first data.
        density : bool, optional
            Forwarded to ``scaled_weights``; see its docstring. Default
            ``False``.

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
        w = self.scaled_weights(*params, density=density, **kwparams)
        return _weighted_sum(w, values, axis, len(self))


def composite(
    base: QuadratureRule | Sequence[QuadratureRule], breakpoints: np.ndarray
) -> QuadratureRule:
    """Tile ``base`` across elements of its reference domain.

    Partitions ``base.measure.support`` at ``breakpoints`` and applies
    ``base``, affinely rescaled, to each element, concatenating the resulting
    nodes and weights. The result is itself a ``QuadratureRule`` on the same
    reference domain -- its nodes are just clustered at the element
    boundaries rather than spread uniformly -- so it can be mapped onto a
    target domain/measure via ``scaled_points``/``scaled_weights``/``integrate``
    exactly like any other rule of ``base.measure``. This works because
    ``measure.affine_params`` maps affinely, and affine maps commute with
    subdivision: rescaling the whole composite pattern onto ``[a, b]`` is
    the same as building the elements directly on the rescaled sub-intervals
    of ``[a, b]``.

    ``base`` may instead be a sequence of rules, one per element, letting
    each element carry its own order (or even its own family) -- e.g.
    ``composite([gauss_legendre(2), gauss_legendre(4)], breakpoints)`` for
    two elements of different degree. A single ``base`` rule is exactly
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
        nodes.append(rule.scaled_points(t0, t1))
        weights.append(rule.scaled_weights(t0, t1))
        # A node on an element boundary belongs to the element it was
        # generated for, which its coordinate alone cannot say.
        elements.append(np.full(len(rule), e, dtype=int))

    names = {rule.name for rule in rules}
    name = names.pop() if len(names) == 1 else "composite"

    return QuadratureRule(
        np.concatenate(nodes),
        np.concatenate(weights),
        measure=measure,
        name=name,
        breakpoints=breakpoints,
        elements=np.concatenate(elements),
    )
