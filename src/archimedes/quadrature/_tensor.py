"""Tensor-product quadrature over several dimensions.

A tensor-product rule integrates over a *box* :math:`\\mathcal{D}_1 \\times
\\cdots \\times \\mathcal{D}_d` by taking the Cartesian product of one
one-dimensional rule per dimension:

.. math::
    \\int f(x) \\, w(x) \\, dx \\approx \\sum_{i_1} \\cdots \\sum_{i_d}
        w^{(1)}_{i_1} \\cdots w^{(d)}_{i_d} \\,
        f\\bigl(x^{(1)}_{i_1}, \\ldots, x^{(d)}_{i_d}\\bigr)

Each dimension keeps its own :class:`~archimedes.measure.Measure`, so the
weight is the product :math:`w(x) = \\prod_k w_k(x_k)` and the dimensions
need not agree: a Hermite weight on one axis and a Legendre weight on
another is the natural rule for a problem mixing Gaussian uncertainties
with uniformly-bounded parameters.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Sequence, cast

import numpy as np

from archimedes import tree
from archimedes.measure import Measure, ReferenceDomain

from ._quadrature_rule import QuadratureRule, _weighted_sum

__all__ = ["TensorQuadratureRule", "tensor_quad"]


def _dim_args(spec: Any) -> tuple[tuple, dict]:
    """Normalize one dimension's domain parameters into ``(args, kwargs)``
    for that dimension's ``measure.affine_params``.

    Accepts the reference domain (``None``), a ``ReferenceDomain.Parameters``
    struct (what a ``FunctionSpace`` domain is made of), a positional tuple
    (``(a, b)``), or an explicit kwargs dict.
    """
    if spec is None:
        return (), {}
    if isinstance(spec, ReferenceDomain.Parameters):
        # `ReferenceDomain.Parameters` is a plain marker base; every concrete
        # subclass is a `@tree.struct`, which `tree.fields` requires.
        fields = tree.fields(cast(Any, spec))
        return (), {f.name: getattr(spec, f.name) for f in fields}
    if isinstance(spec, dict):
        return (), dict(spec)
    if isinstance(spec, (tuple, list)):
        return tuple(spec), {}
    raise TypeError(
        f"per-dimension parameters must be None, a ReferenceDomain.Parameters, "
        f"a tuple of positional arguments, or a dict of keyword arguments; "
        f"got {type(spec).__name__}"
    )


@dataclasses.dataclass(frozen=True)
class TensorQuadratureRule:
    """Cartesian product of one-dimensional quadrature rules.

    Implements the :class:`~archimedes.quadrature.Quadrature` interface with
    ``ndim > 1``: :attr:`nodes` is an ``(n, ndim)`` array (one row per node,
    one column per dimension) rather than the ``(n,)`` of a one-dimensional
    rule, while :attr:`weights` stays ``(n,)`` since each node still carries
    a single scalar weight.

    Exactness follows dimension by dimension. If rule ``k`` is exact through
    degree :math:`p_k` in its own variable, the product is exact for any
    polynomial whose degree in :math:`x_k` is at most :math:`p_k` - i.e.
    exact on the *tensor-product* (maximum-degree) polynomial space, which
    strictly contains the total-degree space usually truncated to.

    The node count is :math:`\\prod_k n_k`, exponential in ``ndim``; sparse
    (Smolyak) quadrature is planned as a future alternative.

    Parameters
    ----------
    rules : tuple of QuadratureRule
        One rule per dimension, in order. Their measures may differ. Passing
        a single rule is allowed and gives a valid ``ndim == 1`` tensor rule.

    See Also
    --------
    tensor_quad : Constructor taking the per-dimension rules variadically.
    composite_quad : Tile a rule across sub-elements *within* one dimension;
        apply it per-dimension before tensoring to get a rectilinear mesh.
    """

    rules: tuple[QuadratureRule, ...]

    def __post_init__(self):
        rules = tuple(self.rules)
        if len(rules) < 1:
            raise ValueError("a tensor rule needs at least one dimension")
        for i, rule in enumerate(rules):
            if not isinstance(rule, QuadratureRule):
                raise TypeError(
                    f"rules[{i}] must be a QuadratureRule, got {type(rule).__name__}"
                )
        object.__setattr__(self, "rules", rules)

    # -- derived node/weight arrays --
    #
    # Cached rather than stored as fields so that `rules` stays the single
    # source of truth for the generated __eq__/__hash__. `_reference_*` are
    # built from each dimension's `reference` payload (never affected by
    # `map_to`); the public `nodes`/`weights` apply each dimension's
    # currently-set mapping (see `QuadratureRule.map_to`) on top.

    @functools.cached_property
    def _reference_nodes(self) -> np.ndarray:
        """Reference nodes, shape ``(n, ndim)``.

        Ordered with the *first* dimension varying slowest (C order, i.e.
        ``np.meshgrid(..., indexing="ij")``).
        """
        grids = np.meshgrid(
            *[rule.reference.nodes for rule in self.rules], indexing="ij"
        )
        return np.stack([g.ravel() for g in grids], axis=-1)

    @functools.cached_property
    def _reference_weights(self) -> np.ndarray:
        """Reference weights, shape ``(n,)``: the product of the
        per-dimension reference weights, in the same order as
        :attr:`_reference_nodes`."""
        w = self.rules[0].reference.weights
        for rule in self.rules[1:]:
            w = np.outer(w, rule.reference.weights).ravel()
        return w

    @functools.cached_property
    def nodes(self) -> np.ndarray:
        """Nodes on the target domain, shape ``(n, ndim)``.

        Ordered with the *first* dimension varying slowest (C order, i.e.
        ``np.meshgrid(..., indexing="ij")``).
        """
        columns = []
        for i, rule in enumerate(self.rules):
            scale, shift = rule._affine_params()
            columns.append(scale * self._reference_nodes[:, i] + shift)
        return np.stack(columns, axis=-1)  # type: ignore[no-any-return]

    @functools.cached_property
    def weights(self) -> np.ndarray:
        """Weights including the Jacobian of the target domain, shape ``(n,)``.

        The Jacobian of a product of affine maps is the product of their
        scales, so this is :math:`\\bigl(\\prod_k \\mathrm{scale}_k\\bigr)`
        times the reference weights.
        """
        factor: Any = 1.0
        for rule in self.rules:
            scale, _ = rule._affine_params()
            factor = factor * scale
        return factor * self._reference_weights  # type: ignore[no-any-return]

    # -- Quadrature interface --

    @property
    def ndim(self) -> int:
        """Number of dimensions, i.e. the number of rules tensored."""
        return len(self.rules)

    @property
    def measures(self) -> tuple[Measure, ...]:
        """The per-dimension measures. There is no single ``measure``: the
        dimensions are independent and may use different families."""
        return tuple(rule.measure for rule in self.rules)

    @functools.cached_property
    def elements(self) -> np.ndarray | None:
        """Owning sub-element per node and dimension, shape ``(n, ndim)``,
        or ``None`` if no dimension has element structure.

        Mirrors :attr:`nodes`: column ``d`` indexes dimension ``d``'s
        elements, and is all-zero for a dimension whose rule is not
        composite.
        """
        if all(rule.elements is None for rule in self.rules):
            return None
        # Same Cartesian expansion as `nodes`, so the two stay row-aligned.
        per_dim = [
            np.zeros(len(rule), dtype=int) if rule.elements is None else rule.elements
            for rule in self.rules
        ]
        grids = np.meshgrid(*per_dim, indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=-1)

    @property
    def breakpoints(self) -> tuple[np.ndarray | None, ...]:
        """Per-dimension element boundaries, one entry per dimension, each
        ``None`` unless that dimension's rule is composite.
        """
        return tuple(rule.breakpoints for rule in self.rules)

    def __len__(self) -> int:
        return int(np.prod([len(rule) for rule in self.rules]))

    def __repr__(self) -> str:
        inner = ", ".join(f"{type(r.measure).__name__}(n={len(r)})" for r in self.rules)
        return f"{type(self).__name__}(ndim={self.ndim}, n={len(self)}, [{inner}])"

    # -- domain mapping --

    def _dims(self, params: tuple, kwparams: dict) -> tuple:
        """Resolve the per-dimension parameter specs from the call.

        Mirrors :meth:`QuadratureRule.map_to`'s free-form signature, except
        that a tensor rule takes a *single* sequence of per-dimension
        parameters (positionally or as ``dims=``).
        """
        if kwparams:
            if params or set(kwparams) != {"dims"}:
                raise TypeError(
                    f"{type(self).__name__} takes the per-dimension domain "
                    f"parameters as a single sequence, positionally or as "
                    f"`dims=`; got args={params!r}, kwargs={sorted(kwparams)}"
                )
            dims = kwparams["dims"]
        elif len(params) > 1:
            raise TypeError(
                f"{type(self).__name__} takes the per-dimension domain "
                f"parameters as a single sequence, not {len(params)} "
                f"positional arguments"
            )
        elif params:
            dims = params[0]
        else:
            dims = None

        if dims is None:
            return (None,) * self.ndim
        dims = tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError(
                f"expected {self.ndim} per-dimension parameters, got {len(dims)}"
            )
        return dims

    def map_to(self, *params: Any, **kwparams: Any) -> "TensorQuadratureRule":
        """A new tensor rule with every dimension mapped onto its target domain.

        See :meth:`QuadratureRule.map_to`. Each dimension is delegated to
        independently.

        Parameters
        ----------
        dims : sequence, optional
            One entry per dimension, positionally or as ``dims=``. Each
            entry is that dimension's domain parameters in any of the forms
            accepted by its measure: ``None`` for the reference domain, a
            ``ReferenceDomain.Parameters`` struct, a tuple of positional
            arguments (``(a, b)`` for an interval), or a dict of keyword
            arguments. Omitted entirely, every dimension is mapped onto its
            reference domain.

        Raises
        ------
        TypeError
            If called with anything other than a single ``dims`` sequence,
            or if an entry is not one of the accepted forms.
        ValueError
            If ``dims`` does not have exactly ``ndim`` entries.
        """
        dims = self._dims(params, kwparams)
        mapped = tuple(
            rule.map_to(*args, **kwargs)
            for rule, spec in zip(self.rules, dims)
            for args, kwargs in [_dim_args(spec)]
        )
        return dataclasses.replace(self, rules=mapped)

    # -- integration --

    def integrate(
        self,
        f: Callable[..., np.ndarray],
        *,
        axis: int = 0,
        args: Sequence[Any] | None = None,
        density: bool = False,
    ) -> np.ndarray:
        """Approximate the weighted integral of ``f`` over the target domain.

        Parameters
        ----------
        f : callable
            Integrand, called once as ``f(x, *args)`` with the full
            ``(n, ndim)`` node array -- so it must be vectorized over rows,
            and must index its own arguments out of the columns (e.g.
            ``lambda x: g(x[:, 0], x[:, 1])``). This is the same convention
            as ``FunctionSpace.project``.
        axis : int, optional
            Axis of ``f``'s output holding the nodes. Default 0, matching
            the nodes-first ``(n, ndim)`` input (unlike the one-dimensional
            :meth:`QuadratureRule.integrate`, whose default is -1).
        args : tuple, optional
            Extra arguments passed to ``f`` after the node array.
        density : bool, optional
            Normalize by the total mass; see :meth:`sum`.

        Returns
        -------
        integral : ndarray
            Shape ``(m,)`` for vector-valued integrands, or ``()`` for
            scalar ones.
        """
        if args is None:
            args = ()
        fp = f(self.nodes, *args)
        return self.sum(fp, axis=axis, density=density)

    def sum(
        self,
        values: np.ndarray,
        *,
        axis: int = 0,
        density: bool = False,
    ) -> np.ndarray:
        """Quadrature applied to values already sampled at the nodes.

        Parameters
        ----------
        values : array_like
            Sampled values with the nodes along ``axis``: shape ``(n,)`` for
            a scalar integrand or ``(n, m)`` for a vector-valued one under
            the default ``axis=0``.
        axis : int, optional
            Axis holding the nodes. Default 0.
        density : bool, optional
            If ``True``, normalize the weights so they sum to 1 --
            equivalently divide by the total mass of the mapped product
            measure, the product of the per-dimension masses. For a
            Wiener-Askey correspondence this makes them the quadrature
            weights of the joint *probability* density of independent
            inputs. Default ``False``.

        Raises
        ------
        ValueError
            If ``values`` has more than 2 dimensions, or if
            ``values.shape[axis]`` does not match the number of nodes.
        """
        w = self.weights
        if density:
            w = w / np.sum(w)
        return _weighted_sum(w, values, axis, len(self))


def tensor_quad(*rules: QuadratureRule) -> TensorQuadratureRule:
    """Build a :class:`TensorQuadratureRule` from one rule per dimension.

    .. code-block:: python

        from archimedes.quadrature import gauss_hermite, gauss_legendre, tensor_quad

        # Two Gaussian inputs and one uniformly-bounded one
        rule = tensor_quad(
            gauss_hermite(4),
            gauss_hermite(4),
            gauss_legendre(4),
        )

    To integrate over a rectilinear mesh rather than a single box, apply
    :func:`composite_quad` per dimension first -- ``tensor_quad(composite_quad(gl,
    bp_x), composite_quad(gl, bp_y))``.

    Parameters
    ----------
    *rules : QuadratureRule
        One rule per dimension, in order. Measures may differ between
        dimensions.

    Returns
    -------
    rule : TensorQuadratureRule
        Product rule with ``prod(len(r) for r in rules)`` nodes.

    Raises
    ------
    ValueError
        If no rules are given.
    TypeError
        If any argument is not a :class:`QuadratureRule`.
    """
    return TensorQuadratureRule(rules)
