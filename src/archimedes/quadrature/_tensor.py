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

__all__ = ["TensorQuadratureRule", "tensor"]


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
    ``ndim > 1``: :meth:`scaled_points` returns an ``(n, ndim)`` array (one
    row per node, one column per dimension) rather than the ``(n,)`` of a
    one-dimensional rule, while :meth:`scaled_weights` stays ``(n,)`` since
    each node still carries a single scalar weight.

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
    tensor : Constructor taking the per-dimension rules variadically.
    composite : Tile a rule across sub-elements *within* one dimension;
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
    # source of truth for the generated __eq__/__hash__

    @functools.cached_property
    def nodes(self) -> np.ndarray:
        """Reference nodes, shape ``(n, ndim)``.

        Ordered with the *first* dimension varying slowest (C order, i.e.
        ``np.meshgrid(..., indexing="ij")``).
        """
        grids = np.meshgrid(*[rule.nodes for rule in self.rules], indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=-1)

    @functools.cached_property
    def weights(self) -> np.ndarray:
        """Reference weights, shape ``(n,)``: the product of the
        per-dimension weights, in the same order as :attr:`nodes`."""
        w = self.rules[0].weights
        for rule in self.rules[1:]:
            w = np.outer(w, rule.weights).ravel()
        return w

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

    @property
    def breakpoints(self) -> tuple[np.ndarray | None, ...]:
        """Per-dimension element boundaries, one entry per dimension, each
        ``None`` unless that dimension's rule is composite.

        Always a tuple of length ``ndim`` (never a bare ``None``), so
        consumers checking alignment against a piecewise basis can zip it
        against the basis's own per-dimension breakpoints.
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

        Mirrors :meth:`QuadratureRule.scaled_points`' free-form signature,
        except that a tensor rule takes a *single* sequence of per-dimension
        parameters (positionally or as ``dims=``) rather than one measure's
        arguments spread out -- the per-dimension arguments would otherwise
        be ambiguous, and their names collide across dimensions.
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

    def scaled_points(self, *params: Any, **kwparams: Any) -> np.ndarray:
        """Nodes mapped onto the target domain, shape ``(n, ndim)``.

        Each dimension is mapped by its own measure's ``affine_params``,
        applied to the corresponding column. Symbolic if any parameter is
        symbolic; the underlying reference nodes are always static.

        Parameters
        ----------
        dims : sequence, optional
            One entry per dimension, positionally or as ``dims=``. Each
            entry is that dimension's domain parameters in any of the forms
            accepted by its measure: ``None`` for the reference domain, a
            ``ReferenceDomain.Parameters`` struct, a tuple of positional
            arguments (``(a, b)`` for an interval), or a dict of keyword
            arguments. Omitted entirely, every dimension uses its reference
            domain.

        Raises
        ------
        TypeError
            If called with anything other than a single ``dims`` sequence,
            or if an entry is not one of the accepted forms.
        ValueError
            If ``dims`` does not have exactly ``ndim`` entries.
        """
        dims = self._dims(params, kwparams)
        columns = []
        for i, (rule, spec) in enumerate(zip(self.rules, dims)):
            args, kwargs = _dim_args(spec)
            scale, shift = rule.measure.affine_params(*args, **kwargs)
            columns.append(scale * self.nodes[:, i] + shift)
        return np.stack(columns, axis=-1)  # type: ignore[no-any-return]

    def scaled_weights(
        self, *params: Any, density: bool = False, **kwparams: Any
    ) -> np.ndarray:
        """Weights including the Jacobian of the map onto the target box,
        shape ``(n,)``.

        The Jacobian of a product of affine maps is the product of their
        scales, so this is :math:`\\bigl(\\prod_k \\mathrm{scale}_k\\bigr)`
        times the reference weights.

        Parameters
        ----------
        dims : sequence, optional
            Per-dimension domain parameters; see :meth:`scaled_points`.
        density : bool, optional
            If ``True``, divide by the total mass of the mapped product
            measure -- the product of the per-dimension masses -- so the
            weights sum to 1. For a Wiener-Askey correspondence this makes
            them the quadrature weights of the joint *probability* density
            of independent inputs. Default ``False``.
        """
        dims = self._dims(params, kwparams)
        factor: Any = 1.0
        for rule, spec in zip(self.rules, dims):
            args, kwargs = _dim_args(spec)
            scale, _ = rule.measure.affine_params(*args, **kwargs)
            factor = factor * scale
            if density:
                factor = factor / rule.measure.mass(*args, **kwargs)
        return factor * self.weights  # type: ignore[no-any-return]

    # -- integration --

    def integrate(
        self,
        f: Callable[..., np.ndarray],
        *params: Any,
        axis: int = 0,
        args: Sequence[Any] | None = None,
        density: bool = False,
        **kwparams: Any,
    ) -> np.ndarray:
        """Approximate the weighted integral of ``f`` over the target box.

        Parameters
        ----------
        f : callable
            Integrand, called once as ``f(x, *args)`` with the full
            ``(n, ndim)`` node array -- so it must be vectorized over rows,
            and must index its own arguments out of the columns (e.g.
            ``lambda x: g(x[:, 0], x[:, 1])``). This is the same convention
            as ``FunctionSpace.project``.
        *params, **kwparams
            Per-dimension domain parameters; see :meth:`scaled_points`.
        axis : int, optional
            Axis of ``f``'s output holding the nodes. Default 0, matching
            the nodes-first ``(n, ndim)`` input (unlike the one-dimensional
            :meth:`QuadratureRule.integrate`, whose default is -1).
        args : tuple, optional
            Extra arguments passed to ``f`` after the node array.
        density : bool, optional
            Normalize by the total mass; see :meth:`scaled_weights`.

        Returns
        -------
        integral : ndarray
            Shape ``(m,)`` for vector-valued integrands, or ``()`` for
            scalar ones.
        """
        if args is None:
            args = ()
        fp = f(self.scaled_points(*params, **kwparams), *args)
        return self.sum(fp, *params, axis=axis, density=density, **kwparams)

    def sum(
        self,
        values: np.ndarray,
        *params: Any,
        axis: int = 0,
        density: bool = False,
        **kwparams: Any,
    ) -> np.ndarray:
        """Quadrature applied to values already sampled at the nodes.

        Parameters
        ----------
        values : array_like
            Sampled values with the nodes along ``axis``: shape ``(n,)`` for
            a scalar integrand or ``(n, m)`` for a vector-valued one under
            the default ``axis=0``.
        *params, **kwparams
            Per-dimension domain parameters; see :meth:`scaled_points`.
        axis : int, optional
            Axis holding the nodes. Default 0.
        density : bool, optional
            Normalize by the total mass; see :meth:`scaled_weights`.

        Raises
        ------
        ValueError
            If ``values`` has more than 2 dimensions, or if
            ``values.shape[axis]`` does not match the number of nodes.
        """
        w = self.scaled_weights(*params, density=density, **kwparams)
        return _weighted_sum(w, values, axis, len(self))


def tensor(*rules: QuadratureRule) -> TensorQuadratureRule:
    """Build a :class:`TensorQuadratureRule` from one rule per dimension.

    .. code-block:: python

        from archimedes.quadrature import gauss_hermite, gauss_legendre, tensor

        # Two Gaussian inputs and one uniformly-bounded one
        rule = tensor(
            gauss_hermite(4, kind="prob"),
            gauss_hermite(4, kind="prob"),
            gauss_legendre(4),
        )

    To integrate over a rectilinear mesh rather than a single box, apply
    :func:`composite` per dimension first -- ``tensor(composite(gl, bp_x),
    composite(gl, bp_y))``.

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
