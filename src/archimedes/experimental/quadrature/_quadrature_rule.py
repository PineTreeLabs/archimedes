"""Fixed-node Gauss quadrature rules for classical orthogonal polynomial weights.

Gauss quadrature approximates a weighted integral

.. math::
    \\int_\\mathcal{D} f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n w_i f(x_i)

exactly for every polynomial `f` of degree :math:`\\leq 2n - 1`, where the
nodes :math:`x_i` are the roots of the degree-`n` polynomial orthogonal with
respect to the weight `w` on the reference domain `D`. Each classical
a "weight/domain pair" (Legendre, Jacobi, Laguerre, Hermite) is represented by
a "measure"; :class:`QuadratureRule` pairs a measure with a
fixed set of nodes and weights on its reference domain.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

import numpy as np
from scipy.special import roots_jacobi, roots_legendre

from archimedes.experimental.polynomial.orthogonal import Measure

__all__ = [
    "QuadratureRule",
    "composite",
]


# Note: dataclass, not struct, because all the data is static
@dataclasses.dataclass(frozen=True)
class QuadratureRule:
    """Fixed-node Gauss quadrature rule on a reference domain.

    Approximates the weighted integral

    .. math::
        \\int_\\mathcal{D} f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n w_i f(x_i)

    where :math:`w` and :math:`\\mathcal{D}` are the weight function and reference
    domain of `measure`, and `nodes`/`weights` are the :math:`x_i`/:math:`w_i`
    above.

    Nodes and weights are always static (NumPy) arrays. Mapping onto a
    target domain/measure is an affine transform of the reference nodes,
    whose parameters are specific to `measure` -- see `scaled_points`.

    Parameters
    ----------
    nodes : array_like
        Quadrature nodes :math:`x_i`, shape `(n,)`, on
        `measure.support`.
    weights : array_like
        Quadrature weights :math:`w_i`, shape `(n,)`.
    name : str
        Name identifying the rule.
    measure : Measure
        Weight function and reference domain the rule is defined on.

    Raises
    ------
    ValueError
        If `nodes` and `weights` do not have the same shape.
    """

    nodes: np.ndarray  # shape (n,), on `measure.support`
    weights: np.ndarray  # shape (n,)
    name: str  # name for the rule
    measure: Measure

    def __post_init__(self):
        # Static data, safe to unconditionally convert to NumPy arrays
        object.__setattr__(self, "nodes", np.asarray(self.nodes, dtype=float))
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float))
        if self.nodes.shape != self.weights.shape:
            raise ValueError(
                f"nodes {self.nodes.shape} and weights {self.weights.shape} "
                "must have the same shape"
            )

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"measure={type(self.measure).__name__}, n={len(self)})"
        )

    # -- domain mapping --

    def scaled_points(self, *params, **kwparams):
        """Nodes mapped by `measure`'s affine parameters.

        Given ``(scale, shift) = measure.affine_params(*params, **kwparams)``,
        the mapped nodes are

        .. math::
            x_i = \\mathrm{scale} \\cdot t_i + \\mathrm{shift}

        for reference node :math:`t_i`. Called with no arguments, returns
        the reference `nodes` unchanged.

        The meaning of `params`/`kwparams` is specific to `measure`:

        - Legendre/Jacobi: `(a, b)` bounds of the target interval.
        - Laguerre: `rate` (and optional `start`) of the target
          exponential weight.
        - Hermite: `mean`, `std` of the target Gaussian-shaped weight.

        See the measure's `affine_params` docstring for details. Symbolic
        if any parameter is symbolic; the underlying nodes are static.
        """
        scale, shift = self.measure.affine_params(*params, **kwparams)
        return scale * self.nodes + shift

    def scaled_weights(self, *params, **kwparams):
        """Weights including the Jacobian factor for the target
        domain/measure.

        .. math::
            \\tilde{w}_i = \\mathrm{scale} \\cdot w_i

        where :math:`\\mathrm{scale}` is the same affine scale used by
        `scaled_points`. See `scaled_points` for the meaning of
        `params`/`kwparams`.
        """
        scale, _ = self.measure.affine_params(*params, **kwparams)
        return scale * self.weights

    # -- integration --

    def integrate(
        self,
        f: Callable[..., np.ndarray],
        *params,
        axis: int = -1,
        args: Sequence[Any] | None = None,
        **kwparams,
    ) -> np.ndarray:
        """Approximate the weighted integral of `f`.

        .. math::
            \\int f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n \\tilde{w}_i
                f(x_i)

        where :math:`x_i` = `scaled_points(*params, **kwparams)` and
        :math:`\\tilde{w}_i` = `scaled_weights(*params, **kwparams)`.

        Parameters
        ----------
        f : callable
            Integrand, called once as ``f(x, *args)`` on the full node
            array. Must be vectorized, returning values with the nodes
            along `axis`. If any of `params`/`kwparams` is symbolic, `f`
            must be symbolically traceable.
        *params, **kwparams
            Target domain/measure parameters; see `scaled_points` for
            their meaning.
        axis : int, optional
            Axis holding the nodes in the output of `f`. Default -1.
        args : tuple, optional
            Extra arguments passed to `f` after the node array.

        Returns
        -------
        integral : ndarray
            Approximated integral. Shape (m,) for vector-valued
            integrands, or () for scalar integrands.
        """
        if args is None:
            args = ()
        fp = f(self.scaled_points(*params, **kwparams), *args)
        return self.sum(fp, *params, axis=axis, **kwparams)

    def sum(
        self,
        values: np.ndarray,
        *params,
        axis: int = -1,
        **kwparams,
    ) -> np.ndarray:
        """Quadrature applied to values already sampled at the nodes.

        .. math::
            \\sum_{i=1}^n \\tilde{w}_i \\, \\mathrm{values}_i

        where :math:`\\tilde{w}_i` = `scaled_weights(*params, **kwparams)`.

        Parameters
        ----------
        values : array_like
            Sampled values, with the quadrature nodes along `axis`.
            Shape (n,) for scalar integrands or (m, n) for vector-valued
            integrands under the default `axis=-1`.
        *params, **kwparams
            Target domain/measure parameters, forwarded to
            `measure.affine_params`; see `scaled_points` for their meaning.
        axis : int, optional
            Axis holding the nodes. Default -1 (nodes last), matching the
            natural output of a vectorized `f`. Use `axis=0` for
            nodes-first data.

        Returns
        -------
        integral : ndarray
            Approximated integral of the sampled values, with the
            quadrature nodes integrated out along `axis`. Shape (m,) for
            vector-valued integrands, or () for scalar integrands.

        Raises
        ------
        ValueError
            If `values` has more than 2 dimensions, or if
            `values.shape[axis]` does not match the number of quadrature
            nodes.
        """
        w = self.scaled_weights(*params, **kwparams)

        if values.ndim > 2:
            raise ValueError(f"expected a 0-D, 1-D, or 2-D array, got {values.ndim}-D")
        if values.shape[axis] != len(self):
            raise ValueError(
                f"values.shape[{axis}] is {values.shape[axis]}, expected "
                f"{len(self)} to match the quadrature nodes"
            )

        if values.ndim == 1 or axis == 0:
            return np.dot(w, values)
        return np.dot(values, w)


def composite(base: QuadratureRule, breakpoints: np.ndarray) -> QuadratureRule:
    """Tile `base` across elements of its reference domain.

    Partitions `base.measure.support` at `breakpoints` and applies
    `base`, affinely rescaled, to each element, concatenating the resulting
    nodes and weights. The result is itself a `QuadratureRule` on the same
    reference domain -- its nodes are just clustered at the element
    boundaries rather than spread uniformly -- so it can be mapped onto a
    target domain/measure via `scaled_points`/`scaled_weights`/`integrate`
    exactly like any other rule of `base.measure`. This works because
    `measure.affine_params` maps affinely, and affine maps commute with
    subdivision: rescaling the whole composite pattern onto `[a, b]` is
    the same as building the elements directly on the rescaled sub-intervals
    of `[a, b]`.

    Only defined for families whose reference weight is uniform (see
    `Measure.uniform_weight`) -- otherwise each interior element
    boundary would pick up a spurious copy of the weight's shape, which is
    only meaningful at the true endpoints of the reference domain.

    Parameters
    ----------
    base : QuadratureRule
        Rule to tile across elements. `base.measure.uniform_weight` must be
        `True`.
    breakpoints : array_like
        Element boundaries, shape `(k + 1,)` for `k` elements. Must be
        strictly increasing and span `base.measure.support`
        exactly (first/last entries equal to its endpoints).

    Returns
    -------
    rule : QuadratureRule
        Composite rule with `k * len(base)` nodes on the same reference
        domain as `base`.

    Raises
    ------
    ValueError
        If `base.measure.uniform_weight` is `False`, if `breakpoints` has
        fewer than 2 entries or is not strictly increasing, or if it does
        not span `base.measure.support` exactly.
    """
    if not base.measure.uniform_weight:
        raise ValueError(
            f"composite quadrature requires a measure with a uniform "
            f"reference weight, got {type(base.measure).__name__}"
        )
    breakpoints = np.asarray(breakpoints, dtype=float)
    if breakpoints.ndim != 1 or len(breakpoints) < 2:
        raise ValueError(
            f"breakpoints must be 1-D with at least 2 entries, got shape "
            f"{breakpoints.shape}"
        )
    if np.any(np.diff(breakpoints) <= 0):
        raise ValueError("breakpoints must be strictly increasing")
    lo, hi = base.measure.support
    if breakpoints[0] != lo or breakpoints[-1] != hi:
        raise ValueError(
            f"breakpoints must span the reference domain {(lo, hi)}, got "
            f"({breakpoints[0]}, {breakpoints[-1]})"
        )

    nodes = []
    weights = []
    for t0, t1 in zip(breakpoints[:-1], breakpoints[1:]):
        nodes.append(base.scaled_points(t0, t1))
        weights.append(base.scaled_weights(t0, t1))

    return QuadratureRule(
        np.concatenate(nodes),
        np.concatenate(weights),
        measure=base.measure,
        name=base.name,
    )
