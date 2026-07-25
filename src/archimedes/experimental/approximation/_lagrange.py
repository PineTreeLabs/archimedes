"""Lagrange (nodal) interpolating polynomial basis."""

from __future__ import annotations

import dataclasses

import numpy as np

from archimedes.measure import UnitInterval

from ._basis import Basis

__all__ = ["LagrangeBasis"]


def _barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    n = len(nodes)
    w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                w[i] *= nodes[i] - nodes[j]
    return 1.0 / w


@dataclasses.dataclass(frozen=True)
class LagrangeBasis(Basis):
    """Lagrange cardinal polynomials :math:`\\{\\ell_0, \\ldots, \\ell_{n-1}\\}`
    for a fixed set of nodes: :math:`\\ell_i(x_j) = \\delta_{ij}`.

    Evaluated via the (first, "true") barycentric formula

    .. math::
        \\ell_i(x) = \\frac{w_i / (x - x_i)}{\\sum_j w_j / (x - x_j)},
        \\qquad w_i = \\frac{1}{\\prod_{j \\neq i} (x_i - x_j)}

    with the usual special-case handling at :math:`x = x_k` (where
    :math:`\\ell_i(x_k) = \\delta_{ik}` directly, avoiding 0/0). Unlike
    :class:`OrthogonalPolynomialBasis`, this family has no weight function
    or orthogonality measure at all -- ``reference_nodes`` are just data,
    and the only thing needed from the domain is the affine
    :class:`~archimedes.measure.UnitInterval` map onto :math:`[a, b]`.

    The barycentric weights are invariant under that map (a uniform
    rescaling of every node cancels in the ratio above), so they're
    computed once from the reference nodes and reused unchanged for every
    target domain.

    Only ``deriv=0`` is implemented so far; the barycentric derivative
    formula (and its own 0/0 handling at the nodes) is left for later.

    Parameters
    ----------
    reference_nodes : array_like
        Interpolation nodes on the reference domain ``[-1, 1]``. Must be
        distinct. Order determines the meaning of a ``Function``'s
        coefficients (``coefficients[i]`` is the value at
        ``reference_nodes[i]``, once mapped to the target domain), but not
        the basis itself.
    """

    reference_nodes: np.ndarray

    def __post_init__(self):
        nodes = np.asarray(self.reference_nodes, dtype=float)
        if nodes.ndim != 1 or len(nodes) < 1:
            raise ValueError(
                f"reference_nodes must be 1-D with at least one entry, got "
                f"shape {nodes.shape}"
            )
        if np.any(nodes < -1.0) or np.any(nodes > 1.0):
            raise ValueError("reference_nodes must lie in [-1, 1]")
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError("reference_nodes must be distinct")
        object.__setattr__(self, "reference_nodes", nodes)
        object.__setattr__(self, "_weights", _barycentric_weights(nodes))

    @property
    def n_basis(self) -> int:
        return len(self.reference_nodes)

    @property
    def Parameters(self) -> type:  # noqa: N802
        return UnitInterval.Parameters

    def evaluate(self, x, deriv: int = 0, a=None, b=None):
        if deriv != 0:
            raise NotImplementedError(
                f"LagrangeBasis only supports deriv=0 currently, got {deriv}"
            )

        scale, shift = UnitInterval().affine_params(a, b)
        xp = scale * self.reference_nodes + shift  # (n_basis,)
        w = self._weights  # scale-invariant; see class docstring

        xdiff = x[:, None] - xp[None, :]  # (npts, n_basis)
        safe_diff = np.where(xdiff == 0, 1.0, xdiff)
        temp = w[None, :] / safe_diff

        is_node = xdiff == 0  # (npts, n_basis), 0/1-valued
        any_node = np.sum(is_node, axis=1)  # (npts,); 1 if x_i is a node
        den = np.sum(temp, axis=1)
        phi_generic = temp / den[:, None]

        return np.where(any_node[:, None] > 0, is_node.astype(float), phi_generic)
