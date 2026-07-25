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


def _differentiation_matrix(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``D[i, j] = ell_j'(x_i)``, the classical barycentric differentiation
    matrix. Used only *at* the nodes, where the general formula is 0/0."""
    n = len(nodes)
    diff = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                diff[i, j] = (weights[j] / weights[i]) / (nodes[i] - nodes[j])
        diff[i, i] = -np.sum(np.delete(diff[i], i))
    return diff


@dataclasses.dataclass(frozen=True)
class LagrangeBasis(Basis):
    """Lagrange cardinal polynomials :math:`\\{\\ell_0, \\ldots, \\ell_{n-1}\\}`
    for a fixed set of nodes: :math:`\\ell_i(x_j) = \\delta_{ij}`.

    Evaluated via the (first, "true") barycentric formula

    .. math::
        \\ell_i(x) = \\frac{w_i / (x - x_i)}{\\sum_j w_j / (x - x_j)},
        \\qquad w_i = \\frac{1}{\\prod_{j \\neq i} (x_i - x_j)}

    with the usual special-case handling at :math:`x = x_k` (where
    :math:`\\ell_i(x_k) = \\delta_{ik}` directly, avoiding 0/0).

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
        weights = _barycentric_weights(nodes)
        object.__setattr__(self, "reference_nodes", nodes)
        object.__setattr__(self, "_weights", weights)
        object.__setattr__(
            self, "_diff_matrix", _differentiation_matrix(nodes, weights)
        )

    def __eq__(self, other: object) -> bool:
        """Compare elementwise on ``reference_nodes``.

        Defined explicitly because the ``@dataclass``-generated ``__eq__``
        compares the array field with ``==``, which yields an array and
        raises "truth value of an array is ambiguous" whenever the two
        instances don't happen to hold the *same* array object.
        """
        if not isinstance(other, LagrangeBasis):
            return NotImplemented
        return np.array_equal(self.reference_nodes, other.reference_nodes)

    def __hash__(self) -> int:
        return hash((type(self), self.reference_nodes.tobytes()))

    @property
    def n_basis(self) -> int:
        return len(self.reference_nodes)

    @property
    def Parameters(self) -> type:  # noqa: N802
        return UnitInterval.Parameters

    def boundary_dofs(self) -> tuple[int | None, int | None]:
        """Indices of the nodes at :math:`t = \\pm 1`, or ``None`` if the
        corresponding endpoint isn't a node.

        Gauss-Lobatto nodes include both endpoints; Gauss-Radau includes
        one; Gauss-Legendre includes neither. Since ``ell_i(x_j) =
        delta_ij``, an endpoint node's coefficient *is* the endpoint value,
        which is what :math:`C^0` assembly identifies across elements.
        """
        nodes = self.reference_nodes
        left = int(np.argmin(np.abs(nodes + 1.0)))
        right = int(np.argmin(np.abs(nodes - 1.0)))
        return (
            left if np.isclose(nodes[left], -1.0) else None,
            right if np.isclose(nodes[right], 1.0) else None,
        )

    def evaluate(self, x, deriv: int = 0, a=None, b=None):
        if deriv not in (0, 1):
            raise NotImplementedError(
                f"LagrangeBasis only supports deriv in (0, 1), got {deriv}"
            )

        scale, shift = UnitInterval().affine_params(a, b)
        xp = scale * self.reference_nodes + shift  # (n_basis,)
        w = self._weights  # scale-invariant; see class docstring

        xdiff = x[:, None] - xp[None, :]  # (npts, n_basis)
        safe_diff = np.where(xdiff == 0, 1.0, xdiff)
        temp = w[None, :] / safe_diff

        is_node = (xdiff == 0).astype(float)  # (npts, n_basis), 0/1-valued
        any_node = np.sum(is_node, axis=1)  # (npts,); 1 if x_i is a node
        den = np.sum(temp, axis=1)
        phi_generic = temp / den[:, None]

        if deriv == 0:
            return np.where(any_node[:, None] > 0, is_node, phi_generic)

        # Differentiating the barycentric quotient ell_j = u_j / S, with
        # u_j = w_j / (x - x_j) and S = sum_k u_k, gives
        #     ell_j' = ell_j * (T / S - 1 / (x - x_j)),   T = sum_k u_k/(x - x_k)
        # which is again 0/0 at a node, where the classical differentiation
        # matrix supplies the value instead. `is_node @ D` selects the row of D
        # belonging to whichever node the point coincides with (and is all
        # zeros elsewhere), so the same masked-select pattern works.
        t_sum = np.sum(temp / safe_diff, axis=1)
        dphi_generic = phi_generic * (t_sum[:, None] / den[:, None] - 1.0 / safe_diff)

        # D is built from the reference nodes, so it carries a 1/scale
        # chain-rule factor when mapped onto [a, b].
        dphi_at_nodes = (is_node @ self._diff_matrix) / scale

        return np.where(any_node[:, None] > 0, dphi_at_nodes, dphi_generic)
