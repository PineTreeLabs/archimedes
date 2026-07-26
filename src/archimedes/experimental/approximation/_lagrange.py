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

    .. warning::
        That special case is selected by a runtime comparison, so it is a
        branch point for automatic differentiation. Differentiating with
        respect to a *domain parameter* (``a``/``b``, which move the nodes)
        at a point that coincides *exactly* with a node returns the
        derivative of the constant :math:`\\delta_{ik}` branch, i.e. zero,
        rather than the true value -- the underlying function is smooth
        there, but this formula is not. Off-node points, and derivatives
        with respect to ``x``, are unaffected.

    Derivatives of every order are supported. Rather than differentiating
    the barycentric quotient (which reintroduces a 0/0 case at each node
    for each order), note that :math:`\\ell_j^{(k)}` is itself a polynomial
    of degree :math:`\\leq n - 1` and so is *exactly* represented in this
    same basis:

    .. math::
        \\ell_j^{(k)}(x) = \\sum_i \\ell_j^{(k)}(x_i) \\, \\ell_i(x)
                        = \\sum_i D^k_{ij} \\, \\ell_i(x),

    where :math:`D_{ij} = \\ell_j'(x_i)` is the classical differentiation
    matrix. In matrix form :math:`\\Phi^{(k)} = \\Phi \\, D^k`, so every
    derivative order reduces to the ``deriv=0`` evaluation above followed
    by a matrix product. At a node :math:`\\Phi` is a unit vector, so this
    returns the corresponding row of :math:`D^k` exactly -- the special
    case is subsumed rather than handled separately.

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

    def default_quadrature(self):
        """Gauss-Legendre rule of ``n_basis`` points.

        This family carries no weight function of its own (see the class
        docstring), so the relevant inner product is the unweighted one on
        :math:`[-1, 1]`, i.e. Legendre. Each cardinal polynomial has degree
        ``n_basis - 1``, so their products have degree
        ``2 * (n_basis - 1)``, which ``n_basis`` Gauss points integrate
        exactly.
        """
        from archimedes.quadrature import gauss_legendre

        return gauss_legendre(self.n_basis)

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
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")

        scale, shift = UnitInterval().affine_params(a, b)
        xp = scale * self.reference_nodes + shift  # (n_basis,)
        w = self._weights  # scale-invariant; see class docstring

        xdiff = x[:, None] - xp[None, :]  # (npts, n_basis)
        safe_diff = np.where(xdiff == 0, 1.0, xdiff)
        temp = w[None, :] / safe_diff

        is_node = (xdiff == 0).astype(float)  # (npts, n_basis), 0/1-valued
        any_node = np.sum(is_node, axis=1)  # (npts,); 1 if x_i is a node
        den = np.sum(temp, axis=1)
        # `den` can be *exactly* zero at a node: substituting 1.0 into
        # `safe_diff` perturbs that term, and the perturbed terms can cancel
        # (they do at the right endpoint of a 3-node Lobatto element). The
        # generic branch is discarded there, but `np.where` evaluates both,
        # so guard this division as well rather than emit an inf that is
        # only conditionally unused.
        safe_den = np.where(any_node > 0, 1.0, den)
        phi = np.where(any_node[:, None] > 0, is_node, temp / safe_den[:, None])

        if deriv == 0:
            return phi

        # Each cardinal polynomial has degree n_basis - 1, so its
        # n_basis-th derivative vanishes identically. Returning exact zeros
        # is both correct and better conditioned than the numerical D**k,
        # which is only nilpotent up to roundoff.
        if deriv >= self.n_basis:
            return np.zeros_like(phi)

        # Phi^(k) = Phi @ D**k -- see the class docstring. D is built from the
        # reference nodes, so it carries a 1/scale chain-rule factor per
        # derivative when mapped onto [a, b]. The division is applied to the
        # result rather than to D so that a symbolic `scale` never has to
        # propagate through `matrix_power`.
        d_power = np.linalg.matrix_power(self._diff_matrix, deriv)
        return (phi @ d_power) / scale**deriv
