"""Lagrange (nodal) interpolating polynomial basis."""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from archimedes.measure import UnitInterval

from ._base import RIGHT, Basis, _check_side

__all__ = ["LagrangeBasis"]


def _barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    n = len(nodes)
    w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                w[i] *= nodes[i] - nodes[j]
    return 1.0 / w


def _lobatto_nodes(n: int) -> np.ndarray:
    """``n`` Gauss-Lobatto nodes, or the single node ``0`` when ``n < 2``.

    Any distinct node set spans the same polynomial space, so the choice
    only affects conditioning and which degrees of freedom are nodal.
    Gauss-Lobatto is well-conditioned and includes both endpoints, which
    keeps :meth:`LagrangeBasis.boundary_dofs` populated so the result can
    still be tiled with :math:`C^0` continuity. ``gauss_lobatto`` is
    undefined below 2 points, and a single node spans the constants.
    """
    from archimedes.quadrature import gauss_lobatto

    return np.zeros(1) if n < 2 else gauss_lobatto(n).nodes


def _gauss_legendre_nodes(n: int) -> np.ndarray:
    from archimedes.quadrature import gauss_legendre

    return gauss_legendre(n).nodes


def _gauss_radau_left_nodes(n: int) -> np.ndarray:
    from archimedes.quadrature import gauss_radau

    return gauss_radau(n, endpoint="left").nodes


def _gauss_radau_right_nodes(n: int) -> np.ndarray:
    from archimedes.quadrature import gauss_radau

    return gauss_radau(n, endpoint="right").nodes


def _equispaced_nodes(n: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n)


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
    r"""Lagrange cardinal polynomials :math:`\{\ell_0, \ldots, \ell_{n-1}\}`
    for a fixed set of nodes: :math:`\ell_i(x_j) = \delta_{ij}`.

    Evaluated via the (first, "true") barycentric formula

    .. math::
        \ell_i(x) = \frac{w_i / (x - x_i)}{\sum_j w_j / (x - x_j)},
        \qquad w_i = \frac{1}{\prod_{j \neq i} (x_i - x_j)}

    with the usual special-case handling at :math:`x = x_k` (where
    :math:`\ell_i(x_k) = \delta_{ik}` directly, avoiding 0/0).

    .. warning::
        That special case is selected by a runtime comparison, so it is a
        branch point for automatic differentiation. Differentiating with
        respect to a *domain parameter* (``a``/``b``, which move the nodes)
        at a point that coincides *exactly* with a node returns the
        derivative of the constant :math:`\delta_{ik}` branch, i.e. zero,
        rather than the true value -- the underlying function is smooth
        there, but this formula is not. Off-node points, and derivatives
        with respect to ``x``, are unaffected.

    Derivatives of every order are supported and exact: :math:`\ell_j^{(k)}`
    is itself a polynomial of degree :math:`\leq n - 1` and so is
    represented exactly in this same basis,

    .. math::
        \ell_j^{(k)}(x) = \sum_i D^k_{ij} \, \ell_i(x),

    where :math:`D_{ij} = \ell_j'(x_i)` is the classical differentiation
    matrix -- so every derivative order reduces to the ``deriv=0``
    evaluation above followed by a matrix product, :math:`\Phi^{(k)} =
    \Phi \, D^k`, with no repeated 0/0 handling.

    Parameters
    ----------
    reference_nodes : array_like
        Interpolation nodes on the reference domain ``[-1, 1]``. Must be
        distinct. Order determines the meaning of a ``Function``'s
        coefficients (``coefficients[i]`` is the value at
        ``reference_nodes[i]``, once mapped to the target domain), but not
        the basis itself.
    node_family : callable, optional
        ``n -> nodes``, used internally to pick the node set whenever a
        *differently sized* basis of the same kind is needed. Not applied
        to ``reference_nodes``, which are taken as given.

        Defaults to Gauss-Lobatto, the conventional nodal set (Chebyshev
        points of the second kind and the spectral-element method's
        standard GLL), which keeps :meth:`boundary_dofs` populated so the
        result can still be tiled with :math:`C^0` continuity.

        Any ``n`` distinct nodes span the same :math:`P_{n-1}`, so the
        choice affects only conditioning and which degrees of freedom are
        nodal. Supply one to keep a derived basis's node family intact
        where that matters, e.g. ``lambda n: gauss_radau(n,
        endpoint="left").nodes`` for a Radau-based pseudospectral scheme.
        A family with no endpoint node leaves :meth:`boundary_dofs` empty,
        so the derived basis cannot be tiled with :math:`C^0` continuity.
    """

    reference_nodes: np.ndarray
    node_family: Callable[[int], np.ndarray] | None = None

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
        return (
            np.array_equal(self.reference_nodes, other.reference_nodes)
            # Two bases with identical nodes but different families agree on
            # every value and disagree on every *derived* basis, so they are
            # not interchangeable.
            and self.node_family == other.node_family
        )

    def __hash__(self) -> int:
        return hash((type(self), self.reference_nodes.tobytes(), self.node_family))

    # --- constructors ---

    @classmethod
    def gauss_lobatto(cls, n: int) -> "LagrangeBasis":
        """``n`` Gauss-Lobatto nodes -- this family's default, so
        ``node_family`` is left at ``None`` rather than set explicitly."""
        return cls(reference_nodes=_lobatto_nodes(n))

    @classmethod
    def gauss_legendre(cls, n: int) -> "LagrangeBasis":
        """``n`` Gauss-Legendre nodes (no endpoints); pseudospectral
        collocation at Gauss points."""
        return cls(
            reference_nodes=_gauss_legendre_nodes(n), node_family=_gauss_legendre_nodes
        )

    @classmethod
    def gauss_radau(cls, n: int, endpoint: str = "left") -> "LagrangeBasis":
        """``n`` Gauss-Radau nodes, fixing ``endpoint`` (``"left"`` or
        ``"right"``); see :func:`archimedes.quadrature.gauss_radau`.
        """
        # Dispatches to one of two named module-level functions rather than
        # parametrizing a single one with a closure or `functools.partial`
        # over `endpoint`: two independently-constructed instances must
        # compare equal to be usable together (see `_product_basis`, basis
        # equality), which requires the *same* `node_family` object each
        # time -- a fresh lambda per call never satisfies that, and neither
        # does `functools.partial`, which has no value-based `__eq__` either.
        if endpoint == "left":
            return cls(
                reference_nodes=_gauss_radau_left_nodes(n),
                node_family=_gauss_radau_left_nodes,
            )
        if endpoint == "right":
            return cls(
                reference_nodes=_gauss_radau_right_nodes(n),
                node_family=_gauss_radau_right_nodes,
            )
        raise ValueError(f"endpoint must be 'left' or 'right', got {endpoint!r}")

    @classmethod
    def equispaced(cls, n: int) -> "LagrangeBasis":
        """``n`` evenly spaced nodes, including both endpoints."""
        return cls(reference_nodes=_equispaced_nodes(n), node_family=_equispaced_nodes)

    # --- implementation ---

    def _nodes_for(self, n: int) -> np.ndarray:
        """``n`` nodes from this basis's family, for a derived basis."""
        if self.node_family is None:
            return _lobatto_nodes(n)
        nodes = np.asarray(self.node_family(n), dtype=float)
        if nodes.shape != (n,):
            raise ValueError(
                f"node_family({n}) returned shape {nodes.shape}, expected ({n},)"
            )
        return nodes

    def _derived(self, n: int) -> "LagrangeBasis":
        """A basis of ``n`` nodes from this one's family, family preserved so
        repeated operations don't drift back to the default."""
        return LagrangeBasis(
            reference_nodes=self._nodes_for(n), node_family=self.node_family
        )

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

    def _product_basis(self, other):
        """``n_1 + n_2 - 1`` nodes from this basis's ``node_family``."""
        if not isinstance(other, LagrangeBasis):
            raise ValueError(
                f"cannot form a product basis between "
                f"{type(self).__name__} and {type(other).__name__}"
            )
        if self.node_family != other.node_family:
            raise ValueError(
                "product requires the same node_family, since the result's "
                "node set would otherwise depend on the order of the operands"
            )
        return self._derived(self.n_basis + other.n_basis - 1)

    def _derivative_basis(self, deriv=1):
        """``n_basis - deriv`` nodes from this basis's ``node_family``.

        The nodes necessarily *move*: a smaller nodal space is a different
        set of points, so the result's coefficients are values at the new
        nodes rather than at this basis's. Collocation methods that need the
        derivative sampled at the *original* nodes want
        ``Function.derivative(space=self_space)`` instead, which for this
        family is exactly the classical barycentric differentiation matrix.
        """
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")
        # Not `self._derived(n_basis)`: regenerating from `node_family` would
        # silently move nodes that were supplied explicitly. This matters for
        # `TensorBasis`, which asks for order 0 on every undifferentiated
        # factor.
        if deriv == 0:
            return self
        if deriv >= self.n_basis:
            raise ValueError(
                f"deriv={deriv} is at or past the degree of a {self.n_basis}-"
                f"node basis, whose elements are polynomials of degree "
                f"{self.n_basis - 1}; the derivative is identically zero and "
                f"has no space of its own. Use `f(x, deriv={deriv})` if the "
                f"zero values are what you want."
            )
        return self._derived(self.n_basis - deriv)

    def _integral_basis(self, order=1):
        """``n_basis + order`` nodes from this basis's ``node_family``.

        Dual of :meth:`_derivative_basis`: growing rather than shrinking
        the node set, so (unlike differentiation) this is always defined.
        As with :meth:`_derivative_basis`, the result's nodes -- and so the
        meaning of its coefficients -- differ from this basis's own.
        """
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if order == 0:
            return self
        return self._derived(self.n_basis + order)

    def boundary_dofs(self, order: int = 0) -> tuple[int | None, int | None]:
        """Indices of the nodes at :math:`t = \\pm 1`, or ``None`` if the
        corresponding endpoint isn't a node.

        Gauss-Lobatto nodes include both endpoints; Gauss-Radau includes
        one; Gauss-Legendre includes neither. Since ``ell_i(x_j) =
        delta_ij``, an endpoint node's coefficient *is* the endpoint value,
        which is what :math:`C^0` assembly identifies across elements.

        Every degree of freedom here is a plain nodal value (``order=0``);
        this family has no derivative-type DOF, so any other ``order``
        returns ``(None, None)``.
        """
        if order != 0:
            return (None, None)
        nodes = self.reference_nodes
        left = int(np.argmin(np.abs(nodes + 1.0)))
        right = int(np.argmin(np.abs(nodes - 1.0)))
        return (
            left if np.isclose(nodes[left], -1.0) else None,
            right if np.isclose(nodes[right], 1.0) else None,
        )

    def evaluate(self, x, deriv: int = 0, a=None, b=None, side: str = RIGHT):
        # `side` is validated but unused: cardinal polynomials are smooth, so
        # both one-sided limits agree everywhere. See `Basis.evaluate`.
        _check_side(side)
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
