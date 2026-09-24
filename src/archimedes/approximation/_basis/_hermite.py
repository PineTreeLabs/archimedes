"""Cubic Hermite (value + slope) basis."""

from __future__ import annotations

import dataclasses

import numpy as np

from archimedes.measure import UnitInterval

from ._base import RIGHT, Basis, _check_side
from ._lagrange import LagrangeBasis, _lobatto_nodes

__all__ = ["CubicHermiteBasis"]


@dataclasses.dataclass(frozen=True)
class CubicHermiteBasis(Basis):
    r"""A basis of cubic Hermite shape functions

    This basis includes both value and derivative degrees of
    freedom at each of the two element endpoints.

    The basis functions are :math:`\{\phi_{00}, \phi_{10}, \phi_{01},
    \phi_{11}\}`, where :math:`\phi_{i0}` is the value-type function at
    endpoint :math:`i` and :math:`\phi_{i1}` the derivative-type one, so a
    :class:`Function` ``u`` on this basis has coefficients ``[u(a), u'(a), u(b),
    u'(b)]``. See [1]_ for more details.

    Typically used to construct piecewise-cubic bases, in which case it
    can produce a globally :math:`C^1` space. See :class:`PiecewiseBasis`.

    Derivatives leave this family: for a piecewise cubic Hermite
    ``Function`` ``f``, ``f.derivative()`` is a piecewise *quadratic*
    function in a :class:`LagrangeBasis` (Gauss-Lobatto nodes), since there
    is no smaller Hermite space for the derivative to live in.

    See Also
    --------
    FunctionSpace.piecewise : Convenience constructor for a :class:`FunctionSpace`
        that supports piecewise cubic Hermite elements.

    Notes
    -----
    The reference functions on :math:`t \in [-1, 1]` are built from the
    classical cubic Hermite basis on :math:`\tau \in [0, 1]` (:math:`\tau =
    (t+1)/2`),

    .. math::
        \begin{align}
        H_{00}(\tau) &= 2\tau^3 - 3\tau^2 + 1, \\
        H_{10}(\tau) &= \tau^3 - 2\tau^2 + \tau, \\
        H_{01}(\tau) &= -2\tau^3 + 3\tau^2, \\
        H_{11}(\tau) &= \tau^3 - \tau^2,
        \end{align}

    via :math:`\phi_{i0}(t) = H_{i0}(\tau)` and :math:`\phi_{i1}(t) = 2 H_{i1}(\tau)`.

    References
    ----------
    .. [1] Wikipedia, "Cubic Hermite spline," https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    """

    @property
    def n_basis(self) -> int:
        """Always 4: a value and a slope at each endpoint."""
        return 4

    @property
    def Parameters(self) -> type:
        return UnitInterval.Parameters

    @property
    def _dof_order(self) -> np.ndarray:
        """Derivative order of each column's degree of freedom.

        ``[0, 1, 0, 1]``: columns 0, 2 are values; columns 1, 3 are
        physical derivatives. See the class docstring."""
        return np.array([0, 1, 0, 1], dtype=int)

    def _default_quadrature(self):
        """Gauss-Legendre rule of 4 points.

        These are cardinal cubics (degree 3), so their products have degree
        6; 4 Gauss points integrate exactly through degree 7.
        """
        from archimedes.quadrature import gauss_legendre

        return gauss_legendre(self.n_basis)

    def boundary_dofs(self, order: int = 0) -> tuple[int | None, int | None]:
        # ``(0, 2)`` for ``order=0`` (the value DOFs), ``(1, 3)`` for
        # ``order=1`` (the physical-derivative DOFs), ``(None, None)``
        # otherwise -- this family has no curvature (or higher) DOF.
        if order == 0:
            return (0, 2)
        if order == 1:
            return (1, 3)
        return (None, None)

    def _derivative_basis(self, deriv=1):
        """A :class:`LagrangeBasis` of ``4 - deriv`` Gauss-Lobatto nodes.

        The derivative must live in a different family, since the derivative
        of a cubic is a plain polynomial with no value/slope DOF structure.
        Gauss-Lobatto nodes are chosen to preserve the boundary DOFs.
        """
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")
        if deriv == 0:
            return self
        if deriv >= self.n_basis:
            raise ValueError(
                f"deriv={deriv} is at or past the degree of a {self.n_basis}-"
                f"function basis, whose elements are polynomials of degree "
                f"{self.n_basis - 1}; the derivative is identically zero."
            )
        return LagrangeBasis(reference_nodes=_lobatto_nodes(self.n_basis - deriv))

    # Explicit domain parameters narrow the base's `**domain_kwargs`, which
    # mypy reports as an incompatible override.
    def evaluate(  # type: ignore[override]
        self, x, deriv: int = 0, *, a=None, b=None, side: str = RIGHT
    ):
        # `side` is validated but unused: a single Hermite element is smooth,
        # so both one-sided limits agree everywhere. See `Basis.evaluate`.
        _check_side(side)
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")

        scale, shift = UnitInterval().affine_params(a, b)
        t = (x - shift) / scale
        tau = 0.5 * (t + 1.0)

        if deriv >= 4:
            # Cubic Hermite functions have degree <= 3; higher derivatives
            # vanish identically.
            zero = np.zeros_like(tau)
            return np.stack([zero, zero, zero, zero], axis=-1)

        # H00, H10, H01, H11 differentiated `deriv` times w.r.t. tau -- each
        # is a cubic in tau, so this is exact and closed-form through order 3.
        if deriv == 0:
            h00 = 2 * tau**3 - 3 * tau**2 + 1
            h10 = tau**3 - 2 * tau**2 + tau
            h01 = -2 * tau**3 + 3 * tau**2
            h11 = tau**3 - tau**2
        elif deriv == 1:
            h00 = 6 * tau**2 - 6 * tau
            h10 = 3 * tau**2 - 4 * tau + 1
            h01 = -6 * tau**2 + 6 * tau
            h11 = 3 * tau**2 - 2 * tau
        elif deriv == 2:
            h00 = 12 * tau - 6
            h10 = 6 * tau - 4
            h01 = -12 * tau + 6
            h11 = 6 * tau - 2
        else:  # deriv == 3
            ones = np.ones_like(tau)
            h00 = 12.0 * ones
            h10 = 6.0 * ones
            h01 = -12.0 * ones
            h11 = 6.0 * ones

        # d^deriv/dt^deriv of H(tau(t)) = (dtau/dt)**deriv * H^(deriv)(tau),
        # since tau is affine in t; `phi_10 = 2*H10(tau)` (see class
        # docstring) carries one extra factor of 2, i.e. one fewer power of
        # the 1/2 chain-rule factor, than the value-type columns.
        value_ref_factor = 0.5**deriv
        slope_ref_factor = 0.5 ** (deriv - 1)

        # Physical-domain rescaling: value-type columns (_dof_order=0) pick
        # up scale**(-deriv); derivative-type columns (_dof_order=1) pick up
        # scale**(1-deriv), so their coefficient is the physical derivative
        # at the owning endpoint regardless of element width -- see the
        # class docstring and `Basis._dof_order`.
        value_scale = scale ** (-deriv)
        slope_scale = scale ** (1 - deriv)

        phi00 = value_ref_factor * value_scale * h00
        phi10 = slope_ref_factor * slope_scale * h10
        phi01 = value_ref_factor * value_scale * h01
        phi11 = slope_ref_factor * slope_scale * h11

        return np.stack([phi00, phi10, phi01, phi11], axis=-1)
