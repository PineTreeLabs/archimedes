"""Orthonormal polynomial basis for any classical-orthogonal-polynomial Measure."""

from __future__ import annotations

import dataclasses

import numpy as np

from archimedes.measure import Measure

from ._base import RIGHT, Basis, _check_side

__all__ = ["OrthogonalPolynomialBasis"]


@dataclasses.dataclass(frozen=True)
class OrthogonalPolynomialBasis(Basis):
    r"""A classical orthonormal polynomial basis

    Orthonormal polynomials :math:`\{p_0, p_1, \ldots, p_{n-1}\}`, constructed
    to be orthonormal with respect to a given measure (reference domain and inner
    product weight).

    The monic polynomials orthogonal w.r.t. any :class:`~archimedes.measure.Measure`
    satisfy the three-term recurrence

    .. math::
        \pi_{k+1}(x) = (x - \alpha_k) \, \pi_k(x) - \beta_k \, \pi_{k-1}(x)

    which holds for any classical orthogonal polynomial family. The squared norm
    for the associated weight function :math:`w(x)` is
    :math:`\int \pi_k^2 \, w \, dx = \beta_0 \beta_1 \cdots \beta_k`.

    The basis functions are additionally normalized with
    :math:`p_k = \pi_k / \sqrt{\beta_0 \cdots \beta_k}` for conditioning
    at high polynomial degree.

    Supported measures include:

    - :class:`~archimedes.measure.LegendreMeasure`
    - :class:`~archimedes.measure.JacobiMeasure` (also Chebyshev as a special case)
    - :class:`~archimedes.measure.PhysicistsHermiteMeasure`
    - :class:`~archimedes.measure.ProbabilistsHermiteMeasure`
    - :class:`~archimedes.measure.LaguerreMeasure`

    However, customized measures may be constructed by defining a domain and weight
    function, with recursions computed automatically using the discretized Stieltjes
    procedure. See the [documentation on quadrature](handbook/quadrature) for details.

    An orthogonal polynomial basis can be constructed from any such custom measure,
    with the basis functions defined automatically via the three-term recurrence.
    """

    measure: Measure
    """Defines the orthogonality weight and reference domain."""
    n_basis: int
    density: bool = False

    def __post_init__(self):
        if self.n_basis < 1:
            raise ValueError(f"n_basis must be >= 1, got {self.n_basis}")

    @property
    def _measures(self) -> tuple[Measure, ...]:
        """This basis's orthogonality weight; see :attr:`Basis._measures`."""
        return (self.measure,)

    @property
    def Parameters(self) -> type:  # noqa: N802
        return type(self.measure.domain).Parameters

    @property
    def _reference_scale_exponent(self) -> float:
        return 0.0 if self.density else 0.5

    def _default_quadrature(self):
        from archimedes.quadrature import golub_welsch_rule

        return golub_welsch_rule(self.measure, self.n_basis)

    def _product_basis(self, other):
        if not isinstance(other, OrthogonalPolynomialBasis):
            raise ValueError(
                f"cannot form a product basis between "
                f"{type(self).__name__} and {type(other).__name__}"
            )
        if self.measure != other.measure:
            raise ValueError(
                f"product requires the same measure, got "
                f"{type(self.measure).__name__} and {type(other.measure).__name__}"
            )
        if self.density != other.density:
            raise ValueError(
                f"product requires the same normalization, got "
                f"density={self.density} and density={other.density}"
            )
        return OrthogonalPolynomialBasis(
            self.measure, self.n_basis + other.n_basis - 1, density=self.density
        )

    def _derivative_basis(self, deriv=1):
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
        return OrthogonalPolynomialBasis(
            self.measure, self.n_basis - deriv, density=self.density
        )

    def _integral_basis(self, order=1):
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if order == 0:
            return self
        return OrthogonalPolynomialBasis(
            self.measure, self.n_basis + order, density=self.density
        )

    def evaluate(self, x, deriv: int = 0, side: str = RIGHT, **domain_kwargs):
        # `side` is validated but unused: these polynomials are smooth, so
        # both one-sided limits agree everywhere. See `Basis.evaluate`.
        _check_side(side)
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")

        alpha, beta = self.measure.recurrence_coeffs(self.n_basis)
        scale, shift = self.measure.affine_params(**domain_kwargs)
        alpha = scale * alpha + shift
        beta = scale**2 * beta

        # norm[k] = sqrt(beta_0 * beta_1 * ... * beta_k), with beta_0 taken as
        # the target measure's total mass -- or 1, with `density=True`, so
        # the basis is instead orthonormal w.r.t. the *probability* density
        # `weight / mass` (see `Basis.density`). Accumulated with an explicit
        # Python loop (n_basis is static) rather than `np.cumprod`, and
        # without writing into `beta`, because both the cumulative product
        # and the item assignment have no symbolic implementation -- `scale`
        # and `mass` are traced whenever the domain parameters are.
        norms = [1.0 if self.density else self.measure.mass(**domain_kwargs)]
        for k in range(1, self.n_basis):
            norms.append(norms[-1] * beta[k])
        norm = np.stack([np.sqrt(value) for value in norms], axis=-1)

        # pi[m][k] = the m-th derivative of the k-th monic polynomial, for
        # every m <= deriv simultaneously (computing derivatives is nearly
        # free once the recurrence is being evaluated regardless).
        pi = [[None] * self.n_basis for _ in range(deriv + 1)]
        pi[0][0] = np.ones_like(x)
        for m in range(1, deriv + 1):
            pi[m][0] = np.zeros_like(x)

        if self.n_basis > 1:
            pi[0][1] = x - alpha[0]
            if deriv >= 1:
                pi[1][1] = np.ones_like(x)
            for m in range(2, deriv + 1):
                pi[m][1] = np.zeros_like(x)

        for k in range(1, self.n_basis - 1):
            for m in range(deriv + 1):
                value = (x - alpha[k]) * pi[m][k] - beta[k] * pi[m][k - 1]
                if m > 0:
                    value = value + m * pi[m - 1][k]
                pi[m][k + 1] = value

        return np.stack(pi[deriv], axis=-1) / norm
