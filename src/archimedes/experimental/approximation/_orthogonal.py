"""Orthonormal polynomial basis for any classical-orthogonal-polynomial Measure."""

from __future__ import annotations

import dataclasses

import numpy as np

from archimedes.measure import Measure

from ._basis import RIGHT, Basis, _check_side

__all__ = ["OrthogonalPolynomialBasis"]


@dataclasses.dataclass(frozen=True)
class OrthogonalPolynomialBasis(Basis):
    """Orthonormal polynomials :math:`\\{p_0, p_1, \\ldots, p_{n-1}\\}` for
    ``measure``, built entirely from ``measure.recurrence_coeffs``.

    The monic polynomials orthogonal w.r.t. any :class:`~archimedes.measure.Measure`
    satisfy the three-term recurrence

    .. math::
        \\pi_{k+1}(x) = (x - \\alpha_k) \\, \\pi_k(x) - \\beta_k \\, \\pi_{k-1}(x)

    Differentiating this recurrence ``deriv`` times (the :math:`\\alpha_k`,
    :math:`\\beta_k` are constants in :math:`x`) gives a recurrence for
    :math:`\\pi_k^{(m)}` for every :math:`m \\leq` ``deriv`` simultaneously:

    .. math::
        \\pi_{k+1}^{(m)}(x) = m \\, \\pi_k^{(m-1)}(x) + (x - \\alpha_k) \\,
            \\pi_k^{(m)}(x) - \\beta_k \\, \\pi_{k-1}^{(m)}(x)

    and the squared norm :math:`\\int \\pi_k^2 \\, w \\, dx = \\beta_0 \\beta_1
    \\cdots \\beta_k` (with :math:`\\beta_0` = the total mass) is a cumulative
    product of the same coefficients. All of this holds for *any* classical
    orthogonal polynomial family.

    Basis functions are the resulting *orthonormal* polynomials
    :math:`p_k = \\pi_k / \\sqrt{\\beta_0 \\cdots \\beta_k}`. This is one of
    only two normalizations derivable from ``(alpha, beta)`` alone (the
    other being the monic polynomials themselves); "classical" normalizations
    like ``scipy.special.eval_legendre``'s (:math:`P_n(1) = 1`) are
    family-specific conventions with no generic definition. Orthonormal is
    preferred over monic here because monic polynomials shrink rapidly with
    degree (on ``[-1, 1]``, degree-30 monic Legendre is O(1e-8)), which is a
    real conditioning problem at higher degree; orthonormal polynomials stay
    O(1) by construction.

    Target-domain evaluation maps ``measure.recurrence_coeffs``' *reference*
    coefficients via the same ``(scale, shift) = measure.affine_params(...)``
    used by ``archimedes.quadrature``: :math:`\\alpha' = \\mathrm{scale}
    \\cdot \\alpha + \\mathrm{shift}`, :math:`\\beta' = \\mathrm{scale}^2 \\cdot
    \\beta` except :math:`\\beta_0' = \\mathrm{measure.mass}(\\ldots)`
    (the total mass on the target domain/measure) -- so the resulting basis
    is orthonormal on the *target* domain, not just the reference one.

    Parameters
    ----------
    measure : Measure
        Defines the orthogonality weight and reference domain.
    n_basis : int
        Number of basis functions (polynomial degrees ``0`` through
        ``n_basis - 1``).
    density : bool, optional
        If ``True``, normalize against the *probability* density
        ``measure.weight / measure.mass(...)`` instead of the raw weight --
        i.e. use ``beta_0' = 1`` in place of ``beta_0' = mass(...)`` in the
        norm below. See :attr:`Basis.density`. Default ``False``.
    """

    measure: Measure
    n_basis: int
    density: bool = False

    def __post_init__(self):
        if self.n_basis < 1:
            raise ValueError(f"n_basis must be >= 1, got {self.n_basis}")

    @property
    def measures(self) -> tuple[Measure, ...]:
        """This basis's orthogonality weight; see :attr:`Basis.measures`."""
        return (self.measure,)

    @property
    def Parameters(self) -> type:  # noqa: N802
        """Delegates to the measure's ``ReferenceDomain`` -- ``a``/``b`` for
        a :class:`~archimedes.measure.UnitInterval` (Legendre/Jacobi),
        ``mean``/``std`` for :class:`~archimedes.measure.RealLine`
        (Hermite), ``rate``/``start`` for
        :class:`~archimedes.measure.HalfLine` (Laguerre)."""
        return type(self.measure.domain).Parameters

    def default_quadrature(self):
        """Gauss rule of ``n_basis`` points for this basis's own measure.

        A rule of :math:`n` Gauss points is exact to degree :math:`2n - 1`,
        which covers the degree-:math:`2(n_\\mathrm{basis} - 1)` mass-matrix
        integrand (and the lower-degree stiffness one).
        """
        from archimedes.quadrature import golub_welsch_rule

        return golub_welsch_rule(self.measure, self.n_basis)

    def _product_basis(self, other):
        """Same measure, ``n_1 + n_2 - 1`` functions.

        Both operands must be built on the same measure: the orthogonality
        weight is what defines the family, so polynomials orthogonal under
        different weights don't share a product space in this form.
        """
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
        """Same measure and normalization, ``n_basis - deriv`` functions.

        The measure is unchanged even for families whose classical
        derivative identity shifts it (Jacobi's
        :math:`\\frac{d}{dx} P_n^{(\\alpha,\\beta)} \\propto
        P_{n-1}^{(\\alpha+1,\\beta+1)}`). That identity says the derivative
        is a *single term* in the shifted family -- a sparsity statement --
        but the span here is all of :math:`P_{n-1}` regardless of which
        weight makes the basis orthogonal, so the derivative is exactly
        representable in this same family as a dense combination.
        """
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")
        if deriv == 0:
            return self
        if deriv >= self.n_basis:
            raise ValueError(
                f"deriv={deriv} is at or past the degree of a {self.n_basis}-"
                f"function basis, whose elements are polynomials of degree "
                f"{self.n_basis - 1}; the derivative is identically zero and "
                f"has no space of its own. Use `f(x, deriv={deriv})` if the "
                f"zero values are what you want."
            )
        return OrthogonalPolynomialBasis(
            self.measure, self.n_basis - deriv, density=self.density
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
