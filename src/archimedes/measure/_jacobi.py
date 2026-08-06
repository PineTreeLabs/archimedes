"""Measure for the Jacobi polynomial family."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.special import beta as beta_fn

from ._base import Measure
from ._domain import UnitInterval

__all__ = ["JacobiMeasure"]


@dataclasses.dataclass(frozen=True)
class JacobiMeasure(Measure):
    """Measure for the Jacobi polynomial family.

    Weight :math:`w(x) = (1-x)^\\alpha (1+x)^\\beta` on
    :math:`[-1, 1]`, with :math:`\\alpha, \\beta > -1`.

    The associated orthogonal polynomials are the Jacobi polynomials
    :math:`P_n^{(\\alpha,\\beta)}(x)`. Gauss-Legendre is the special case
    :math:`\\alpha = \\beta = 0`; Chebyshev quadrature of the first and
    second kind are the special cases :math:`\\alpha = \\beta = -1/2` and
    :math:`\\alpha = \\beta = 1/2`, respectively. The zeroth moment
    (normalization) of the weight has a closed form in terms of the Beta
    function:

    .. math::
        \\int_{-1}^1 (1-x)^\\alpha (1+x)^\\beta \\, dx
            = 2^{\\alpha + \\beta + 1} \\, B(\\alpha + 1, \\beta + 1)

    Shares the :class:`UnitInterval` reference domain (and hence
    ``affine_params``) with :class:`LegendreMeasure`; the two differ only
    in their weight.

    Parameters
    ----------
    alpha, beta : float
        Exponents of the weight function. Must be :math:`> -1` for the
        weight to be integrable at the corresponding endpoint.

    Raises
    ------
    ValueError
        If ``alpha`` or ``beta`` is :math:`\\leq -1`.
    """

    alpha: float
    beta: float

    domain = UnitInterval()

    # Not a dataclass field: unannotated, so `dataclasses` leaves it as a
    # plain class attribute overriding `LegendreMeasure.uniform_weight`.
    # The Jacobi weight is singular at the reference endpoints, so it
    # cannot be tiled into a composite rule -- see `Measure.uniform_weight`.
    uniform_weight = False

    # Also not a dataclass field, same reasoning: `alpha`/`beta` are
    # dimensionless shape exponents that don't interact with the affine
    # domain map, so the closed-form recurrence stays valid under any
    # `a`/`b` remap -- see `Measure.affine_invariant`.
    affine_invariant = True

    def __post_init__(self):
        if self.alpha <= -1 or self.beta <= -1:
            raise ValueError(
                f"invalid alpha={self.alpha} or beta={self.beta}, must be > -1"
            )

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = (1-x)^\\alpha
        (1+x)^\\beta`, evaluated at ``x``."""
        return (1 - x) ** self.alpha * (1 + x) ** self.beta

    @property
    def reference_mass(self) -> float:
        """:math:`2^{\\alpha + \\beta + 1} \\, B(\\alpha + 1, \\beta + 1)`."""
        return float(
            2 ** (self.alpha + self.beta + 1) * beta_fn(self.alpha + 1, self.beta + 1)
        )

    def recurrence_coeffs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Monic Jacobi recurrence coefficients (DLMF 18.9.2_1-2).

        .. math::
            \\alpha_0 = \\frac{\\beta - \\alpha}{\\alpha + \\beta + 2}, \\qquad
            \\alpha_k = \\frac{\\beta^2 - \\alpha^2}
                {(2k+\\alpha+\\beta)(2k+\\alpha+\\beta+2)}, \\quad k \\geq 1

        .. math::
            \\beta_0 = \\mathrm{reference\\_mass}, \\qquad
            \\beta_1 = \\frac{4(1+\\alpha)(1+\\beta)}
                {(2+\\alpha+\\beta)^2(3+\\alpha+\\beta)}, \\qquad
            \\beta_k = \\frac{4k(k+\\alpha)(k+\\beta)(k+\\alpha+\\beta)}
                {(2k+\\alpha+\\beta)^2(2k+\\alpha+\\beta+1)(2k+\\alpha+\\beta-1)},
            \\quad k \\geq 2

        The :math:`\\alpha_0` and :math:`\\beta_1` forms above are the
        already-simplified (common-factor-cancelled) versions of the
        general-:math:`k` formulas. Evaluated directly, the general formulas
        have removable :math:`0/0` singularities at :math:`k=0` when
        :math:`\\alpha+\\beta=0` (the Legendre point) and at :math:`k=1`
        when :math:`\\alpha+\\beta=-1` (the Chebyshev-first-kind point,
        :math:`\\alpha=\\beta=-1/2`, and any other pair summing to :math:`-1`)
        -- both within the valid domain (:math:`\\alpha,\\beta > -1`, so
        :math:`\\alpha+\\beta > -2`), and both explicitly named as important
        special cases in this class's docstring. So ``alpha[0]`` and
        ``beta[1]`` are assigned the simplified forms directly rather than
        computed via the general-:math:`k` expression (which would evaluate
        the singular branch elementwise even under ``np.where``).
        """
        a, b = self.alpha, self.beta
        alpha = np.empty(n)
        alpha[0] = (b - a) / (a + b + 2)
        if n > 1:
            k = np.arange(1, n, dtype=float)
            alpha[1:] = (b**2 - a**2) / ((2 * k + a + b) * (2 * k + a + b + 2))

        beta = np.empty(n)
        beta[0] = self.reference_mass
        if n > 1:
            beta[1] = 4 * (1 + a) * (1 + b) / ((2 + a + b) ** 2 * (3 + a + b))
        if n > 2:
            k = np.arange(2, n, dtype=float)
            beta[2:] = (
                4
                * k
                * (k + a)
                * (k + b)
                * (k + a + b)
                / ((2 * k + a + b) ** 2 * (2 * k + a + b + 1) * (2 * k + a + b - 1))
            )
        return alpha, beta
