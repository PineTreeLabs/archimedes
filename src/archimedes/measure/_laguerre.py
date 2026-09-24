"""Measure for the Laguerre polynomial family."""

from __future__ import annotations

import numpy as np

from ._base import Measure
from ._domain import HalfLine

__all__ = ["LaguerreMeasure"]


class LaguerreMeasure(Measure):
    """Measure for the Laguerre polynomial family.

    Weight :math:`w(x) = e^{-x}` on :math:`[0, \\infty)`.

    The associated orthogonal polynomials are the (physicists') Laguerre
    polynomials :math:`L_n(x)`. Moments of the weight are given by the
    Gamma function: :math:`\\int_0^\\infty x^k e^{-x} \\, dx = k! =
    \\Gamma(k+1)`.
    """

    affine_invariant = True
    domain = HalfLine()

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function :math:`w(x) = e^{-x}`, evaluated at
        ``x``."""
        return np.exp(-x)  # type: ignore[no-any-return]

    @property
    def reference_mass(self) -> float:
        """:math:`\\int_0^\\infty e^{-t} \\, dt = 1`."""
        return 1.0

    def recurrence_coeffs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Monic Laguerre recurrence coefficients.

        :math:`\\alpha_k = 2k + 1`; :math:`\\beta_0 = 1` (=
        ``reference_mass``), :math:`\\beta_k = k^2` for :math:`k \\geq 1`.
        """
        k = np.arange(n, dtype=float)
        alpha = 2 * k + 1
        beta = k**2
        beta[0] = self.reference_mass
        return alpha, beta
