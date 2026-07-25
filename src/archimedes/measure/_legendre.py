"""Measure for the Legendre polynomial family."""

from __future__ import annotations

import numpy as np

from ._base import Measure
from ._domain import UnitInterval

__all__ = ["LegendreMeasure"]


class LegendreMeasure(Measure):
    """Measure for the Legendre polynomial family.

    Weight :math:`w(x) = 1` on :math:`[-1, 1]` (a :class:`UnitInterval`
    domain, shared with :class:`JacobiMeasure`).
    """

    uniform_weight = True
    domain = UnitInterval()

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = 1`, evaluated at ``x``."""
        return np.ones_like(x)

    @property
    def reference_mass(self) -> float:
        """:math:`\\int_{-1}^1 1 \\, dt = 2`."""
        return 2.0

    def recurrence_coeffs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Monic Legendre recurrence coefficients.

        :math:`\\alpha_k = 0`; :math:`\\beta_0 = 2` (= ``reference_mass``),
        :math:`\\beta_k = k^2 / (4k^2 - 1)` for :math:`k \\geq 1`.
        """
        alpha = np.zeros(n)
        beta = np.empty(n)
        beta[0] = self.reference_mass
        if n > 1:
            k = np.arange(1, n, dtype=float)
            beta[1:] = k**2 / (4 * k**2 - 1)
        return alpha, beta
