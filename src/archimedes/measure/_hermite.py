"""Measures for the physicists' and probabilists' Hermite polynomial families."""

from __future__ import annotations

import numpy as np

from ._base import Measure
from ._domain import RealLine

__all__ = ["PhysicistsHermiteMeasure", "ProbabilistsHermiteMeasure"]


class PhysicistsHermiteMeasure(Measure):
    """Measure for the physicists' Hermite polynomial family.

    Weight :math:`w(x) = e^{-x^2}` on :math:`(-\\infty, \\infty)`.

    The associated orthogonal polynomials are the *physicists'* Hermite
    polynomials :math:`H_n(x)` (as opposed to the *probabilists'*
    convention used by :class:`ProbabilistsHermiteMeasure`, which instead uses
    weight :math:`e^{-x^2/2}`). The zeroth moment of the weight is
    :math:`\\int_{-\\infty}^\\infty e^{-x^2} \\, dx = \\sqrt{\\pi}`.
    """

    affine_invariant = True
    domain = RealLine()

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = e^{-x^2}`, evaluated at
        ``x``."""
        return np.exp(-(x**2))  # type: ignore[no-any-return]

    @property
    def reference_mass(self) -> float:
        """:math:`\\int_{-\\infty}^\\infty e^{-t^2} \\, dt = \\sqrt{\\pi}`."""
        return float(np.sqrt(np.pi))

    def recurrence_coeffs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Monic physicists' Hermite recurrence coefficients.

        :math:`\\alpha_k = 0`; :math:`\\beta_0 = \\sqrt{\\pi}` (=
        ``reference_mass``), :math:`\\beta_k = k / 2` for :math:`k \\geq 1`.
        """
        alpha = np.zeros(n)
        beta = np.arange(n, dtype=float) / 2
        beta[0] = self.reference_mass
        return alpha, beta


class ProbabilistsHermiteMeasure(Measure):
    """Measure for the probabilists' Hermite polynomial family.

    Weight :math:`w(x) = e^{-x^2/2}` on :math:`(-\\infty, \\infty)`.

    The associated orthogonal polynomials are the *probabilists'* Hermite
    polynomials :math:`\\mathit{He}_n(x)` (as opposed to the *physicists'*
    convention used by :class:`PhysicistsHermiteMeasure`, with weight
    :math:`e^{-x^2}`). Up to normalization, this weight is exactly the
    density of a standard normal distribution:
    :math:`e^{-x^2/2} = \\sqrt{2\\pi} \\, \\phi(x)`, where :math:`\\phi` is
    the standard normal PDF. The zeroth moment of the weight is
    :math:`\\int_{-\\infty}^\\infty e^{-x^2/2} \\, dx = \\sqrt{2\\pi}`.
    """

    affine_invariant = True
    domain = RealLine()

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function :math:`w(x) = e^{-x^2/2}`, evaluated at ``x``."""
        return np.exp(-(x**2) / 2)  # type: ignore[no-any-return]

    @property
    def reference_mass(self) -> float:
        """:math:`\\int_{-\\infty}^\\infty e^{-t^2/2} \\, dt =
        \\sqrt{2\\pi}`."""
        return float(np.sqrt(2 * np.pi))

    def recurrence_coeffs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Monic probabilists' Hermite recurrence coefficients.

        :math:`\\alpha_k = 0`; :math:`\\beta_0 = \\sqrt{2\\pi}` (=
        ``reference_mass``), :math:`\\beta_k = k` for :math:`k \\geq 1`.
        """
        alpha = np.zeros(n)
        beta = np.arange(n, dtype=float)
        beta[0] = self.reference_mass
        return alpha, beta
