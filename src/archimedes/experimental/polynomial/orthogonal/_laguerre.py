"""Measure for the Laguerre polynomial family."""

from __future__ import annotations

import dataclasses

import numpy as np
from ._measure import Measure

__all__ = ["LaguerreMeasure"]


class LaguerreMeasure(Measure):
    """Measure for the Laguerre polynomial family.

    Weight :math:`w(x) = e^{-x}` on :math:`[0, \\infty)`.

    The associated orthogonal polynomials are the (physicists') Laguerre
    polynomials :math:`L_n(x)`. Moments of the weight are given by the
    Gamma function: :math:`\\int_0^\\infty x^k e^{-x} \\, dx = k! =
    \\Gamma(k+1)`.
    """

    @property
    def support(self) -> tuple[float, float]:
        """Support :math:`[0, \\infty)`."""
        return (0.0, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Wweight function :math:`w(x) = e^{-x}`, evaluated at
        ``x``."""
        return np.exp(-x)

    def affine_params(self, rate=None, start=None) -> tuple[float, float]:
        """Map the reference weight onto a rate/shifted exponential weight
        :math:`w(x) = e^{-\\mathrm{rate}(x - \\mathrm{start})}` on
        :math:`[\\mathrm{start}, \\infty)`.

        Substituting :math:`x = \\mathrm{start} + t / \\mathrm{rate}` into
        the reference integral gives

        .. math::
            \\int_{\\mathrm{start}}^\\infty f(x) \\,
                e^{-\\mathrm{rate}(x - \\mathrm{start})} \\, dx
            = \\frac{1}{\\mathrm{rate}} \\int_0^\\infty
                f(\\mathrm{start} + t/\\mathrm{rate}) \\, e^{-t} \\, dt,

        so :math:`\\mathrm{scale} = 1/\\mathrm{rate}` and
        :math:`\\mathrm{shift} = \\mathrm{start}`.

        Parameters
        ----------
        rate : float, optional
            Rate of the target exponential weight. Default 1.
        start : float, optional
            Left endpoint of the target domain. Default 0.

        Returns
        -------
        scale, shift : float
            Affine parameters as derived above.

        Raises
        ------
        ValueError
            If ``rate`` is not positive.
        """
        if rate is None and start is None:
            return 1.0, 0.0
        if rate is None:
            rate = 1.0
        if start is None:
            start = 0.0
        if isinstance(rate, float) and rate <= 0:
            raise ValueError(f"Gauss-Laguerre rate must be positive, got {rate}")
        return 1.0 / rate, start
