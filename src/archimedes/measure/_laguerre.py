"""Measure for the Laguerre polynomial family."""

from __future__ import annotations

import numpy as np

from archimedes import tree

from ._base import Measure

__all__ = ["LaguerreMeasure"]


class LaguerreMeasure(Measure):
    """Measure for the Laguerre polynomial family.

    Weight :math:`w(x) = e^{-x}` on :math:`[0, \\infty)`.

    The associated orthogonal polynomials are the (physicists') Laguerre
    polynomials :math:`L_n(x)`. Moments of the weight are given by the
    Gamma function: :math:`\\int_0^\\infty x^k e^{-x} \\, dx = k! =
    \\Gamma(k+1)`.
    """

    @tree.struct
    class Parameters(Measure.Parameters):
        """Rate/location of the target exponential weight; see ``affine_params``."""

        rate: float = 1.0
        start: float = 0.0

        def __post_init__(self):
            if isinstance(self.rate, float) and self.rate <= 0:
                raise ValueError(
                    f"Gauss-Laguerre rate must be positive, got {self.rate}"
                )

    @property
    def support(self) -> tuple[float, float]:
        """Support :math:`[0, \\infty)`."""
        return (0.0, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function :math:`w(x) = e^{-x}`, evaluated at
        ``x``."""
        return np.exp(-x)  # type: ignore[no-any-return]

    @property
    def reference_mass(self) -> float:
        """:math:`\\int_0^\\infty e^{-t} \\, dt = 1`."""
        return 1.0

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
        kwargs = {}
        if rate is not None:
            kwargs["rate"] = rate
        if start is not None:
            kwargs["start"] = start
        params = self.Parameters(**kwargs)
        return 1.0 / params.rate, params.start

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
