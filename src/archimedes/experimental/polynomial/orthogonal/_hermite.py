from __future__ import annotations

import dataclasses

import numpy as np
from ._measure import Measure

__all__ = ["HermiteMeasure", "HermiteNormMeasure"]


class HermiteMeasure(Measure):
    """Measure for the physicists' Hermite polynomial family.
    
    Weight :math:`w(x) = e^{-x^2}` on :math:`(-\\infty, \\infty)`.

    The associated orthogonal polynomials are the *physicists'* Hermite
    polynomials :math:`H_n(x)` (as opposed to the *probabilists'*
    convention used by :class:`HermiteNormMeasure`, which instead uses
    weight :math:`e^{-x^2/2}`). The zeroth moment of the weight is
    :math:`\\int_{-\\infty}^\\infty e^{-x^2} \\, dx = \\sqrt{\\pi}`.
    """

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (-np.inf, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = e^{-x^2}`, evaluated at
        `x`."""
        return np.exp(-(x**2))

    def affine_params(self, mean=None, std=None) -> tuple[float, float]:
        """Map the reference weight onto a location-scaled Gaussian-shaped
        weight :math:`w(x) = \\exp(-((x - \\mathrm{mean})/
        \\mathrm{std})^2)`.

        Substituting :math:`x = \\mathrm{mean} + \\mathrm{std} \\cdot t`
        gives

        .. math::
            \\int_{-\\infty}^\\infty f(x) \\,
                e^{-((x-\\mathrm{mean})/\\mathrm{std})^2} \\, dx
            = \\mathrm{std} \\int_{-\\infty}^\\infty
                f(\\mathrm{mean} + \\mathrm{std} \\cdot t) \\, e^{-t^2} \\,
                dt,

        so :math:`\\mathrm{scale} = \\mathrm{std}` and
        :math:`\\mathrm{shift} = \\mathrm{mean}`.

        Note that because the reference weight uses the physicists'
        normalization :math:`e^{-t^2}` rather than the probabilists'
        :math:`e^{-t^2/2}`, `std` is *not* the standard deviation of a
        Gaussian density with this shape -- that would be :math:`\\sigma =
        \\mathrm{std} / \\sqrt{2}`.

        Parameters
        ----------
        mean : float, optional
            Location of the target weight. Default 0.
        std : float, optional
            Scale of the target weight. Default 1. Equal to
            :math:`\\sqrt{2}` times the standard deviation of the
            corresponding Gaussian density.

        Returns
        -------
        scale, shift : float
            Affine parameters as derived above.

        Raises
        ------
        ValueError
            If `std` is not positive.
        """
        if mean is None and std is None:
            return 1.0, 0.0
        if mean is None:
            mean = 0.0
        if std is None:
            std = 1.0
        if isinstance(std, float) and std <= 0:
            raise ValueError(f"Gauss-Hermite std must be positive, got {std}")
        return std, mean


class HermiteNormMeasure(Measure):
    """Measure for the probabilists' Hermite polynomial family.
    
    Weight :math:`w(x) = e^{-x^2/2}` on :math:`(-\\infty, \\infty)`.

    The associated orthogonal polynomials are the *probabilists'* Hermite
    polynomials :math:`\\mathit{He}_n(x)` (as opposed to the *physicists'*
    convention used by :class:`HermiteMeasure`, with weight
    :math:`e^{-x^2}`). Up to normalization, this weight is exactly the
    density of a standard normal distribution:
    :math:`e^{-x^2/2} = \\sqrt{2\\pi} \\, \\phi(x)`, where :math:`\\phi` is
    the standard normal PDF. The zeroth moment of the weight is
    :math:`\\int_{-\\infty}^\\infty e^{-x^2/2} \\, dx = \\sqrt{2\\pi}`.
    """

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (-np.inf, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = e^{-x^2/2}`, evaluated
        at `x`."""
        return np.exp(-(x**2) / 2)

    def affine_params(self, mean=None, std=None) -> tuple[float, float]:
        """Map the reference weight onto a Gaussian weight with the given
        mean and standard deviation:
        :math:`w(x) = \\exp(-(x - \\mathrm{mean})^2 /
        (2 \\, \\mathrm{std}^2))`.

        Substituting :math:`x = \\mathrm{mean} + \\mathrm{std} \\cdot t`
        gives

        .. math::
            \\int_{-\\infty}^\\infty f(x) \\,
                e^{-(x-\\mathrm{mean})^2/(2 \\, \\mathrm{std}^2)} \\, dx
            = \\mathrm{std} \\int_{-\\infty}^\\infty
                f(\\mathrm{mean} + \\mathrm{std} \\cdot t) \\, e^{-t^2/2}
                \\, dt,

        so :math:`\\mathrm{scale} = \\mathrm{std}` and
        :math:`\\mathrm{shift} = \\mathrm{mean}`. Unlike
        `HermiteNormMeasure.affine_params`, `std` here is exactly the standard
        deviation of the corresponding Gaussian density -- the reference
        weight already uses the probabilists' normalization, so no
        :math:`\\sqrt{2}` correction is needed.

        Parameters
        ----------
        mean : float, optional
            Mean of the target Gaussian weight. Default 0.
        std : float, optional
            Standard deviation of the target Gaussian weight. Default 1.

        Returns
        -------
        scale, shift : float
            Affine parameters as derived above.

        Raises
        ------
        ValueError
            If `std` is not positive.
        """
        if mean is None and std is None:
            return 1.0, 0.0
        if mean is None:
            mean = 0.0
        if std is None:
            std = 1.0
        if isinstance(std, float) and std <= 0:
            raise ValueError(f"Gauss-Hermite std must be positive, got {std}")
        return std, mean