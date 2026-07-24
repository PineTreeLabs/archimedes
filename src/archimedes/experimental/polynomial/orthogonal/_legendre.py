"""Measure for the Legendre polynomial family."""

from __future__ import annotations

import numpy as np
from ._measure import Measure

__all__ = ["LegendreMeasure"]


class LegendreMeasure(Measure):
    """Measure for the Legendre polynomial family.

    Weight :math:`w(x) = 1` on :math:`[-1, 1]`.
    """

    uniform_weight = True

    @property
    def support(self) -> tuple[float, float]:
        """Support :math:`[-1, 1]`."""
        return (-1.0, 1.0)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = 1`, evaluated at ``x``."""
        return np.ones_like(x)

    @property
    def reference_mass(self) -> float:
        """:math:`\\int_{-1}^1 1 \\, dt = 2`."""
        return 2.0

    def affine_params(self, a=None, b=None) -> tuple[float, float]:
        """Map the reference interval :math:`[-1, 1]` onto :math:`[a, b]`.

        The affine map :math:`x = \\mathrm{scale} \\cdot t +
        \\mathrm{shift}` with

        .. math::
            \\mathrm{scale} = \\frac{b - a}{2}, \\qquad
            \\mathrm{shift} = \\frac{a + b}{2}

        carries the reference measure onto :math:`[a, b]` while
        preserving the (constant) shape of the weight function.

        Parameters
        ----------
        a, b : float, optional
            Bounds of the target interval. Must both be given, or neither
            (falls back to the reference domain, ``(1.0, 0.0)``).

        Returns
        -------
        scale, shift : float
            Affine parameters mapping reference nodes onto :math:`[a, b]`.

        Raises
        ------
        ValueError
            If only one of ``a``, ``b`` is given, or if ``a``/``b`` are not finite.
        """
        if a is None and b is None:
            return 1.0, 0.0
        if a is None or b is None:
            raise ValueError("specify both `a` and `b`, or neither")
        if (isinstance(a, float) and not np.isfinite(a)) or (
            isinstance(b, float) and not np.isfinite(b)
        ):
            raise ValueError(
                f"{type(self).__name__} requires a finite domain, got ({a}, {b})"
            )
        lo, hi = self.support
        scale = (b - a) / (hi - lo)
        return scale, a - scale * lo
