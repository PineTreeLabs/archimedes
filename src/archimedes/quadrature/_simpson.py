"""Composite Simpson's rule quadrature."""

from __future__ import annotations

import numpy as np

from ._gauss_legendre import gauss_lobatto
from ._quadrature_rule import QuadratureRule, composite_quad

__all__ = ["simpson"]


def simpson(n: int, a: float, b: float) -> QuadratureRule:
    r"""Composite Simpson's rule quadrature.

    Simpson's rule fits a parabola through the two ends and the midpoint
    of an interval:
    :math:`\int_a^b f(x) \, dx \approx \tfrac{h}{3}(f_0 + 4 f_1 + f_2)`
    
    The quadrature rule tiles ``n`` equal-width segments across the domain
    :math:`[a, b]`, with each segment using the Simpson quadrature rule.

    Parameters
    ----------
    n : int
        Number of integration segments
    a, b : float
        Bounds of the target interval

    Returns
    -------
    rule : QuadratureRule
        Composite Simpson's rule with ``3 * n`` nodes on
        :math:`[a, b]`, exact for a cubic on each segment.

    Raises
    ------
    ValueError
        If ``n < 1``.

    See Also
    --------
    gauss_lobatto : Equivalent single-segment quadrature rule.
    composite_quad : General tiling for piecewise quadrature rules.

    Notes
    -----
    Simpson's rule is equivalent to the 3-point Gauss-Lobatto quadrature rule,
    which is exact for cubics. The composite Simpson's rule is exact for
    piecewise cubics.
    """
    if n < 1:
        raise ValueError(f"simpson requires n >= 1, got {n}")
    breakpoints = np.linspace(-1.0, 1.0, n + 1)
    rule = composite_quad(gauss_lobatto(3), breakpoints)
    return rule.map_to(a, b)
