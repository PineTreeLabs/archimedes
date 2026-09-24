"""Gauss-Laguerre quadrature on the half-line."""

from __future__ import annotations

from scipy.special import roots_laguerre

from archimedes.measure import LaguerreMeasure

from ._quadrature_rule import QuadratureRule

__all__ = ["gauss_laguerre"]


def gauss_laguerre(
    n: int, rate: float | None = None, start: float | None = None
) -> QuadratureRule:
    """Gauss-Laguerre quadrature rule with ``n`` nodes.

    Nodes are the roots of the degree-``n`` Laguerre polynomial
    :math:`L_n(x)`. The rule is exact for polynomials up to degree
    :math:`2n - 1`, weighted by the reference weight :math:`e^{-x}` on
    :math:`[0, \\infty)`.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.
    rate, start : float, optional
        Rate/location of the target exponential weight. Default
        ``rate=1``, ``start=0``.

    Returns
    -------
    rule : QuadratureRule
        Gauss-Laguerre rule with ``n`` nodes on :math:`[0, \\infty)`, exact
        to degree :math:`2n - 1`, mapped by ``rate``/``start`` if given.
    """
    x, w = roots_laguerre(n)
    measure = LaguerreMeasure()
    rule = QuadratureRule.from_arrays(x, w, measure=measure, name="gauss_laguerre")
    if rate is not None or start is not None:
        rule = rule.map_to(rate=rate, start=start)
    return rule
