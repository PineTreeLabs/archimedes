"""Gauss-Laguerre quadrature on the half-line."""

from __future__ import annotations

from scipy.special import roots_laguerre

from archimedes.measure import LaguerreMeasure

from ._quadrature_rule import QuadratureRule

__all__ = ["gauss_laguerre"]


def gauss_laguerre(n: int) -> QuadratureRule:
    """Gauss-Laguerre quadrature rule with ``n`` nodes.

    Nodes are the roots of the degree-``n`` Laguerre polynomial
    :math:`L_n(x)`. The rule is exact for polynomials up to degree
    :math:`2n - 1`, weighted by the reference weight :math:`e^{-x}` on
    :math:`[0, \\infty)`.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.

    Returns
    -------
    rule : QuadratureRule
        Gauss-Laguerre rule with ``n`` nodes on :math:`[0, \\infty)`, exact
        to degree :math:`2n - 1`.
    """
    x, w = roots_laguerre(n)
    measure = LaguerreMeasure()
    return QuadratureRule(x, w, measure=measure, name="gauss_laguerre")
