"""Gauss-Hermite quadrature on the real line."""

from __future__ import annotations

from scipy.special import roots_hermite, roots_hermitenorm

from archimedes.experimental.polynomial.orthogonal import (
    HermiteMeasure,
    HermiteNormMeasure,
    Measure,
)
from ._quadrature_rule import QuadratureRule

__all__ = ["gauss_hermite"]


def gauss_hermite(n: int, norm: bool = False) -> QuadratureRule:
    """Gauss-Hermite quadrature rule with ``n`` nodes.

    Nodes are the roots of the degree-``n`` Hermite polynomial. The rule is
    exact for polynomials up to degree :math:`2n - 1`, weighted by the
    Gaussian-shaped reference weight of ``measure``.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.
    norm : bool, optional
        If ``False`` (default), use the *physicists'* convention, with
        reference weight :math:`e^{-x^2}` (:class:`HermiteMeasure`). If
        ``True``, use the *probabilists'* convention, with reference weight
        :math:`e^{-x^2/2}` (:class:`HermiteNormMeasure`) -- up to
        normalization, the standard normal density.

    Returns
    -------
    rule : QuadratureRule
        Gauss-Hermite rule with ``n`` nodes on :math:`(-\\infty, \\infty)`,
        exact to degree :math:`2n - 1`.
    """
    measure: Measure
    if norm:
        x, w = roots_hermitenorm(n)
        measure = HermiteNormMeasure()
    else:
        x, w = roots_hermite(n)
        measure = HermiteMeasure()
    return QuadratureRule(x, w, measure=measure, name="gauss_hermite")
