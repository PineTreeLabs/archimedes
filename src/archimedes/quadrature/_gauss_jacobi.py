"""Gauss-Jacobi quadrature on the interval [-1, 1]."""

from __future__ import annotations

from scipy.special import roots_jacobi

from archimedes.measure import JacobiMeasure

from ._quadrature_rule import QuadratureRule

__all__ = ["gauss_jacobi"]


def gauss_jacobi(
    n: int,
    alpha: float,
    beta: float,
    a: float | None = None,
    b: float | None = None,
) -> QuadratureRule:
    r"""Gauss-Jacobi quadrature rule with ``n`` nodes.

    Nodes are the roots of the degree-``n`` Jacobi polynomial
    :math:`P_n^{(\alpha,\beta)}(x)`. The rule is exact for polynomials up
    to degree :math:`2n - 1`, weighted by the reference weight
    :math:`(1-x)^\alpha (1+x)^\beta` on :math:`[-1, 1]`.

    Gauss-Legendre is the special case :math:`\alpha = \beta = 0` (see
    :func:`gauss_legendre`); Gauss-Radau and Gauss-Lobatto also build on
    Jacobi polynomials with one or both exponents shifted to fix an
    endpoint (see :func:`gauss_radau`, :func:`gauss_lobatto`).

    Parameters
    ----------
    n : int
        Number of quadrature nodes.
    alpha, beta : float
        Exponents of the weight function. Must be :math:`> -1`.
    a, b : float, optional
        Bounds of the target interval. Must both be given, or neither. Defaults to
        :math:`[-1, 1]`.

    Returns
    -------
    rule : QuadratureRule
        Gauss-Jacobi rule with ``n`` nodes on :math:`[-1, 1]`, exact to
        degree :math:`2n - 1`, mapped onto :math:`[a, b]` if given.

    Raises
    ------
    ValueError
        If ``alpha`` or ``beta`` is :math:`\leq -1`.

    See Also
    --------
    gauss_legendre : The :math:`\alpha = \beta = 0` special case.
    """
    measure = JacobiMeasure(alpha=alpha, beta=beta)
    x, w = roots_jacobi(n, alpha, beta)
    rule = QuadratureRule.from_arrays(x, w, measure=measure, name="gauss_jacobi")
    if a is not None or b is not None:
        rule = rule.map_to(a, b)
    return rule
