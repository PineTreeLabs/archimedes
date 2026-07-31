"""Periodic (trapezoidal) quadrature rule for trigonometric integrands."""

from __future__ import annotations

import numpy as np

from archimedes.measure import LegendreMeasure

from ._quadrature_rule import QuadratureRule

__all__ = ["periodic_trapezoidal"]


def periodic_trapezoidal(n: int) -> QuadratureRule:
    """Equally-spaced quadrature rule for a periodic (Fourier) integrand.

    Nodes :math:`t_j = -1 + 2j/n`, :math:`j = 0, \\ldots, n-1`, equally
    spaced over the **half-open** reference period :math:`[-1, 1)`. Unlike
    :func:`gauss_lobatto`/:func:`clenshaw_curtis`, which place distinct nodes
    at *both* :math:`-1` and :math:`+1`, this rule places a node only at
    :math:`-1`: the domain is periodic, so :math:`-1` and :math:`+1` denote
    the same point and including both would double-count it. All weights
    equal :math:`2/n`, so :math:`\\sum_j w_j = 2`, the same total mass as
    every other rule sharing :class:`~archimedes.measure.LegendreMeasure`.

    Exact for :math:`\\cos(k \\pi t)` and :math:`\\sin(k \\pi t)`
    for every :math:`1 \\leq k \\leq n - 1`: for equally-spaced points
    over one period, :math:`\\sum_j e^{i k \\cdot 2\\pi j/n} = n` if
    :math:`k \\equiv 0 \\pmod n`, else :math:`0`, so any nonzero mode strictly
    below the Nyquist mode :math:`n` integrates to exactly zero.

    Shares :class:`~archimedes.measure.LegendreMeasure` (uniform weight on
    :math:`[-1, 1]`) with :func:`gauss_legendre`, :func:`gauss_lobatto`,
    :func:`gauss_radau`, and :func:`clenshaw_curtis`, even though it is not a
    Gauss rule for that measure. Its accuracy is calibrated to trigonometric, not
    ordinary, polynomials, so tiling it with :func:`composite_quad` is
    mechanically possible (the shared measure is uniform-weight) but not the
    intended use: a periodic rule's nodes already span the whole reference
    domain by construction.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.

    Returns
    -------
    rule : QuadratureRule
        Periodic-trapezoidal rule with ``n`` nodes on :math:`[-1, 1)`, exact
        for trigonometric polynomials of mode :math:`\\leq n - 1`.

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"periodic_trapezoidal requires n >= 1, got {n}")
    j = np.arange(n)
    x = -1.0 + 2.0 * j / n
    w = np.full(n, 2.0 / n)
    return QuadratureRule(x, w, measure=LegendreMeasure(), name="periodic_trapezoidal")
