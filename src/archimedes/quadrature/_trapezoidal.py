"""Trapezoidal quadrature rule, plain and periodic."""

from __future__ import annotations

import numpy as np

from archimedes.measure import LegendreMeasure

from ._quadrature_rule import QuadratureRule

__all__ = ["trapezoidal"]


def trapezoidal(n: int, periodic: bool = False) -> QuadratureRule:
    """Equally-spaced trapezoidal quadrature rule.

    With ``periodic=False`` (default), the usual composite trapezoidal
    rule: :math:`n` nodes :math:`t_j = -1 + 2j/(n-1)`, :math:`j = 0,
    \\ldots, n-1`, spanning the **closed** reference interval
    :math:`[-1, 1]`, with half-weight at the two endpoints.

    With ``periodic=True``, nodes :math:`t_j = -1 + 2j/n`, equally spaced
    over the **half-open** period :math:`[-1, 1)`, with a node only at
    :math:`-1`: the domain is periodic, so :math:`-1` and :math:`+1`
    denote the same point and including both would double-count it. All
    weights equal :math:`2/n`. This form is exact for :math:`\\cos(k \\pi
    t)` and :math:`\\sin(k \\pi t)` for every :math:`1 \\leq k \\leq n -
    1`: for equally-spaced points over one period, :math:`\\sum_j e^{i k
    \\cdot 2\\pi j/n} = n` if :math:`k \\equiv 0 \\pmod n`, else :math:`0`,
    so any nonzero mode strictly below the Nyquist mode :math:`n`
    integrates to exactly zero.

    Parameters
    ----------
    n : int
        Number of quadrature nodes. Must be :math:`\\geq 2` if
        ``periodic`` is ``False`` (at least one interval to span), or
        :math:`\\geq 1` if ``periodic`` is ``True``.
    periodic : bool, optional
        If ``True``, build the periodic form described above -- the
        natural rule for a Fourier/trigonometric integrand -- instead of
        the plain closed-interval form. Default ``False``.

    Returns
    -------
    rule : QuadratureRule
        Trapezoidal rule with ``n`` nodes, exact for degree-1 polynomials
        (``periodic=False``) or for trigonometric polynomials of mode
        :math:`\\leq n - 1` (``periodic=True``).

    Raises
    ------
    ValueError
        If ``n`` is smaller than the minimum for the chosen ``periodic``
        setting.

    Notes
    -----
    Tiling this rule with :func:`composite_quad` is meaningful only for
    ``periodic=False``: its shared endpoint nodes double up across
    elements exactly like :func:`gauss_lobatto`'s. The periodic form's
    nodes already span the whole reference domain by construction, so
    tiling it is mechanically possible (the shared measure is
    uniform-weight) but not the intended use.
    """
    measure = LegendreMeasure()
    if periodic:
        if n < 1:
            raise ValueError(f"trapezoidal requires n >= 1 for periodic=True, got {n}")
        j = np.arange(n)
        x = -1.0 + 2.0 * j / n
        w = np.full(n, 2.0 / n)
        name = "trapezoidal_periodic"
    else:
        if n < 2:
            raise ValueError(f"trapezoidal requires n >= 2 for periodic=False, got {n}")
        x = np.linspace(-1.0, 1.0, n)
        w = np.full(n, 2.0 / (n - 1))
        w[0] *= 0.5
        w[-1] *= 0.5
        name = "trapezoidal"
    return QuadratureRule(x, w, measure=measure, name=name)
