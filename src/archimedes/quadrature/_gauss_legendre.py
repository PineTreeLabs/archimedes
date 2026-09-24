"""Gaussian quadrature rules on the interval [-1, 1].

Includes Gauss-Legendre, Gauss-Radau, and Gauss-Lobatto rules, as well as
Clenshaw-Curtis quadrature (Chebyshev-Lobatto nodes). All rules share the
same Legendre measure, so they can be tiled into composite rules with
:func:`composite_quad`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import roots_jacobi, roots_legendre

from archimedes.measure import LegendreMeasure

from ._quadrature_rule import QuadratureRule


def gauss_legendre(
    n: int, a: float | None = None, b: float | None = None
) -> QuadratureRule:
    """Gauss-Legendre quadrature rule with ``n`` nodes.

    Nodes are the roots of the degree-``n`` Legendre polynomial
    :math:`P_n(x)`, none of which coincide with the endpoints
    :math:`\\pm 1`. The rule is exact for polynomials up to degree
    :math:`2n - 1`.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.
    a, b : float, optional
        Bounds of the target interval, forwarded to
        :meth:`~archimedes.quadrature.QuadratureRule.map_to`. Must both be
        given, or neither (the default, giving the reference interval
        :math:`[-1, 1]`).

    Returns
    -------
    rule : QuadratureRule
        Gauss-Legendre rule with ``n`` nodes on :math:`[-1, 1]`, exact to
        degree :math:`2n - 1`, mapped onto :math:`[a, b]` if given.
    """
    x, w = roots_legendre(n)
    measure = LegendreMeasure()
    rule = QuadratureRule.from_arrays(x, w, measure=measure, name="gauss_legendre")
    if a is not None or b is not None:
        rule = rule.map_to(a, b)
    return rule


def gauss_radau(
    n: int, endpoint: str = "left", a: float | None = None, b: float | None = None
) -> QuadratureRule:
    """Gauss-Radau quadrature rule including exactly one endpoint.

    Radau rules fix one endpoint of :math:`[-1, 1]` as a node and choose
    the remaining :math:`n - 1` nodes to maximize the polynomial degree of
    exactness. Fixing the left endpoint, these are the roots of the Jacobi
    polynomial :math:`P_{n-1}^{(0,1)}(x)`; fixing the right endpoint, the
    roots of :math:`P_{n-1}^{(1,0)}(x)`. The rule is exact for polynomials
    up to degree :math:`2n - 2`.

    Parameters
    ----------
    n : int
        Number of quadrature nodes, including the fixed endpoint.
    endpoint : {"left", "right"}, optional
        Which endpoint of :math:`[-1, 1]` to fix as a node:

        - ``"left"`` includes :math:`-1` (LGR, the pseudospectral
          convention).
        - ``"right"`` includes :math:`+1` (Radau IIA, the IRK/DAE
          convention).
    a, b : float, optional
        Bounds of the target interval, forwarded to
        :meth:`~archimedes.quadrature.QuadratureRule.map_to`. Must both be
        given, or neither (the default, giving the reference interval
        :math:`[-1, 1]`).

    Returns
    -------
    rule : QuadratureRule
        Gauss-Radau rule with ``n`` nodes on :math:`[-1, 1]`, exact to
        degree :math:`2n - 2`, mapped onto :math:`[a, b]` if given.

    Raises
    ------
    ValueError
        If ``n < 1``, or if ``endpoint`` is not ``"left"`` or ``"right"``.
    """
    measure = LegendreMeasure()
    if n < 1:
        raise ValueError("Gauss-Radau requires n >= 1")
    if n == 1:
        x, w = np.array([-1.0]), np.array([2.0])
    else:
        x, w = roots_jacobi(n - 1, 0.0, 1.0)
        w = w / (1 + x)
        x = np.insert(x, 0, -1.0)
        w = np.insert(w, 0, 2.0 / n**2)
    if endpoint == "right":
        x, w = -x[::-1], w[::-1]
    elif endpoint != "left":
        raise ValueError(f"endpoint must be 'left' or 'right', got {endpoint!r}")
    rule = QuadratureRule.from_arrays(
        x, w, measure=measure, name=f"gauss_radau_{endpoint}"
    )
    if a is not None or b is not None:
        rule = rule.map_to(a, b)
    return rule


def gauss_lobatto(
    n: int, a: float | None = None, b: float | None = None
) -> QuadratureRule:
    """Gauss-Lobatto quadrature rule including both endpoints.

    Lobatto rules fix both endpoints :math:`\\pm 1` as nodes and choose the
    remaining :math:`n - 2` interior nodes to maximize the polynomial
    degree of exactness -- the roots of the Jacobi polynomial
    :math:`P_{n-2}^{(1,1)}(x)`, equivalently the roots of :math:`P_{n-1}'
    (x)`, the derivative of the degree-:math:`(n-1)` Legendre polynomial.
    The rule is exact for polynomials up to degree :math:`2n - 3`.

    Parameters
    ----------
    n : int
        Number of quadrature nodes, including both endpoints.
    a, b : float, optional
        Bounds of the target interval, forwarded to
        :meth:`~archimedes.quadrature.QuadratureRule.map_to`. Must both be
        given, or neither (the default, giving the reference interval
        :math:`[-1, 1]`).

    Returns
    -------
    rule : QuadratureRule
        Gauss-Lobatto rule with ``n`` nodes on :math:`[-1, 1]`, exact to
        degree :math:`2n - 3`, mapped onto :math:`[a, b]` if given.

    Raises
    ------
    ValueError
        If ``n < 2``.
    """
    measure = LegendreMeasure()
    if n < 2:
        raise ValueError("Gauss-Lobatto requires n >= 2")
    if n == 2:
        x, w = np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    else:
        x, w = roots_jacobi(n - 2, 1.0, 1.0)
        w = w / (1 - x**2)
        x = np.concatenate([[-1.0], x, [1.0]])
        end_w = 2.0 / (n * (n - 1))
        w = np.concatenate([[end_w], w, [end_w]])
    rule = QuadratureRule.from_arrays(x, w, measure=measure, name="gauss_lobatto")
    if a is not None or b is not None:
        rule = rule.map_to(a, b)
    return rule


def clenshaw_curtis(
    n: int, a: float | None = None, b: float | None = None
) -> QuadratureRule:
    """Clenshaw-Curtis quadrature rule with ``n`` nodes.

    Nodes are the extrema of the degree-:math:`(n-1)` Chebyshev polynomial
    :math:`T_{n-1}(x)` (the Chebyshev-Lobatto points), including both
    endpoints :math:`\\pm 1`:

    .. math::
        x_k = \\cos(k \\pi / (n - 1)), \\quad k = 0, \\ldots, n - 1.

    Unlike Gauss quadrature, these nodes are not chosen to maximize
    polynomial exactness -- the rule is only guaranteed exact for
    polynomials up to degree :math:`n - 1`, half that of Gauss-Legendre
    for the same node count. The weight function is still uniform,
    so this rule shares the Legendre measure with ``gauss_legendre``,
    ``gauss_radau``, and ``gauss_lobatto``, and can be tiled with
    ``composite_quad``.
    The tradeoff for the lower degree of exactness is that
    Chebyshev-Lobatto nodes are nested across doublings of ``n`` and cheap,
    numerically stable to compute for very large ``n``.

    Weights are computed via a :math:`O(n \\log n)` algorithm from
    Waldvogel [1]_, which expresses them as the inverse DFT of an
    explicit, rational moment vector.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.
    a, b : float, optional
        Bounds of the target interval. Must both be given, or neither. Defaults to
        :math:`[-1, 1]`.

    Returns
    -------
    rule : QuadratureRule
        Clenshaw-Curtis rule with ``n`` nodes on :math:`[-1, 1]`, exact to
        degree :math:`n - 1`, mapped onto :math:`[a, b]` if given.

    Raises
    ------
    ValueError
        If ``n < 2``.

    References
    ----------
    .. [1] J. Waldvogel, "Fast Construction of the Fej\\'er and
        Clenshaw-Curtis Quadrature Rules", BIT Numerical Mathematics,
        Vol. 43, No. 1, 2003, pp. 1-18.
    """
    measure = LegendreMeasure()
    if n < 2:
        raise ValueError("Clenshaw-Curtis requires n >= 2")
    if n == 2:
        rule = QuadratureRule.from_arrays(
            np.array([-1.0, 1.0]),
            np.array([1.0, 1.0]),
            measure=measure,
            name="clenshaw_curtis",
        )
        if a is not None or b is not None:
            rule = rule.map_to(a, b)
        return rule

    # `d`, Waldvogel's node/weight count, is one less than here: nodes are
    # x_k = cos(k*pi/d), k = 0, ..., d (d + 1 = n nodes total). Ported from
    # the `fejer` MATLAB listing in [1]_, keeping only the `wcc` branch.
    d = n - 1
    odd = np.arange(1, d, 2, dtype=float)  # MATLAB `N = 1:2:d-1`
    n_odd = len(odd)
    n_even = d - n_odd

    v0 = np.concatenate([2.0 / (odd * (odd - 2)), [1.0 / odd[-1]], np.zeros(n_even)])
    v = -v0[:-1] - v0[-1:0:-1]

    g0 = -np.ones(d)
    g0[n_odd] += d
    g0[n_even] += d
    g = g0 / (d**2 - 1 + (d % 2))

    w = np.fft.ifft(v + g).real
    w = np.concatenate([w, [w[0]]])

    k = np.arange(d + 1)
    x = np.cos(np.pi * k / d)

    # Descending (x[0] = 1) to ascending, matching the other rules' node order
    rule = QuadratureRule.from_arrays(
        x[::-1], w[::-1], measure=measure, name="clenshaw_curtis"
    )
    if a is not None or b is not None:
        rule = rule.map_to(a, b)
    return rule
