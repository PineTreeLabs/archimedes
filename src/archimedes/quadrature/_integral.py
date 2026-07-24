"""High-level interface to fixed-order Gauss quadrature."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ._gauss_legendre import (
    clenshaw_curtis,
    gauss_legendre,
    gauss_lobatto,
    gauss_radau,
)
from ._quadrature_rule import QuadratureRule

__all__ = ["integral"]

_RULES: dict[str, Callable[[int], QuadratureRule]] = {
    "legendre": gauss_legendre,
    "radau_left": lambda n: gauss_radau(n, endpoint="left"),
    "radau_right": lambda n: gauss_radau(n, endpoint="right"),
    "lobatto": gauss_lobatto,
    "clenshaw_curtis": clenshaw_curtis,
}


def integral(
    func: Callable[..., np.ndarray],
    a: float,
    b: float,
    n: int = 20,
    rule: str = "legendre",
    axis: int = -1,
    args: Sequence[Any] | None = None,
) -> np.ndarray:
    """Approximate the definite integral of ``func`` over ``[a, b]``.

    .. math::
        \\int_a^b f(x) \\, dx \\approx \\sum_{i=1}^n w_i f(x_i)

    using a fixed-order Gauss quadrature rule (by default
    `Gauss-Legendre quadrature
    <https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature>`_).

    This is a convenience wrapper around the ``rule(n).integrate(f, a, b)``
    pattern (see :py:meth:`QuadratureRule.integrate`) for one-off integrals
    on a finite interval; for repeated evaluation of the same rule (e.g.
    inside a loop or a function decorated with :py:func:`archimedes.compile`),
    build the ``QuadratureRule`` once with :py:func:`gauss_legendre`,
    :py:func:`gauss_radau`, :py:func:`gauss_lobatto`, or
    :py:func:`clenshaw_curtis` and call ``.integrate`` directly instead.

    Parameters
    ----------
    func : callable
        Integrand. Called once as ``func(x, *args)`` on the full array of
        quadrature nodes, so must be vectorized. Must return values with
        the nodes along ``axis``.
    a, b : float
        Bounds of integration. Must both be finite (see Notes).
    n : int, optional
        Number of quadrature nodes. Default 20.
    rule : {"legendre", "radau_left", "radau_right", "lobatto", \
"clenshaw_curtis"}, optional
        Quadrature family to use, all defined on ``[a, b]`` with uniform
        weight. Default ``"legendre"`` (no fixed endpoints, exact to degree
        ``2n - 1``). Use ``"radau_left"``/``"radau_right"`` to fix the
        left/right endpoint as a node, or ``"lobatto"`` to fix both; see
        :py:func:`gauss_radau` and :py:func:`gauss_lobatto` for the
        accompanying loss of exactness.
    axis : int, optional
        Axis holding the nodes in the output of ``func``. Default -1.
    args : tuple, optional
        Extra arguments passed to ``func`` after ``x``.

    Returns
    -------
    value : ndarray
        Approximated integral. Shape (m,) for vector-valued integrands, or
        () for scalar integrands.

    Raises
    ------
    ValueError
        If ``rule`` is not one of the supported names, or if ``a``/``b``
        are not finite.

    Notes
    -----
    This function only supports finite ``[a, b]``. Unlike
    ``scipy.integrate.quad``, it does not adapt the node count or use a
    variable transformation to reach infinite/semi-infinite intervals --
    it is a thin wrapper around a single fixed-order rule. Gauss-Laguerre
    and Gauss-Hermite quadrature *can* integrate over semi-infinite and
    infinite intervals, but only the weighted integral
    :math:`\\int f(x) e^{-x} \\, dx` or :math:`\\int f(x) e^{-x^2} \\, dx`
    respectively -- correcting for the weight by evaluating
    :math:`f(x) e^{x}` (or :math:`e^{x^2}`) at the nodes reintroduces the
    overflow/cancellation problems Gauss quadrature exists to avoid, so
    this is not done automatically. If your integrand already has the
    appropriate decay, construct a ``QuadratureRule`` directly with an
    :py:class:`~archimedes.measure.LaguerreMeasure`
    or :py:class:`~archimedes.measure.HermiteMeasure`
    instead of using this function.

    See Also
    --------
    QuadratureRule.integrate : Underlying integration method.
    gauss_legendre, gauss_radau, gauss_lobatto, clenshaw_curtis :
        Rule constructors dispatched by ``rule``.
    scipy.integrate.quad : Adaptive quadrature, including infinite and
        semi-infinite intervals.
    """
    if rule not in _RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {sorted(_RULES)}")

    return _RULES[rule](n).integrate(func, a, b, axis=axis, args=args)
