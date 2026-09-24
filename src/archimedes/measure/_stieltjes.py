"""Discretized Stieltjes procedure for arbitrary weight functions."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad

__all__ = ["stieltjes_recurrence"]


def stieltjes_recurrence(
    weight: Callable[[np.ndarray], np.ndarray],
    support: tuple[float, float],
    n: int,
    **quad_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Monic recurrence coefficients for an arbitrary weight function.

    Computes the coefficients of the monic three-term recurrence

    .. math::
        \pi_{k+1}(x) = (x - \alpha_k) \, \pi_k(x) - \beta_k \, \pi_{k-1}(x),
        \qquad \pi_{-1} = 0, \ \pi_0 = 1

    satisfied by the polynomials orthogonal with respect to ``weight`` on
    ``support``, via the (discretized) Stieltjes procedure [1]_:

    .. math::
        \alpha_k = \frac{\langle x \pi_k, \pi_k \rangle}
            {\langle \pi_k, \pi_k \rangle}, \qquad
        \beta_k = \frac{\langle \pi_k, \pi_k \rangle}
            {\langle \pi_{k-1}, \pi_{k-1} \rangle}, \quad k \geq 1

    where :math:`\langle f, g \rangle = \int_{\text{support}} f(x) \, g(x)
    \, w(x) \, dx`. The inner products are evaluated directly by adaptive
    quadrature (:func:`scipy.integrate.quad`).

    This is the default implementation of
    :meth:`~archimedes.measure.Measure.recurrence_coeffs`; any ``Measure``
    subclass that supplies ``weight`` and ``support`` (i.e. ``domain``)
    gets a working ``recurrence_coeffs`` -- and hence, via Golub-Welsch
    (:func:`~archimedes.quadrature.golub_welsch_rule`), a Gauss
    quadrature rule -- without deriving a closed-form recursion.

    Parameters
    ----------
    weight : callable
        Weight function :math:`w(x)`, evaluated at ``x``.
    support : tuple of float
        Integration bounds ``(a, b)``; either may be infinite.
    n : int
        Number of coefficients to compute, i.e. degrees ``0, ..., n - 1``.
        Must be ``>= 1``; not validated here.
    **quad_kwargs
        Forwarded to every internal :func:`scipy.integrate.quad` call (e.g.
        ``epsabs``, ``epsrel``, ``limit``) -- tightening these can extend
        the usable range of ``n`` for weights singular at an endpoint (see
        Notes).

    Returns
    -------
    alpha, beta : ndarray, shape ``(n,)``
        Monic recurrence coefficients. ``beta[0]`` is the zeroth moment
        (total mass) of ``weight`` rather than a recursion coefficient --
        see :meth:`~archimedes.measure.Measure.recurrence_coeffs`.

    Notes
    -----
    Accuracy is limited by the compounding of each step's adaptive
    quadrature error into the next, and degrades with ``n`` -- how quickly
    depends on the weight. For smooth, bounded weights (e.g.
    Legendre-like), coefficients are accurate to near machine precision
    through about :math:`n \sim 15`. For weights singular at an endpoint
    (e.g. Jacobi-like), accuracy degrades starting around :math:`n \sim 8`.
    For larger ``n`` or better accuracy, either supply a closed-form
    ``recurrence_coeffs`` override, or pass tighter
    ``epsabs``/``epsrel``/``limit`` via ``**quad_kwargs`` as a partial
    mitigation.

    References
    ----------
    .. [1] W. Gautschi, "Orthogonal Polynomials: Computation and
        Approximation", Oxford University Press, 2004.

    See Also
    --------
    Measure.recurrence_coeffs : Default implementation built on this function.
    golub_welsch : Turns these coefficients into Gauss quadrature nodes/weights.
    """
    a, b = support
    alpha = np.empty(n)
    beta = np.empty(n)

    def pi(k, x):
        p0, p1 = 0.0, 1.0  # pi_{-1}, pi_0
        for j in range(k):
            p0, p1 = p1, (x - alpha[j]) * p1 - beta[j] * p0
        return p1

    beta[0] = quad(weight, a, b, **quad_kwargs)[0]
    alpha[0] = quad(lambda x: x * weight(x), a, b, **quad_kwargs)[0] / beta[0]

    prev_norm = beta[0]
    for k in range(1, n):
        norm_k = quad(lambda x: weight(x) * pi(k, x) ** 2, a, b, **quad_kwargs)[0]
        num_k = quad(lambda x: x * weight(x) * pi(k, x) ** 2, a, b, **quad_kwargs)[0]
        alpha[k] = num_k / norm_k
        beta[k] = norm_k / prev_norm
        prev_norm = norm_k

    return alpha, beta
