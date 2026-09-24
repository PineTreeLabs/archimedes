"""Gauss quadrature from recurrence coefficients (Golub & Welsch 1969)."""

from __future__ import annotations

import numpy as np
import scipy.linalg

from archimedes.measure import Measure

from ._quadrature_rule import QuadratureRule

__all__ = ["golub_welsch", "golub_welsch_rule"]


def golub_welsch(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gauss nodes/weights from monic recurrence coefficients.

    Given monic three-term recurrence coefficients ``(alpha, beta)`` (see
    :py:meth:`~archimedes.measure.Measure.recurrence_coeffs`), the Gauss
    nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix
    with diagonal ``alpha`` and off-diagonal ``sqrt(beta[1:])``; the
    quadrature weights are ``beta[0]`` times the squared first component of
    each corresponding normalized eigenvector [1]_.

    Parameters
    ----------
    alpha, beta : array_like
        Monic recurrence coefficients, each shape ``(n,)``. ``beta[0]`` is
        the total mass of the measure (its zeroth moment); the rest of
        ``beta`` and all of ``alpha`` are the recursion coefficients
        themselves.

    Returns
    -------
    nodes, weights : ndarray, shape ``(n,)``
        Gauss quadrature nodes (ascending) and weights.

    References
    ----------
    .. [1] G. H. Golub and J. H. Welsch, "Calculation of Gauss Quadrature
        Rules", Mathematics of Computation, Vol. 23, No. 106, 1969,
        pp. 221-230.
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    nodes, vecs = scipy.linalg.eigh_tridiagonal(alpha, np.sqrt(beta[1:]))
    weights = beta[0] * vecs[0, :] ** 2
    return nodes, weights


def golub_welsch_rule(
    measure: Measure, n: int, name: str | None = None
) -> QuadratureRule:
    """Gauss quadrature rule for any ``Measure`` via its recurrence coefficients.

    Unlike :py:func:`~archimedes.quadrature.gauss_legendre` and friends,
    which use closed-form node/weight formulas (via ``scipy.special.roots_*``)
    specific to their classical family, this works for *any* ``Measure``
    subclass that implements
    :py:meth:`~archimedes.measure.Measure.recurrence_coeffs` -- including,
    eventually, user-defined ones.

    Parameters
    ----------
    measure : Measure
        Measure to build the rule for.
    n : int
        Number of quadrature nodes. Must be ``>= 1``.
    name : str, optional
        Name for the rule. Defaults to ``"golub_welsch_<MeasureClassName>"``.

    Returns
    -------
    rule : QuadratureRule
        Gauss quadrature rule with ``n`` nodes on ``measure.support``, exact
        for polynomials up to degree ``2n - 1``.

    Raises
    ------
    ValueError
        If ``n < 1``.

    See Also
    --------
    golub_welsch : Underlying node/weight computation.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    alpha, beta = measure.recurrence_coeffs(n)
    nodes, weights = golub_welsch(alpha, beta)
    return QuadratureRule.from_arrays(
        nodes,
        weights,
        measure=measure,
        name=name or f"golub_welsch_{type(measure).__name__}",
    )
