"""Gauss-Hermite quadrature on the real line."""

from __future__ import annotations

from typing import Literal

from scipy.special import roots_hermite, roots_hermitenorm

from archimedes.measure import (
    Measure,
    PhysicistsHermiteMeasure,
    ProbabilistsHermiteMeasure,
)

from ._quadrature_rule import QuadratureRule

__all__ = ["gauss_hermite"]


def gauss_hermite(n: int, kind: Literal["phys", "prob"] = "prob") -> QuadratureRule:
    """Gauss-Hermite quadrature rule with ``n`` nodes.

    Nodes are the roots of the degree-``n`` Hermite polynomial. The rule is
    exact for polynomials up to degree :math:`2n - 1`, weighted by the
    Gaussian-shaped reference weight of ``measure``.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.
    kind : {"phys", "prob"}, optional
        Which classical Hermite convention to use. ``"prob"`` (default) is
        the *probabilists'* convention, with reference weight
        :math:`e^{-x^2/2}` (:class:`ProbabilistsHermiteMeasure`) -- up to
        normalization, the standard normal density. ``"phys"`` is the
        *physicists'* convention, with reference weight
        :math:`e^{-x^2}` (:class:`PhysicistsHermiteMeasure`). Neither weight
        integrates to 1 on its own; pass ``density=True`` to
        ``QuadratureRule.integrate``/``sum``/``scaled_weights`` for weights
        that do.

    Returns
    -------
    rule : QuadratureRule
        Gauss-Hermite rule with ``n`` nodes on :math:`(-\\infty, \\infty)`,
        exact to degree :math:`2n - 1`.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"phys"`` or ``"prob"``.
    """
    measure: Measure
    if kind == "prob":
        x, w = roots_hermitenorm(n)
        measure = ProbabilistsHermiteMeasure()
    elif kind == "phys":
        x, w = roots_hermite(n)
        measure = PhysicistsHermiteMeasure()
    else:
        raise ValueError(f"kind must be 'phys' or 'prob', got {kind!r}")
    return QuadratureRule(x, w, measure=measure, name="gauss_hermite")
