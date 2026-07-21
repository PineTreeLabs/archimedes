"""Measures for classical orthogonal polynomial families.

A :class:`Measure` pairs a weight function :math:`w(x)` with its reference
domain, defining the orthogonality relation for one classical family of
orthogonal polynomials -- Legendre, Jacobi, Laguerre, or Hermite (in both
physicists' and probabilists' normalizations). Measures are consumed by
:mod:`archimedes.experimental.quadrature` to build fixed-node Gauss
quadrature rules.
"""

from ._measure import Measure
from ._hermite import HermiteMeasure, HermiteNormMeasure
from ._jacobi import JacobiMeasure
from ._laguerre import LaguerreMeasure
from ._legendre import LegendreMeasure

__all__ = [
    "Measure",
    "HermiteMeasure",
    "HermiteNormMeasure",
    "JacobiMeasure",
    "LaguerreMeasure",
    "LegendreMeasure",
]
