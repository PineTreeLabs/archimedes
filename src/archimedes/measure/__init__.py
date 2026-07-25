"""Weight functions and reference domains for classical measures.

Shared low-level abstraction consumed by ``archimedes.quadrature`` (Gauss
quadrature rules) and, via the Wiener-Askey correspondence between these
weights and classical probability distributions, intended for future use by
probability/uncertainty code as well.
"""

from ._base import Measure
from ._domain import HalfLine, RealLine, ReferenceDomain, UnitInterval
from ._hermite import HermiteMeasure, HermiteNormMeasure
from ._jacobi import JacobiMeasure
from ._laguerre import LaguerreMeasure
from ._legendre import LegendreMeasure

__all__ = [
    "Measure",
    "ReferenceDomain",
    "UnitInterval",
    "HalfLine",
    "RealLine",
    "HermiteMeasure",
    "HermiteNormMeasure",
    "JacobiMeasure",
    "LaguerreMeasure",
    "LegendreMeasure",
]
