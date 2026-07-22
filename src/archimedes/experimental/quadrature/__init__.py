"""Numerical quadrature methods for approximating integrals."""

from ._quadrature_rule import (
    QuadratureRule,
    composite,
)
from ._gauss_legendre import (
    gauss_legendre,
    gauss_radau,
    gauss_lobatto,
    clenshaw_curtis,
)
from ._integral import integral

__all__ = [
    "QuadratureRule",
    "gauss_legendre",
    "gauss_radau",
    "gauss_lobatto",
    "clenshaw_curtis",
    "composite",
    "integral",
]
