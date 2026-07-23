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
from ._gauss_hermite import gauss_hermite
from ._gauss_laguerre import gauss_laguerre
from ._integral import integral

__all__ = [
    "QuadratureRule",
    "gauss_legendre",
    "gauss_radau",
    "gauss_lobatto",
    "clenshaw_curtis",
    "gauss_hermite",
    "gauss_laguerre",
    "composite",
    "integral",
]
