"""Numerical quadrature methods for approximating integrals."""

from ._gauss_hermite import gauss_hermite
from ._gauss_laguerre import gauss_laguerre
from ._gauss_legendre import (
    clenshaw_curtis,
    gauss_legendre,
    gauss_lobatto,
    gauss_radau,
)
from ._golub_welsch import from_measure, golub_welsch
from ._integral import integral
from ._quadrature_rule import (
    Quadrature,
    QuadratureRule,
    composite,
)
from ._tensor import TensorQuadratureRule, tensor

__all__ = [
    "Quadrature",
    "QuadratureRule",
    "TensorQuadratureRule",
    "gauss_legendre",
    "gauss_radau",
    "gauss_lobatto",
    "clenshaw_curtis",
    "gauss_hermite",
    "gauss_laguerre",
    "golub_welsch",
    "from_measure",
    "composite",
    "tensor",
    "integral",
]
