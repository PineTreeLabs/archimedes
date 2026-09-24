"""Numerical quadrature methods for approximating integrals."""

from ._gauss_hermite import gauss_hermite
from ._gauss_jacobi import gauss_jacobi
from ._gauss_laguerre import gauss_laguerre
from ._gauss_legendre import (
    clenshaw_curtis,
    gauss_legendre,
    gauss_lobatto,
    gauss_radau,
)
from ._golub_welsch import golub_welsch, golub_welsch_rule
from ._integral import quadint
from ._quadrature_rule import (
    Quadrature,
    QuadratureReferenceData,
    QuadratureRule,
    composite_quad,
)
from ._simpson import simpson
from ._tensor import TensorQuadratureRule, tensor_quad
from ._trapezoidal import trapezoidal

__all__ = [
    "Quadrature",
    "QuadratureReferenceData",
    "QuadratureRule",
    "TensorQuadratureRule",
    "gauss_legendre",
    "gauss_radau",
    "gauss_lobatto",
    "clenshaw_curtis",
    "gauss_hermite",
    "gauss_jacobi",
    "gauss_laguerre",
    "golub_welsch",
    "golub_welsch_rule",
    "composite_quad",
    "simpson",
    "tensor_quad",
    "quadint",
    "trapezoidal",
]
