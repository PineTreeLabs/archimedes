"""Numerical quadrature methods for approximating integrals."""

from ._quadrature_rule import (
    QuadratureRule,
    gauss_legendre,
    gauss_radau,
    gauss_lobatto,
    clenshaw_curtis,
    composite,
)

__all__ = [
    "QuadratureRule",
    "gauss_legendre",
    "gauss_radau",
    "gauss_lobatto",
    "clenshaw_curtis",
    "composite",
]
