"""Linear basis-expansion / function approximation infrastructure.

- :class:`Basis` - a basis family (e.g. :class:`OrthogonalPolynomialBasis`)
- :class:`FunctionSpace` - a ``Basis`` on a domain along with a quadrature-based
  inner product and associated operations (e.g. :meth:`FunctionSpace.project`).
- :class:`BasisMatrix` - a basis evaluated at a set of quadrature nodes; a
  generalized Vandermonde matrix.
- :class:`Function` - a ``FunctionSpace`` plus a coefficient vector.

Bases are univariate by default. :class:`TensorBasis` combines one per
dimension into a multivariate basis but otherwise works similarly to the
univariate bases.

Two composition primitives build custom bases out of existing ones.
:class:`ConstrainedBasis` recombines one basis's functions by a fixed
matrix (e.g. the nullspace of a boundary condition constraint).
:class:`ConcatBasis` stacks functions from several bases side by side
(e.g. spectral-element vertex + bubble functions).

Examples
--------
Project a function onto a degree-5 Legendre space and evaluate it:

>>> import numpy as np
>>> from archimedes import approximation as approx
>>> space = approx.FunctionSpace.legendre(n_basis=6)
>>> f = space.project(lambda x: np.sin(np.pi * x))
>>> round(float(f(np.array([0.5]))[0]), 3)
1.005
"""

from ._basis import (
    Basis,
    BasisMatrix,
    BSplineBasis,
    ConcatBasis,
    ConstrainedBasis,
    CubicHermiteBasis,
    FourierBasis,
    LagrangeBasis,
    MonomialBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
    ProductParameters,
    TensorBasis,
)
from ._function import Function
from ._function_space import FunctionSpace

__all__ = [
    "Basis",
    "BasisMatrix",
    "BSplineBasis",
    "ConcatBasis",
    "ConstrainedBasis",
    "CubicHermiteBasis",
    "FourierBasis",
    "Function",
    "FunctionSpace",
    "LagrangeBasis",
    "MonomialBasis",
    "OrthogonalPolynomialBasis",
    "PiecewiseBasis",
    "ProductParameters",
    "TensorBasis",
]
