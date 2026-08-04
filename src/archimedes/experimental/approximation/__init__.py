"""Linear basis-expansion / function approximation infrastructure.

- :class:`Basis` -- a basis family and its size (e.g.
  :class:`OrthogonalPolynomialBasis`); only evaluates, has no notion of a
  target domain or coefficients.
- :class:`FunctionSpace` -- a ``Basis`` on a fixed target domain, with
  quadrature-based operations (``project``, ``quadrature``, ``basis_matrix``).
- :class:`BasisMatrix` -- a basis evaluated at a fixed set of quadrature
  nodes, bundled with the matching weights; the building block for a custom
  (Petrov-)Galerkin residual (``FunctionSpace.basis_matrix``).
- :class:`Function` -- a ``FunctionSpace`` plus a coefficient vector.

Bases are univariate by default. :class:`TensorBasis` combines one per
dimension into a multivariate basis, with :class:`ProductParameters`
carrying the per-dimension target domains; everything above it works
unchanged, since a tensor basis still evaluates to a ``(npts, n_basis)``
design matrix.

Two composition primitives build custom bases out of existing ones:
:class:`ConstrainedBasis` recombines *one* basis's functions by a fixed
matrix (e.g. the null space of a boundary-condition constraint), and
:class:`ConcatBasis` stacks functions from *several* bases side by side
(e.g. spectral-element vertex + bubble functions).

Examples
--------
Project a function onto a degree-5 Legendre space and evaluate it:

>>> import numpy as np
>>> from archimedes.experimental import approximation as approx
>>> space = approx.FunctionSpace.legendre(n_basis=6)
>>> f = space.project(lambda x: np.sin(np.pi * x))
>>> round(float(f(np.array([0.5]))[0]), 3)
1.005
"""

from ._basis import (
    Basis,
    BasisMatrix,
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
