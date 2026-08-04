"""Linear basis-expansion / function approximation infrastructure.

- ``Basis`` -- a basis family and its size (e.g. ``OrthogonalPolynomialBasis``);
  only evaluates, has no notion of a target domain or coefficients.
- ``FunctionSpace`` -- a ``Basis`` on a fixed target domain, with
  quadrature-based operations (``project``, ``quadrature``, ``basis_matrix``).
- ``BasisMatrix`` -- a basis evaluated at a fixed set of quadrature nodes,
  bundled with the matching weights; the public building block for a custom
  (Petrov-)Galerkin residual (``FunctionSpace.basis_matrix``).
- ``Function`` -- a ``FunctionSpace`` plus a coefficient vector

Bases are univariate by default. ``TensorBasis`` combines one per dimension
into a multivariate basis, with ``ProductParameters`` carrying the
per-dimension target domains; everything above it works unchanged, since a
tensor basis still evaluates to a ``(npts, n_basis)`` design matrix.

Two composition primitives build custom bases out of existing ones:
``ConstrainedBasis`` recombines *one* basis's functions by a fixed matrix
(e.g. the null space of a boundary-condition constraint), and ``ConcatBasis``
stacks functions from *several* bases side by side (e.g. spectral-element
vertex + bubble functions). Composing the two covers the common custom-basis
recipes without a dedicated class per recipe.
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
