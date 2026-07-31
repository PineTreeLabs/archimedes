"""Linear basis-expansion / function approximation infrastructure.

- ``Basis`` -- a basis family and its size (e.g. ``OrthogonalPolynomialBasis``);
  only evaluates, has no notion of a target domain or coefficients.
- ``FunctionSpace`` -- a ``Basis`` on a fixed target domain, with
  quadrature-based operations (``project``, ``quadrature``, ``basis_matrix``).
- ``BasisMatrix`` -- a basis evaluated at a fixed set of quadrature nodes,
  bundled with the matching weights; the public building block for a custom
  (Petrov-)Galerkin residual (``FunctionSpace.basis_matrix``).
- ``BasisExpansion`` -- a ``FunctionSpace`` plus a coefficient vector

Bases are univariate by default. ``TensorBasis`` combines one per dimension
into a multivariate basis, with ``ProductParameters`` carrying the
per-dimension target domains; everything above it works unchanged, since a
tensor basis still evaluates to a ``(npts, n_basis)`` design matrix.
"""

from ._basis import Basis, BasisMatrix
from ._basis_expansion import BasisExpansion
from ._basis_expansion import FunctionSpace
from ._hermite import CubicHermiteBasis
from ._lagrange import LagrangeBasis
from ._orthogonal import OrthogonalPolynomialBasis
from ._piecewise import PiecewiseBasis
from ._tensor_basis import ProductParameters, TensorBasis

__all__ = [
    "Basis",
    "BasisExpansion",
    "BasisMatrix",
    "CubicHermiteBasis",
    "FunctionSpace",
    "LagrangeBasis",
    "OrthogonalPolynomialBasis",
    "PiecewiseBasis",
    "ProductParameters",
    "TensorBasis",
]
