"""Linear basis-expansion / function approximation infrastructure.

- ``Basis`` -- a basis family and its size (e.g. ``OrthogonalPolynomialBasis``);
  only evaluates, has no notion of a target domain or coefficients.
- ``FunctionSpace`` -- a ``Basis`` on a fixed target domain, with
  quadrature-based operations (``evaluate``, ``project``, ``mass_matrix``,
  ``stiffness_matrix``).
- ``Function`` -- a ``FunctionSpace`` plus a coefficient vector

Bases are univariate by default. ``TensorBasis`` combines one per dimension
into a multivariate basis, with ``ProductParameters`` carrying the
per-dimension target domains; everything above it works unchanged, since a
tensor basis still evaluates to a ``(npts, n_basis)`` design matrix.
"""

from ._basis import Basis
from ._function import Function
from ._function_space import FunctionSpace
from ._lagrange import LagrangeBasis
from ._orthogonal import OrthogonalPolynomialBasis
from ._piecewise import PiecewiseBasis
from ._tensor_basis import ProductParameters, TensorBasis

__all__ = [
    "Basis",
    "Function",
    "FunctionSpace",
    "LagrangeBasis",
    "OrthogonalPolynomialBasis",
    "PiecewiseBasis",
    "ProductParameters",
    "TensorBasis",
]
