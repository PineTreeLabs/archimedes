"""Basis family: the ``Basis`` interface and every concrete or composite
implementation.

- ``Basis``, ``BasisMatrix`` -- the abstract interface (``_base.py``); a
  basis only evaluates, with no notion of a target domain or coefficients.
- Concrete basis implementations: ``OrthogonalPolynomialBasis``, ``LagrangeBasis``,
  ``CubicHermiteBasis``, ``FourierBasis``, ``PiecewiseBasis``.
- Composition primitives, each building a new ``Basis`` out of existing
  ones: ``ConstrainedBasis`` recombines *one* basis's functions by a fixed
  matrix, ``ConcatBasis`` stacks functions from *several* bases side by
  side, and ``TensorBasis`` combines one basis per dimension into a
  multivariate basis (with ``ProductParameters`` carrying the
  per-dimension target domains).
"""

from ._base import RIGHT, Basis, BasisMatrix
from ._concat import ConcatBasis
from ._constrained import ConstrainedBasis
from ._fourier import FourierBasis
from ._hermite import CubicHermiteBasis
from ._lagrange import LagrangeBasis
from ._orthogonal import OrthogonalPolynomialBasis
from ._piecewise import PiecewiseBasis
from ._tensor import ProductParameters, TensorBasis

__all__ = [
    "RIGHT",
    "Basis",
    "BasisMatrix",
    "ConcatBasis",
    "ConstrainedBasis",
    "CubicHermiteBasis",
    "FourierBasis",
    "LagrangeBasis",
    "OrthogonalPolynomialBasis",
    "PiecewiseBasis",
    "ProductParameters",
    "TensorBasis",
]
