"""Basis family: the ``Basis`` interface and every concrete or composite
implementation.

- :class:`Basis`, :class:`BasisMatrix` -- the abstract interface; a basis
  only evaluates, with no notion of a target domain or coefficients.
- Concrete basis implementations: :class:`OrthogonalPolynomialBasis`,
  :class:`LagrangeBasis`, :class:`CubicHermiteBasis`, :class:`FourierBasis`,
  :class:`MonomialBasis`, :class:`PiecewiseBasis`, :class:`BSplineBasis`.
- Composition primitives, each building a new ``Basis`` out of existing
  ones: :class:`ConstrainedBasis` recombines *one* basis's functions by a
  fixed matrix, :class:`ConcatBasis` stacks functions from *several* bases
  side by side, and :class:`TensorBasis` combines one basis per dimension
  into a multivariate basis (with :class:`ProductParameters` carrying the
  per-dimension target domains).
"""

from ._base import RIGHT, Basis, BasisMatrix
from ._bspline import BSplineBasis
from ._concat import ConcatBasis
from ._constrained import ConstrainedBasis
from ._fourier import FourierBasis
from ._hermite import CubicHermiteBasis
from ._lagrange import LagrangeBasis
from ._monomial import MonomialBasis
from ._orthogonal import OrthogonalPolynomialBasis
from ._piecewise import PiecewiseBasis
from ._tensor import ProductParameters, TensorBasis

__all__ = [
    "RIGHT",
    "Basis",
    "BasisMatrix",
    "BSplineBasis",
    "ConcatBasis",
    "ConstrainedBasis",
    "CubicHermiteBasis",
    "FourierBasis",
    "LagrangeBasis",
    "MonomialBasis",
    "OrthogonalPolynomialBasis",
    "PiecewiseBasis",
    "ProductParameters",
    "TensorBasis",
]
