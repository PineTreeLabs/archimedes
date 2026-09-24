"""Basis family: the ``Basis`` interface and concrete implementations.

- :class:`Basis` -- the abstract interface; a basis only defines and evaluates
  the basis functions, independent of a target domain or expansion coefficients.
- :class:`BasisMatrix` -- a :class:`Basis` evaluated at a set of quadrature
  nodes, bundled with the matching quadrature weights.
- Concrete basis implementations, e.g.: :class:`OrthogonalPolynomialBasis`,
  :class:`LagrangeBasis`, :class:`CubicHermiteBasis`, :class:`FourierBasis`,
  :class:`MonomialBasis`, :class:`PiecewiseBasis`, :class:`BSplineBasis`.
- Composition primitives that build a new ``Basis`` from existing ones:
  :class:`ConstrainedBasis` recombines one basis's functions by a
  fixed matrix (e.g. eliminating a nullspace associated with a boundary condition),
  :class:`ConcatBasis` stacks functions from several bases, and :class:`TensorBasis`
  combines one basis per dimension into a multivariate basis.
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
