"""Linear basis-expansion / function approximation infrastructure.

- ``Basis`` -- a basis family and its size (e.g. ``OrthogonalPolynomialBasis``);
  only evaluates, has no notion of a target domain or coefficients.
- ``FunctionSpace`` -- a ``Basis`` on a fixed target domain, with
  quadrature-based operations (``evaluate``, ``project``, ``mass_matrix``,
  ``stiffness_matrix``).
- ``Function`` -- a ``FunctionSpace`` plus a coefficient vector
"""

from ._basis import Basis
from ._function import Function
from ._function_space import FunctionSpace
from ._lagrange import LagrangeBasis
from ._orthogonal import OrthogonalPolynomialBasis
from ._piecewise import PiecewiseBasis

__all__ = [
    "Basis",
    "Function",
    "FunctionSpace",
    "LagrangeBasis",
    "OrthogonalPolynomialBasis",
    "PiecewiseBasis",
]
