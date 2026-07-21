from __future__ import annotations

import dataclasses

import numpy as np
from ._measure import Measure

__all__ = ["JacobiMeasure"]


@dataclasses.dataclass(frozen=True)
class JacobiMeasure(Measure):
    """Measure for the Jacobi polynomial family.
    
    Weight :math:`w(x) = (1-x)^\\alpha (1+x)^\\beta` on
    :math:`[-1, 1]`, with :math:`\\alpha, \\beta > -1`.

    The associated orthogonal polynomials are the Jacobi polynomials
    :math:`P_n^{(\\alpha,\\beta)}(x)`. Gauss-Legendre is the special case
    :math:`\\alpha = \\beta = 0`; Chebyshev quadrature of the first and
    second kind are the special cases :math:`\\alpha = \\beta = -1/2` and
    :math:`\\alpha = \\beta = 1/2`, respectively. The zeroth moment
    (normalization) of the weight has a closed form in terms of the Beta
    function:

    .. math::
        \\int_{-1}^1 (1-x)^\\alpha (1+x)^\\beta \\, dx
            = 2^{\\alpha + \\beta + 1} \\, B(\\alpha + 1, \\beta + 1)

    Since the reference domain :math:`[-1, 1]` is the same as
    the Legendre family, `affine_params` is the same.

    Parameters
    ----------
    alpha, beta : float
        Exponents of the weight function. Must be :math:`> -1` for the
        weight to be integrable at the corresponding endpoint.

    Raises
    ------
    ValueError
        If `alpha` or `beta` is :math:`\\leq -1`.
    """

    alpha: float
    beta: float

    # Not a dataclass field: unannotated, so `dataclasses` leaves it as a
    # plain class attribute overriding `_LegendreFamily.uniform_weight`.
    # The Jacobi weight is singular at the reference endpoints, so it
    # cannot be tiled into a composite rule -- see `_QuadratureFamily.
    # uniform_weight`.
    uniform_weight = False

    def __post_init__(self):
        if self.alpha <= -1 or self.beta <= -1:
            raise ValueError(
                f"invalid alpha={self.alpha} or beta={self.beta}, must be > -1"
            )

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = (1-x)^\\alpha
        (1+x)^\\beta`, evaluated at `x`."""
        return (1 - x) ** self.alpha * (1 + x) ** self.beta
