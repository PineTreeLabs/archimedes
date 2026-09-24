"""Monomial (power series) basis on an affine-mapped reference interval."""

from __future__ import annotations

import dataclasses

import numpy as np

from archimedes.measure import UnitInterval

from ._base import RIGHT, Basis, _check_side

__all__ = ["MonomialBasis"]


@dataclasses.dataclass(frozen=True)
class MonomialBasis(Basis):
    r"""Monomial (power series) basis on an interval.

    The basis functions are :math:`\{1, t, t^2, \ldots, t^{n-1}\}`, with
    reference domain :math:`[-1, 1]`. Other intervals :math:`x \in [a, b]` are
    affinely mapped onto the reference domain.

    Parameters
    ----------
    n_basis : int
        Number of basis functions. The basis spans polynomials of degree
        ``0`` through ``n_basis - 1``.

    Warnings
    --------
    Monomial bases are inherently ill-conditioned for high degrees; prefer
    orthogonal polynomial bases for high-degree polynomial approximation.

    See Also
    --------
    OrthogonalPolynomialBasis : Orthogonal polynomial families (Legendre,
        Chebyshev, Jacobi, Hermite, Laguerre); preferred for numerical
        conditioning and spectral accuracy.
    FunctionSpace.monomial : Convenience constructor for a
        :class:`FunctionSpace` built on this basis.
    """

    n_basis: int

    def __post_init__(self):
        if self.n_basis < 1:
            raise ValueError(f"n_basis must be >= 1, got {self.n_basis}")

    @property
    def Parameters(self) -> type:  # noqa: N802
        # A finite target interval is required even though the monomials
        # themselves are defined on all of R: FunctionSpace's mass-matrix
        # and projection integrals have no decaying weight to make them
        # converge otherwise, unlike RealLine/HalfLine (Hermite/Laguerre),
        # whose Gaussian/exponential weights do.
        return UnitInterval.Parameters

    def _default_quadrature(self):
        """Gauss-Legendre rule of ``n_basis`` points."""
        from archimedes.quadrature import gauss_legendre

        return gauss_legendre(self.n_basis)

    def _product_basis(self, other):
        if not isinstance(other, MonomialBasis):
            raise ValueError(
                f"cannot form a product basis between "
                f"{type(self).__name__} and {type(other).__name__}"
            )
        return MonomialBasis(self.n_basis + other.n_basis - 1)

    def _derivative_basis(self, deriv=1):
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")
        if deriv == 0:
            return self
        if deriv >= self.n_basis:
            raise ValueError(
                f"deriv={deriv} is at or past the degree of a {self.n_basis}-"
                f"function basis, whose elements are polynomials of degree "
                f"{self.n_basis - 1}; the derivative is identically zero."
            )
        return MonomialBasis(self.n_basis - deriv)

    def _integral_basis(self, order=1):
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if order == 0:
            return self
        return MonomialBasis(self.n_basis + order)

    def evaluate(self, x, deriv: int = 0, side: str = RIGHT, **domain_kwargs):
        # `side` is validated but unused: monomials are smooth, so both
        # one-sided limits agree everywhere. See `Basis.evaluate`.
        _check_side(side)
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")

        scale, shift = UnitInterval().affine_params(**domain_kwargs)
        t = (x - shift) / scale

        # powers[k] = t**k, k = 0..n_basis-1, built by an explicit recurrence
        # (n_basis is static) rather than `t[:, None] ** np.arange(n_basis)`,
        # since elementwise power against a vector of exponents has no
        # symbolic implementation -- `t` may be a traced SymbolicArray.
        powers = [np.ones_like(x)]
        for _ in range(1, self.n_basis):
            powers.append(t * powers[-1])

        if deriv == 0:
            return np.stack(powers, axis=-1)

        # d^deriv/dx^deriv t^k = falling_factorial(k, deriv) * t**(k - deriv)
        # / scale**deriv (the chain-rule factor from dt/dx = 1/scale); zero
        # for k < deriv. The falling-factorial coefficients are static
        # Python floats (k and deriv are both static ints), never traced.
        cols = []
        for k in range(self.n_basis):
            if k < deriv:
                cols.append(np.zeros_like(x))
                continue
            coeff = 1.0
            for j in range(k - deriv + 1, k + 1):
                coeff *= j
            cols.append((coeff / scale**deriv) * powers[k - deriv])

        return np.stack(cols, axis=-1)
