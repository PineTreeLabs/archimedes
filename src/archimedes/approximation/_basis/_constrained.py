from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from archimedes import tree

from ._base import RIGHT, Basis, _check_side

__all__ = ["ConstrainedBasis"]


def _reference_kwargs(base: Basis) -> dict:
    """The reference-domain parameters from ``base``

    For example:
    - ``{"a": None, "b": None}`` for an interval
    - ``{"rate": 1.0, "start": 0.0}`` for a half-line

    The result is the default from ``base.Parameters()`` (no args) defaults to,
    guaranteed to produce the reference basis functions when evaluating ``base``.
    """
    ref = base.Parameters()
    return {f.name: getattr(ref, f.name) for f in tree.fields(ref)}


@dataclasses.dataclass(frozen=True)
class ConstrainedBasis(Basis):
    r"""A basis whose functions are fixed linear combinations of another's

    Usually built so that every combination satisfies a set of linear
    constraints on the reference domain, for example to satisfy a boundary
    condition by construction.

    Given a constraint matrix :math:`A` (one row per constraint, one column
    per ``base`` function), the new basis functions are

    .. math::
        \phi_i(x) = \sum_j N_{ji} \, \psi_j(x)

    where the columns of :math:`N` span :math:`\mathrm{null}(A)`, i.e.
    :math:`A N \approx 0`. Every function in the new basis therefore
    satisfies every constraint row to numerical precision.

    Should typically not be constructed directly; use one of the classmethod
    constructors instead.
    """

    base: Basis
    """The basis that is recombined to form the constrained basis."""

    matrix: np.ndarray
    """Constraint matrix defining the new basis as linear combinations of ``base``."""

    def __post_init__(self):
        if self.matrix.shape[0] != self.base.n_basis:
            raise ValueError(
                f"matrix has {self.matrix.shape[0]} rows, expected "
                f"base.n_basis={self.base.n_basis}"
            )

    @classmethod
    def from_constraints(
        cls, base: Basis, constraints: Callable[[Basis], np.ndarray]
    ) -> ConstrainedBasis:
        """Construct a basis from the nullspace of a provided matrix.

        The constraint matrix is ``null(A)``, where ``A = constraints(base)``.

        A useful way to construct the ``constraints`` function is by evaluating
        the ``base`` basis. :meth:`Basis.evaluate` returns a Vandermonde-like matrix
        with one row per point and one column per basis function. If
        ``A = base.evaluate(x)``, then ``A @ c`` evaluates the function defined
        by coefficient vector ``c`` at ``x``. Hence the condition that the function
        vanish at ``x`` is the same as ``A @ c = 0``; in other words ``null(A)`` is the
        set of coefficient vectors whose function vanishes at ``x``. See example below.

        Parameters
        ----------
        base : Basis
            The basis to combine.
        constraints : callable
            ``constraints(base) -> ndarray`` of shape ``(n_constraints,
            base.n_basis)``.

        Examples
        --------
        Build a basis where every function vanishes at the left endpoint only:

        >>> import numpy as np
        >>> from archimedes.approximation import (
        ...     ConstrainedBasis,
        ...     FunctionSpace,
        ...     OrthogonalPolynomialBasis,
        ... )
        >>> from archimedes.measure import LegendreMeasure, UnitInterval
        >>> legendre = OrthogonalPolynomialBasis(LegendreMeasure(), 6)
        >>> basis = ConstrainedBasis.from_constraints(
        ...     legendre, lambda base: base.evaluate(np.array([-1.0]))
        ... )
        >>> basis.n_basis
        5

        Any combination of the new functions satisfies the constraint:

        >>> coeffs = np.random.default_rng(0).standard_normal(basis.n_basis)
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> u = basis.evaluate(x) @ coeffs
        >>> bool(np.isclose(u[0], 0.0))
        True

        The constraint is imposed on the reference domain, so it holds at the
        left endpoint of any target interval:

        >>> space = FunctionSpace(basis, UnitInterval.Parameters(a=0.0, b=2.0))
        >>> f = space.project(np.sin)
        >>> bool(np.isclose(f(np.array([0.0]))[0], 0.0))
        True
        """
        A = np.asarray(constraints(base))
        if A.ndim != 2 or A.shape[1] != base.n_basis:
            raise ValueError(
                f"constraints must return shape (n_constraints, "
                f"{base.n_basis}), got {A.shape}"
            )
        rank = np.linalg.matrix_rank(A)
        if rank != A.shape[0]:
            raise ValueError(
                f"constraint rows are not linearly independent: rank "
                f"{rank} for {A.shape[0]} rows"
            )
        # SVD's trailing right-singular-vectors span null(A) whenever A has
        # full row rank (checked above): singular values past `rank` are
        # (numerically) zero, so their vectors satisfy A @ v = 0.
        _, _, Vh = np.linalg.svd(A, full_matrices=True)
        N = Vh[rank:].T  # (base.n_basis, base.n_basis - rank)
        return cls(base=base, matrix=N)

    @classmethod
    def dirichlet(cls, base: Basis) -> "ConstrainedBasis":
        r"""Construct a basis that vanishes at both reference endpoints.

        The resulting basis satisfies the Dirichlet boundary condition
        :math:`\phi(-1) = \phi(1) = 0`.

        Spans the same space as the classical hand-derived combinations
        (e.g. Shen's Chebyshev-Galerkin basis :math:`T_k - T_{k+2}`), but
        is not necessarily numerically identical to them. Both are valid
        bases of :math:`\mathrm{null}(A)`.

        This cannot be constructed to hold nonzero values at the endpoints;
        see :class:`ConcatBasis` for combining this with another "vertex"
        basis to hold nonzero endpoint values.

        Examples
        --------
        >>> import numpy as np
        >>> from archimedes.approximation import (
        ...     ConstrainedBasis,
        ...     OrthogonalPolynomialBasis,
        ... )
        >>> from archimedes.measure import LegendreMeasure
        >>> legendre = OrthogonalPolynomialBasis(LegendreMeasure(), 6)
        >>> basis = ConstrainedBasis.dirichlet(legendre)
        >>> basis.n_basis
        4
        >>> x = np.array([-1.0, 1.0])
        >>> bool(np.allclose(basis.evaluate(x), 0.0))
        True
        """

        def constraints(base):
            kwargs = _reference_kwargs(base)
            left = base.evaluate(np.array([-1.0]), **kwargs)[0]
            right = base.evaluate(np.array([1.0]), **kwargs)[0]
            return np.stack([left, right])

        return cls.from_constraints(base, constraints)

    @classmethod
    def neumann(cls, base: Basis) -> "ConstrainedBasis":
        r"""Construct a basis whose derivatives vanish at both reference endpoints.

        The resulting basis satisfies the Neumann boundary condition
        :math:`\phi'(-1) = \phi'(1) = 0`.

        Examples
        --------
        >>> import numpy as np
        >>> from archimedes.approximation import (
        ...     ConstrainedBasis,
        ...     OrthogonalPolynomialBasis,
        ... )
        >>> from archimedes.measure import LegendreMeasure
        >>> legendre = OrthogonalPolynomialBasis(LegendreMeasure(), 6)
        >>> basis = ConstrainedBasis.neumann(legendre)
        >>> basis.n_basis
        4
        >>> x = np.array([-1.0, 1.0])
        >>> dphi = basis.evaluate(x, deriv=1)
        >>> bool(np.allclose(dphi, 0.0))
        True
        """

        def constraints(base):
            kwargs = _reference_kwargs(base)
            left = base.evaluate(np.array([-1.0]), deriv=1, **kwargs)[0]
            right = base.evaluate(np.array([1.0]), deriv=1, **kwargs)[0]
            return np.stack([left, right])

        return cls.from_constraints(base, constraints)

    @property
    def n_basis(self) -> int:
        return self.matrix.shape[1]

    @property
    def Parameters(self) -> type:  # noqa: N802
        return self.base.Parameters

    @property
    def _measures(self):
        # Delegates to `base`, since a linear combination of the
        # functions from `base` don't change the right quadrature weight.
        return self.base._measures

    @property
    def _required_breakpoints(self):
        return self.base._required_breakpoints

    def _default_quadrature(self):
        """Delegates to ``base``: the combination spans a subspace of the
        same polynomials (or elements), never a larger one, so any rule
        exact for ``base`` remains exact here."""
        return self.base._default_quadrature()

    def evaluate(self, x, deriv: int = 0, side: str = RIGHT, **domain_kwargs):
        _check_side(side)
        return (
            self.base.evaluate(x, deriv=deriv, side=side, **domain_kwargs) @ self.matrix
        )
