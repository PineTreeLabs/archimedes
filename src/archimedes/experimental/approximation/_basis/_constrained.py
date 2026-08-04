"""A basis built as a fixed linear combination of another basis's functions

Typically the recombination is chosen so that every combination satisfies a
set of linear constraints (e.g. boundary conditions) by construction rather
than by enforcement in an assembled system.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from archimedes import tree

from ._base import RIGHT, Basis, _check_side

__all__ = ["ConstrainedBasis"]


def _reference_kwargs(base: Basis) -> dict:
    """``base``'s own reference-domain parameters, e.g. ``{"a": None, "b":
    None}`` for an interval or ``{"rate": 1.0, "start": 0.0}`` for a
    half-line -- whatever ``base.Parameters()`` (no args) defaults to.
    ``ReferenceDomain.affine_params`` guarantees this is the identity map,
    so evaluating ``base`` with these kwargs gives the *reference* basis
    functions, independent of any later target domain.
    """
    ref = base.Parameters()
    return {f.name: getattr(ref, f.name) for f in tree.fields(ref)}


@dataclasses.dataclass(frozen=True)
class ConstrainedBasis(Basis):
    r"""A basis whose functions are fixed linear combinations of another's

    Usually built so that every combination satisfies a set of linear
    constraints on the *reference* domain.

    Given a constraint matrix :math:`A` (one row per constraint, one column
    per ``base`` function), the new basis functions are

    .. math::
        \phi_i(x) = \sum_j N_{ji} \, \psi_j(x)

    where the columns of :math:`N` span :math:`\mathrm{null}(A)`, i.e.
    :math:`A N \approx 0`. Every function in the new basis therefore
    satisfies every constraint row exactly (to numerical precision), rather
    than the constraint being imposed afterwards on an assembled system.

    Because differentiation is linear, ``evaluate(..., deriv=k)`` is simply
    ``base.evaluate(..., deriv=k) @ matrix`` -- the same transform works at
    every derivative order, so a constraint on a derivative (Neumann) is no
    different in kind from a constraint on a value (Dirichlet).

    This trades ``base.n_basis`` degrees of freedom for
    ``base.n_basis - n_constraints``; use :meth:`from_constraints` (or one
    of the convenience constructors below), which checks that the
    constraint rows are linearly independent so the result has the
    expected size.

    Parameters
    ----------
    base : Basis
        The underlying basis being combined. Its own domain-mapping,
        ``default_quadrature``, and ``measures`` continue to apply
        unchanged.
    matrix : ndarray
        Shape ``(base.n_basis, n_basis)``; ``matrix[:, i]`` is the
        coefficient vector (in ``base``) of this basis's ``i``-th
        function.

    Notes
    -----
    Any subspace that is *smaller* than ``base``'s can be described as
    ``null(A)`` for some constraint matrix ``A``. A dimension-*preserving*
    change of basis (``matrix`` square and invertible, e.g. reshuffling
    normalization conventions) isn't a constraint in any useful sense, but
    can still be constructed directly with ``ConstrainedBasis(base, matrix)``,
    bypassing :meth:`from_constraints`.

    The constraint matrix is evaluated once, at ``base``'s own
    reference-domain defaults -- *not* at whatever target domain a later
    ``FunctionSpace`` supplies. A pure value or derivative constraint is
    invariant under the target domain's affine rescaling (an endpoint is
    still an endpoint, and rescaling a row does not change its null
    space), so the same ``matrix`` is reused verbatim for every target
    domain. A constraint that mixes *different* derivative orders with
    fixed nonzero relative weights (e.g. Robin, :math:`u'(a) + c\,u(a) =
    0` for :math:`c \neq 0`) does **not** have this property, since the
    relative weight of the two terms changes with domain scale. For this
    reason, the satisfied-by-basis-construction approach is not recommended
    for Robin boundary conditions.
    """

    base: Basis
    matrix: np.ndarray

    def __post_init__(self):
        if self.matrix.shape[0] != self.base.n_basis:
            raise ValueError(
                f"matrix has {self.matrix.shape[0]} rows, expected "
                f"base.n_basis={self.base.n_basis}"
            )

    @classmethod
    def from_constraints(
        cls, base: Basis, constraints: Callable[[Basis], np.ndarray]
    ) -> "ConstrainedBasis":
        """Build ``matrix`` as an orthonormal basis for ``null(A)``, with
        ``A = constraints(base)``.

        Parameters
        ----------
        base : Basis
            The basis to combine.
        constraints : callable
            ``constraints(base) -> ndarray`` of shape ``(n_constraints,
            base.n_basis)``. Typically built by calling ``base.evaluate``
            at reference-domain points (see :meth:`dirichlet`/:meth:`neumann`
            for worked examples).

        Raises
        ------
        ValueError
            If the constraint rows are not linearly independent (rank <
            number of rows), so the null space is larger than the caller
            expects and some constraints are redundant rather than each
            cutting the dimension by one.
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
        """Every function vanishes at both reference endpoints,
        :math:`\\phi(-1) = \\phi(1) = 0`.

        Spans the same space as the classical hand-derived combinations
        (e.g. Shen's Chebyshev-Galerkin basis :math:`T_k - T_{k+2}`), but
        is not necessarily numerically identical to them -- both are valid
        bases of :math:`\\mathrm{null}(A)`.
        """

        def constraints(base):
            kwargs = _reference_kwargs(base)
            left = base.evaluate(np.array([-1.0]), **kwargs)[0]
            right = base.evaluate(np.array([1.0]), **kwargs)[0]
            return np.stack([left, right])

        return cls.from_constraints(base, constraints)

    @classmethod
    def neumann(cls, base: Basis) -> "ConstrainedBasis":
        """Every function has zero derivative at both reference endpoints,
        :math:`\\phi'(-1) = \\phi'(1) = 0`."""

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
    def measures(self):
        """Delegates to ``base``: the constraint only recombines rows of
        the same design matrix, so the quadrature weight ``base`` is
        orthogonal under still applies unchanged."""
        return self.base.measures

    @property
    def required_breakpoints(self):
        return self.base.required_breakpoints

    def default_quadrature(self):
        """Delegates to ``base``: the combination spans a subspace of the
        same polynomials (or elements), never a larger one, so any rule
        exact for ``base`` remains exact here."""
        return self.base.default_quadrature()

    def evaluate(self, x, deriv: int = 0, side: str = RIGHT, **domain_kwargs):
        _check_side(side)
        return (
            self.base.evaluate(x, deriv=deriv, side=side, **domain_kwargs) @ self.matrix
        )
