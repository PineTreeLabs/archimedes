"""Abstract base class for finite-dimensional bases."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

from archimedes import tree

if TYPE_CHECKING:
    from archimedes.measure import Measure, ReferenceDomain
    from archimedes.quadrature import QuadratureRule

__all__ = ["Basis", "BasisMatrix"]

RIGHT = "right"
"""``side`` value selecting the limit from above at a point of discontinuity."""

LEFT = "left"
"""``side`` value selecting the limit from below at a point of discontinuity."""


@tree.struct
class BasisMatrix:
    r"""A :class:`Basis` evaluated at a set of quadrature nodes.

    Also called a "design matrix" or "generalized Vandermonde matrix".

    The basis matrix is bundled with the matching quadrature weights to
    provide proper weighted inner product semantics for matrix multiplication.
    In particular, ``Phi.T`` gives the adjoint of :math:`\Phi` under the weighted
    inner product: :math:`\langle \Phi c, r\rangle_w = \langle c, \Phi^\top r\rangle`.
    That is, ``Phi.T @ r`` is not plain matrix multiplication but includes the
    weights: ``Phi.T @ r == Phi.matrix.T @ np.diag(Phi.weights) @ r``.

    This definition of the transpose operation ``.T`` as an adjoint means that
    the Gram (mass) matrix is implemented as ``Phi.T @ Phi``, and a Galerkin
    projection of a function ``f`` is implemented as ``Phi.T @ f(x)``, where
    ``x = Phi.nodes``.

    Parameters
    ----------
    matrix : ndarray
        The design matrix :math:`\Phi`, shape ``(npts, n_basis)`` -- a
        generalized Vandermonde matrix, :math:`\Phi_{ni} = \phi_i(x_n)` for
        an arbitrary basis :math:`\{\phi_i\}`.
    weights : ndarray
        Quadrature weights for the associated inner product, shape ``(npts,)``.
    nodes : ndarray
        Quadrature nodes at which the basis was evaluated, shape ``(npts,)``.

    See Also
    --------
    FunctionSpace.basis_matrix : Builds a :class:`BasisMatrix` for a given basis
        and quadrature rule.
    """

    matrix: np.ndarray
    weights: np.ndarray
    nodes: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape  # type: ignore[return-value]

    def __matmul__(self, coefficients: np.ndarray) -> np.ndarray:
        r""":math:`\Phi c`: sampled values at the quadrature nodes."""
        return self.matrix @ coefficients  # type: ignore[no-any-return]

    @property
    def T(self) -> _BasisMatrixAdjoint:  # noqa: N802
        r"""The adjoint :math:`\Phi^\top`; see the class docstring."""
        return _BasisMatrixAdjoint(self)


@tree.struct
class _BasisMatrixAdjoint:
    """``BasisMatrix.T``: apply via ``@``, undo via ``.T`` again."""

    basis_matrix: BasisMatrix

    def __matmul__(self, values: np.ndarray | BasisMatrix) -> np.ndarray:
        r""":math:`\Phi^\top r = \phi^\top (w \odot r)`.

        ``values`` (``r``) must already be sampled at the same quadrature
        nodes as ``self.basis_matrix`` and have shape ``(npts, ...)``.
        """
        if isinstance(values, BasisMatrix):
            values = values.matrix

        phi, w = self.basis_matrix.matrix, self.basis_matrix.weights
        if values.ndim == 1:
            return phi.T @ (w * values)  # type: ignore[no-any-return]
        return phi.T @ (w[:, None] * values)  # type: ignore[no-any-return]

    @property
    def T(self) -> BasisMatrix:  # noqa: N802
        return self.basis_matrix


def _check_side(side: str) -> str:
    if side not in (LEFT, RIGHT):
        raise ValueError(f"side must be {LEFT!r} or {RIGHT!r}, got {side!r}")
    return side


class Basis(metaclass=abc.ABCMeta):
    r"""A finite-dimensional family of basis functions :math:`\{\phi_i\}_{i=1}^n`.

    ``Basis`` only defines the basis functions and evaluates them, without any
    notion of a target domain or coefficient vector. :class:`FunctionSpace` combines
    a ``Basis`` with a domain and quadrature rule to define an inner product space,
    and :class:`Function` further combines a ``FunctionSpace`` with expansion
    coefficients.

    Available implementations include:
    
    - :class:`BSplineBasis`: B-spline basis functions
    - :class:`ConcatBasis`: Concatenation of multiple bases
    - :class:`ConstrainedBasis`: Constrained linear combination of another basis
    - :class:`CubicHermiteBasis`: Cubic Hermite basis functions
    - :class:`FourierBasis`: Trigonometric basis functions on a periodic interval
    - :class:`LagrangeBasis`: Lagrange polynomial basis functions
    - :class:`MonomialBasis`: Monomial (power series) basis functions
    - :class:`OrthogonalPolynomialBasis`: Classical orthogonal polynomial bases
    - :class:`PiecewiseBasis`: Basis constructed by tiling multiple local bases
    - :class:`TensorBasis`: Multi-dimensional basis formed by a tensor product
    """

    # Each attribute below is declared twice. The type checker sees a
    # read-only property, which subclasses may override with either a
    # dataclass field (e.g. OrthogonalPolynomialBasis.n_basis) or a computed
    # property (e.g. TensorBasis.n_basis). At runtime it is a plain class
    # attribute, so it neither shadows subclass dataclass fields nor hides
    # the docstring from Sphinx.
    if TYPE_CHECKING:

        @property
        def n_basis(self) -> int: ...

        @property
        def ndim(self) -> int: ...

        @property
        def density(self) -> bool: ...

    else:
        n_basis: int
        """Number of basis functions in this basis."""

        ndim: int = 1
        """Number of independent variables the basis functions take.

        Typically 1, since most bases are univariate. :class:`TensorBasis` is
        the exception, with one variable per tensored factor.
        """

        density: bool = False
        """Whether this basis is orthonormal with respect to a probability
        measure (unit mass) rather than the raw weight of the associated
        :class:`~archimedes.measure.Measure`.

        Only meaningful for orthogonal polynomial families based on a
        ``Measure`` (in particular :class:`OrthogonalPolynomialBasis`);
        other families should leave this ``False``.
        """

    @property
    @abc.abstractmethod
    def Parameters(self) -> type[ReferenceDomain.Parameters]:  # noqa: N802
        r"""The target-domain parameters this basis expects.

        A :class:`~archimedes.measure.ReferenceDomain.Parameters` subclass
        (e.g. ``UnitInterval.Parameters``). Returns the type, not an instance.

        Determines which reference domain the basis is defined on, for example:

        - :class:`~archimedes.measure.UnitInterval` for :math:`[-1, 1]`.
        - :class:`~archimedes.measure.HalfLine` for :math:`[0, \infty)`.
        - :class:`~archimedes.measure.RealLine` for :math:`(-\infty, \infty)`.
        """
        raise NotImplementedError

    @property
    def _measures(self) -> tuple[Measure | None, ...]:
        """The :class:`Measure`(s) this basis is defined from, if applicable.

        A tuple of length ``ndim``, with ``None`` for a family that has
        no defining measure (e.g. nodal, piecewise).
        """
        return (None,) * self.ndim

    @property
    def _required_breakpoints(self) -> np.ndarray | None:
        """Points on the reference domain where this basis is not smooth, or
        ``None`` if it is smooth throughout.

        Used for checking consistency of quadrature rules with the basis.
        A quadrature rule integrates products of basis functions exactly
        only if none of its subintervals straddles one of these kinks --
        equivalently, if the rule's own breakpoints are a *superset* of
        these. Equality is not required: refining an element is harmless,
        and a finer rule that is not aligned is still wrong, so node count
        is beside the point.

        Returns ``None`` by default, which skips the consistency check.
        """
        return None

    @abc.abstractmethod
    def _default_quadrature(self) -> "QuadratureRule":
        """A quadrature rule that integrates this basis's mass and stiffness
        integrands exactly.

        Fully determined by the basis: the degree requirement follows from
        ``n_basis``, and any element structure from the basis's breakpoints.
        :class:`FunctionSpace` uses this when no explicit rule is given.

        That default rule is *not* generally sufficient for
        :meth:`FunctionSpace.project`, where the accuracy depends on the
        target function rather than on the space. ``project`` takes an explicit
        override to manually control accuracy in that case.
        """
        raise NotImplementedError

    def _evaluate_expansion(
        self,
        coefficients: np.ndarray,
        x: np.ndarray,
        deriv: int = 0,
        side: str = RIGHT,
        **domain_kwargs,
    ) -> np.ndarray:
        r"""Evaluate :math:`\sum_i c_i \, \phi_i(x)` directly.

        Mathematically equivalent to ``evaluate(x, deriv) @ coefficients``,
        which is the default implementation, but lets a family with local
        support (e.g. piecewise polynomials) fuse the two steps and avoid
        materializing the full ``(npts, n_basis)`` design matrix. Used by
        :class:`FunctionSpace` (via its own private ``_evaluate``), which is
        how a :class:`Function` call reaches this; not something a caller
        needs directly.

        ``coefficients`` has shape ``(n_basis,)`` or ``(n_basis, m)`` for a
        vector-valued expansion, and the return shape matches: ``(npts,)``
        or ``(npts, m)``.
        """
        return self.evaluate(x, deriv=deriv, side=side, **domain_kwargs) @ coefficients

    def _evaluate_at_nodes(self, rule, deriv=0, **domain_kwargs) -> np.ndarray:
        """Design matrix at a quadrature rule's nodes.

        Equivalent to ``evaluate(rule.nodes, deriv)``, which is the default
        implementation, but lets a family use the provenance from a rule to
        override. For example, discontinuous piecewise bases need to track
        which nodes belong to which element, since the coordinates alone don't
        determine this. Smooth bases don't have this ambiguity and don't need
        to override this.
        """
        return self.evaluate(rule.nodes, deriv=deriv, **domain_kwargs)

    def _product_basis(self, other: "Basis") -> "Basis":
        """A basis sufficient to represent products exactly.

        For polynomial families the requirement is purely a degree count.
        For example, a degree-:math:`(n_1 - 1)` polynomial times a
        degree-:math:`(n_2 - 1)` polynomial in general has degree
        :math:`n_1 + n_2 - 2`, so the product space needs
        :math:`n_1 + n_2 - 1` functions.

        Raises ``NotImplementedError`` by default, since this can't be
        known in general for non-polynomial families.

        Raises
        ------
        ValueError
            If the two bases are not compatible (different measures,
            different breakpoints, ...), so no common product space exists.
        NotImplementedError
            If this family has no product-space construction.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a product basis; multiply "
            f"the Functions into an explicitly chosen FunctionSpace instead"
        )

    def _derivative_basis(self, deriv=1) -> "Basis":
        r"""The smallest basis that represents this family's derivatives exactly.

        Note that for polynomial families, this is a strictly smaller space than
        the original. Returning the minimal space keeps downstream operations
        from carrying extra unnecessary degrees of freedom.

        Raises ``NotImplementedError`` by default, since this can't be
        known in general for non-polynomial families.

        Raises
        ------
        ValueError
            If ``deriv`` is not a valid derivative order for this basis.
        NotImplementedError
            If this family has no derivative-space construction.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a derivative basis; "
            f"evaluate with `deriv=` instead, or differentiate into an "
            f"explicitly chosen FunctionSpace"
        )

    def _integral_basis(self, order: int = 1) -> "Basis":
        """The smallest basis whose derivatives span this basis exactly.

        For polynomial families this returns a larger basis, since integrating
        raises degree by one per order. The expanded basis usually has an extra
        degree of freedom for the integral constant; see :meth:`Function.integral`.

        Raises ``NotImplementedError`` by default, since this can't be
        known in general for non-polynomial families.

        Raises
        ------
        ValueError
            If ``order`` is not a valid integration order for this basis.
        NotImplementedError
            If this family has no integral-space construction.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define an integral basis."
        )

    @property
    def _dof_order(self) -> np.ndarray:
        """The intrinsic derivative order of each basis function's coefficient

        Most commonly zero, meaning each coefficient corresponds to the function
        itself rather than its derivatives. A counterexample is a cubic Hermite
        basis, where some coefficients correspond to derivatives rather than the
        function itself.

        Returns an array of shape ``(n_basis,)``.
        """
        return np.zeros(self.n_basis, dtype=int)  # type: ignore[attr-defined]

    @property
    def _reference_scale_exponent(self) -> float:
        r"""Optional extra, uniform scaling exponent for the reference domain.

        Zero by default: a family whose coefficients are plain values with no
        domain-dependent normalization of their own.

        Applied on top of the ``(2/width) ** (deriv - dof_order[k])`` Jacobian
        factor for the reference-to-physical domain transformation.
        """
        return 0.0

    def boundary_dofs(self, order: int = 0) -> tuple[int | None, int | None]:
        r"""Indices of the degrees of freedom at the left/right ends of the domain.

        Can be used for example to set boundary conditions or enforce continuity
        at the domain endpoints.

        Parameters
        ----------
        order : int, optional
            Derivative order to look up. Default 0 (the endpoint value).

        Returns
        -------
        left, right : int or None
            Index into this basis's functions, or ``None`` where there is
            no degree of freedom of that order at that end.

        Notes
        -----
        Only meaningful for nodal families (e.g. :class:`LagrangeBasis`) with
        nodes at the endpoints. Modal families (e.g.
        :class:`OrthogonalPolynomialBasis`) and nodal bases with interior-only
        nodes (Gauss-Legendre points, for instance) do not have boundary DOFs.
        """
        return (None, None)

    @abc.abstractmethod
    def evaluate(
        self, x: np.ndarray, deriv: int = 0, side: str = RIGHT, **domain_kwargs
    ) -> np.ndarray:
        """Evaluate all ``n_basis`` basis functions at ``x``.

        Parameters
        ----------
        x : array_like
            Evaluation points, shape ``(npts,)``.
        deriv : int, optional
            Order of derivative to evaluate. Default 0.
        side : {"right", "left"}, optional
            Which one-sided limit to take where the basis is two-valued.
            Default ``"right"``.  Irrelevant for smooth bases.
        **domain_kwargs
            Target-domain parameters; see the subclass docstring.

        Returns
        -------
        phi : ndarray
            Basis values (or ``deriv``-th derivatives), shape ``(npts, n_basis)``.
        """
        raise NotImplementedError
