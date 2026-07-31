"""Abstract base class for finite-dimensional bases.

Defines the :class:`Basis` interface implemented by each basis family
(orthogonal polynomials, Lagrange/nodal interpolants, piecewise/local
bases, etc.).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

from archimedes import tree

if TYPE_CHECKING:
    from archimedes.measure import Measure
    from archimedes.quadrature import QuadratureRule

__all__ = ["Basis", "BasisMatrix"]

RIGHT = "right"
"""``side`` value selecting the limit from above at a point of discontinuity."""

LEFT = "left"
"""``side`` value selecting the limit from below at a point of discontinuity."""


@tree.struct
class BasisMatrix:
    r"""A basis evaluated at a fixed set of quadrature nodes, bundled with
    the matching weights so the two can never be supplied out of sync.

    ``Phi[n, i]`` is the ``deriv``-th derivative of basis function ``i`` at
    node ``n`` -- see :meth:`FunctionSpace.basis_matrix`, which builds one.
    ``Phi`` maps coefficients to sampled values, :math:`\Phi c = \phi \cdot c`.

    ``Phi.T`` is the adjoint of ``Phi`` under the Euclidean inner product on
    coefficients and the weighted one on sampled values:
    :math:`\langle \Phi c, r\rangle_w = \langle c, \Phi^\top r\rangle`, so
    :math:`\Phi^\top r = \phi^\top (w \odot r)`. That makes the Galerkin
    projection equation read almost like the math it's approximating:
    ``M = phi.T @ phi`` is the Gram matrix :math:`\Phi^\top\Phi`
    (the mass matrix), and ``phi.T @ f(x)`` is the load vector
    :math:`\Phi^\top f` -- see :meth:`FunctionSpace.project`.

    Parameters
    ----------
    matrix : ndarray
        The design matrix ``Phi``, shape ``(npts, n_basis)``.
    weights : ndarray
        Quadrature weights matching ``matrix``'s node axis, shape
        ``(npts,)``.
    """

    matrix: np.ndarray
    weights: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape  # type: ignore[return-value]

    def __matmul__(self, coefficients: np.ndarray) -> np.ndarray:
        """:math:`\\Phi c`: sampled values at the quadrature nodes."""
        return self.matrix @ coefficients  # type: ignore[no-any-return]

    @property
    def T(self) -> _BasisMatrixAdjoint:  # noqa: N802
        """The adjoint :math:`\\Phi^\\top`; see the class docstring."""
        return _BasisMatrixAdjoint(self)


@tree.struct
class _BasisMatrixAdjoint:
    """``BasisMatrix.T``: apply via ``@``, undo via ``.T`` again."""

    basis_matrix: BasisMatrix

    def __matmul__(self, values: np.ndarray | BasisMatrix) -> np.ndarray:
        """:math:`\\Phi^\\top r = \\phi^\\top (w \\odot r)`.

        ``values`` (``r``) must already be sampled at the same quadrature
        nodes as ``self.basis_matrix`` -- shape ``(npts,)`` for a scalar
        integrand, or ``(npts, m)`` for a vector-valued one (contracted
        independently per component), or ``(npts, k)`` to apply the adjoint
        to another design matrix at once (as in the Gram matrix
        ``phi.T @ phi``).
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
    """Validate a ``side`` value.

    Checked by every family, including the smooth ones for which the two
    sides coincide: a typo should fail the same way regardless of which
    basis it is handed to.
    """
    if side not in (LEFT, RIGHT):
        raise ValueError(f"side must be {LEFT!r} or {RIGHT!r}, got {side!r}")
    return side


class Basis(metaclass=abc.ABCMeta):
    """A finite family of basis functions :math:`\\{\\phi_i\\}_{i=1}^n`.

    Concrete subclasses implement one basis family each (Legendre, Lagrange,
    ...). Unlike :class:`archimedes.measure.Measure`, which represents an
    infinite family with no notion of size, a ``Basis`` always has a fixed
    ``n_basis``.

    ``Basis`` only evaluates -- it has no notion of a coefficient vector or
    a fixed target domain. See :class:`FunctionSpace`, which combines a
    ``Basis`` with a domain and quadrature-based operations, and :class:`Function`,
    which further combines a ``FunctionSpace`` with coefficients.
    """

    ndim: int = 1
    """Number of independent variables the basis functions take.

    Nearly every family here is univariate; :class:`TensorBasis` is the
    exception, taking one variable per tensored factor. ``evaluate``'s ``x``
    is ``(npts,)`` when this is 1 and ``(npts, ndim)`` otherwise, and
    ``deriv`` is a plain order in the first case and a multi-index in the
    second.
    """

    density: bool = False
    """Whether this basis is orthonormal with respect to a *probability*
    measure (unit mass) rather than its associated
    :class:`~archimedes.measure.Measure`'s raw weight.

    Only meaningful for families built on a classical-orthogonal-polynomial
    ``Measure`` (see :class:`OrthogonalPolynomialBasis`); other families
    (nodal, piecewise) have no such notion and leave this ``False``.
    """

    @property
    @abc.abstractmethod
    def Parameters(self) -> type:  # noqa: N802
        """The domain parameters this basis expects."""
        raise NotImplementedError

    @property
    def measures(self) -> tuple[Measure | None, ...]:
        """The orthogonality weight this basis is built against, per
        dimension -- always a tuple of length ``ndim``, with ``None`` for a
        family that has no weight of its own (nodal, piecewise).

        Used by :class:`FunctionSpace` to reject a quadrature rule whose
        weight does not match the basis's. ``None`` disables that
        check for a dimension, since there is then nothing to disagree with.
        """
        return (None,) * self.ndim

    @property
    def required_breakpoints(self) -> np.ndarray | None:
        """Points on the reference domain where this basis is not smooth, or
        ``None`` if it is smooth throughout.

        A quadrature rule integrates products of basis functions exactly
        only if none of its subintervals straddles one of these kinks --
        equivalently, if the rule's own breakpoints are a *superset* of
        these. Equality is not required: refining an element is harmless,
        and a finer rule that is not aligned is still wrong, so node count
        is beside the point.

        Returns ``None`` by default (a globally smooth family, e.g. a
        polynomial basis); :class:`PiecewiseBasis` overrides it.
        """
        return None

    @abc.abstractmethod
    def default_quadrature(self) -> "QuadratureRule":
        """A quadrature rule that integrates this basis's mass and stiffness
        integrands exactly.

        Fully determined by the basis: the degree requirement follows from
        ``n_basis``, and any element structure from the basis's own
        breakpoints. :class:`FunctionSpace` uses this when no explicit rule
        is given.

        This is *not* generally sufficient for :meth:`FunctionSpace.project`,
        whose accuracy requirement depends on the target function rather
        than on the space -- ``project`` takes an explicit override for
        that case.
        """
        raise NotImplementedError

    def evaluate_expansion(
        self,
        coefficients: np.ndarray,
        x: np.ndarray,
        deriv: int = 0,
        side: str = RIGHT,
        **domain_kwargs,
    ) -> np.ndarray:
        """Evaluate :math:`\\sum_i c_i \\, \\phi_i(x)` directly.

        Mathematically equivalent to ``evaluate(x, deriv) @ coefficients``,
        which is the default implementation, but allows families with local
        support (e.g. piecewise polynomials) to fuse the two steps.

        Parameters
        ----------
        coefficients : ndarray
            Shape ``(n_basis,)`` or ``(n_basis, m)`` for a vector-valued
            expansion.
        x : array_like
            Evaluation points, shape ``(npts,)``.
        deriv : int, optional
            Derivative order. Default 0.
        side : {"right", "left"}, optional
            One-sided limit to take at a point of discontinuity, as for
            :meth:`evaluate`.
        **domain_kwargs
            Target-domain parameters, as for :meth:`evaluate`.

        Returns
        -------
        ndarray
            Shape ``(npts,)`` or ``(npts, m)``, matching ``coefficients``.
        """
        return self.evaluate(x, deriv=deriv, side=side, **domain_kwargs) @ coefficients

    def _evaluate_at_nodes(self, rule, deriv=0, **domain_kwargs) -> np.ndarray:
        """Design matrix at a quadrature rule's nodes.

        Equivalent to ``evaluate(rule.scaled_points(...), deriv)``, which is
        the default implementation, but lets a family use the *provenance*
        a rule carries and coordinates do not: which sub-element each node
        came from (see
        :attr:`~archimedes.quadrature.Quadrature.elements`).

        Only :class:`PiecewiseBasis` needs this, and only because it is
        discontinuous at its breakpoints: a rule may legitimately place
        nodes exactly there, where the value depends on which element the
        node belongs to and the coordinate cannot say. Every smooth family
        is single-valued everywhere, so the default is exact for them.

        Used by :class:`FunctionSpace` wherever it integrates (``basis_matrix``,
        ``project``, and their private counterparts). Evaluation at
        *user-supplied* points
        goes through :meth:`evaluate`/:meth:`evaluate_expansion` instead,
        which have no provenance to draw on and resolve breakpoints by the
        documented ``side`` convention.
        """
        return self.evaluate(
            rule.scaled_points(**domain_kwargs), deriv=deriv, **domain_kwargs
        )

    def _product_basis(self, other: "Basis") -> "Basis":
        """A basis large enough to represent products from this basis and
        ``other`` *exactly*.

        For polynomial families the requirement is purely a degree count:
        a degree-:math:`(n_1 - 1)` polynomial times a degree-:math:`(n_2 -
        1)` one has degree :math:`n_1 + n_2 - 2`, so the product space needs
        :math:`n_1 + n_2 - 1` functions. That number is *static*, which is
        what makes products expressible here at all -- the space cannot be
        sized from the data the way an adaptive system would.

        Raising ``NotImplementedError`` is the correct default: whether
        products are closed in an enlarged version of the same family is a
        per-family fact, not something that can be assumed.

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
        """The smallest basis that represents this family's ``deriv``-th
        derivatives *exactly*.

        Note this is a strictly smaller space, not merely a different one:
        :math:`P_{n-2} \\subset P_{n-1}`, so the derivative would also be
        exactly representable in the *original* basis. Returning the minimal
        space keeps the rule uniform so that every closed operation gives
        the tightest exact space and keeps downstream products from carrying
        extra unnecessary degrees of freedom. See also
        :meth:`Function.derivative`, whose square (same-space) form is
        built from the classical differentiation matrix.

        Raises
        ------
        ValueError
            If ``deriv`` is not a valid derivative order for this basis.
        NotImplementedError
            If this family has no derivative-space construction.

        Notes
        -----
        The result is usually, but not necessarily, in the the *same family*
        at a lower order. A family whose degrees of freedom are not all the
        same *kind* of quantity (e.g.
        :class:`~archimedes.experimental.approximation.CubicHermiteBasis`,
        whose coefficients are a mix of values and physical derivatives) may
        have no derivative-space construction *of its own kind*: the
        derivative of a cubic Hermite element is a plain polynomial with no
        value/slope split, so its derivative basis is a quadratic
        :class:`LagrangeBasis`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a derivative basis; "
            f"evaluate with `deriv=` instead, or differentiate into an "
            f"explicitly chosen FunctionSpace"
        )

    @property
    def _dof_order(self) -> np.ndarray:
        """The intrinsic derivative order of each basis function's
        coefficient, shape ``(n_basis,)``.

        Every family so far is *homogeneous*: each coefficient is a plain
        value (a nodal value, a modal amplitude), which is why a single
        ``scale`` factor per requested ``deriv`` (see :meth:`evaluate`) is
        enough to remap a whole basis onto a different target domain. The
        default here reflects that: all zeros.

        Purely an implementation detail of that remapping -- not something
        a caller needs, only the small set of families and internals
        (:class:`PiecewiseBasis`'s fused
        :meth:`~PiecewiseBasis.evaluate_expansion` path) that must apply a
        *per-column* power of ``scale`` rather than one factor for the whole
        matrix. A family whose coefficients are not all the same *kind* of
        quantity --
        :class:`~archimedes.experimental.approximation.CubicHermiteBasis`
        is the first example, whose odd-indexed coefficients are physical
        derivatives rather than values -- overrides this so that
        :meth:`evaluate` can apply ``scale ** (dof_order - deriv)`` per column.
        """
        return np.zeros(self.n_basis, dtype=int)  # type: ignore[attr-defined]

    def boundary_dofs(self, order: int = 0) -> tuple[int | None, int | None]:
        """Indices of the degrees of freedom that *are* the ``order``-th
        derivative at the left and right ends of the domain, or ``None``
        where there is no such DOF.

        Only meaningful for nodal families: a Lagrange basis whose nodes
        include both endpoints has :math:`\\ell_i(x_{\\mathrm{left}}) =
        \\delta_{i,\\mathrm{left}}`, so coefficient ``left`` is exactly the
        endpoint value (``order=0``). A modal family (e.g.
        :class:`OrthogonalPolynomialBasis`) has no such DOF -- its endpoint
        value is a combination of every coefficient -- and neither does a
        nodal basis whose nodes are all interior (Gauss-Legendre points).
        ``order=1`` asks instead for the DOF that *is* the physical first
        derivative at that endpoint -- meaningful only for a family with
        derivative-type degrees of freedom, e.g.
        :class:`~archimedes.experimental.approximation.CubicHermiteBasis`.

        Used by :class:`PiecewiseBasis` to impose continuity by identifying
        adjacent elements' endpoint DOFs, one ``order`` at a time up to its
        ``continuity``. The default returns ``(None, None)`` for every
        ``order``, i.e. "no continuity of any order supported"; families
        that can support it override this.
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
            Evaluation points, shape ``(npts,)``. May be a NumPy array or a
            symbolic array (e.g. inside ``@arc.compile``); implementations
            must not branch in Python on which.
        deriv : int, optional
            Order of derivative to evaluate. Default 0 (the basis functions
            themselves). Not every family supports every order -- see the
            subclass docstring for what's implemented.
        side : {"right", "left"}, optional
            Which one-sided limit to take where the basis is two-valued.
            Default ``"right"``.  Irrelevant for smooth bases.
        **domain_kwargs
            Target-domain parameters. Families built on a classical
            orthogonal-polynomial :class:`~archimedes.measure.Measure`
            forward these to that measure's ``affine_params`` (e.g. ``a``,
            ``b`` for Legendre); families with no natural reference domain
            may ignore them. See the subclass docstring.

        Returns
        -------
        phi : ndarray
            Basis values (or ``deriv``-th derivatives), shape
            ``(npts, n_basis)`` -- points first, basis index last, matching
            the "design matrix" / (generalized) Vandermonde convention used
            by e.g. ``numpy.polynomial.legendre.legvander`` and modal/POD
            decompositions, rather than
            ``archimedes.quadrature.QuadratureRule``'s points-last axis
            convention.
        """
        raise NotImplementedError
