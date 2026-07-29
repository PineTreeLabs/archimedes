"""Abstract base class for finite-dimensional bases.

Defines the :class:`Basis` interface implemented by each basis family
(orthogonal polynomials, Lagrange/nodal interpolants, piecewise/local
bases, etc.).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from archimedes.measure import Measure
    from archimedes.quadrature import QuadratureRule

__all__ = ["Basis"]

RIGHT = "right"
"""``side`` value selecting the limit from above at a point of discontinuity."""

LEFT = "left"
"""``side`` value selecting the limit from below at a point of discontinuity."""


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
    a fixed target domain. See :class:`FunctionSpace`, which pairs a
    ``Basis`` with a domain and quadrature-based operations (``project``,
    ``mass_matrix``, ``stiffness_matrix``), and :class:`Function`, which
    pairs a ``FunctionSpace`` with coefficients.
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

    :class:`FunctionSpace` forwards this to
    ``QuadratureRule.scaled_weights``/``scaled_points``' ``density``
    argument wherever it draws quadrature weights (``mass_matrix``,
    ``stiffness_matrix``, ``inner_product``, ``project``), so that with
    ``density=True`` the mass matrix is still the identity but the
    quadrature is now over a probability measure -- e.g. the coefficients
    from ``project`` become directly interpretable moments, as in a
    polynomial chaos expansion: ``c_0`` is the mean and ``sum(c[1:]**2)``
    is the variance of the projected function, with no extra rescaling.
    Default ``False``, matching the classical spectral-method convention
    (orthonormal against the raw weight).
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

        Used by :class:`FunctionSpace` wherever it integrates
        (``mass_matrix``, ``stiffness_matrix``, ``inner_product``,
        ``project``, ``diff_matrix``). Evaluation at *user-supplied* points
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
        :meth:`FunctionSpace.diff_matrix`, whose square (same-space) form is
        the classical differentiation matrix.

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

    def boundary_dofs(self) -> tuple[int | None, int | None]:
        """Indices of the degrees of freedom that *are* the values at the
        left and right ends of the domain, or ``None`` where there is no
        such DOF.

        Only meaningful for nodal families: a Lagrange basis whose nodes
        include both endpoints has :math:`\\ell_i(x_{\\mathrm{left}}) =
        \\delta_{i,\\mathrm{left}}`, so coefficient ``left`` is exactly the
        endpoint value. A modal family (e.g.
        :class:`OrthogonalPolynomialBasis`) has no such DOF -- its endpoint
        value is a combination of every coefficient -- and neither does a
        nodal basis whose nodes are all interior (Gauss-Legendre points).

        Used by :class:`PiecewiseBasis` to impose :math:`C^0` continuity by
        identifying adjacent elements' endpoint DOFs. The default returns
        ``(None, None)``, i.e. "continuity >= 0 not supported"; families
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
