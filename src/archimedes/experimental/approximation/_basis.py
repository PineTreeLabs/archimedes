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
    from archimedes.quadrature import QuadratureRule

__all__ = ["Basis"]


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

    @property
    @abc.abstractmethod
    def Parameters(self) -> type:  # noqa: N802
        """The domain parameters this basis expects."""
        raise NotImplementedError

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
    def evaluate(self, x: np.ndarray, deriv: int = 0, **domain_kwargs) -> np.ndarray:
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
