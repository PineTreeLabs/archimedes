"""Abstract base class for finite-dimensional bases.

Defines the :class:`Basis` interface implemented by each basis family
(orthogonal polynomials, Lagrange/nodal interpolants, piecewise/local
bases, etc.).
"""

from __future__ import annotations

import abc

import numpy as np

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
