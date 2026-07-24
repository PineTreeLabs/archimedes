"""Abstract base class for orthogonal polynomial family measures.

Defines the :class:`Measure` interface implemented by each classical family
(Legendre, Jacobi, Laguerre, Hermite): a weight function, its reference
domain, and the affine map relating the reference measure to other instances
of the same family.
"""

from __future__ import annotations

import abc

import numpy as np

__all__ = ["Measure"]


class Measure(metaclass=abc.ABCMeta):
    """The weight and reference domain defining an orthogonal polynomial family.

    A weight function :math:`w(x) \\geq 0` together with its support (the
    reference domain :math:`\\mathcal{D}`) defines an orthogonality measure
    :math:`d\\mu(x) = w(x) \\, dx`.

    The polynomials orthogonal with respect to this measure also determine the nodes
    of the associated Gauss quadrature rule: for the degree-``n`` orthogonal polynomial,
    the ``n`` roots :math:`x_i` and quadrature weights :math:`w_i` satisfy

    .. math::
        \\int_\\mathcal{D} f(x) \\, w(x) \\, dx = \\sum_{i=1}^n w_i f(x_i)

    exactly for every polynomial ``f`` of degree :math:`\\leq 2n - 1`.

    Subclasses implement one classical family each (Legendre, Jacobi,
    Laguerre, Hermite). ``affine_params`` additionally describes how the
    reference measure relates to other instances of the same family, e.g.
    rescaling the interval for Legendre/Jacobi, or the rate/location for
    Laguerre/Hermite.
    """

    uniform_weight: bool = False
    """True if ``weight(x) == 1`` for every ``x`` in ``support``"""

    @property
    @abc.abstractmethod
    def support(self) -> tuple[float, float]:
        """Support :math:`\\mathcal{D} = [a, b]` of the measure."""
        raise NotImplementedError

    @abc.abstractmethod
    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function :math:`w(x)`, evaluated at ``x``."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reference_mass(self) -> float:
        """Zeroth moment :math:`\\int_\\mathcal{D} w(t) \\, dt` of the
        *reference* weight (i.e. before any ``affine_params`` shift/scale).

        Since ``affine_params`` only ever rescales/shifts the reference
        domain -- the shape of the weight itself (e.g. Jacobi's ``alpha``,
        ``beta``) is fixed per instance -- the zeroth moment of the mapped
        weight is always ``scale * reference_mass``, with no dependence on
        ``shift``. This is what lets ``QuadratureRule``'s ``density``
        option normalize weights without re-deriving a moment per call.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def affine_params(self, *args, **kwargs) -> tuple[float, float]:
        """Return ``(scale, shift)`` mapping reference nodes onto the
        requested instance of this family.

        For reference node ``t``, the corresponding node in the target
        domain/measure is :math:`x = \\mathrm{scale} \\cdot t +
        \\mathrm{shift}`. Quadrature weights pick up the same ``scale`` as a
        Jacobian factor, since :math:`dx = \\mathrm{scale} \\cdot dt`.

        Called with no arguments, must return the identity ``(1.0, 0.0)``,
        i.e. the reference domain/measure itself. Also validates that
        ``args``/``kwargs`` are compatible with this family. Their meaning is
        family-specific; see the subclass docstring.
        """
        raise NotImplementedError
