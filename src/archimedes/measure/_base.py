"""Abstract base class for classical weight/domain measures.

Defines the :class:`Measure` interface implemented by each classical family
(Legendre, Jacobi, Laguerre, Hermite): a weight function, its reference
domain, and the affine map relating the reference measure to other instances
of the same family. These are the weights of the classical orthogonal
polynomial families and, via the Wiener-Askey correspondence, the
(unnormalized) densities of the associated classical probability
distributions.
"""

from __future__ import annotations

import abc

import numpy as np

from ._domain import ReferenceDomain
from ._stieltjes import stieltjes_recurrence

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
    Laguerre, Hermite), pairing a weight with the ``domain`` it lives on.
    The domain -- the reference support and the affine map onto other
    instances of it -- is factored out into
    :class:`~archimedes.measure.ReferenceDomain`, since several families
    share one (Legendre and Jacobi are both :class:`UnitInterval`; both
    Hermite conventions are :class:`RealLine`) and since consumers with no
    weight function at all, such as a nodal
    :class:`~archimedes.experimental.approximation.Basis`, need the domain
    without the measure.
    """

    uniform_weight: bool = False
    """True if the weight is constant (``weight(x) == weight(y)``) for every
    ``x``, ``y`` in ``support`` -- e.g. true for Legendre, false for Jacobi
    (singular at the endpoints) or Hermite/Laguerre (unbounded support).
    Used by consumers that need to know whether the weight's *shape* is
    trivial, e.g. ``archimedes.quadrature.composite_quad``, which can only tile a
    rule across sub-elements when there's no interior discontinuity in the
    weight to worry about."""

    domain: ReferenceDomain
    """The reference domain this measure's weight is supported on. Set as a
    class attribute by each concrete subclass; carries the ``support``,
    ``affine_params`` and ``Parameters`` that ``Measure`` delegates to."""

    def __eq__(self, other: object) -> bool:
        """Measures compare by *type*, since a measure with no parameters
        (Legendre, Laguerre, both Hermites) is fully determined by its class
        -- two separately-constructed ``LegendreMeasure()`` instances denote
        the same measure and must compare equal.

        Parametrized families override this: :class:`JacobiMeasure` is a
        ``@dataclass``, whose generated ``__eq__`` compares ``alpha``/``beta``
        as well (``@dataclass`` leaves an explicitly-defined ``__eq__`` alone,
        but Jacobi doesn't define one, so it gets the field-wise version).
        """
        if not isinstance(other, Measure):
            return NotImplemented
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    @property
    def support(self) -> tuple[float, float]:
        """Support :math:`\\mathcal{D} = [a, b]`; forwarded from ``domain``."""
        return self.domain.support

    def affine_params(self, *args, **kwargs) -> tuple[float, float]:
        """Return ``(scale, shift)`` mapping the reference measure onto the
        requested instance of this family; forwarded to
        ``domain.affine_params``.

        Quadrature weights pick up the same ``scale`` as a Jacobian factor,
        since :math:`dx = \\mathrm{scale} \\cdot dt`. The meaning of the
        arguments is domain-specific -- ``a``/``b`` for
        :class:`UnitInterval`, ``loc``/``scale`` for :class:`RealLine`,
        ``rate``/``start`` for :class:`HalfLine` -- see the corresponding
        ``ReferenceDomain`` subclass.
        """
        return self.domain.affine_params(*args, **kwargs)

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
        ``shift``. This is the normalizing constant that turns the (raw)
        weight into a probability density, ``weight(x) / reference_mass``.
        See also ``mass``, which generalizes this to a mapped instance.
        """
        raise NotImplementedError

    def mass(self, *args, **kwargs) -> float:
        """Total mass of the measure mapped by ``affine_params(*args,
        **kwargs)``.

        Equal to ``scale * reference_mass``, where ``scale`` is the affine
        scale factor -- i.e. the Jacobian picked up by rescaling the
        reference domain. Dividing a mapped weight by this quantity turns it
        into a probability density on the mapped domain. Concrete
        subclasses don't need to override this; it's fully determined by
        ``affine_params`` and ``reference_mass``.
        """
        scale, _ = self.affine_params(*args, **kwargs)
        return scale * self.reference_mass

    def recurrence_coeffs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Monic three-term recurrence coefficients, each shape ``(n,)``.

        The monic polynomials orthogonal with respect to this (reference)
        measure satisfy

        .. math::
            \\pi_{k+1}(x) = (x - \\alpha_k) \\, \\pi_k(x) - \\beta_k \\,
                \\pi_{k-1}(x), \\qquad k = 0, \\ldots, n-1,

        with :math:`\\pi_{-1} = 0`, :math:`\\pi_0 = 1`. ``beta[0]`` plays no
        role in the recursion itself (since :math:`\\pi_{-1} = 0`) and is
        instead defined as ``reference_mass`` -- the normalization Gauss
        quadrature (e.g. the Golub-Welsch algorithm) needs to recover
        quadrature weights from these coefficients.

        Only reference-instance coefficients are provided; a mapped
        instance's coefficients follow from ``affine_params`` (``alpha' =
        scale * alpha + shift``, ``beta' = scale**2 * beta`` with
        ``beta'[0] = mass(...)``), which callers can apply themselves.

        The default implementation falls back to a discretized Stieltjes
        procedure (:func:`~archimedes.measure.stieltjes_recurrence`), which
        numerically integrates the required moments via
        ``scipy.integrate.quad`` instead of a closed form -- so any
        ``weight``/``domain`` pair yields a valid orthogonal polynomial
        family and Gauss quadrature rule (via
        :func:`~archimedes.quadrature.golub_welsch_rule`) with no further
        work. It is accurate to near machine precision for smooth, bounded
        weights through about :math:`n \\sim 15`, degrading (silently, for
        smooth weights) beyond that -- see
        :func:`~archimedes.measure.stieltjes_recurrence` for the full
        accuracy envelope. Classical families override this method with a
        closed-form recursion for speed and much better high-degree
        accuracy; see :class:`~archimedes.measure.JacobiMeasure` for the
        pattern.

        Parameters
        ----------
        n : int
            Number of coefficients to compute, i.e. degrees ``0, ..., n-1``.
            Must be ``>= 1``; not validated here.
        """
        alpha, beta = stieltjes_recurrence(self.weight, self.support, n)
        beta[0] = self.reference_mass
        return alpha, beta
