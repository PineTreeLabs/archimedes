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
    """True if the weight is constant (``weight(x) == weight(y)``) for every
    ``x``, ``y`` in ``support`` -- e.g. true for Legendre, false for Jacobi
    (singular at the endpoints) or Hermite/Laguerre (unbounded support).
    Used by consumers that need to know whether the weight's *shape* is
    trivial, e.g. ``archimedes.quadrature.composite``, which can only tile a
    rule across sub-elements when there's no interior discontinuity in the
    weight to worry about."""

    class Parameters:
        """Base for a family's affine-reparametrization parameters.

        Each concrete :class:`Measure` defines its own ``@tree.struct``
        subclass of ``Parameters`` with the fields ``affine_params`` accepts
        (e.g. ``a``/``b`` for :class:`~archimedes.measure.LegendreMeasure`,
        ``mean``/``std`` for :class:`~archimedes.measure.HermiteMeasure`).
        This base is never instantiated directly -- it exists so code that
        doesn't know which family it's working with can still refer to "the
        parameters of some measure" as a single type.
        """

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
        ``args``/``kwargs`` are compatible with this family by constructing
        and validating a ``self.Parameters(*args, **kwargs)`` instance internally.
        Their meaning is family-specific; see the subclass docstring.
        """
        raise NotImplementedError

    @abc.abstractmethod
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

        Parameters
        ----------
        n : int
            Number of coefficients to compute, i.e. degrees ``0, ..., n-1``.
            Must be ``>= 1``; not validated here.
        """
        raise NotImplementedError
