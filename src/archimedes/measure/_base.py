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
import functools

import numpy as np
from scipy.integrate import quad

from ._domain import ReferenceDomain
from ._stieltjes import stieltjes_recurrence

__all__ = ["Measure"]


def _check_affine_invariant(measure: "Measure", params: tuple, kwparams: dict) -> None:
    """Raise if `params`/`kwparams` describe a non-reference-domain mapping
    and `measure.affine_invariant` is False.

    Syntactic (were *any* arguments given), not semantic (is the resulting
    map the identity) -- deliberately, so this stays correct under symbolic
    tracing, where `scale == 1.0` isn't a decidable Python bool. A bare
    reference-domain call is always allowed.

    Shared by `Measure.__call__` and `QuadratureRule.map_to`/`scaled_points`/
    `scaled_weights` (imported from here rather than duplicated) -- both are
    "take a measure, apply an affine remap" operations with the same
    affine_invariant requirement.
    """
    if (params or kwparams) and not measure.affine_invariant:
        raise ValueError(
            f"{type(measure).__name__}.affine_invariant is False: its "
            f"recurrence_coeffs relies on the generic Stieltjes-based "
            f"fallback, so mapping a rule built for it onto a different "
            f"domain is not verified to give the same rule you'd get by "
            f"building on the target domain directly. Call with no "
            f"arguments for the reference domain, or set "
            f"`affine_invariant = True` on a subclass whose closed-form "
            f"recurrence you have verified is affine-invariant."
        )


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

    affine_invariant: bool = False
    """True if this measure's weight is provably the same *shape*, just
    relocated/rescaled, under ``affine_params``'s reparametrization.
    False (the default) for anything relying on the generic Stieltjes-based
    ``recurrence_coeffs`` fallback, since it is not guaranteed that an arbitrary
    weight family stays affine-closed."""

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

    @functools.cached_property
    def reference_mass(self) -> float:
        """Zeroth moment :math:`\\int_\\mathcal{D} w(t) ~ dt` of the reference weight.

        Since ``affine_params`` only ever rescales/shifts the reference
        domain, the zeroth moment of the mapped weight is always
        ``scale * reference_mass``, with no dependence on ``shift``. This is
        the normalizing constant that turns the (raw) weight into a probability
        density, ``weight(x) / reference_mass``. See also ``mass``, which
        generalizes this to a mapped instance.

        The default implementation numerically integrates ``weight`` over
        ``support`` via :func:`scipy.integrate.quad`, so (together with the
        default :meth:`recurrence_coeffs`) a custom ``Measure`` subclass
        needs only ``weight`` and ``domain`` to be fully usable.
        """
        return float(quad(self.weight, *self.support)[0])

    def mass(self, *args, **kwargs) -> float:
        """Total mass of the measure mapped by ``affine_params(*args,
        **kwargs)``.

        Equal to ``scale * reference_mass``, where ``scale`` is the affine
        scale factor. Dividing a mapped weight by this quantity turns it
        into a probability density on the mapped domain. Concrete
        subclasses don't need to override this; it's fully determined by
        ``affine_params`` and ``reference_mass``.
        """
        scale, _ = self.affine_params(*args, **kwargs)
        return scale * self.reference_mass

    def __call__(
        self, x: np.ndarray, *params, density: bool = False, **kwparams
    ) -> np.ndarray:
        """The weight function evaluated at physical points ``x`` on the
        domain mapped by ``affine_params(*params, **kwparams)``.

        Unlike :meth:`weight`, which only ever takes reference-domain
        points, this maps ``x`` back onto the reference domain first --
        :math:`w(\\mathrm{scale}^{-1} (x - \\mathrm{shift}))` for
        ``(scale, shift) = affine_params(*params, **kwparams)`` -- so it
        can be evaluated anywhere on the *target* domain. Called with no
        ``params``/``kwparams``, this is exactly ``weight(x)`` (the
        identity map).

        Parameters
        ----------
        x : array_like
            Physical-domain points to evaluate at.
        *params, **kwparams
            Target domain/measure parameters; see :meth:`affine_params`.
        density : bool, optional
            If ``True``, additionally divide by ``mass(*params, **kwparams)``,
            so the result is a probability density (unit total mass) rather
            than the raw weight -- e.g. for plotting the PDF a measure
            corresponds to via the Wiener-Askey correspondence. This is
            the *un*normalized weight by default, matching
            ``QuadratureRule.integrate``/``sum``, which likewise only
            normalize when ``density=True``: the raw weight is what a
            quadrature rule built on this measure actually integrates
            against, so it's the more fundamental case, e.g. for sanity-
            checking a rule's weights against a continuous plot of the
            same weight function.

        Raises
        ------
        ValueError
            If any of ``params``/``kwparams`` is given and
            ``affine_invariant`` is ``False``; see ``affine_invariant``.
        """
        _check_affine_invariant(self, params, kwparams)
        scale, shift = self.affine_params(*params, **kwparams)
        w = self.weight((x - shift) / scale)
        if density:
            w = w / self.mass(*params, **kwparams)
        return w

    def recurrence_coeffs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Monic three-term recurrence coefficients, each shape ``(n,)``.

        The monic polynomials orthogonal with respect to this (reference)
        measure satisfy

        .. math::
            \\pi_{k+1}(x) = (x - \\alpha_k) \\, \\pi_k(x) - \\beta_k \\,
                \\pi_{k-1}(x), \\qquad k = 0, \\ldots, n-1,

        with :math:`\\pi_{-1} = 0`, :math:`\\pi_0 = 1`. ``beta[0]`` plays no
        role in the recursion itself (since :math:`\\pi_{-1} = 0`) and is
        instead defined as ``reference_mass`` (see there for how it's
        obtained by default) -- the normalization Gauss quadrature (e.g. the
        Golub-Welsch algorithm) needs to recover quadrature weights from
        these coefficients.

        Only reference-instance coefficients are provided; a mapped
        instance's coefficients follow from ``affine_params`` (``alpha' =
        scale * alpha + shift``, ``beta' = scale**2 * beta`` with
        ``beta'[0] = mass(...)``), which callers can apply themselves.

        The default implementation falls back to a discretized Stieltjes
        procedure (:func:`~archimedes.measure.stieltjes_recurrence`), which
        numerically integrates the required moments via
        ``scipy.integrate.quad`` instead of a closed form. Classical families
        override this method with a closed-form recursion for speed and much
        better high-degree accuracy.

        Parameters
        ----------
        n : int
            Number of coefficients to compute, i.e. degrees ``0, ..., n-1``.
            Must be ``>= 1``; not validated here.
        """
        alpha, beta = stieltjes_recurrence(self.weight, self.support, n)
        beta[0] = self.reference_mass
        return alpha, beta
