"""Reference domains and their affine reparametrizations.

A :class:`ReferenceDomain` itself holds no parameters: it describes a *reference*
support (``[-1, 1]``, ``[0, inf)``, ``(-inf, inf)``) together with the shape of the
parameters that specialize it to a concrete target domain. Those parameter
*values* live in a separate ``Parameters`` instance.
"""

from __future__ import annotations

import abc
from typing import cast

import numpy as np

from archimedes import tree

__all__ = ["ReferenceDomain", "UnitInterval", "HalfLine", "RealLine"]


class ReferenceDomain(metaclass=abc.ABCMeta):
    """A reference support plus the affine map onto instances of it.

    Concrete subclasses fix a reference support and define the
    family-specific parameters that map it onto a target domain --
    ``a``/``b`` bounds for :class:`UnitInterval`, ``loc``/``scale`` for
    :class:`RealLine`, ``rate``/``start`` for :class:`HalfLine`.

    Instances carry no state; they exist so that a ``Measure`` or a
    ``Basis`` can say *which* reference domain it lives on without
    re-implementing the map. See :class:`archimedes.measure.Measure`, which
    pairs one of these with a weight function.
    """

    class Parameters:
        """Base for a domain's affine-reparametrization parameters.

        Each concrete :class:`ReferenceDomain` defines its own
        struct subclass with the fields ``affine_params``
        accepts. This base is never instantiated directly -- it exists so
        code that doesn't know which domain it's working with can still
        refer to "the parameters of some reference domain" as a single
        type.

        Because these are ``@struct`` types, their fields are pytree
        leaves: a domain parametrization can be symbolically traced (and
        so optimized over) rather than being fixed static data.
        """

    @property
    @abc.abstractmethod
    def support(self) -> tuple[float, float]:
        """Reference support :math:`\\mathcal{D} = [a, b]`."""
        raise NotImplementedError

    @abc.abstractmethod
    def resolve_params(self, *args, **kwargs) -> ReferenceDomain.Parameters:
        """Build (and validate) this domain's ``Parameters``.

        Called with no arguments, must return the ``Parameters`` for the
        reference domain itself.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def affine_params(self, *args, **kwargs) -> tuple[float, float]:
        """Return ``(scale, shift)`` mapping the reference domain onto the
        requested target instance.

        For reference point ``t``, the corresponding point in the target
        domain is :math:`x = \\mathrm{scale} \\cdot t + \\mathrm{shift}`.
        Integration weights pick up the same ``scale`` as a Jacobian
        factor, since :math:`dx = \\mathrm{scale} \\cdot dt`.

        Called with no arguments, must return the identity ``(1.0, 0.0)``,
        i.e. the reference domain itself. Validates the arguments via
        :meth:`resolve_params`.
        """
        raise NotImplementedError


class UnitInterval(ReferenceDomain):
    """The reference interval :math:`[-1, 1]`, mapped onto :math:`[a, b]`.

    Shared by the Legendre and Jacobi measures (which differ only in
    weight, not domain) and by nodal bases defined on a reference
    interval.
    """

    @tree.struct
    class Parameters(ReferenceDomain.Parameters):
        """Bounds of the target interval; see ``affine_params``."""

        a: float | None = None
        b: float | None = None

        def __post_init__(self):
            if self.a is None and self.b is None:
                return
            if self.a is None or self.b is None:
                raise ValueError("specify both `a` and `b`, or neither")
            if (isinstance(self.a, float) and not np.isfinite(self.a)) or (
                isinstance(self.b, float) and not np.isfinite(self.b)
            ):
                raise ValueError(
                    f"{type(self).__qualname__} requires a finite domain, "
                    f"got ({self.a}, {self.b})"
                )

    @property
    def support(self) -> tuple[float, float]:
        """Support :math:`[-1, 1]`."""
        return (-1.0, 1.0)

    def resolve_params(self, a=None, b=None) -> UnitInterval.Parameters:
        """Build this domain's ``Parameters``, defaulting to a=-1.0, b=1.0"""
        return self.Parameters(a, b)

    def affine_params(self, a=None, b=None) -> tuple[float, float]:
        """Map the reference interval :math:`[-1, 1]` onto :math:`[a, b]`.

        The affine map :math:`x = \\mathrm{scale} \\cdot t +
        \\mathrm{shift}` with

        .. math::
            \\mathrm{scale} = \\frac{b - a}{2}, \\qquad
            \\mathrm{shift} = \\frac{a + b}{2}

        carries the reference domain onto :math:`[a, b]`.

        Parameters
        ----------
        a, b : float, optional
            Bounds of the target interval. Must both be given, or neither
            (falls back to the reference domain, ``(1.0, 0.0)``).

        Returns
        -------
        scale, shift : float
            Affine parameters mapping reference points onto :math:`[a, b]`.

        Raises
        ------
        ValueError
            If only one of ``a``, ``b`` is given, or if they aren't finite.
        """
        params = self.resolve_params(a, b)
        if params.a is None and params.b is None:
            return 1.0, 0.0
        # `Parameters.__post_init__` has already rejected the one-sided case,
        # so both are non-None here; `cast` states that for mypy's benefit.
        a_, b_ = cast(float, params.a), cast(float, params.b)
        lo, hi = self.support
        scale = (b_ - a_) / (hi - lo)
        return scale, a_ - scale * lo


class HalfLine(ReferenceDomain):
    """The reference half-line :math:`[0, \\infty)`, reparametrized by a
    rate and a starting point. Used by the Laguerre measure."""

    @tree.struct
    class Parameters(ReferenceDomain.Parameters):
        """Rate/location of the target domain; see ``affine_params``."""

        rate: float = 1.0
        start: float = 0.0

        def __post_init__(self):
            if isinstance(self.rate, float) and self.rate <= 0:
                raise ValueError(f"rate must be positive, got {self.rate}")

    @property
    def support(self) -> tuple[float, float]:
        """Support :math:`[0, \\infty)`."""
        return (0.0, np.inf)

    def resolve_params(self, rate=None, start=None) -> HalfLine.Parameters:
        """Build this domain's ``Parameters``, defaulting to start=0.0, rate=1.0"""
        kwargs = {}
        if rate is not None:
            kwargs["rate"] = rate
        if start is not None:
            kwargs["start"] = start
        return self.Parameters(**kwargs)

    def affine_params(self, rate=None, start=None) -> tuple[float, float]:
        """Map :math:`[0, \\infty)` onto :math:`[\\mathrm{start}, \\infty)`
        with the given rate.

        Substituting :math:`x = \\mathrm{start} + t / \\mathrm{rate}` gives
        :math:`\\mathrm{scale} = 1/\\mathrm{rate}` and
        :math:`\\mathrm{shift} = \\mathrm{start}`.

        Parameters
        ----------
        rate : float, optional
            Rate of the target domain. Default 1.
        start : float, optional
            Left endpoint of the target domain. Default 0.

        Returns
        -------
        scale, shift : float
            Affine parameters as derived above.

        Raises
        ------
        ValueError
            If ``rate`` is not positive.
        """
        params = self.resolve_params(rate, start)
        return 1.0 / params.rate, params.start


class RealLine(ReferenceDomain):
    """The reference real line :math:`(-\\infty, \\infty)`, reparametrized
    by a location and scale. Shared by both Hermite measures."""

    @tree.struct
    class Parameters(ReferenceDomain.Parameters):
        """Location/scale of the target domain; see ``affine_params``."""

        loc: float = 0.0
        scale: float = 1.0

        def __post_init__(self):
            if isinstance(self.scale, float) and self.scale <= 0:
                raise ValueError(f"scale must be positive, got {self.scale}")

    @property
    def support(self) -> tuple[float, float]:
        """Support :math:`(-\\infty, \\infty)`."""
        return (-np.inf, np.inf)

    def resolve_params(self, loc=None, scale=None) -> RealLine.Parameters:
        """Build this domain's ``Parameters``, defaulting to loc=1.0, scale=1.0"""
        kwargs = {}
        if loc is not None:
            kwargs["loc"] = loc
        if scale is not None:
            kwargs["scale"] = scale
        return self.Parameters(**kwargs)

    def affine_params(self, loc=None, scale=None) -> tuple[float, float]:
        """Location-scale map :math:`x = \\mathrm{loc} + \\mathrm{scale} \\cdot t`.

        Note that the meaning of ``scale`` depends on the *weight* paired
        with this domain, not on the domain itself: for the physicists'
        Hermite weight :math:`e^{-t^2}` it is :math:`\\sqrt{2}` times the
        standard deviation of the corresponding Gaussian density, while
        for the probabilists' weight :math:`e^{-t^2/2}` it is exactly that
        standard deviation. See the respective ``Measure`` docstrings.

        Parameters
        ----------
        loc : float, optional
            Location of the target domain. Default 0.
        scale : float, optional
            Scale of the target domain. Default 1.

        Returns
        -------
        scale, shift : float
            Affine parameters as derived above.

        Raises
        ------
        ValueError
            If ``scale`` is not positive.
        """
        params = self.resolve_params(loc, scale)
        return params.scale, params.loc
