"""Fixed-node Gauss quadrature rules for classical orthogonal polynomial weights.

Gauss quadrature approximates a weighted integral

.. math::
    \\int_I f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n w_i f(x_i)

exactly for every polynomial `f` of degree :math:`\\leq 2n - 1`, where the
nodes :math:`x_i` are the roots of the degree-`n` polynomial orthogonal with
respect to the weight `w` on the reference domain `I`. Each classical
a "weight/domain pair" (Legendre, Jacobi, Laguerre, Hermite) is represented by
a "family"; :class:`QuadratureRule` pairs a family with a
fixed set of nodes and weights on its reference domain.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Callable

import numpy as np
from scipy.special import roots_jacobi, roots_legendre


class _QuadratureFamily(metaclass=abc.ABCMeta):
    """The weight and reference domain defining a classical orthogonal
    polynomial family and its Gauss quadrature rule.

    A weight function :math:`w(x) \\geq 0` together with its support (the
    reference domain :math:`I`) defines an orthogonality measure
    :math:`d\\mu(x) = w(x) \\, dx`. The polynomials orthogonal with respect
    to this measure determine the nodes of the associated Gauss quadrature
    rule: for the degree-`n` orthogonal polynomial, the `n` roots
    :math:`x_i` and quadrature weights :math:`w_i` satisfy

    .. math::
        \\int_I f(x) \\, w(x) \\, dx = \\sum_{i=1}^n w_i f(x_i)

    exactly for every polynomial `f` of degree :math:`\\leq 2n - 1`.

    Subclasses implement one classical family each (Legendre, Jacobi,
    Laguerre, Hermite). `affine_params` additionally describes how the
    reference measure relates to other instances of the same family, e.g.
    rescaling the interval for Legendre/Jacobi, or the rate/location for
    Laguerre/Hermite.
    """

    uniform_weight: bool = False
    """True if `weight(x) == 1` for every `x` in `reference_domain`.

    A rule can only be tiled into a composite rule (see `composite`) if its
    family's reference weight is uniform: `affine_params` rescales the
    *whole* reference domain, so compositing applies it element-by-element, and
    that's only correct if the weight has no shape of its own to distort --
    i.e. it's constant. Families with a non-uniform reference weight (e.g.
    Jacobi, whose weight is singular at the reference endpoints) would pick
    up a spurious copy of that shape at every interior element boundary if
    tiled the same way.
    """

    @property
    @abc.abstractmethod
    def reference_domain(self) -> tuple[float, float]:
        """Support :math:`I = (\\mathrm{lo}, \\mathrm{hi})` of the reference
        weight function."""
        raise NotImplementedError

    @abc.abstractmethod
    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x)`, evaluated at `x`."""
        raise NotImplementedError

    @abc.abstractmethod
    def affine_params(self, *args, **kwargs) -> tuple[float, float]:
        """Return `(scale, shift)` mapping reference nodes onto the
        requested instance of this family.

        For reference node `t`, the corresponding node in the target
        domain/measure is :math:`x = \\mathrm{scale} \\cdot t +
        \\mathrm{shift}`. Quadrature weights pick up the same `scale` as a
        Jacobian factor, since :math:`dx = \\mathrm{scale} \\cdot dt`.

        Called with no arguments, must return the identity `(1.0, 0.0)`,
        i.e. the reference domain/measure itself. Also validates that
        `args`/`kwargs` are compatible with this family. Their meaning is
        family-specific; see the subclass docstring.
        """
        raise NotImplementedError


class _LegendreFamily(_QuadratureFamily):
    """Gauss-Legendre weight: :math:`w(x) = 1` on :math:`[-1, 1]`.

    The associated orthogonal polynomials are the Legendre polynomials
    :math:`P_n(x)`, and the resulting quadrature rule is exact for
    polynomials up to degree :math:`2n - 1`.
    """

    uniform_weight = True

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = 1`, evaluated at `x`."""
        return np.ones_like(x)

    def affine_params(self, a=None, b=None) -> tuple[float, float]:
        """Map the reference interval :math:`[-1, 1]` onto :math:`[a, b]`.

        The affine map :math:`x = \\mathrm{scale} \\cdot t +
        \\mathrm{shift}` with

        .. math::
            \\mathrm{scale} = \\frac{b - a}{2}, \\qquad
            \\mathrm{shift} = \\frac{a + b}{2}

        carries the reference node/weight pairs onto :math:`[a, b]` while
        preserving the (constant) shape of the weight function.

        Parameters
        ----------
        a, b : float, optional
            Bounds of the target interval. Must both be given, or neither
            (falls back to the reference domain, `(1.0, 0.0)`).

        Returns
        -------
        scale, shift : float
            Affine parameters mapping reference nodes onto :math:`[a, b]`.

        Raises
        ------
        ValueError
            If only one of `a`, `b` is given, or if `a`/`b` are not finite.
        """
        if a is None and b is None:
            return 1.0, 0.0
        if a is None or b is None:
            raise ValueError("specify both `a` and `b`, or neither")
        if (isinstance(a, float) and not np.isfinite(a)) or (
            isinstance(b, float) and not np.isfinite(b)
        ):
            raise ValueError(
                f"{type(self).__name__} requires a finite domain, got ({a}, {b})"
            )
        lo, hi = self.reference_domain
        scale = (b - a) / (hi - lo)
        return scale, a - scale * lo


@dataclasses.dataclass(frozen=True)
class _JacobiFamily(_LegendreFamily):
    """Gauss-Jacobi weight :math:`w(x) = (1-x)^\\alpha (1+x)^\\beta` on
    :math:`[-1, 1]`, with :math:`\\alpha, \\beta > -1`.

    The associated orthogonal polynomials are the Jacobi polynomials
    :math:`P_n^{(\\alpha,\\beta)}(x)`. Gauss-Legendre is the special case
    :math:`\\alpha = \\beta = 0`; Chebyshev quadrature of the first and
    second kind are the special cases :math:`\\alpha = \\beta = -1/2` and
    :math:`\\alpha = \\beta = 1/2`, respectively. The zeroth moment
    (normalization) of the weight has a closed form in terms of the Beta
    function:

    .. math::
        \\int_{-1}^1 (1-x)^\\alpha (1+x)^\\beta \\, dx
            = 2^{\\alpha + \\beta + 1} \\, B(\\alpha + 1, \\beta + 1)

    Since the reference domain :math:`[-1, 1]` is the same as
    :class:`_LegendreFamily`, `affine_params` is inherited unchanged.

    Parameters
    ----------
    alpha, beta : float
        Exponents of the weight function. Must be :math:`> -1` for the
        weight to be integrable at the corresponding endpoint.

    Raises
    ------
    ValueError
        If `alpha` or `beta` is :math:`\\leq -1`.
    """

    alpha: float
    beta: float

    # Not a dataclass field: unannotated, so `dataclasses` leaves it as a
    # plain class attribute overriding `_LegendreFamily.uniform_weight`.
    # The Jacobi weight is singular at the reference endpoints, so it
    # cannot be tiled into a composite rule -- see `_QuadratureFamily.
    # uniform_weight`.
    uniform_weight = False

    def __post_init__(self):
        if self.alpha <= -1 or self.beta <= -1:
            raise ValueError(
                f"invalid alpha={self.alpha} or beta={self.beta}, must be > -1"
            )

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = (1-x)^\\alpha
        (1+x)^\\beta`, evaluated at `x`."""
        return (1 - x) ** self.alpha * (1 + x) ** self.beta


class _LaguerreFamily(_QuadratureFamily):
    """Gauss-Laguerre weight: :math:`w(x) = e^{-x}` on
    :math:`[0, \\infty)`.

    The associated orthogonal polynomials are the (physicists') Laguerre
    polynomials :math:`L_n(x)`. Moments of the weight are given by the
    Gamma function: :math:`\\int_0^\\infty x^k e^{-x} \\, dx = k! =
    \\Gamma(k+1)`.
    """

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (0.0, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = e^{-x}`, evaluated at
        `x`."""
        return np.exp(-x)

    def affine_params(self, rate=None, start=None) -> tuple[float, float]:
        """Map the reference weight onto a rate/shifted exponential weight
        :math:`w(x) = e^{-\\mathrm{rate}(x - \\mathrm{start})}` on
        :math:`[\\mathrm{start}, \\infty)`.

        Substituting :math:`x = \\mathrm{start} + t / \\mathrm{rate}` into
        the reference integral gives

        .. math::
            \\int_{\\mathrm{start}}^\\infty f(x) \\,
                e^{-\\mathrm{rate}(x - \\mathrm{start})} \\, dx
            = \\frac{1}{\\mathrm{rate}} \\int_0^\\infty
                f(\\mathrm{start} + t/\\mathrm{rate}) \\, e^{-t} \\, dt,

        so :math:`\\mathrm{scale} = 1/\\mathrm{rate}` and
        :math:`\\mathrm{shift} = \\mathrm{start}`.

        Parameters
        ----------
        rate : float, optional
            Rate of the target exponential weight. Default 1.
        start : float, optional
            Left endpoint of the target domain. Default 0.

        Returns
        -------
        scale, shift : float
            Affine parameters as derived above.

        Raises
        ------
        ValueError
            If `rate` is not positive.
        """
        if rate is None and start is None:
            return 1.0, 0.0
        if rate is None:
            rate = 1.0
        if start is None:
            start = 0.0
        if isinstance(rate, float) and rate <= 0:
            raise ValueError(f"Gauss-Laguerre rate must be positive, got {rate}")
        return 1.0 / rate, start


class _HermiteFamily(_QuadratureFamily):
    """Gauss-Hermite weight: :math:`w(x) = e^{-x^2}` on
    :math:`(-\\infty, \\infty)`.

    The associated orthogonal polynomials are the *physicists'* Hermite
    polynomials :math:`H_n(x)` (as opposed to the *probabilists'*
    convention used by :class:`_HermiteNormFamily`, which instead uses
    weight :math:`e^{-x^2/2}`). The zeroth moment of the weight is
    :math:`\\int_{-\\infty}^\\infty e^{-x^2} \\, dx = \\sqrt{\\pi}`.
    """

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (-np.inf, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = e^{-x^2}`, evaluated at
        `x`."""
        return np.exp(-(x**2))

    def affine_params(self, mean=None, std=None) -> tuple[float, float]:
        """Map the reference weight onto a location-scaled Gaussian-shaped
        weight :math:`w(x) = \\exp(-((x - \\mathrm{mean})/
        \\mathrm{std})^2)`.

        Substituting :math:`x = \\mathrm{mean} + \\mathrm{std} \\cdot t`
        gives

        .. math::
            \\int_{-\\infty}^\\infty f(x) \\,
                e^{-((x-\\mathrm{mean})/\\mathrm{std})^2} \\, dx
            = \\mathrm{std} \\int_{-\\infty}^\\infty
                f(\\mathrm{mean} + \\mathrm{std} \\cdot t) \\, e^{-t^2} \\,
                dt,

        so :math:`\\mathrm{scale} = \\mathrm{std}` and
        :math:`\\mathrm{shift} = \\mathrm{mean}`.

        Note that because the reference weight uses the physicists'
        normalization :math:`e^{-t^2}` rather than the probabilists'
        :math:`e^{-t^2/2}`, `std` is *not* the standard deviation of a
        Gaussian density with this shape -- that would be :math:`\\sigma =
        \\mathrm{std} / \\sqrt{2}`.

        Parameters
        ----------
        mean : float, optional
            Location of the target weight. Default 0.
        std : float, optional
            Scale of the target weight. Default 1. Equal to
            :math:`\\sqrt{2}` times the standard deviation of the
            corresponding Gaussian density.

        Returns
        -------
        scale, shift : float
            Affine parameters as derived above.

        Raises
        ------
        ValueError
            If `std` is not positive.
        """
        if mean is None and std is None:
            return 1.0, 0.0
        if mean is None:
            mean = 0.0
        if std is None:
            std = 1.0
        if isinstance(std, float) and std <= 0:
            raise ValueError(f"Gauss-Hermite std must be positive, got {std}")
        return std, mean


class _HermiteNormFamily(_QuadratureFamily):
    """Gauss-Hermite (probabilists') weight: :math:`w(x) = e^{-x^2/2}` on
    :math:`(-\\infty, \\infty)`.

    The associated orthogonal polynomials are the *probabilists'* Hermite
    polynomials :math:`\\mathit{He}_n(x)` (as opposed to the *physicists'*
    convention used by :class:`_HermiteFamily`, with weight
    :math:`e^{-x^2}`). Up to normalization, this weight is exactly the
    density of a standard normal distribution:
    :math:`e^{-x^2/2} = \\sqrt{2\\pi} \\, \\phi(x)`, where :math:`\\phi` is
    the standard normal PDF. The zeroth moment of the weight is
    :math:`\\int_{-\\infty}^\\infty e^{-x^2/2} \\, dx = \\sqrt{2\\pi}`.
    """

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (-np.inf, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Reference weight function :math:`w(x) = e^{-x^2/2}`, evaluated
        at `x`."""
        return np.exp(-(x**2) / 2)

    def affine_params(self, mean=None, std=None) -> tuple[float, float]:
        """Map the reference weight onto a Gaussian weight with the given
        mean and standard deviation:
        :math:`w(x) = \\exp(-(x - \\mathrm{mean})^2 /
        (2 \\, \\mathrm{std}^2))`.

        Substituting :math:`x = \\mathrm{mean} + \\mathrm{std} \\cdot t`
        gives

        .. math::
            \\int_{-\\infty}^\\infty f(x) \\,
                e^{-(x-\\mathrm{mean})^2/(2 \\, \\mathrm{std}^2)} \\, dx
            = \\mathrm{std} \\int_{-\\infty}^\\infty
                f(\\mathrm{mean} + \\mathrm{std} \\cdot t) \\, e^{-t^2/2}
                \\, dt,

        so :math:`\\mathrm{scale} = \\mathrm{std}` and
        :math:`\\mathrm{shift} = \\mathrm{mean}`. Unlike
        `_HermiteFamily.affine_params`, `std` here is exactly the standard
        deviation of the corresponding Gaussian density -- the reference
        weight already uses the probabilists' normalization, so no
        :math:`\\sqrt{2}` correction is needed.

        Parameters
        ----------
        mean : float, optional
            Mean of the target Gaussian weight. Default 0.
        std : float, optional
            Standard deviation of the target Gaussian weight. Default 1.

        Returns
        -------
        scale, shift : float
            Affine parameters as derived above.

        Raises
        ------
        ValueError
            If `std` is not positive.
        """
        if mean is None and std is None:
            return 1.0, 0.0
        if mean is None:
            mean = 0.0
        if std is None:
            std = 1.0
        if isinstance(std, float) and std <= 0:
            raise ValueError(f"Gauss-Hermite std must be positive, got {std}")
        return std, mean


# Note: dataclass, not struct, because all the data is static
@dataclasses.dataclass(frozen=True)
class QuadratureRule:
    """Fixed-node Gauss quadrature rule on a reference domain.

    Approximates the weighted integral

    .. math::
        \\int_I f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n w_i f(x_i)

    where :math:`w` and :math:`I` are the weight function and reference
    domain of `family`, and `nodes`/`weights` are the :math:`x_i`/:math:`w_i`
    above.

    Nodes and weights are always static (NumPy) arrays. Mapping onto a
    target domain/measure is an affine transform of the reference nodes,
    whose parameters are specific to `family` -- see `scaled_points`.

    Parameters
    ----------
    nodes : array_like
        Quadrature nodes :math:`x_i`, shape `(n,)`, on
        `family.reference_domain`.
    weights : array_like
        Quadrature weights :math:`w_i`, shape `(n,)`.
    name : str
        Name identifying the rule.
    family : _QuadratureFamily
        Weight function and reference domain the rule is defined on.

    Raises
    ------
    ValueError
        If `nodes` and `weights` do not have the same shape.
    """

    nodes: np.ndarray  # shape (n,), on `family.reference_domain`
    weights: np.ndarray  # shape (n,)
    name: str  # name for the rule
    family: _QuadratureFamily

    def __post_init__(self):
        # Static data, safe to unconditionally convert to NumPy arrays
        object.__setattr__(self, "nodes", np.asarray(self.nodes, dtype=float))
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float))
        if self.nodes.shape != self.weights.shape:
            raise ValueError(
                f"nodes {self.nodes.shape} and weights {self.weights.shape} "
                "must have the same shape"
            )

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"family={type(self.family).__name__}, n={len(self)})"
        )

    # -- domain mapping --

    def scaled_points(self, *params, **kwparams):
        """Nodes mapped by `family`'s affine parameters.

        Given ``(scale, shift) = family.affine_params(*params, **kwparams)``,
        the mapped nodes are

        .. math::
            x_i = \\mathrm{scale} \\cdot t_i + \\mathrm{shift}

        for reference node :math:`t_i`. Called with no arguments, returns
        the reference `nodes` unchanged.

        The meaning of `params`/`kwparams` is specific to `family`:

        - Legendre/Jacobi: `(a, b)` bounds of the target interval.
        - Laguerre: `rate` (and optional `start`) of the target
          exponential weight.
        - Hermite: `mean`, `std` of the target Gaussian-shaped weight.

        See the family's `affine_params` docstring for details. Symbolic
        if any parameter is symbolic; the underlying nodes are static.
        """
        scale, shift = self.family.affine_params(*params, **kwparams)
        return scale * self.nodes + shift

    def scaled_weights(self, *params, **kwparams):
        """Weights including the Jacobian factor for the target
        domain/measure.

        .. math::
            \\tilde{w}_i = \\mathrm{scale} \\cdot w_i

        where :math:`\\mathrm{scale}` is the same affine scale used by
        `scaled_points`. See `scaled_points` for the meaning of
        `params`/`kwparams`.
        """
        scale, _ = self.family.affine_params(*params, **kwparams)
        return scale * self.weights

    # -- integration --

    def integrate(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        *params,
        axis=-1,
        **kwparams,
    ) -> np.ndarray:
        """Approximate the weighted integral of `f`.

        .. math::
            \\int f(x) \\, w(x) \\, dx \\approx \\sum_{i=1}^n \\tilde{w}_i
                f(x_i)

        where :math:`x_i` = `scaled_points(*params, **kwparams)` and
        :math:`\\tilde{w}_i` = `scaled_weights(*params, **kwparams)`.

        Parameters
        ----------
        f : callable
            Integrand, called once on the full node array. Must be
            vectorized, returning values with the nodes along `axis`. If
            any of `params`/`kwparams` is symbolic, `f` must be
            symbolically traceable.
        *params, **kwparams
            Target domain/measure parameters; see `scaled_points` for
            their meaning.
        axis : int, optional
            Axis holding the nodes in the output of `f`. Default -1.

        Returns
        -------
        integral : ndarray
            Approximated integral. Shape (m,) for vector-valued
            integrands, or () for scalar integrands.
        """
        fp = f(self.scaled_points(*params, **kwparams))
        return self.dot(fp, *params, axis=axis, **kwparams)

    def dot(
        self,
        values: np.ndarray,
        *params,
        axis: int = -1,
        **kwparams,
    ) -> np.ndarray:
        """Quadrature applied to values already sampled at the nodes.

        .. math::
            \\sum_{i=1}^n \\tilde{w}_i \\, \\mathrm{values}_i

        where :math:`\\tilde{w}_i` = `scaled_weights(*params, **kwparams)`.

        Parameters
        ----------
        values : array_like
            Sampled values, with the quadrature nodes along `axis`.
            Shape (n,) for scalar integrands or (m, n) for vector-valued
            integrands under the default `axis=-1`.
        *params, **kwparams
            Target domain/measure parameters, forwarded to
            `family.affine_params`; see `scaled_points` for their meaning.
        axis : int, optional
            Axis holding the nodes. Default -1 (nodes last), matching the
            natural output of a vectorized `f`. Use `axis=0` for
            nodes-first data.

        Returns
        -------
        integral : ndarray
            Approximated integral of the sampled values, with the
            quadrature nodes integrated out along `axis`. Shape (m,) for
            vector-valued integrands, or () for scalar integrands.

        Raises
        ------
        ValueError
            If `values` has more than 2 dimensions, or if
            `values.shape[axis]` does not match the number of quadrature
            nodes.
        """
        w = self.scaled_weights(*params, **kwparams)

        if values.ndim > 2:
            raise ValueError(f"expected a 0-D, 1-D, or 2-D array, got {values.ndim}-D")
        if values.shape[axis] != len(self):
            raise ValueError(
                f"values.shape[{axis}] is {values.shape[axis]}, expected "
                f"{len(self)} to match the quadrature nodes"
            )

        if values.ndim == 1 or axis == 0:
            return np.dot(w, values)
        return np.dot(values, w)


def gauss_legendre(n: int) -> QuadratureRule:
    """Gauss-Legendre quadrature rule with `n` nodes.

    Nodes are the roots of the degree-`n` Legendre polynomial
    :math:`P_n(x)`, none of which coincide with the endpoints
    :math:`\\pm 1`. The rule is exact for polynomials up to degree
    :math:`2n - 1`.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.

    Returns
    -------
    rule : QuadratureRule
        Gauss-Legendre rule with `n` nodes on :math:`[-1, 1]`, exact to
        degree :math:`2n - 1`.
    """
    x, w = roots_legendre(n)
    family = _LegendreFamily()
    return QuadratureRule(x, w, family=family, name="gauss_legendre")


def gauss_radau(n: int, endpoint: str = "left") -> QuadratureRule:
    """Gauss-Radau quadrature rule including exactly one endpoint.

    Radau rules fix one endpoint of :math:`[-1, 1]` as a node and choose
    the remaining :math:`n - 1` nodes to maximize the polynomial degree of
    exactness. Fixing the left endpoint, these are the roots of the Jacobi
    polynomial :math:`P_{n-1}^{(0,1)}(x)`; fixing the right endpoint, the
    roots of :math:`P_{n-1}^{(1,0)}(x)`. The rule is exact for polynomials
    up to degree :math:`2n - 2`.

    Parameters
    ----------
    n : int
        Number of quadrature nodes, including the fixed endpoint.
    endpoint : {"left", "right"}, optional
        Which endpoint of :math:`[-1, 1]` to fix as a node:

        - `"left"` includes :math:`-1` (LGR, the pseudospectral
          convention).
        - `"right"` includes :math:`+1` (Radau IIA, the IRK/DAE
          convention).

    Returns
    -------
    rule : QuadratureRule
        Gauss-Radau rule with `n` nodes on :math:`[-1, 1]`, exact to
        degree :math:`2n - 2`.

    Raises
    ------
    ValueError
        If `n < 1`, or if `endpoint` is not `"left"` or `"right"`.
    """
    family = _LegendreFamily()
    if n < 1:
        raise ValueError("Gauss-Radau requires n >= 1")
    if n == 1:
        x, w = np.array([-1.0]), np.array([2.0])
    else:
        x, w = roots_jacobi(n - 1, 0.0, 1.0)
        w = w / (1 + x)
        x = np.insert(x, 0, -1.0)
        w = np.insert(w, 0, 2.0 / n**2)
    if endpoint == "right":
        x, w = -x[::-1], w[::-1]
    elif endpoint != "left":
        raise ValueError(f"endpoint must be 'left' or 'right', got {endpoint!r}")
    return QuadratureRule(x, w, family=family, name=f"gauss_radau_{endpoint}")


def gauss_lobatto(n: int) -> QuadratureRule:
    """Gauss-Lobatto quadrature rule including both endpoints.

    Lobatto rules fix both endpoints :math:`\\pm 1` as nodes and choose the
    remaining :math:`n - 2` interior nodes to maximize the polynomial
    degree of exactness -- the roots of the Jacobi polynomial
    :math:`P_{n-2}^{(1,1)}(x)`, equivalently the roots of :math:`P_{n-1}'
    (x)`, the derivative of the degree-:math:`(n-1)` Legendre polynomial.
    The rule is exact for polynomials up to degree :math:`2n - 3`.

    Parameters
    ----------
    n : int
        Number of quadrature nodes, including both endpoints.

    Returns
    -------
    rule : QuadratureRule
        Gauss-Lobatto rule with `n` nodes on :math:`[-1, 1]`, exact to
        degree :math:`2n - 3`.

    Raises
    ------
    ValueError
        If `n < 2`.
    """
    family = _LegendreFamily()
    if n < 2:
        raise ValueError("Gauss-Lobatto requires n >= 2")
    if n == 2:
        x, w = np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    else:
        x, w = roots_jacobi(n - 2, 1.0, 1.0)
        w = w / (1 - x**2)
        x = np.concatenate([[-1.0], x, [1.0]])
        end_w = 2.0 / (n * (n - 1))
        w = np.concatenate([[end_w], w, [end_w]])
    return QuadratureRule(x, w, family=family, name="gauss_lobatto")


def composite(base: QuadratureRule, breakpoints: np.ndarray) -> QuadratureRule:
    """Tile `base` across elements of its reference domain.

    Partitions `base.family.reference_domain` at `breakpoints` and applies
    `base`, affinely rescaled, to each element, concatenating the resulting
    nodes and weights. The result is itself a `QuadratureRule` on the same
    reference domain -- its nodes are just clustered at the element
    boundaries rather than spread uniformly -- so it can be mapped onto a
    target domain/measure via `scaled_points`/`scaled_weights`/`integrate`
    exactly like any other rule of `base.family`. This works because
    `family.affine_params` maps affinely, and affine maps commute with
    subdivision: rescaling the whole composite pattern onto `[a, b]` is
    the same as building the elements directly on the rescaled sub-intervals
    of `[a, b]`.

    Only defined for families whose reference weight is uniform (see
    `_QuadratureFamily.uniform_weight`) -- otherwise each interior element
    boundary would pick up a spurious copy of the weight's shape, which is
    only meaningful at the true endpoints of the reference domain.

    Parameters
    ----------
    base : QuadratureRule
        Rule to tile across elements. `base.family.uniform_weight` must be
        `True`.
    breakpoints : array_like
        Element boundaries, shape `(k + 1,)` for `k` elements. Must be
        strictly increasing and span `base.family.reference_domain`
        exactly (first/last entries equal to its endpoints).

    Returns
    -------
    rule : QuadratureRule
        Composite rule with `k * len(base)` nodes on the same reference
        domain as `base`.

    Raises
    ------
    ValueError
        If `base.family.uniform_weight` is `False`, if `breakpoints` has
        fewer than 2 entries or is not strictly increasing, or if it does
        not span `base.family.reference_domain` exactly.
    """
    if not base.family.uniform_weight:
        raise ValueError(
            f"composite quadrature requires a family with a uniform "
            f"reference weight, got {type(base.family).__name__}"
        )
    breakpoints = np.asarray(breakpoints, dtype=float)
    if breakpoints.ndim != 1 or len(breakpoints) < 2:
        raise ValueError(
            f"breakpoints must be 1-D with at least 2 entries, got shape "
            f"{breakpoints.shape}"
        )
    if np.any(np.diff(breakpoints) <= 0):
        raise ValueError("breakpoints must be strictly increasing")
    lo, hi = base.family.reference_domain
    if breakpoints[0] != lo or breakpoints[-1] != hi:
        raise ValueError(
            f"breakpoints must span the reference domain {(lo, hi)}, got "
            f"({breakpoints[0]}, {breakpoints[-1]})"
        )

    nodes = []
    weights = []
    for t0, t1 in zip(breakpoints[:-1], breakpoints[1:]):
        nodes.append(base.scaled_points(t0, t1))
        weights.append(base.scaled_weights(t0, t1))

    return QuadratureRule(
        np.concatenate(nodes),
        np.concatenate(weights),
        family=base.family,
        name=base.name,
    )
