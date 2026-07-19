from __future__ import annotations

import abc
import dataclasses
from typing import Callable

import numpy as np
from scipy.special import roots_jacobi, roots_legendre


class _QuadratureFamily(metaclass=abc.ABCMeta):
    """The measure of an orthogonal polynomial quadrature rule.

    Combines a weight function and reference domain (support) -- together,
    the data `w(x) dx` on `reference_domain` that defines a classical
    family of orthogonal polynomials and its Gauss quadrature rules.
    """

    @property
    @abc.abstractmethod
    def reference_domain(self) -> tuple[float, float]:
        """Reference domain for the quadrature rule."""
        raise NotImplementedError

    @abc.abstractmethod
    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function for the quadrature rule, evaluated at `x`."""
        raise NotImplementedError

    @abc.abstractmethod
    def affine_params(self, *args, **kwargs) -> tuple[float, float]:
        """Return `(scale, shift)` mapping reference nodes onto the
        requested instance of this family: `x = scale * t + shift` for
        reference node `t`. Weights pick up the same `scale` as a
        Jacobian factor.

        Called with no arguments, must return the identity `(1.0, 0.0)`
        -- i.e. the reference domain/measure itself. Also validates that
        `args`/`kwargs` are compatible with this family. Their meaning is
        family-specific; see the subclass docstring.
        """
        raise NotImplementedError


class _LegendreFamily(_QuadratureFamily):
    """Gauss-Legendre weights: 1 on [-1, 1]."""

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function for the quadrature rule, evaluated at `x`."""
        return np.ones_like(x)

    def affine_params(self, a=None, b=None) -> tuple[float, float]:
        """`(a, b)`: bounds of the target interval, default the reference
        domain."""
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
    """Gauss-Jacobi weights (1-x)^alpha * (1+x)^beta on [-1, 1]."""

    alpha: float
    beta: float

    def __post_init__(self):
        if self.alpha <= -1 or self.beta <= -1:
            raise ValueError(
                f"invalid alpha={self.alpha} or beta={self.beta}, must be > -1"
            )

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function for the quadrature rule, evaluated at `x`."""
        return (1 - x) ** self.alpha * (1 + x) ** self.beta


class _LaguerreFamily(_QuadratureFamily):
    """Gauss-Laguerre weights: exp(-x) on [0, inf)."""

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (0.0, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function for the quadrature rule, evaluated at `x`."""
        return np.exp(-x)

    def affine_params(self, rate=None, start=None) -> tuple[float, float]:
        """`rate`: rate of the target exponential weight
        `exp(-rate*(x-start))`, default 1. `start`: left endpoint of the
        target domain `[start, inf)`, default 0.
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
    """Gauss-Hermite weights: exp(-x^2) on (-inf, inf)."""

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (-np.inf, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function for the quadrature rule, evaluated at `x`."""
        return np.exp(-(x**2))

    def affine_params(self, mean=None, std=None) -> tuple[float, float]:
        """`mean`, `std`: location and scale of the target Gaussian
        weight, default 0 and 1."""
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
    """Fixed-node quadrature rule on a reference domain.

    Nodes and weights are always static (NumPy) arrays. Mapping onto a
    target domain/measure is an affine transform of the reference nodes,
    whose parameters are specific to `family` -- see `scaled_points`.
    """

    nodes: np.ndarray  # shape (n,), on `family.reference_domain`
    weights: np.ndarray  # shape (n,)
    name: str  # name for the rule
    family: _QuadratureFamily
    degree: int | None = None  # exact for polynomials up to this degree

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

    # -- domain mapping --

    def scaled_points(self, *params, **kwparams):
        """Nodes mapped by `family`'s affine parameters, or the reference
        nodes if no parameters are given.

        The meaning of `params`/`kwparams` is specific to `family`:
        - Legendre/Jacobi: `(a, b)` bounds of the target interval.
        - Laguerre: `rate` (and optional `start`) of the target
          exponential weight.
        - Hermite: `mean`, `std` of the target Gaussian weight.

        See the family's `affine_params` docstring for details. Symbolic
        if any parameter is symbolic; the underlying nodes are static.
        """
        scale, shift = self.family.affine_params(*params, **kwparams)
        return scale * self.nodes + shift

    def scaled_weights(self, *params, **kwparams):
        """Weights including the Jacobian factor for the target
        domain/measure. See `scaled_points` for the meaning of
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

        `f` is called once on the full node array and must be vectorized,
        returning values with the nodes along `axis`. If any of
        `params`/`kwparams` is symbolic, `f` must be symbolically
        traceable. See `scaled_points` for their meaning.
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
            Approximated integral of the sampled values, with the quadrature
            nodes integrated out along `axis`. Shape (m,) for vector-valued
            integrands, or () for scalar integrands.
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
    x, w = roots_legendre(n)
    family = _LegendreFamily()
    degree = 2 * n - 1
    name = f"gauss_legendre_{n}"
    return QuadratureRule(x, w, degree=degree, family=family, name=name)


def gauss_radau(n: int, endpoint: str = "left") -> QuadratureRule:
    """Radau rule including exactly one endpoint.

    `endpoint="left"`  includes -1  (LGR, pseudospectral convention)
    `endpoint="right"` includes +1  (Radau IIA, IRK/DAE convention)
    """
    family = _LegendreFamily()
    name = f"gauss_radau_{n}_{endpoint}"
    degree = 2 * n - 2
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
    return QuadratureRule(x, w, degree=degree, family=family, name=name)


def gauss_lobatto(n: int) -> QuadratureRule:
    family = _LegendreFamily()
    name = f"gauss_lobatto_{n}"
    degree = 2 * n - 3
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
    return QuadratureRule(x, w, degree=degree, family=family, name=name)
