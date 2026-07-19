from __future__ import annotations
import abc
import dataclasses
import numpy as np
from typing import Callable
from scipy.special import roots_jacobi, roots_legendre


class _WeightType(metaclass=abc.ABCMeta):
    """A type of quadrature rule, coupling a weight function and reference domain"""

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
    def validate_domain(self, domain: tuple[float, float]) -> None:
        """Validate that the given domain is compatible with this quadrature type."""
        raise NotImplementedError



class _LegendreWeight(_WeightType):
    """Gauss-Legendre weights: 1 on [-1, 1]."""

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function for the quadrature rule, evaluated at `x`."""
        return np.ones_like(x)

    def validate_domain(self, domain: tuple[float, float]) -> None:
        """Validate that the given domain is compatible with this quadrature type."""
        a, b = domain
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError(
                f"Gauss-Legendre quadrature requires finite domain, got {domain}"
            )


@dataclasses.dataclass(frozen=True)
class _JacobiWeight(_WeightType):
    """Gauss-Jacobi weights (1-x)^alpha * (1+x)^beta on [-1, 1]."""
    alpha: float
    beta: float

    def __post_init__(self):
        if self.alpha <= -1 or self.beta <= -1:
            raise ValueError(f"invalid alpha={self.alpha} or beta={self.beta}, must be > -1")

    @property
    def reference_domain(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function for the quadrature rule, evaluated at `x`."""
        return (1 - x) ** self.alpha * (1 + x) ** self.beta

    def validate_domain(self, domain: tuple[float, float]) -> None:
        """Validate that the given domain is compatible with this quadrature type."""
        a, b = domain
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError(
                f"Gauss-Jacobi quadrature requires finite domain, got {domain}"
            )

class _LaguerreWeight(_WeightType):
    """Gauss-Laguerre weights: exp(-x) on [0, inf)."""
    @property
    def reference_domain(self) -> tuple[float, float]:
        return (0.0, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function for the quadrature rule, evaluated at `x`."""
        return np.exp(-x)

    def validate_domain(self, domain: tuple[float, float]) -> None:
        """Validate that the given domain is compatible with this quadrature type."""
        a, b = domain
        if a != 0 or b != np.inf:
            raise ValueError(
                f"Gauss-Laguerre quadrature requires domain [0, inf), got {domain}"
            )


class _HermiteWeight(_WeightType):
    """Gauss-Hermite weights: exp(-x^2) on (-inf, inf)."""
    @property
    def reference_domain(self) -> tuple[float, float]:
        return (-np.inf, np.inf)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Weight function for the quadrature rule, evaluated at `x`."""
        return np.exp(-x**2)

    def validate_domain(self, domain: tuple[float, float]) -> None:
        """Validate that the given domain is compatible with this quadrature type."""
        a, b = domain
        if a != -np.inf or b != np.inf:
            raise ValueError(
                f"Gauss-Hermite quadrature requires domain (-inf, inf), got {domain}"
            )

def _resolve_quadrature_type(rule: str, **kwargs) -> _WeightType:
    """Resolve a quadrature rule type from a string and optional parameters."""
    QuadratureType = {
        "gauss_legendre": _LegendreWeight,
        "gauss_jacobi": _JacobiWeight,
        "gauss_laguerre": _LaguerreWeight,
        "gauss_hermite": _HermiteWeight,
    }
    if rule not in QuadratureType:
        raise ValueError(f"Unknown quadrature rule '{rule}', must be one of {list(QuadratureType.keys())}")
    return QuadratureType[rule](**kwargs)


# Note: dataclass, not struct, because all the data is static
@dataclasses.dataclass(frozen=True)
class QuadratureRule:
    """Fixed-node quadrature rule on a reference domain.

    Nodes and weights are always static (NumPy) arrays. The integration
    domain may be symbolic, in which case only the affine scaling is traced.
    """
    nodes: np.ndarray          # shape (n,), on `domain`
    weights: np.ndarray        # shape (n,)
    name: str  # name for the rule
    weight_type: _WeightType
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

    def _affine(self, a, b):
        """Scale/shift coefficients mapping `domain` -> (a, b)."""
        lo, hi = self.domain
        scale = (b - a) / (hi - lo)
        return scale, a - scale * lo

    def scaled_points(self, a=None, b=None):
        """Nodes mapped onto (a, b), or the reference nodes if omitted.

        Symbolic if `a` or `b` are symbolic; the underlying nodes are static.
        """
        if a is None and b is None:
            return self.nodes
        if a is None or b is None:
            raise ValueError("specify both `a` and `b`, or neither")
        scale, shift = self._affine(a, b)
        return scale * self.nodes + shift

    def scaled_weights(self, a=None, b=None):
        """Weights including the Jacobian factor for (a, b)."""
        if a is None and b is None:
            return self.weights
        if a is None or b is None:
            raise ValueError("specify both `a` and `b`, or neither")
        scale, _ = self._affine(a, b)
        return scale * self.weights

    def _resolve_domain(self, domain: tuple[float, float] | None) -> tuple[float, float]:
        """Resolve the integration domain, defaulting to the reference domain."""
        if domain is None:
            return self.rule_type.reference_domain
        self.rule_type.validate_domain(domain)
        return domain

    # -- integration --

    def integrate(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        domain: tuple[float, float] | None = None,
        axis=-1
    ) -> np.ndarray:
        """Approximate the weighted integral of `f` over `domain`.

        `f` is called once on the full node array and must be vectorized,
        returning values with the nodes along `axis`. If the domain is
        symbolic, `f` must be symbolically traceable.
        """
        a, b = self._resolve_domain(domain)
        fp = f(self.scaled_points(a, b))
        return self.dot(fp, domain, axis=axis)

    def dot(
        self,
        values: np.ndarray,
        domain: tuple[float, float] | None = None,
        axis: int =-1
    ) -> np.ndarray:
        """Quadrature applied to values already sampled at the nodes.

        Parameters
        ----------
        values : array_like
            Sampled values, with the quadrature nodes along `axis`.
            Shape (n,) for scalar integrands or (m, n) for vector-valued
            integrands under the default `axis=-1`.
        domain : tuple[float, float]
            Integration domain (a, b) to scale the quadrature weights.
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
        a, b = self._resolve_domain(domain)
        w = self.scaled_weights(a, b)

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
    weight_type = _LegendreWeight()
    degree = 2 * n - 1
    name = f"gauss_legendre_{n}"
    return QuadratureRule(x, w, degree=degree, weight_type=weight_type, name=name)


def gauss_radau(n: int, endpoint: str = "left") -> QuadratureRule:
    """Radau rule including exactly one endpoint.

    `endpoint="left"`  includes -1  (LGR, pseudospectral convention)
    `endpoint="right"` includes +1  (Radau IIA, IRK/DAE convention)
    """
    weight_type = _LegendreWeight()
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
    return QuadratureRule(x, w, degree=degree, weight_type=weight_type, name=name)


def gauss_lobatto(n: int) -> QuadratureRule:
    weight_type = _LegendreWeight()
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
    return QuadratureRule(x, w, degree=degree, weight_type=weight_type, name=name)