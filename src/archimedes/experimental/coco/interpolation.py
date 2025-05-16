from functools import partial

import numpy as np
from scipy.special import roots_jacobi, roots_legendre
import casadi as cs

import archimedes as arc
from archimedes._core._array_impl import _as_casadi_array, SymbolicArray
from archimedes._core.utils import find_equal

__all__ = [
    "gauss_legendre",
    "gauss_radau",
    "gauss_lobatto",
    "LagrangePolynomial",
]


def gauss_legendre(n):
    return roots_legendre(n)


def gauss_radau(n):
    if n == 0:
        raise ValueError("Gauss-Radau not defined for n=0")

    if n == 1:
        x = np.array([-1.0])
        w = np.array([1.0])

    elif n == 2:
        x = np.array([-1.0, 1.0 / 3.0])
        w = np.array([0.5, 1.5])

    else:
        x, w = roots_jacobi(n - 1, 0.0, 1.0)
        w = w / (1 + x)
        x = np.insert(x, 0, -1.0)
        w = np.insert(w, 0, 2.0 / n**2)

    return x, w


def gauss_lobatto(n):
    if n <= 1:
        raise ValueError("Gauss-Lobatto not defined for n <= 1")

    elif n == 2:
        x = np.array([-1.0, 1.0])
        w = np.array([1.0, 1.0])

    elif n == 3:
        x = np.array([-1.0, 0.0, 1.0])
        w = np.array([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0])

    else:
        x, w = roots_jacobi(n - 2, 1.0, 1.0)
        w = w / (1 - x**2)
        x = np.insert(x, 0, -1.0)
        x = np.append(x, 1.0)
        w = np.insert(w, 0, 2.0 / (n * (n - 1)))
        w = np.append(w, 2.0 / (n * (n - 1)))

    return x, w


def barycentric_weights(x):
    """Compute the weights for barycentric interpolation"""
    n = len(x)
    w = np.zeros_like(x)
    for i in range(n):
        w[i] = 1.0
        for j in range(n):
            if j != i:
                w[i] *= x[i] - x[j]
        w[i] = 1.0 / w[i]

    return w


@arc.compile
def interpolate_helper(x, yp, xp, w):
    # Quick exit for empty arrays
    if yp.size == 0:
        return np.array([])

    n0 = len(xp)
    if len(yp) != n0:
        raise ValueError("Number of data points must match number of nodes")

    # Reshape x to be a 1D array
    x = np.atleast_1d(x).ravel()

    n = x.size
    m = 1 if len(yp.shape) == 1 == 1 else len(yp[0])

    # Initialize arrays to store the numerator and denominator terms
    num = np.zeros((n, m), dtype=float)
    den = np.zeros(n, dtype=float)

    # Compute the numerator and denominator terms for each x value
    for j in range(n0):
        xdiff = x - xp[j]
        temp = np.where(xdiff == 0, 0.0, w[j] / xdiff)
        num += np.outer(temp, yp[j])
        den += temp

    # Handle the case where x matches one of the nodes exactly
    for i in range(n):
        yp_i = find_equal(x[i], xp, yp)
        is_match = sum(x[i] == xp)
        num[i] = np.where(is_match, yp_i, num[i])
        den[i] = np.where(is_match, 1.0, den[i])

    result = num / den[:, None]
    return result.squeeze()


class LagrangePolynomial:
    def __init__(self, nodes):
        self.n = len(nodes) - 1
        self.nodes = nodes
        self.weights = barycentric_weights(nodes)

    def interpolate(self, yp, x):
        """Interpolate the polynomial at `x` when `f(nodes[i]) = yp[i]`

        If `x` is an array, it will be flattened before interpolation. The result
        will have the shape (n, m), where `n` is the number of elements in `x` and
        `m` is the shape of `yp[0]`.  If the data is scalar-valued, the result will
        be a 1D array of length `n`.
        """
        x = np.atleast_1d(x).ravel()
        if x.size > 1:
            _interpolate_helper = arc.vmap(
                interpolate_helper, in_axes=(0, None, None, None)
            )
        else:
            _interpolate_helper = interpolate_helper
        return _interpolate_helper(x, yp, self.nodes, self.weights)

    @property
    def diff_matrix(self):
        """Return the differentiation matrix for the polynomial"""
        D = np.zeros((self.n + 1, self.n + 1))
        x, w = self.nodes, self.weights
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                if i != j:
                    D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
            D[i, i] = -sum(D[i, :])
        return D
