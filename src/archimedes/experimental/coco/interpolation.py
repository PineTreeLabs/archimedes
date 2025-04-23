import numpy as np
from scipy.special import roots_jacobi, roots_legendre

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


class LagrangePolynomial:
    def __init__(self, nodes):
        self.n = len(nodes) - 1
        self.nodes = nodes
        self.weights = barycentric_weights(nodes)

    def interpolate(self, y0, x):
        """Interpolate the polynomial at `x` when `f(nodes[i]) = y0[i]`

        If `x` is an array, it will be flattened before interpolation. The result
        will have the shape (n, m), where `n` is the number of elements in `x` and
        `m` is the shape of `y0[0]`.  If the data is scalar-valued, the result will
        be a 1D array of length `n`.
        """

        if len(y0) != self.n + 1:
            raise ValueError("Number of data points must match number of nodes")

        x0, w = self.nodes, self.weights

        # Reshape x to be a 1D array
        x = np.asarray(x).ravel()

        n = len(x)
        m = 1 if len(y0.shape) == 1 == 1 else len(y0[0])

        # Initialize arrays to store the numerator and denominator terms
        num = np.zeros((n, m), dtype=float)
        den = np.zeros(n, dtype=float)

        # Compute the numerator and denominator terms for each x value
        for j in range(self.n + 1):
            xdiff = x - x0[j]
            mask = xdiff != 0  # Mask to avoid division by zero
            temp = np.zeros_like(x)
            temp[mask] = w[j] / xdiff[mask]
            num += np.outer(temp, y0[j])
            den += temp

        # Handle the case where x matches one of the nodes exactly
        mask = np.isin(x, x0)  # These are indexes into `x`
        if np.any(mask):
            for i in np.nonzero(mask)[0]:
                i0 = np.where(x0 == x[i])[0][0]  # Index into `x0`
                num[i] = y0[i0]
                den[i] = 1.0

        result = num / den[:, None]

        return result.squeeze()

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
