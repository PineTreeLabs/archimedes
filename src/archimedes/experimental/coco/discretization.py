import abc
import dataclasses

import numpy as np

import archimedes as arc
from .interpolation import LagrangePolynomial, gauss_radau


# Base class for finite element discretization?
@dataclasses.dataclass
class SplineDiscretization:
    n_nodes: int = None
    nodes: np.ndarray = None
    weights: np.ndarray = None
    diff_matrix: np.ndarray = None

    def time_nodes(self, t0, tf):
        τ = self.nodes
        return 0.5 * (tf + t0) + 0.5 * (tf - t0) * τ

    @abc.abstractmethod
    def create_interpolants(self, x, u, t0, tf):
        """Create interpolating functions for the state and control"""


@dataclasses.dataclass
class RadauInterval(SplineDiscretization):
    poly: LagrangePolynomial = None
    diff_inv: np.ndarray = None

    def __post_init__(self):
        x, w = gauss_radau(self.n_nodes)
        x = np.append(x, 1.0)  # Add non-collocated end point
        p = LagrangePolynomial(x)
        D = p.diff_matrix[:-1, :]  # Remove non-collocated last row of D

        self.nodes = x
        self.weights = w
        self.poly = p
        self.diff_matrix = D
        self.diff_inv = np.linalg.inv(D[:, 1:])  # Integration matrix

    def create_interpolant(self, x, t0, tf):
        p = self.poly

        # Scale time to the "local" interval [-1, 1]
        def _scale(t):
            return (t - t0) / (tf - t0) * 2 - 1

        # Construct functions for interpolating the state and control
        def x_fn(t):
            return p.interpolate(x, _scale(t))

        return x_fn

    def create_interpolants(self, x, u, t0, tf):
        τ = self.nodes

        # Extrapolate the control to the end time
        p_radau = LagrangePolynomial(τ[:-1])  # Interpolant for collocated nodes
        uf = p_radau.interpolate(u, τ[-1])
        # u = np.append(u, np.atleast_2d(uf), axis=0)
        u = np.vstack([u, uf])

        x_fn = self.create_interpolant(x, t0, tf)
        u_fn = self.create_interpolant(u, t0, tf)

        return x_fn, u_fn


@dataclasses.dataclass
class RadauFiniteElements(SplineDiscretization):
    N: dataclasses.InitVar[list[int]] = None
    knots: np.ndarray = None
    elements: list[RadauInterval] = None
    n_elements: int = None

    def __post_init__(self, N):
        if isinstance(N, int):
            N = [N]
        self.elements = [RadauInterval(n_nodes=N_k) for N_k in N]
        if self.knots is None:
            # Default to equally spaced knots
            self.knots = np.linspace(-1, 1, len(N) + 1)[1:-1]
        else:
            if len(N) != len(self.knots) + 1:
                raise ValueError(
                    "Number of knots must be one fewer than the number of elements"
                )
            self.knots = np.asarray(self.knots)

        # Check that knots are in ascending order
        if not np.all(np.diff(self.knots) > 0):
            raise ValueError("Knots must be in strictly ascending order")

        if not np.all((self.knots > -1) * (self.knots < 1)):
            raise ValueError("Knots must be in the interval (-1, 1)")

        # N+1 knots, including beginning and end point
        self.knots = np.insert(self.knots, (0, len(self.knots)), (-1, 1))
        self.n_elements = len(N)

        # Initialize the interface variables for the SplineDiscretization
        self.n_nodes = sum(N)  # Total number of _collocated_ nodes
        self.nodes = self._get_nodes()
        self.weights = self._get_weights()
        self.diff_matrix = self._get_diff_matrix()

    def local_to_global(self, n):
        k = self.knots
        if n >= self.n_elements:
            raise ValueError("Element index out of range")
        x = self.elements[n].nodes
        return 0.5 * (k[n + 1] - k[n]) * x + 0.5 * (k[n + 1] + k[n])

    def _get_nodes(self):
        # Exclude the non-collocated end point of each interval here except the last
        x = np.array([])
        for k in range(self.n_elements - 1):
            x = np.append(x, self.local_to_global(k)[:-1])
        return np.append(
            x, self.local_to_global(self.n_elements - 1)
        )  # Keep the final node

    def _get_weights(self):
        # Quadrature weights
        knots = self.knots
        w = np.array([])
        for k in range(self.n_elements):
            w_k = self.elements[k].weights
            tscale = 0.5 * (knots[k + 1] - knots[k])
            w = np.append(w, w_k * tscale)
        return w

    def _get_diff_matrix(self):
        # Block-diagonal differentiation matrix
        N = self.n_nodes
        knots = self.knots
        D = np.zeros((N, N + 1))
        idx = 0

        for k in range(self.n_elements):
            N_k = self.elements[k].n_nodes
            D_k = self.elements[k].diff_matrix
            # Scale time according to (local) interval
            tscale = 0.5 * (knots[k + 1] - knots[k])
            D[idx : idx + N_k, idx : idx + N_k + 1] = D_k / tscale
            idx += N_k

        return D

    def create_interpolant(self, x_fns, xp, t0, tf, extrapolation="flat"):
        if extrapolation not in {"flat", "zero"}:
            raise ValueError(f"Unsupported extrapolation type: {extrapolation}")

        τ = self.time_nodes(t0, tf)  # Nodes scaled to (t0, tf)
        kt = self.knots * 0.5 * (tf - t0) + 0.5 * (tf + t0)  # Global knots

        @arc.compile
        def global_interp(t):
            scalar_t = t.shape == ()

            t = np.atleast_1d(t)
            n = len(t)
            m = xp.shape[1]

            # Determine which interval each time point falls into and use
            # the element interpolant
            out = np.zeros((n, m))  # (nt, nx)
            for k in range(self.n_elements):
                # Find all indices of t such that kt[k] <= t < kt[k + 1]
                idx = np.logical_and(t >= kt[k], t <= kt[k + 1])

                # Evaluate this interpolant at all time points
                # This looks inefficient, but ends up being fast once compiled
                x = x_fns[k](t)  # (nt, nx)
                if x.ndim == 1:
                    x = np.reshape(x, (n, m))  # Convert (n,) to (n, 1)

                # print(idx.shape, x.shape, out.shape)
                out = np.where(idx[:, None], x, out)

            low_idx = t < τ[0]
            high_idx = t > τ[-1]
            # print(out.shape, low_idx.shape, high_idx.shape)

            if extrapolation == "flat":
                out = np.where(low_idx[:, None], xp[0], out)
                out = np.where(high_idx[:, None], xp[-1], out)

            elif extrapolation == "zero":
                out = np.where(low_idx[:, None], 0.0, out)
                out = np.where(high_idx[:, None], 0.0, out)

            if scalar_t:
                out = out[0]

            return out

        return global_interp

    def create_interpolants(self, x, u, t0, tf, extrapolation="flat"):
        # Extract the collocation points for each interval
        x_fns = []
        u_fns = []

        kt = self.knots * 0.5 * (tf - t0) + 0.5 * (tf + t0)  # Global knots

        # Construct interpolating functions for each interval
        idx = 0
        for k, el in enumerate(self.elements):
            N = el.n_nodes
            x_k = x[idx : idx + N + 1]
            u_k = u[idx : idx + N]
            x_fn, u_fn = el.create_interpolants(x_k, u_k, kt[k], kt[k + 1])
            x_fns.append(x_fn)
            u_fns.append(u_fn)
            idx += N

        global_x_fn = self.create_interpolant(
            x_fns, x, t0, tf, extrapolation=extrapolation
        )
        global_u_fn = self.create_interpolant(
            u_fns, u, t0, tf, extrapolation=extrapolation
        )

        return global_x_fn, global_u_fn
