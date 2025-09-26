import numpy as np

from archimedes import struct
from archimedes.experimental.signal import tf2ss


class TransferFunctionBase:
    @property
    def ss(self):
        """Convert transfer function to state-space representation."""
        return tf2ss(self.num, self.den)

    def dynamics(self, t, x, u):
        A, B, _, _ = self.ss
        x = np.atleast_1d(x)
        u = np.atleast_1d(u)
        return A @ x + B @ u

    def observation(self, t, x, u):
        _, _, C, D = self.ss
        x = np.atleast_1d(x)
        u = np.atleast_1d(u)
        return C @ x + D @ u


@struct
class TransferFunction(TransferFunctionBase):
    num: np.ndarray
    den: np.ndarray


@struct
class FirstOrderLag(TransferFunctionBase):
    gain: float
    tau: float

    @property
    def num(self):
        """Numerator coefficients of the transfer function."""
        return np.atleast_1d(self.gain)

    @property
    def den(self):
        """Denominator coefficients of the transfer function."""
        return np.hstack([self.tau, 1.0])


class StaticNonlinearity:
    def __call__(self, x):
        """Apply nonlinearity to input."""
        raise NotImplementedError("StaticNonlinearity must implement __call__ method.")


@struct
class HammersteinWiener:
    input: StaticNonlinearity
    lin_sys: TransferFunction
    output: StaticNonlinearity

    def dynamics(self, t, x, u):
        """Hammerstein-Wiener system dynamics."""
        # Linear system dynamics with input nonlinearity
        return self.lin_sys.dynamics(t, x, self.input(u))

    def observation(self, t, x, u):
        """Hammerstein-Wiener system observation."""
        # Apply output nonlinearity to linear system output
        return self.output(self.lin_sys.observation(t, x, u))


@struct
class TanhNonlinearity(StaticNonlinearity):
    saturation: float

    def __call__(self, x):
        """Apply tanh nonlinearity with saturation."""
        return self.saturation * np.tanh(x / self.saturation)


@struct
class CubicNonlinearity(StaticNonlinearity):
    coeff: np.ndarray

    def __call__(self, x):
        """Apply cubic nonlinearity."""
        return x + self.coeff * x**3
