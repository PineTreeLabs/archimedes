import abc
import numpy as np
from archimedes import struct
from archimedes.experimental.signal import tf2ss


class TransferFunctionBase(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def num(self):
        """Numerator coefficients of the transfer function."""
        pass

    @property
    @abc.abstractmethod
    def den(self):
        """Denominator coefficients of the transfer function."""
        pass

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


@struct.pytree_node
class TransferFunction(TransferFunctionBase):
    num: np.ndarray
    den: np.ndarray

@struct.pytree_node
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