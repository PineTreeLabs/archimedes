import abc

import numpy as np

from archimedes import struct

__all__ = [
    "GravityModel",
    "ConstantGravityModel",
]


class GravityModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, p_E):
        """Gravitational acceleration at the body CM in the inertial frame E

        Args:
            p_E position vector relative to the earth center E

        Returns:
            g_E: gravitational acceleration in earth frame E
        """


@struct.pytree_node
class ConstantGravity(GravityModel):
    """Constant gravitational acceleration model

    This model assumes a constant gravitational acceleration vector
    in the +z direction (e.g. for a NED frame with "flat Earth" approximation)
    """

    g0: float = 9.81  # m/s^2

    def __call__(self, p_E):
        return np.hstack([0, 0, self.g0])
