import abc
from typing import Tuple

import numpy as np

from archimedes import struct


__all__ = [
    "AtmosphereModel",
    "ConstantAtmosphere",
]


@struct.pytree_node
class AtmosphereModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, Vt: float, alt: float) -> Tuple[float, float]:
        """Compute Mach number and dynamic pressure at given altitude and velocity."""


@struct.pytree_node
class ConstantAtmosphere(AtmosphereModel):
    """Constant atmosphere model"""

    # Defaults based on US Standard Atmosphere, 1976: 20km altitude
    p: float = 5474.89  # Pressure [Pa]
    T: float = 216.65  # Temperature [K]
    Rs: float = 287.05  # Specific gas constant for air [J/(kg·K)]
    gamma: float = 1.4  # Adiabatic index for air [-]

    def __call__(self, Vt: float, alt: float) -> Tuple[float, float]:
        rho = self.p / (self.Rs * self.T)
        amach = Vt / np.sqrt(self.gamma * self.Rs * self.T)
        qbar = 0.5 * rho * Vt**2
        return amach, qbar
