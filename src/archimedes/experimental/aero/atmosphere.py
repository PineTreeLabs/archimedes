import abc
from typing import Tuple

import numpy as np

from archimedes import struct
from archimedes._core.utils import find_equal


__all__ = [
    "AtmosphereModel",
    "ConstantAtmosphere",
]


Rs = 287.05287  # Specific gas constant for air [J/(kg·K)]
gamma = 1.4  # Adiabatic index for air [-]
g0 = 9.80665  # Gravity constant m/s^2


@struct.pytree_node
class AtmosphereModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def calc_p(self, alt: float) -> float:
        """Compute pressure at given altitude."""

    @abc.abstractmethod
    def calc_T(self, alt: float) -> float:
        """Compute pressure at given altitude."""

    def __call__(self, Vt: float, alt: float) -> Tuple[float, float]:
        """Compute Mach number and dynamic pressure at given altitude and velocity."""
        p = self.calc_p(alt)
        T = self.calc_T(alt)
        rho = p / (Rs * T)
        amach = Vt / np.sqrt(gamma * Rs * T)  # Adiabatic Mach number
        qbar = 0.5 * rho * Vt**2
        return amach, qbar


@struct.pytree_node
class ConstantAtmosphere(AtmosphereModel):
    """Constant atmosphere model"""

    # Defaults based on US Standard Atmosphere, 1976: 20km altitude
    p: float = 5474.89  # Pressure [Pa]
    T: float = 216.65  # Temperature [K]
    
    def calc_p(self, alt: float) -> float:
        """Return constant pressure"""
        return self.p

    def calc_T(self, alt: float) -> float:
        """Return constant temperature"""
        return self.T


# Altitude [m]
h_USSA1976 = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
# Temperature [K]
T_USSA1976 = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95])
# Pressure [Pa]
p_USSA1976 = np.array([101325, 22632.06, 5474.89, 868.02, 110.91, 66.94, 3.96, 0.3734])
# Temperature lapse rate [K/m]
L_USSA1976 = np.array([-0.0065, 0, 0.001, 0.0028, 0, 0.0028, 0, 0])


@struct.pytree_node
class StandardAtmosphere1976(AtmosphereModel):
    """U.S. Standard Atmosphere, 1976"""

    def calc_p(self, alt: float) -> float:
        alt = np.fmax(0, alt)
        h1 = find_equal(alt, h_USSA1976, h_USSA1976)
        T1 = find_equal(alt, h_USSA1976, T_USSA1976)
        p1 = find_equal(alt, h_USSA1976, p_USSA1976)
        L = find_equal(alt, h_USSA1976, L_USSA1976)

        return np.where(L == 0,
            p1 * np.exp(-g0 * (alt - h1) / (Rs * T1)),
            p1 * (T1 / (T1 + L * (alt - h1))) ** (g0 / (Rs * L)),
        )

    def calc_T(self, alt: float) -> float:
        return np.interp(alt, h_USSA1976, T_USSA1976)
