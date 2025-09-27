import abc
from typing import Tuple

import numpy as np

from archimedes import struct, StructConfig, UnionConfig
from archimedes._core.utils import find_equal


__all__ = [
    "AtmosphereModel",
    "ConstantAtmosphere",
    "ConstantAtmosphereConfig",
    "StandardAtmosphere1976",
    "StandardAtmosphere1976Config",
    "AtmosphereConfig",
]


@struct
class AtmosphereModel(metaclass=abc.ABCMeta):
    Rs: float = 287.05287  # Specific gas constant for air [J/(kg·K)]
    gamma: float = 1.4  # Adiabatic index for air [-]

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
        rho = p / (self.Rs * T)
        amach = Vt / np.sqrt(self.gamma * self.Rs * T)  # Adiabatic Mach number
        qbar = 0.5 * rho * Vt**2
        return amach, qbar


class AtmosphereConfigBase(StructConfig):
    Rs: float = 287.05287  # Specific gas constant for air [J/(kg·K)]
    gamma: float = 1.4  # Adiabatic index for air [-]


@struct
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


class ConstantAtmosphereConfig(AtmosphereConfigBase, type="constant"):
    p: float = 5474.89  # Pressure [Pa]
    T: float = 216.65  # Temperature [K]

    def build(self) -> ConstantAtmosphere:
        return ConstantAtmosphere(
            Rs=self.Rs,
            gamma=self.gamma,
            p=self.p,
            T=self.T,
        )


# Altitude [m]
h_USSA1976 = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
# Temperature [K]
T_USSA1976 = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95])
# Pressure [Pa]
p_USSA1976 = np.array([101325, 22632.06, 5474.89, 868.02, 110.91, 66.94, 3.96, 0.3734])
# Temperature lapse rate [K/m]
L_USSA1976 = np.array([-0.0065, 0, 0.001, 0.0028, 0, 0.0028, 0, 0])


@struct
class StandardAtmosphere1976(AtmosphereModel):
    """U.S. Standard Atmosphere, 1976"""

    g0: float = 9.80665  # Gravity constant m/s^2

    def calc_p(self, alt: float) -> float:
        alt = np.fmax(0, alt)
        h1 = find_equal(alt, h_USSA1976, h_USSA1976)
        T1 = find_equal(alt, h_USSA1976, T_USSA1976)
        p1 = find_equal(alt, h_USSA1976, p_USSA1976)
        L = find_equal(alt, h_USSA1976, L_USSA1976)

        return np.where(
            L == 0,
            p1 * np.exp(-self.g0 * (alt - h1) / (self.Rs * T1)),
            p1 * (T1 / (T1 + L * (alt - h1))) ** (self.g0 / (self.Rs * L)),
        )

    def calc_T(self, alt: float) -> float:
        return np.interp(alt, h_USSA1976, T_USSA1976)


class StandardAtmosphere1976Config(AtmosphereConfigBase, type="ussa1976"):
    g0: float = 9.80665  # Gravity constant m/s^2

    def build(self) -> StandardAtmosphere1976:
        return StandardAtmosphere1976(
            g0=self.g0,
            Rs=self.Rs,
            gamma=self.gamma,
        )


AtmosphereConfig = UnionConfig[
    ConstantAtmosphereConfig,
    StandardAtmosphere1976Config,
]
