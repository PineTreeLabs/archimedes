# ruff: noqa: N803, N806, N815, N816
from __future__ import annotations

from typing import Protocol

import numpy as np

from archimedes import StructConfig, UnionConfig, struct
from archimedes._core.utils import find_equal


__all__ = [
    "AtmosphereModel",
    "AtmosphereConfig",
    "LinearAtmosphere",
    "LinearAtmosphereConfig",
    "StandardAtmosphere1976",
    "StandardAtmosphere1976Config",
]


class AtmosphereModel(Protocol):
    def __call__(self, Vt: float, alt: float) -> tuple[float, float]:
        """Compute Mach number and dynamic pressure at given altitude and velocity.

        Args:
            Vt: true airspeed [ft/s]
            alt: altitude [ft]

        Returns:
            mach: Mach number [-]
            qbar: dynamic pressure [lbf/ft²]
        """


@struct
class LinearAtmosphere:
    """
    Linear temperature gradient atmosphere model using barometric formula.

    Density varies as ρ = ρ₀(T/T₀)^n where n = g/(R·L) - 1
    for a linear temperature profile T = T₀(1 - βz).
    """

    g0: float = 32.17  # Gravitational acceleration [ft/s²]
    R0: float = 2.377e-3  # Density scale [slug/ft^3]
    gamma: float = 1.4  # Adiabatic index for air [-]
    Rs: float = 1716.3  # Specific gas constant for air [ft·lbf/slug-R]
    dTdz: float = 0.703e-5  # Temperature gradient scale [1/ft]
    Tmin: float = 390.0  # Minimum temperature [R]
    Tmax: float = 519.0  # Maximum temperature [R]
    max_alt: float = 35000.0  # Maximum altitude [ft]

    def __call__(self, Vt, alt):
        L = self.Tmax * self.dTdz  # Temperature gradient [°R/ft]
        n = self.g0 / (self.Rs * L) - 1  # Density exponent [-]
        Tfac = 1 - self.dTdz * alt  # Temperature factor [-]
        T = np.where(alt >= self.max_alt, self.Tmin, self.Tmax * Tfac)
        rho = self.R0 * Tfac**n
        mach = Vt / np.sqrt(self.gamma * self.Rs * T)
        qbar = 0.5 * rho * Vt**2
        return mach, qbar


class LinearAtmosphereConfig(StructConfig, type="linear"):
    def build(self) -> LinearAtmosphere:
        return LinearAtmosphere()



# Altitude [ft]
M_TO_FT = 3.28084
K_TO_R = 9 / 5
PA_TO_PSF = 0.02088
h_USSA1976 = M_TO_FT * np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
# Temperature [R]
T_USSA1976 = K_TO_R * np.array([
    288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95
])
# Pressure [psf]
p_USSA1976 = PA_TO_PSF * np.array([
    101325, 22632.06, 5474.89, 868.02, 110.91, 66.94, 3.96, 0.3734
])
# Temperature lapse rate [R/ft]
L_USSA1976 = (K_TO_R / M_TO_FT) * np.array([
    -0.0065, 0, 0.001, 0.0028, 0, 0.0028, 0, 0
])


@struct
class StandardAtmosphere1976(AtmosphereModel):
    """U.S. Standard Atmosphere, 1976"""

    g0: float = 32.174  # Gravity constant ft/s^2
    Rs: float = 1716.3  # Specific gas constant for air [ft·lbf/slug-R]
    gamma: float = 1.4  # Adiabatic index for air [-]

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

    def __call__(self, Vt: float, alt: float) -> tuple[float, float]:
        """Compute Mach number and dynamic pressure at given altitude and velocity."""
        p = self.calc_p(alt)
        T = self.calc_T(alt)
        rho = p / (self.Rs * T)
        amach = Vt / np.sqrt(self.gamma * self.Rs * T)  # Adiabatic Mach number
        qbar = 0.5 * rho * Vt**2
        return amach, qbar


class StandardAtmosphere1976Config(StructConfig, type="ussa1976"):
    g0: float = 32.174  # Gravity constant ft/s^2
    Rs: float = 1716.3  # Specific gas constant for air [ft·lbf/slug-R]
    gamma: float = 1.4  # Adiabatic index for air [-]

    def build(self) -> StandardAtmosphere1976:
        return StandardAtmosphere1976(
            g0=self.g0,
            Rs=self.Rs,
            gamma=self.gamma,
        )


AtmosphereConfig = UnionConfig[
    LinearAtmosphereConfig,
    StandardAtmosphere1976Config,
]
