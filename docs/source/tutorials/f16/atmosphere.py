from __future__ import annotations

from typing import Protocol

import numpy as np

from archimedes import StructConfig, UnionConfig, struct

__all__ = [
    "AtmosphereModel",
    "AtmosphereConfig",
    "LinearAtmosphere",
    "LinearAtmosphereConfig",
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


AtmosphereConfig = UnionConfig[LinearAtmosphereConfig,]
