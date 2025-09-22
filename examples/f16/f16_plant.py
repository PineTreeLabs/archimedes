from __future__ import annotations
import abc

import numpy as np

import archimedes as arc
from archimedes import struct

from archimedes.experimental import aero
from archimedes.experimental.aero import (
    GravityModel,
    FlightVehicle,
)

from f16_engine import F16Engine
from f16_aero import F16Aerodynamics


@struct.module
class ConstantGravity:
    """Constant gravitational acceleration model

    This model assumes a constant gravitational acceleration vector
    in the +z direction (e.g. for a NED frame with "flat Earth" approximation)
    """

    g0: float = 32.17  # ft/s^2

    def __call__(self, p_E):
        return np.hstack([0, 0, self.g0])


@struct.module
class AtmosphereModel:
    R0: float = 2.377e-3  # Density scale [slug/ft^3]
    gamma: float = 1.4  # Adiabatic index for air [-]
    Rs: float = 1716.3  # Specific gas constant for air [ft·lbf/slug-R]
    dTdz: float = 0.703e-5  # Temperature gradient scale [1/ft]
    Tmin: float = 390.0  # Minimum temperature [R]
    Tmax: float = 519.0  # Maximum temperature [R]
    max_alt: float = 35000.0  # Maximum altitude [ft]

    def __call__(self, Vt, alt):
        Tfac = 1 - self.dTdz * alt  # Temperature factor [-]

        T = np.where(alt >= self.max_alt, self.Tmin, self.Tmax * Tfac)

        rho = self.R0 * Tfac**4.14
        amach = Vt / np.sqrt(self.gamma * self.Rs * T)
        qbar = 0.5 * rho * Vt**2

        return amach, qbar


@struct.module
class SubsonicF16(FlightVehicle):
    gravity: GravityModel = struct.field(default_factory=ConstantGravity)
    atmos: AtmosphereModel = struct.field(default_factory=AtmosphereModel)
    engine: F16Engine = struct.field(default_factory=F16Engine)
    aero: F16Aerodynamics = struct.field(default_factory=F16Aerodynamics)

    xcg: float = 0.35  # CG location (% of cbar)

    S: float = 300.0  # Planform area
    b: float = 30.0  # Span
    cbar: float = 11.32  # Mean aerodynamic chord
    xcgr: float = 0.35  # Reference CG location (% of cbar)
    hx: float = 160.0  # Engine angular momentum (assumed constant)

    def net_forces(self, t, x, u, C_BN):
        """Net forces and moments in body frame B, plus any extra state derivatives

        Args:
            t: time
            x: state: (p_N, att, v_B, w_B, aux)
            u: (throttle, elevator, aileron, rudder) control inputs
            C_BN: rotation matrix from inertial (N) to body (B) frame

        Returns:
            F_B: net forces in body frame B
            M_B: net moments in body frame B
            aux_state_derivs: time derivatives of auxiliary state variables
        """

        # Unpack state and controls
        thtl, el, ail, rdr = u

        vt, alpha, beta = aero.wind_frame(x.v_B)

        # Atmosphere model
        alt = -x.p_N[2]
        amach, qbar = self.atmos(vt, alt)

        # Engine thrust model
        pow = x.aux
        Feng_B = self.engine.calc_thrust(pow, alt, amach)

        force_coeffs, moment_coeffs = self.aero(
            vt, alpha, beta, x.w_B, el, ail, rdr, self
        )
        cxt, cyt, czt = force_coeffs
        clt, cmt, cnt = moment_coeffs

        Fgrav_N = self.m * self.gravity(x.p_N)
        Faero_B = qbar * self.S * np.stack([cxt, cyt, czt])
        F_B = Faero_B + Feng_B + C_BN @ Fgrav_N

        # Moments
        p, q, r = x.w_B  # Angular velocity in body frame (ω_B)
        Meng_B = self.hx * np.array([0.0, -r, q])
        Maero_B = (
            qbar * self.S * np.array([self.b * clt, self.cbar * cmt, self.b * cnt])
        )
        M_B = Meng_B + Maero_B

        # Dynamic component of engine state (auxiliary state)
        pow_t = self.engine.dynamics(t, pow, thtl)

        aux_state_derivs = np.atleast_1d(pow_t)
        return F_B, M_B, aux_state_derivs
