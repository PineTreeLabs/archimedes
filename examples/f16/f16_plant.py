from __future__ import annotations
import abc

import numpy as np

import archimedes as arc

from archimedes.spatial import RigidBody
from archimedes.experimental import aero
from archimedes.experimental.aero import GravityModel

from f16_engine import F16Engine
from f16_aero import F16Aerodynamics


GRAV_FTS2 = 32.17  # ft/s^2

# NOTE: The weight in the textbook is 25,000 lbs, but this
# does not give consistent values - the default value here
# matches the values given in the tables
weight = 20490.4459

Axx = 9496.0
Ayy = 55814.0
Azz = 63100.0
Axz = -982.0
default_mass = weight / GRAV_FTS2

default_J_B = np.array(
    [
        [Axx, 0.0, Axz],
        [0.0, Ayy, 0.0],
        [Axz, 0.0, Azz],
    ]
)


@arc.struct
class ConstantGravity:
    """Constant gravitational acceleration model

    This model assumes a constant gravitational acceleration vector
    in the +z direction (e.g. for a NED frame with "flat Earth" approximation)
    """

    g0: float = GRAV_FTS2  # ft/s^2

    def __call__(self, p_E):
        return np.hstack([0, 0, self.g0])


@arc.struct
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


@arc.struct
class SubsonicF16:
    rigid_body: RigidBody = arc.field(default_factory=RigidBody)
    gravity: GravityModel = arc.field(default_factory=ConstantGravity)
    atmos: AtmosphereModel = arc.field(default_factory=AtmosphereModel)
    engine: F16Engine = arc.field(default_factory=F16Engine)
    aero: F16Aerodynamics = arc.field(default_factory=F16Aerodynamics)

    # NOTE: The weight in the textbook is 25,000 lbs, but this
    # does not give consistent values - the default value here
    # matches the values given in the tables
    m: float = default_mass  # Vehicle mass [slug]
    # Vehicle inertia matrix [slug·ft²]
    J_B: np.ndarray = arc.field(default_factory=lambda: default_J_B)

    xcg: float = 0.35  # CG location (% of cbar)

    S: float = 300.0  # Planform area
    b: float = 30.0  # Span
    cbar: float = 11.32  # Mean aerodynamic chord
    xcgr: float = 0.35  # Reference CG location (% of cbar)
    hx: float = 160.0  # Engine angular momentum (assumed constant)

    @arc.struct
    class State:
        rigid_body: RigidBody.State
        engine_power: np.ndarray  # Engine power state (0 to 1)

        @property
        def p_N(self):
            return self.rigid_body.p_N

        @property
        def att(self):
            return self.rigid_body.att

        @property
        def v_B(self):
            return self.rigid_body.v_B

        @property
        def w_B(self):
            return self.rigid_body.w_B

    def net_forces(self, t, x: State, u):
        """Net forces and moments in body frame B, plus any extra state derivatives

        Args:
            t: time
            x: state: (p_N, att, v_B, w_B, aux)
            u: (throttle, elevator, aileron, rudder) control inputs

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
        pow = x.engine_power
        F_eng_B = self.engine.calc_thrust(pow, alt, amach)

        force_coeffs, moment_coeffs = self.aero(
            vt, alpha, beta, x.w_B, el, ail, rdr, self
        )
        cxt, cyt, czt = force_coeffs
        clt, cmt, cnt = moment_coeffs

        F_grav_N = self.m * self.gravity(x.p_N)
        F_aero_B = qbar * self.S * np.stack([cxt, cyt, czt])

        F_grav_B = x.att.apply(F_grav_N, inverse=True)

        F_B = F_aero_B + F_eng_B + F_grav_B

        # Moments
        p, q, r = x.w_B  # Angular velocity in body frame (ω_B)
        Meng_B = self.hx * np.array([0.0, -r, q])
        Maero_B = (
            qbar * self.S * np.array([self.b * clt, self.cbar * cmt, self.b * cnt])
        )
        M_B = Meng_B + Maero_B

        # Dynamic component of engine state (auxiliary state)
        pow_t = self.engine.dynamics(t, pow, thtl)

        pow_t = np.atleast_1d(pow_t)
        return F_B, M_B, pow_t

    def dynamics(self, t, x: State, u: np.ndarray) -> State:
        """Compute time derivative of the state

        Args:
            t: time
            x: state: (p_N, att, v_B, w_B, engine_power)
            u: (throttle, elevator, aileron, rudder) control inputs

        Returns:
            x_dot: time derivative of the state
        """
        # Compute the net forces
        F_B, M_B, engine_deriv = self.net_forces(t, x, u)

        rb_input = RigidBody.Input(
            F_B=F_B,
            M_B=M_B,
            m=self.m,
            J_B=self.J_B,
        )
        rb_derivs = self.rigid_body.dynamics(t, x.rigid_body, rb_input)

        return self.State(
            rigid_body=rb_derivs,
            engine_power=engine_deriv,
        )
