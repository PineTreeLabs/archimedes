from __future__ import annotations

import numpy as np

import archimedes as arc

from archimedes import struct, field
from archimedes.spatial import (
    RigidBody, Rotation, euler_kinematics, dcm_from_euler
)
from archimedes.experimental import aero
from archimedes.experimental.aero import GravityModel

from engine import F16Engine, F16EngineConfig, LagEngine
from aero import F16Aero, TabulatedAero
from actuator import Actuator, RateLimitedActuator, IdealActuator, ActuatorConfig


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
class FlightCondition:
    alt: float  # Altitude [ft]
    vt: float  # True airspeed [ft/s]
    alpha: float  # Angle of attack [rad]
    beta: float  # Sideslip angle [rad]
    mach: float  # Mach number
    qbar: float  # Dynamic pressure [lbf/ft²]


@arc.struct
class F16Geometry:
    S: float = 300.0  # Planform area
    b: float = 30.0  # Span
    cbar: float = 11.32  # Mean aerodynamic chord
    xcgr: float = 0.35  # Reference CG location (% of cbar)


@arc.struct
class SubsonicF16:
    rigid_body: RigidBody = field(default_factory=RigidBody)
    gravity: GravityModel = field(default_factory=ConstantGravity)
    atmos: AtmosphereModel = field(default_factory=AtmosphereModel)
    engine: F16Engine = field(default_factory=LagEngine)
    aero: F16Aero = field(default_factory=TabulatedAero)
    geometry: F16Geometry = field(default_factory=F16Geometry)

    # Control surface actuators
    elevator: Actuator = field(default_factory=IdealActuator)
    aileron: Actuator = field(default_factory=IdealActuator)
    rudder: Actuator = field(default_factory=IdealActuator)

    # NOTE: The weight in the textbook is 25,000 lbs, but this
    # does not give consistent values - the default value here
    # matches the values given in the tables
    m: float = default_mass  # Vehicle mass [slug]
    # Vehicle inertia matrix [slug·ft²]
    J_B: np.ndarray = field(default_factory=lambda: default_J_B)

    xcg: float = 0.35  # CG location (% of cbar)
    hx: float = 160.0  # Engine angular momentum (assumed constant)

    @arc.struct
    class State(RigidBody.State):
        eng: F16Engine.State
        aero: F16Aero.State = field(default_factory=TabulatedAero.State)
        elevator: Actuator.State = field(default_factory=IdealActuator.State)
        aileron: Actuator.State = field(default_factory=IdealActuator.State)
        rudder: Actuator.State = field(default_factory=IdealActuator.State)

    @arc.struct
    class Input:
        throttle: float  # Throttle command [0-1]
        elevator: float  # Elevator deflection [deg]
        aileron: float  # Aileron deflection [deg]
        rudder: float  # Rudder deflection [deg]

    def calc_gravity(self, x: State):
        F_grav_N = self.m * self.gravity(x.p_N)
        if self.rigid_body.rpy_attitude:
            F_grav_B = dcm_from_euler(x.att) @ F_grav_N
        else:
            F_grav_B = x.att.apply(F_grav_N, inverse=True)
        return F_grav_B

    def flight_condition(self, x: RigidBody.State) -> FlightCondition:
        vt, alpha, beta = aero.wind_frame(x.v_B)

        # Atmosphere model
        alt = -x.p_N[2]
        mach, qbar = self.atmos(vt, alt)

        return FlightCondition(
            vt=vt,
            alpha=alpha,
            beta=beta,
            mach=mach,
            qbar=qbar,
            alt=alt,
        )

    def net_forces(self, t, x: State, u: Input, z: FlightCondition | None = None):
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
        if z is None:
            z = self.flight_condition(x)

        # Engine thrust model
        u_eng = self.engine.Input(
            throttle=u.throttle,
            alt=z.alt,
            mach=z.mach,
        )
        y_eng = self.engine.output(t, x.eng, u_eng)
        F_eng_B = np.hstack([y_eng.thrust, 0.0, 0.0])

        # Control surface actuator positions
        u_elev = Actuator.Input(command=u.elevator)
        y_elev = self.elevator.output(t, x.elevator, u_elev)
        el = y_elev.position
        u_ail = Actuator.Input(command=u.aileron)
        y_ail = self.aileron.output(t, x.aileron, u_ail)
        ail = y_ail.position
        u_rud = Actuator.Input(command=u.rudder)
        y_rud = self.rudder.output(t, x.rudder, u_rud)
        rud = y_rud.position

        # Aerodynamic model
        u_aero = self.aero.Input(
            condition=z,
            w_B=x.w_B,
            elevator=el,
            aileron=ail,
            rudder=rud,
            xcg=self.xcg,
        )
        y_aero = self.aero.output(t, x.aero, u_aero, self.geometry)
        cxt, cyt, czt = y_aero.CF_B
        clt, cmt, cnt = y_aero.CM_B

        F_grav_B = self.calc_gravity(x)
        S = self.geometry.S
        b = self.geometry.b
        cbar = self.geometry.cbar
        F_aero_B = z.qbar * S * np.stack([cxt, cyt, czt])

        F_B = F_aero_B + F_eng_B + F_grav_B

        # Moments
        p, q, r = x.w_B  # Angular velocity in body frame (ω_B)
        M_eng_B = self.hx * np.hstack([0.0, -r, q])
        M_aero_B = z.qbar * S * np.hstack([b * clt, cbar * cmt, b * cnt])
        M_B = M_eng_B + M_aero_B

        return F_B, M_B

    def dynamics(self, t, x: State, u: Input) -> State:
        """Compute time derivative of the state

        Args:
            t: time
            x: state: (p_N, att, v_B, w_B, engine_power)
            u: (throttle, elevator, aileron, rudder) control inputs

        Returns:
            x_dot: time derivative of the state
        """
        z = self.flight_condition(x)

        # Compute the net forces
        F_B, M_B = self.net_forces(t, x, u, z)

        rb_input = RigidBody.Input(
            F_B=F_B,
            M_B=M_B,
            m=self.m,
            J_B=self.J_B,
        )
        rb_deriv = self.rigid_body.dynamics(t, x, rb_input)

        # Engine dynamics
        eng_input = self.engine.Input(
            throttle=u.throttle,
            alt=z.alt,
            mach=z.mach,
        )
        eng_deriv = self.engine.dynamics(t, x.eng, eng_input)

        # Unsteady aero
        aero_input = self.aero.Input(
            condition=z,
            w_B=x.w_B,
            elevator=u.elevator,
            aileron=u.aileron,
            rudder=u.rudder,
            xcg=self.xcg,
        )
        aero_deriv = self.aero.dynamics(t, x.aero, aero_input, self.geometry)

        # Actuator dynamics
        elev_input = Actuator.Input(command=u.elevator)
        elev_deriv = self.elevator.dynamics(t, x.elevator, elev_input)
        ail_input = Actuator.Input(command=u.aileron)
        ail_deriv = self.aileron.dynamics(t, x.aileron, ail_input)
        rud_input = Actuator.Input(command=u.rudder)
        rud_deriv = self.rudder.dynamics(t, x.rudder, rud_input)

        return self.State(
            p_N=rb_deriv.p_N,
            att=rb_deriv.att,
            v_B=rb_deriv.v_B,
            w_B=rb_deriv.w_B,
            eng=eng_deriv,
            aero=aero_deriv,
            elevator=elev_deriv,
            aileron=ail_deriv,
            rudder=rud_deriv,
        )
    
    def trim(
        self,
        vt: float,  # True airspeed [ft/s]
        alt: float = 0.0,  # Altitude [ft]
        gamma: float = 0.0,  # Flight path angle [deg]
        roll_rate: float = 0.0,  # Roll rate [rad/s]
        pitch_rate: float = 0.0,  # Pitch rate [rad/s]
        turn_rate: float = 0.0,  # Turn rate [rad/s]
    ) -> TrimPoint:
        """Trim the aircraft for steady flight conditions

        Args:
            vt: True airspeed [ft/s]
            alt: Altitude [ft]
            gamma: Flight path angle [deg]
            roll_rate: Roll rate [rad/s]
            pitch_rate: Pitch rate [rad/s]
            turn_rate: Turn rate [rad/s]
        Returns:
            TrimPoint: trimmed state, inputs, and variables
        """
        return trim(
            model=self,
            vt=vt,
            alt=alt,
            gamma=gamma,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            turn_rate=turn_rate,
        )


@struct
class TrimCondition:
    vt: float  # True airspeed [ft/s]
    alt: float = 0.0  # Altitude [ft]
    gamma: float = 0.0  # Flight path angle [deg]
    roll_rate: float = 0.0  # Roll rate [rad/s]
    pitch_rate: float = 0.0  # Pitch rate [rad/s]
    turn_rate: float = 0.0  # Turn rate [rad/s]


@struct
class TrimVariables:
    alpha: float  # Angle of attack [rad]
    beta: float  # Sideslip angle [rad]
    throttle: float  # Throttle setting [0-1]
    elevator: float  # Elevator deflection [deg]
    aileron: float  # Aileron deflection [deg]
    rudder: float  # Rudder deflection [deg]

    @property
    def inputs(self) -> SubsonicF16.Input:
        return SubsonicF16.Input(
            throttle=self.throttle,
            elevator=self.elevator,
            aileron=self.aileron,
            rudder=self.rudder,
        )


@struct
class TrimPoint:
    condition: TrimCondition
    variables: TrimVariables
    xcg: float  # CG location [-]
    name: str = ""
    description: str = ""
    state: SubsonicF16.State | None = None
    residuals: np.ndarray | None = None

    @property
    def inputs(self) -> SubsonicF16.Input:
        return self.variables.inputs
    

def trim_state(
    params: TrimVariables,
    condition: TrimCondition,
    model: SubsonicF16,
) -> SubsonicF16.State:
    gamma = np.deg2rad(condition.gamma)
    alpha = params.alpha
    beta = params.beta

    # Turn constraint (determines roll angle)
    G = condition.turn_rate * condition.vt / GRAV_FTS2  # Centripetal acceleration [g's]
    a = 1 - G * np.tan(alpha) * np.sin(beta)
    b = np.sin(gamma) / np.cos(beta)
    c = 1 + G ** 2 * np.cos(beta) ** 2

    num = G * np.cos(beta) * np.sqrt(
        (a - b ** 2) + b * np.tan(alpha) * (c * (1 - b ** 2) + G ** 2 * np.sin(beta) ** 2)
    )
    den = np.cos(alpha) * (a ** 2 - b ** 2 * (1 + c * np.tan(alpha) ** 2))
    phi = np.arctan2(num, den)

    # Rate-of-climb constraint (determines pitch angle)
    a = np.cos(alpha) * np.cos(beta)
    b = np.sin(phi) * np.sin(beta) + np.cos(phi) * np.sin(alpha) * np.cos(beta)
    num = a * b + np.sin(gamma) * np.sqrt(a ** 2 + b ** 2 - np.sin(gamma) ** 2)
    den = a ** 2 - np.sin(gamma) ** 2 
    theta = np.arctan2(num, den)

    # Calculate the angular velocity based on the Euler rates and
    # roll-pitch-yaw angles (inverse Euler kinematics)
    rpy = np.hstack([phi, theta, 0.0])  # Arbitrary yaw angle
    H_inv = euler_kinematics(rpy, inverse=True)
    w_B = H_inv @ np.hstack([
        condition.roll_rate, condition.pitch_rate, condition.turn_rate
    ])

    # Body-frame velocity (rotate from wind frame)
    v_W = np.array([condition.vt, 0.0, 0.0])  # Wind-frame velocity [ft/s]
    R_WB = Rotation.from_euler("zy", [-beta, alpha])
    v_B = R_WB.apply(v_W, inverse=True)

    if model.rigid_body.rpy_attitude:
        att = rpy
    else:
        att = Rotation.from_euler("xyz", rpy)

    return model.State(
        p_N=np.hstack([0.0, 0.0, -condition.alt]),
        att=att,
        v_B=v_B,
        w_B=w_B,
        eng=model.engine.trim(params.throttle),
        aero=model.aero.trim(),
        elevator=model.elevator.trim(params.elevator),
        aileron=model.aileron.trim(params.aileron),
        rudder=model.rudder.trim(params.rudder),
    )


def trim_residual(
    params: TrimVariables,
    condition: TrimCondition,
    model: SubsonicF16,
) -> np.ndarray:
    x = trim_state(params, condition, model)
    u = params.inputs
    x_t = model.dynamics(0.0, x, u)
    return np.hstack([x_t.v_B, x_t.w_B])


def trim(
    model: SubsonicF16,
    vt: float,  # True airspeed [ft/s]
    alt: float = 0.0,  # Altitude [ft]
    gamma: float = 0.0,  # Flight path angle [deg]
    roll_rate: float = 0.0,  # Roll rate [rad/s]
    pitch_rate: float = 0.0,  # Pitch rate [rad/s]
    turn_rate: float = 0.0,  # Turn rate [rad/s]
) -> TrimPoint:
    condition = TrimCondition(
        vt=vt,
        alt=alt,
        gamma=gamma,
        roll_rate=roll_rate,
        pitch_rate=pitch_rate,
        turn_rate=turn_rate,
    )
    params_guess = TrimVariables(
        alpha=0.0,
        beta=0.0,
        throttle=0.0,
        elevator=0.0,
        aileron=0.0,
        rudder=0.0,
    )

    params_guess_flat, unravel = arc.tree.ravel(params_guess)
    def residual(params_flat):
        params = unravel(params_flat)
        return trim_residual(params, condition, model)

    params_opt_flat = arc.root(residual, params_guess_flat)
    params_opt = unravel(params_opt_flat)

    state_opt = trim_state(params_opt, condition, model)

    return TrimPoint(
        condition=condition,
        variables=params_opt,
        state=state_opt,
        xcg=model.xcg,
        residuals=residual(params_opt_flat),
    )