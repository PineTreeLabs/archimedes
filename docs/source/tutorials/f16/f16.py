from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from actuator import (
    Actuator,
    ActuatorConfig,
    IdealActuator,
    IdealActuatorConfig,
)
from aero import F16Aero, TabulatedAero
from atmosphere import (
    AtmosphereConfig,
    AtmosphereModel,
    LinearAtmosphere,
    LinearAtmosphereConfig,
)
from engine import (
    F16Engine,
    F16EngineConfig,
    NASAEngine,
    NASAEngineConfig,
)

from archimedes import StructConfig, field, struct
from archimedes.experimental import aero
from archimedes.experimental.aero import (
    ConstantGravity,
    ConstantGravityConfig,
    GravityConfig,
    GravityModel,
)
from archimedes.spatial import RigidBody, RigidBodyConfig, dcm_from_euler

if TYPE_CHECKING:
    from trim import TrimPoint

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


@struct
class FlightCondition:
    alt: float  # Altitude [ft]
    vt: float  # True airspeed [ft/s]
    alpha: float  # Angle of attack [rad]
    beta: float  # Sideslip angle [rad]
    mach: float  # Mach number
    qbar: float  # Dynamic pressure [lbf/ft²]


@struct
class F16Geometry:
    S: float = 300.0  # Planform area
    b: float = 30.0  # Span
    cbar: float = 11.32  # Mean aerodynamic chord
    xcgr: float = 0.35  # Reference CG location (% of cbar)


@struct
class SubsonicF16:
    rigid_body: RigidBody = field(default_factory=RigidBody)
    gravity: GravityModel = field(default_factory=lambda: ConstantGravity(GRAV_FTS2))
    atmos: AtmosphereModel = field(default_factory=LinearAtmosphere)
    engine: F16Engine = field(default_factory=NASAEngine)
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

    @struct
    class State(RigidBody.State):
        eng: F16Engine.State
        aero: F16Aero.State = field(default_factory=TabulatedAero.State)
        elevator: Actuator.State = field(default_factory=IdealActuator.State)
        aileron: Actuator.State = field(default_factory=IdealActuator.State)
        rudder: Actuator.State = field(default_factory=IdealActuator.State)

    @struct
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

    def net_forces(
        self, t, x: State, u: Input, condition: FlightCondition | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Net forces and moments in body frame B

        Args:
            t: time
            x: state
            u: control inputs
            z: flight condition (optional, will be computed if not provided)

        Returns:
            F_B: net forces in body frame B
            M_B: net moments in body frame B
        """
        if condition is None:
            condition = self.flight_condition(x)

        # === Engine ===
        u_eng = self.engine.Input(
            throttle=u.throttle,
            alt=condition.alt,
            mach=condition.mach,
        )
        y_eng = self.engine.output(t, x.eng, u_eng)
        F_eng_B = np.hstack([y_eng.thrust, 0.0, 0.0])
        p, q, r = x.w_B  # Angular velocity in body frame (ω_B)
        M_eng_B = self.hx * np.hstack([0.0, -r, q])

        # === Control surface actuators ===
        el = self.elevator.output(t, x.elevator, u.elevator)
        ail = self.aileron.output(t, x.aileron, u.aileron)
        rud = self.rudder.output(t, x.rudder, u.rudder)

        # === Aerodynamics ===
        u_aero = self.aero.Input(
            condition=condition,
            w_B=x.w_B,
            elevator=el,
            aileron=ail,
            rudder=rud,
            xcg=self.xcg,
        )
        y_aero = self.aero.output(t, x.aero, u_aero, self.geometry)
        cxt, cyt, czt = y_aero.CF_B
        clt, cmt, cnt = y_aero.CM_B

        S = self.geometry.S
        b = self.geometry.b
        cbar = self.geometry.cbar
        F_aero_B = condition.qbar * S * np.stack([cxt, cyt, czt])
        M_aero_B = condition.qbar * S * np.hstack([b * clt, cbar * cmt, b * cnt])

        # === Gravity ===
        F_grav_B = self.calc_gravity(x)

        # === Net forces and moments ===
        F_B = F_aero_B + F_eng_B + F_grav_B
        M_B = M_eng_B + M_aero_B

        return F_B, M_B

    def dynamics(self, t, x: State, u: Input) -> State:
        """Compute time derivative of the state

        Args:
            t: time
            x: state
            u: control inputs

        Returns:
            x_dot: time derivative of the state
        """
        condition = self.flight_condition(x)

        # Compute the net forces
        F_B, M_B = self.net_forces(t, x, u, condition)

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
            alt=condition.alt,
            mach=condition.mach,
        )
        eng_deriv = self.engine.dynamics(t, x.eng, eng_input)

        # Unsteady aero
        aero_input = self.aero.Input(
            condition=condition,
            w_B=x.w_B,
            elevator=u.elevator,
            aileron=u.aileron,
            rudder=u.rudder,
            xcg=self.xcg,
        )
        aero_deriv = self.aero.dynamics(t, x.aero, aero_input, self.geometry)

        # Actuator dynamics
        elev_deriv = self.elevator.dynamics(t, x.elevator, u.elevator)
        ail_deriv = self.aileron.dynamics(t, x.aileron, u.aileron)
        rud_deriv = self.rudder.dynamics(t, x.rudder, u.rudder)

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
        from trim import trim  # Avoid circular import

        return trim(
            model=self,
            vt=vt,
            alt=alt,
            gamma=gamma,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            turn_rate=turn_rate,
        )

    @classmethod
    def from_yaml(cls, path: str | Path, key: str | None = None) -> SubsonicF16:
        """Load SubsonicF16 configuration from a YAML file"""
        path = Path(path)
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        if key is not None:
            config_dict = config_dict[key]

        config = SubsonicF16Config.model_validate(config_dict)
        return config.build()


class SubsonicF16Config(StructConfig):
    rigid_body: RigidBodyConfig = field(default_factory=RigidBodyConfig)
    gravity: GravityConfig = field(
        default_factory=lambda: ConstantGravityConfig(g0=GRAV_FTS2)
    )
    atmos: AtmosphereConfig = field(
        default_factory=lambda: LinearAtmosphereConfig(g0=GRAV_FTS2)
    )
    engine: F16EngineConfig = field(default_factory=NASAEngineConfig)
    # Skip aero: only one model and no configuration needed
    geometry: F16Geometry = field(default_factory=F16Geometry)

    # Control surface actuators
    elevator: ActuatorConfig = field(default_factory=IdealActuatorConfig)
    aileron: ActuatorConfig = field(default_factory=IdealActuatorConfig)
    rudder: ActuatorConfig = field(default_factory=IdealActuatorConfig)

    m: float = default_mass  # Vehicle mass [slug]
    # Vehicle inertia matrix [slug·ft²]
    J_B: np.ndarray = field(default_factory=lambda: default_J_B)

    xcg: float = 0.35  # CG location (% of cbar)
    hx: float = 160.0  # Engine angular momentum (assumed constant)

    def build(self) -> SubsonicF16:
        return SubsonicF16(
            rigid_body=self.rigid_body.build(),
            gravity=self.gravity.build(),
            atmos=self.atmos.build(),
            engine=self.engine.build(),
            aero=TabulatedAero(),
            geometry=self.geometry,
            elevator=self.elevator.build(),
            aileron=self.aileron.build(),
            rudder=self.rudder.build(),
            m=self.m,
            J_B=self.J_B,
            xcg=self.xcg,
            hx=self.hx,
        )
