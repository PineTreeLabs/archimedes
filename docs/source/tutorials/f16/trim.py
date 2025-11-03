# ruff: noqa: N806
from __future__ import annotations

import numpy as np
from f16 import GRAV_FTS2, SubsonicF16

import archimedes as arc
from archimedes import struct
from archimedes.spatial import EulerAngles, euler_kinematics

__all__ = ["trim", "TrimPoint", "TrimCondition", "TrimVariables"]


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
    c = 1 + G**2 * np.cos(beta) ** 2

    num = (
        G
        * np.cos(beta)
        * np.sqrt(
            (a - b**2) + b * np.tan(alpha) * (c * (1 - b**2) + G**2 * np.sin(beta) ** 2)
        )
    )
    den = np.cos(alpha) * (a**2 - b**2 * (1 + c * np.tan(alpha) ** 2))
    phi = np.arctan2(num, den)

    # Rate-of-climb constraint (determines pitch angle)
    a = np.cos(alpha) * np.cos(beta)
    b = np.sin(phi) * np.sin(beta) + np.cos(phi) * np.sin(alpha) * np.cos(beta)
    num = a * b + np.sin(gamma) * np.sqrt(a**2 + b**2 - np.sin(gamma) ** 2)
    den = a**2 - np.sin(gamma) ** 2
    theta = np.arctan2(num, den)

    # Calculate the angular velocity based on the Euler rates and
    # roll-pitch-yaw angles (inverse Euler kinematics)
    rpy = np.hstack([phi, theta, 0.0])  # Arbitrary yaw angle
    H_inv = euler_kinematics(rpy, inverse=True)
    w_B = H_inv @ np.hstack(
        [condition.roll_rate, condition.pitch_rate, condition.turn_rate]
    )

    # Body-frame velocity (rotate from wind frame)
    v_W = np.array([condition.vt, 0.0, 0.0])  # Wind-frame velocity [ft/s]
    R_WB = EulerAngles([-beta, alpha], "zy")
    v_B = R_WB.rotate(v_W, inverse=True)

    att = EulerAngles(rpy)

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
