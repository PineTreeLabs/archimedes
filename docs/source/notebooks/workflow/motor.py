import numpy as np

import dataclasses
from typing import NamedTuple

import archimedes as arc
from archimedes import struct

GEAR_RATIO = 46.8512  # 47:1 nominal

MOTOR_VOLTAGE = 12.0

ENC_PPR = 48
RAD_PER_COUNT = (2 * np.pi) / (ENC_PPR * GEAR_RATIO)

# Current sense conversion
CS_V_PER_AMP = 0.14  # VNH5019 spec: 0.14 V/A

# Voltage divider for VOUT measurement
VOUT_R1 = 47.0  # First leg of voltage divider
VOUT_R2 = 15.0  # Second leg of voltage divider
VOUT_SCALE = VOUT_R2 / (VOUT_R1 + VOUT_R2)

nx = 3  # State dimension (current, velocity, position)
nu = 1  # Input dimension (voltage)
ny = 2  # Output dimension (position, current)


@struct.pytree_node
class MotorParams:
    m: float  # Effective mass/inertia
    b: float  # Viscous friction
    L: float  # Motor inductance [H]
    R: float  # Motor resistance [Ohm]
    Kt: float  # Current -> torque scale [N-m/A]

    def asdict(self):
        return dataclasses.asdict(self)


def motor_ode(
    t: float, x: np.ndarray, u: np.ndarray, params: MotorParams
) -> np.ndarray:
    params = params.asdict()

    i, _pos, vel = x
    (V,) = u
    m = params["m"]  # Effective mass/inertia
    b = params["b"]  # Viscous friction
    L = params["L"]  # Motor inductance
    Kt = params["Kt"]  # Current -> torque scale
    R = params["R"]

    Ke = Kt / GEAR_RATIO  # Velocity -> Back EMF scale

    i_t = (1 / L) * (V - (i * R) - Ke * vel)
    pos_t = vel
    vel_t = (1 / m) * (Kt * i - b * vel)

    return np.hstack([i_t, pos_t, vel_t])


hil_dt = 1e-4  # Control loop time step
motor_dyn = arc.discretize(motor_ode, dt=hil_dt, method="euler")


# Observation model (used for system ID)
def motor_obs(t, x, u, p):
    p = p.asdict()
    return np.hstack(
        [
            abs(x[0]),  # Current
            x[1],  # Position
        ]
    )


@struct.pytree_node
class MotorInputs:
    pwm_duty: float  # PWM duty cycle (0-1)
    ENA: bool
    ENB: bool
    INA: bool
    INB: bool


class MotorOutputs(NamedTuple):
    I_motor: float
    V_motor: float
    pos: float
    vel: float
    ENCA: int
    ENCB: int
    V_CS: float
    VOUTA: float
    VOUTB: float


# Motor logic table:
#   |--INA--|--INB--|--STATE--|
#   | HIGH  |  LOW  | FORWARD | (CW)
#   | HIGH  | HIGH  |  BRAKE  |
#   |  LOW  |  LOW  |  COAST  |
#   |  LOW  | HIGH  | REVERSE | (CCW)


# Motor enable/disable/direction logic
@arc.compile
def motor_dir(INA, INB, ENA, ENB):
    d = (INA + (1 - INB)) - (INB + (1 - INA))

    # Disable if either of ENA or ENB are low
    return ENA * ENB * (d / 2)


@arc.compile(static_argnames="PPR")
def encoder(pos: float, PPR: int) -> tuple[int, int]:
    # Convert position to encoder counts
    counts = np.fmod((pos / (2 * np.pi)) * PPR / 4, PPR / 4)

    # Generate quadrature signals
    ENCA = np.fmod(np.floor(counts * 4) + 1, 4) < 2
    ENCB = np.fmod(np.floor(counts * 4), 4) < 2

    return ENCA, ENCB


@arc.compile(static_argnames=("pwm_table",))
def plant_step(
    t,
    state: np.ndarray,
    inputs: MotorInputs,
    params: MotorParams,
    pwm_table: np.ndarray,
) -> tuple[np.ndarray, MotorOutputs]:
    # Determine motor direction
    d = motor_dir(inputs.INA, inputs.INB, inputs.ENA, inputs.ENB)
    pwm_duty = np.clip(inputs.pwm_duty, 0.0, 1.0)
    # V_motor = d * np.interp(pwm_duty, pwm_table[:, 0], pwm_table[:, 1])
    V_motor = d * pwm_duty * MOTOR_VOLTAGE

    # Motor dynamics model
    u = (V_motor,)
    state = motor_dyn(t, state, u, params)

    I_motor, pos, vel = state

    # Encoder emulation
    PPR = ENC_PPR * GEAR_RATIO

    ENCA, ENCB = encoder(pos, PPR)

    outputs = MotorOutputs(
        V_motor=V_motor,
        I_motor=I_motor,
        pos=pos,
        vel=vel,
        VOUTA=V_motor * VOUT_SCALE,
        VOUTB=0.0,
        V_CS=abs(I_motor) * CS_V_PER_AMP,
        ENCA=ENCA,
        ENCB=ENCB,
    )

    return state, outputs
