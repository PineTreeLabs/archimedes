# ruff: noqa: N803, N806
from typing import NamedTuple

import numpy as np

import archimedes as arc
from archimedes import struct

GEAR_RATIO = 46.8512  # 47:1 nominal
SUPPLY_VOLTAGE = 12.0  # Power supply nominal voltage [V]

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


@struct
class MotorParams:
    J: float  # Effective inertia
    b: float  # Viscous friction
    L: float  # Motor inductance [H]
    R: float  # Motor resistance [Ohm]
    kt: float  # Current -> torque scale [N-m/A]


def motor_ode(
    t: float, x: np.ndarray, u: np.ndarray, params: MotorParams
) -> np.ndarray:
    i, _pos, vel = x
    (V,) = u

    ke = params.kt / GEAR_RATIO  # Velocity -> Back EMF scale

    i_t = (1 / params.L) * (V - (i * params.R) - ke * vel)
    vel_t = (1 / params.J) * (params.kt * i - params.b * vel)

    return np.hstack([i_t, vel, vel_t])


hil_dt = 1e-4  # Control loop time step
motor_dyn = arc.discretize(motor_ode, dt=hil_dt, method="euler")


# Observation model (used for system ID)
def motor_obs(
    t: float, x: np.ndarray, u: np.ndarray, params: MotorParams
) -> np.ndarray:
    # Measure absolute current and position
    return np.hstack([abs(x[0]), x[1]])


@struct
class MotorInputs:
    pwm_duty: float  # PWM duty cycle (0-1)
    ENA: bool
    ENB: bool
    INA: bool
    INB: bool


class MotorOutputs(NamedTuple):
    ENCA: int
    ENCB: int
    V_CS: float
    VOUTA: float
    VOUTB: float


# Motor logic table:
#   |--INA--|--INB--|--STATE--|
#   |  LOW  | HIGH  | FORWARD | (CCW)
#   | HIGH  |  LOW  | REVERSE | (CW)
#   | HIGH  | HIGH  |  BRAKE  |
#   |  LOW  |  LOW  |  COAST  |


# Motor enable/disable/direction logic
@arc.compile
def motor_dir(INA, INB, ENA, ENB):
    d = (INB + (1 - INA)) - (INA + (1 - INB))

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


@arc.compile
def quad_count(A, B, count, prev_A, prev_B):
    rising_A = np.logical_and(A, np.logical_not(prev_A))
    count += rising_A * np.where(B, -1, 1)  # CCW

    falling_A = np.logical_and(np.logical_not(A), prev_A)
    count += falling_A * np.where(B, 1, -1)  # CW

    rising_B = np.logical_and(B, np.logical_not(prev_B))
    count += rising_B * np.where(A, 1, -1)  # CW

    falling_B = np.logical_and(np.logical_not(B), prev_B)
    count += falling_B * np.where(A, -1, 1)  # CCW

    return count


@arc.compile(name="plant", return_names=("state_new", "outputs"))
def plant_step(
    t,
    state: np.ndarray,
    inputs: MotorInputs,
    params: MotorParams,
) -> tuple[np.ndarray, MotorOutputs]:
    # Determine motor direction
    d = motor_dir(inputs.INA, inputs.INB, inputs.ENA, inputs.ENB)
    pwm_duty = np.clip(inputs.pwm_duty, 0.0, 1.0)
    V_motor = d * pwm_duty * SUPPLY_VOLTAGE

    # Motor dynamics model (discretized)
    u = (V_motor,)
    state = motor_dyn(t, state, u, params)

    I_motor, pos, vel = state

    # Encoder emulation
    PPR = ENC_PPR * GEAR_RATIO
    ENCA, ENCB = encoder(pos, PPR)

    # H-bridge output voltages
    VOUTA = np.where(d >= 0, V_motor * VOUT_SCALE, 0.0)
    VOUTB = np.where(d < 0, -V_motor * VOUT_SCALE, 0.0)

    # Current sense voltage
    V_CS = abs(I_motor) * CS_V_PER_AMP

    outputs = MotorOutputs(
        VOUTA=VOUTA,
        VOUTB=VOUTB,
        V_CS=V_CS,
        ENCA=ENCA,
        ENCB=ENCB,
    )

    return state, outputs
