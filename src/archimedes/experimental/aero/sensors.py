from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

import archimedes as arc
from archimedes import struct
from archimedes.experimental.aero import (
    dcm_from_quaternion,
    GravityModel,
    ConstantGravity,
    FlightVehicle,
)

if TYPE_CHECKING:
    from archimedes.typing import ArrayLike

__all__ = [
    "Accelerometer",
    "Gyroscope",
]


@struct.pytree_node
class Accelerometer:
    """Basic three-axis accelerometer model

    Currently assumes that the accel is located at the center of mass (CM) of the vehicle.
    """

    gravity: GravityModel = struct.field(default_factory=ConstantGravity)
    noise: float = 0.0  # Noise standard deviation [m/s^2]

    def __call__(
        self,
        x: FlightVehicle.State,
        a_B: ArrayLike,
        w: ArrayLike,
    ) -> ArrayLike:
        g_N = self.gravity(x.p_N)  # Inertial gravity vector
        C_BN = dcm_from_quaternion(x.att)

        # Measure inertial acceleration in body coordinates
        a_N_B = a_B + np.cross(x.w_B, x.v_B)
        a_meas_B = a_N_B - C_BN @ g_N  # "proper" inertial acceleration

        return a_meas_B + self.noise * w


@struct.pytree_node
class Gyroscope:
    """Basic three-axis gyroscope model

    Currently assumes that the gyro is located at the center of mass (CM) of the vehicle.
    """

    noise: float = 0.0  # Noise standard deviation [rad/s]

    def __call__(
        self,
        x: FlightVehicle.State,
        w: ArrayLike,
    ) -> ArrayLike:
        # Measure angular velocity in body coordinates
        return x.w_B + self.noise * w


@struct.pytree_node
class LineOfSight:
    """Basic line-of-sight sensor model"""

    noise: float = 0.0  # Noise standard deviation [rad]

    def __call__(
        self,
        vehicle: FlightVehicle.State,
        target: FlightVehicle.State,
        w: ArrayLike,
    ) -> ArrayLike:
        C_BN = dcm_from_quaternion(vehicle.att)

        r_N = target.pos - vehicle.p_N  # Relative position in inertial coordinates
        r_B = C_BN @ r_N  # Relative position in body-fixed coordinates
        az = np.atan2(r_B[1], r_B[0])  # Azimuth angle
        el = np.arctan2(r_B[2], np.sqrt(r_B[0] ** 2 + r_B[1] ** 2))  # Elevation angle

        return np.hstack([az, el]) + self.noise * w
