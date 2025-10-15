from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from archimedes import struct, field, StructConfig
from archimedes.experimental.aero import (
    GravityModel,
    GravityConfig,
    ConstantGravity,
    ConstantGravityConfig,
)

if TYPE_CHECKING:
    from archimedes.typing import ArrayLike
    from archimedes.spatial import RigidBody

__all__ = [
    "Accelerometer",
    "AccelerometerConfig",
    "Gyroscope",
    "GyroscopeConfig",
    "LineOfSight",
    "LineOfSightConfig",
]


@struct
class Accelerometer:
    """Basic three-axis accelerometer model

    Currently assumes that the accel is located at the center of mass (CM) of the vehicle.
    """

    gravity: GravityModel = field(default_factory=ConstantGravity)
    noise: float = 0.0  # Noise standard deviation [m/s^2]

    def __call__(
        self,
        x: RigidBody.State,
        a_B: ArrayLike,
        w: ArrayLike,
    ) -> ArrayLike:
        g_N = self.gravity(x.p_N)  # Inertial gravity vector
        C_BN = x.att.as_dcm()

        # Measure inertial acceleration in body coordinates
        a_N_B = a_B + np.cross(x.w_B, x.v_B)
        a_meas_B = a_N_B - C_BN @ g_N  # "proper" inertial acceleration

        return a_meas_B + self.noise * w


class AccelerometerConfig(StructConfig, type="basic"):
    gravity: GravityConfig = field(default_factory=ConstantGravityConfig)
    noise: float = 0.0  # Noise standard deviation [m/s^2]

    def build(self) -> Accelerometer:
        return Accelerometer(gravity=self.gravity.build(), noise=self.noise)


@struct
class Gyroscope:
    """Basic three-axis gyroscope model

    Currently assumes that the gyro is located at the center of mass (CM) of the vehicle.
    """

    noise: float = 0.0  # Noise standard deviation [rad/s]

    def __call__(
        self,
        x: RigidBody.State,
        w: ArrayLike,
    ) -> ArrayLike:
        # Measure angular velocity in body coordinates
        return x.w_B + self.noise * w


class GyroscopeConfig(StructConfig, type="basic"):
    noise: float = 0.0  # Noise standard deviation [rad/s]

    def build(self) -> Gyroscope:
        return Gyroscope(noise=self.noise)


@struct
class LineOfSight:
    """Basic line-of-sight sensor model"""

    noise: float = 0.0  # Noise standard deviation [rad]

    def __call__(
        self,
        vehicle: RigidBody.State,
        target: RigidBody.State,
        w: ArrayLike,
    ) -> ArrayLike:
        C_BN = dcm_from_quaternion(vehicle.att)

        r_N = target.p_N - vehicle.p_N  # Relative position in inertial coordinates
        r_B = C_BN @ r_N  # Relative position in body-fixed coordinates
        az = np.atan2(r_B[1], r_B[0])  # Azimuth angle
        el = np.arctan2(r_B[2], np.sqrt(r_B[0] ** 2 + r_B[1] ** 2))  # Elevation angle

        return np.hstack([az, el]) + self.noise * w


class LineOfSightConfig(StructConfig, type="basic"):
    noise: float = 0.0  # Noise standard deviation [rad]

    def build(self) -> LineOfSight:
        return LineOfSight(noise=self.noise)
