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

    Outputs the specific force (proper acceleration) measured by the accelerometer in units
    of g's as defined by standard gravity on Earth's surface
    """

    gravity: GravityModel = field(default_factory=ConstantGravity)
    g0: float = 9.80665  # Standard gravity [m/s^2]
    noise: float = 0.0  # Noise standard deviation [g's]

    def __call__(
        self,
        x: RigidBody.State,
        a_B: np.ndarray,
        w: np.ndarray,
    ) -> np.ndarray:
        g_N = self.gravity(x.pos)  # Inertial gravity vector

        C_BN = x.att.as_matrix()

        # Measure inertial acceleration in body coordinates
        a_meas_B = (a_B - C_BN @ g_N) / self.g0  # "proper" inertial acceleration

        return a_meas_B + self.noise * w


class AccelerometerConfig(StructConfig, type="basic"):
    gravity: GravityConfig = field(default_factory=ConstantGravityConfig)
    g0: float = 9.80665  # Standard gravity [m/s^2]
    noise: float = 0.0  # Noise standard deviation [g's]

    def build(self) -> Accelerometer:
        return Accelerometer(gravity=self.gravity.build(), g0=self.g0, noise=self.noise)

@struct
class Gyroscope:
    """Basic three-axis gyroscope model

    Currently assumes that the gyro is located at the center of mass (CM) of the vehicle.
    """

    noise: float = 0.0  # Noise standard deviation [rad/s]

    def __call__(
        self,
        x: RigidBody.State,
        w: np.ndarray,
    ) -> np.ndarray:
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
        w: np.ndarray,
    ) -> np.ndarray:
        C_BN = vehicle.att.as_matrix()

        r_N = target.pos - vehicle.pos  # Relative position in inertial coordinates
        r_B = C_BN @ r_N  # Relative position in body-fixed coordinates
        az = np.atan2(r_B[1], r_B[0])  # Azimuth angle
        el = np.arctan2(r_B[2], np.sqrt(r_B[0] ** 2 + r_B[1] ** 2))  # Elevation angle

        return np.hstack([az, el]) + self.noise * w


class LineOfSightConfig(StructConfig, type="basic"):
    noise: float = 0.0  # Noise standard deviation [rad]

    def build(self) -> LineOfSight:
        return LineOfSight(noise=self.noise)
