from typing import Protocol

import numpy as np

from archimedes import struct, StructConfig, UnionConfig

__all__ = [
    "GravityModel",
    "ConstantGravity",
    "ConstantGravityConfig",
    "PointGravity",
    "PointGravityCartesianConfig",
    "PointGravityLatLonConfig",
    "GravityConfig",
]


class GravityModel(Protocol):
    def __call__(self, p_E: np.ndarray) -> np.ndarray:
        """Gravitational acceleration at the body CM in the inertial frame E

        Args:
            p_N position vector relative to the inertial frame N [m]

        Returns:
            g_N: gravitational acceleration in inertial frame N [m/s^2]
        """


@struct
class ConstantGravity:
    """Constant gravitational acceleration model

    This model assumes a constant gravitational acceleration vector
    in the +z direction (e.g. for a NED frame with "flat Earth" approximation)
    """

    g0: float = 9.81  # m/s^2

    def __call__(self, p_N: np.ndarray) -> np.ndarray:
        return np.hstack([0, 0, self.g0])


class ConstantGravityConfig(StructConfig, type="constant"):
    g0: float = 9.81  # m/s^2

    def build(self) -> ConstantGravity:
        return ConstantGravity(g0=self.g0)


def lla2eci(
    lat: float, lon: float, alt: float = 0.0, RE: float = 6.3781e6
) -> tuple[np.ndarray, np.ndarray]:
    """Convert latitude/longitude/altitude to Cartesian ECI coordinates.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        alt: Altitude above the surface in meters.
        RE: Earth radius in meters.

    Returns:
        p_EN: Cartesian coordinates [m]
        R_EN: Rotation matrix from NED to E
    """
    r = RE + alt  # Radius from Earth center [m]
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    p_EN = r * np.array(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )

    # TODO: Use a built-in DCM function
    R_EN = np.array(
        [
            [-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
            [-np.sin(lat) * np.sin(lon), np.cos(lon), -np.cos(lat) * np.sin(lon)],
            [np.cos(lat), 0, -np.sin(lat)],
        ]
    )

    return p_EN, R_EN


@struct
class PointGravity:
    """Point mass gravitational acceleration model

    This model assumes a point mass at the origin of the inertial frame E
    """

    p_EN: np.ndarray  # Relative position of N with respect to E (measured in E) [m]
    R_EN: np.ndarray  # Rotation from N to E
    mu: float = 3.986e14  # m^3/s^2

    def __call__(self, p_N):
        r_E = self.p_EN + self.R_EN @ p_N
        r = np.linalg.norm(r_E)
        g_E = -self.mu * r_E / r**3
        return self.R_EN.T @ g_E


# Example of a base class config to demonstrate inheritance.
# Note that the "type" field is not specified here.
class PointGravityConfig(StructConfig):
    mu: float = 3.986e14  # m^3/s^2


class PointGravityCartesianConfig(PointGravityConfig, type="point_cartesian"):
    p_EN: np.ndarray  # Relative position of N with respect to E (measured in E) [m]
    R_EN: np.ndarray  # Rotation from N to E

    def build(self) -> PointGravity:
        return PointGravity(self.p_EN, self.R_EN, mu=self.mu)


class PointGravityLatLonConfig(PointGravityConfig, type="point_latlon"):
    lat: float  # Latitude [deg]
    lon: float  # Longitude [deg]
    RE: float = 6.3781e6  # Earth radius [m]

    def build(self) -> PointGravity:
        p_EN, R_EN = lla2eci(self.lat, self.lon, RE=self.RE)
        return PointGravity(p_EN, R_EN, mu=self.mu)


GravityConfig = UnionConfig[
    ConstantGravityConfig,
    PointGravityCartesianConfig,
    PointGravityLatLonConfig,
]
