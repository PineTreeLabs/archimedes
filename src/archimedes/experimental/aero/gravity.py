import abc

import numpy as np

from archimedes import struct

__all__ = [
    "GravityModel",
    "ConstantGravityModel",
]


class GravityModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, p_E):
        """Gravitational acceleration at the body CM in the inertial frame E

        Args:
            p_N position vector relative to the inertial frame N [m]

        Returns:
            g_N: gravitational acceleration in inertial frame N [m/s^2]
        """


@struct.pytree_node
class ConstantGravity(GravityModel):
    """Constant gravitational acceleration model

    This model assumes a constant gravitational acceleration vector
    in the +z direction (e.g. for a NED frame with "flat Earth" approximation)
    """

    g0: float = 9.81  # m/s^2

    def __call__(self, p_N):
        return np.hstack([0, 0, self.g0])


@struct.pytree_node
class PointGravity(GravityModel):
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

    @classmethod
    def from_lat_lon(cls, lat: float, lon: float):
        RE = 6.378e6  # Earth radius [m]
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        p_EN = np.array([
            RE * np.cos(lat) * np.cos(lon),
            RE * np.cos(lat) * np.sin(lon),
            RE * np.sin(lat),
        ])

        # TODO: Use a built-in DCM function
        R_EN = np.array([
            [-np.sin(lat)*np.cos(lon), -np.sin(lon), -np.cos(lat)*np.cos(lon)],
            [-np.sin(lat)*np.sin(lon),  np.cos(lon), -np.cos(lat)*np.sin(lon)],
            [np.cos(lat), 0, -np.sin(lat)],
        ])

        return cls(p_EN, R_EN)