"""Spatial representations and kinematics/dynamics models."""

import warnings

from ._attitude import (
    Attitude,
    EulerAngles,
    Quaternion,
)
from ._euler import (
    euler_kinematics,
    euler_to_dcm,
)
from ._quaternion import (
    dcm_to_quaternion,
    euler_to_quaternion,
    quaternion_inverse,
    quaternion_kinematics,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_to_euler,
)
from ._rigid_body import (
    RigidBody,
)

__all__ = [
    "Attitude",
    "RigidBody",
    "dcm_to_quaternion",
    "euler_kinematics",
    "euler_to_dcm",
    "euler_to_quaternion",
    "EulerAngles",
    "Quaternion",
    "quaternion_inverse",
    "quaternion_kinematics",
    "quaternion_multiply",
    "quaternion_to_dcm",
    "quaternion_to_euler",
    "Quaternion",
    "Velocity",
    "velocity",
]


def __getattr__(name):
    if name == "Rotation":
        warnings.warn(
            "The Rotation class is deprecated and will be removed in version 1.0. "
            "Please migrate to Quaternion instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ._rotation import Rotation

        return Rotation
