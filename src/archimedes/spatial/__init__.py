"""Spatial representations and kinematics/dynamics models."""

from ._attitude import Attitude
from ._rigid_body import (
    RigidBody,
    RigidBodyConfig,
)
from ._euler import (
    euler_kinematics,
    euler_to_dcm,
)
from ._quaternion import (
    Quaternion,
    dcm_to_quaternion,
    euler_to_quaternion,
    quaternion_inverse,
    quaternion_kinematics,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_to_euler,
)

__all__ = [
    "Attitude",
    "dcm_to_quaternion",
    "euler_kinematics",
    "euler_to_dcm",
    "euler_to_quaternion",
    "Quaternion",
    "quaternion_inverse",
    "quaternion_kinematics",
    "quaternion_multiply",
    "quaternion_to_dcm",
    "quaternion_to_euler",
    "Quaternion",
    "RigidBody",
    "RigidBodyConfig",
]
