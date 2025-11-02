"""Spatial representations and kinematics/dynamics models."""

from ._rigid_body import (
    RigidBody,
    RigidBodyConfig,
)
from ._euler import (
    euler_kinematics,
    euler_to_dcm,
)
from ._quaternion import (
    dcm_to_quaternion,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_kinematics,
    quaternion_to_dcm,
    quaternion_to_euler,
)
from ._rotation import Rotation

__all__ = [
    "dcm_to_quaternion",
    "euler_kinematics",
    "euler_to_dcm",
    "euler_to_quaternion",
    "quaternion_kinematics",
    "quaternion_multiply",
    "quaternion_to_dcm",
    "quaternion_to_euler",
    "Rotation",
    "RigidBody",
    "RigidBodyConfig",
]
