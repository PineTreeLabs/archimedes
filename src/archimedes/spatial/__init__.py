"""Spatial representations and kinematics/dynamics models."""

from ._rigid_body import (
    RigidBody,
    RigidBodyConfig,
)
from ._attitude import (
    Rotation,
    euler_kinematics,
    euler_to_dcm,
    euler_to_quaternion,
    quaternion_kinematics,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_to_euler,
)


__all__ = [
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
