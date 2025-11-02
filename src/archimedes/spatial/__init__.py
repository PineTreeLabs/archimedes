"""Spatial representations and kinematics/dynamics models."""

from ._rigid_body import (
    RigidBody,
    RigidBodyConfig,
)
from ._attitude import (
    Rotation,
    euler_kinematics,
    euler_to_dcm,
    quat_to_dcm,
)


__all__ = [
    "euler_kinematics",
    "euler_to_dcm",
    "quat_to_dcm",
    "Rotation",
    "RigidBody",
    "RigidBodyConfig",
]
