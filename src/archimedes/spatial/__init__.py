"""Spatial representations and kinematics/dynamics models."""

from ._rigid_body import (
    RigidBody,
    RigidBodyConfig,
)
from ._attitude import (
    Rotation,
    euler_to_dcm,
    euler_kinematics,
)


__all__ = [
    "Rotation",
    "RigidBody",
    "RigidBodyConfig",
    "euler_kinematics",
    "euler_to_dcm",
]
