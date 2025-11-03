"""Spatial representations and kinematics/dynamics models."""
import warnings

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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
