import warnings

from .gravity import (
    GravityModel,
    ConstantGravity,
    ConstantGravityConfig,
    PointGravity,
    PointGravityCartesianConfig,
    PointGravityLatLonConfig,
    GravityConfig,
)
from .atmosphere import (
    AtmosphereModel,
    ConstantAtmosphere,
    ConstantAtmosphereConfig,
    StandardAtmosphere1976,
    StandardAtmosphere1976Config,
    AtmosphereConfig,
)
from .sensors import (
    Accelerometer,
    AccelerometerConfig,
    Gyroscope,
    GyroscopeConfig,
    LineOfSight,
    LineOfSightConfig,
)
from .frames import wind_frame

_deprecated = {
    "RigidBody",
    "RigidBodyConfig",
    "euler_kinematics",
    "dcm_from_euler",
}


def __getattr__(name):
    if name in _deprecated:
        warnings.warn(
            f"Importing {name} from archimedes.experimental.aero is deprecated "
            "and will be removed in version 1.0. "
            "Please import from archimedes.spatial instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(__import__("archimedes.spatial", fromlist=[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *_deprecated,
    "wind_frame",
    "GravityModel",
    "ConstantGravity",
    "ConstantGravityConfig",
    "PointGravity",
    "PointGravityCartesianConfig",
    "PointGravityLatLonConfig",
    "GravityConfig",
    "AtmosphereModel",
    "ConstantAtmosphere",
    "ConstantAtmosphereConfig",
    "StandardAtmosphere1976",
    "StandardAtmosphere1976Config",
    "AtmosphereConfig",
    "Accelerometer",
    "AccelerometerConfig",
    "Gyroscope",
    "GyroscopeConfig",
    "LineOfSight",
    "LineOfSightConfig",
]
