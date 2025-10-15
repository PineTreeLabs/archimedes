import warnings
import sys

from . import aero, coco, observers, signal
from .lqr import lqr_design
from .balanced_truncation import balanced_truncation


def __getattr__(name):
    if name == "spatial":
        warnings.warn(
            "Importing from archimedes.experimental.spatial is deprecated "
            "and will be removed in version 1.0. "
            "Please import from archimedes.spatial instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import the new module
        from archimedes import spatial

        # Cache it in sys.modules under the old path
        # This ensures subsequent imports use the same module object
        sys.modules["archimedes.experimental.spatial"] = spatial

        return spatial

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "coco",
    "aero",
    "observers",
    "signal",
    "spatial",
    "lqr_design",
    "balanced_truncation",
]
