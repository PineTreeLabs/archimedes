import warnings

warnings.warn(
    "Importing from archimedes.experimental.spatial is deprecated "
    "and will be removed in version 1.0. "
    "Please import from archimedes.spatial instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from archimedes.spatial import *
