from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, Union

import casadi as cs
import numpy as np
from numpy.typing import DTypeLike
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from ._core import SymbolicArray

    ArrayLike: TypeAlias = Union[np.ndarray, SymbolicArray]


Tree: TypeAlias = Any

# Type aliases for common types
CasadiMatrix: TypeAlias = Union[cs.SX, cs.MX, cs.DM]
ShapeLike: TypeAlias = Tuple[int, ...]

__all__ = [
    "ArrayLike",
    "Tree",
    "DTypeLike",
    "CasadiMatrix",
    "ShapeLike",
]
