from typing import TYPE_CHECKING, Any, Tuple, TypeAlias

import casadi as cs
from numpy.typing import DTypeLike, NDArray

if TYPE_CHECKING:
    from ._core import SymbolicArray

    ArrayLike: TypeAlias = NDArray | SymbolicArray


Tree: TypeAlias = Any

# Type aliases for common types
CasadiMatrix: TypeAlias = cs.SX | cs.MX | cs.DM
ShapeLike: TypeAlias = Tuple[int, ...]

__all__ = [
    "NDArray",
    "ArrayLike",
    "Tree",
    "DTypeLike",
    "CasadiMatrix",
    "ShapeLike",
]
