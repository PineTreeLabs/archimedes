from typing import TYPE_CHECKING, Any, Tuple, TypeAlias

import casadi as cs

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from ._core import SymbolicArray

    ArrayLike: TypeAlias = NDArray | SymbolicArray
    PyTree: TypeAlias = Any

    # Type aliases for common types
    CasadiMatrix: TypeAlias = cs.SX | cs.MX | cs.DM
    ShapeLike: TypeAlias = Tuple[int, ...]


__all__ = [
    "NDArray",
    "ArrayLike",
    "PyTree",
    "DTypeLike",
    "CasadiMatrix",
    "ShapeLike",
]
