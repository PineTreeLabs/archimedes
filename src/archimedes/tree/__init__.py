"""Utilities for working with hierarchical "pytree" data structures."""

from . import struct
from ._flatten_util import ravel_pytree as ravel
from ._registry import (
    register_dataclass,
    register_pytree_node,
)
from ._tree_util import (
    tree_all as all,
)
from ._tree_util import (
    tree_flatten as flatten,
)
from ._tree_util import (
    tree_leaves as leaves,
)
from ._tree_util import (
    tree_map as map,
)
from ._tree_util import (
    tree_reduce as reduce,
)
from ._tree_util import (
    tree_structure as structure,
)
from ._tree_util import (
    tree_unflatten as unflatten,
)

__all__ = [
    "register_pytree_node",
    "register_dataclass",
    "flatten",
    "unflatten",
    "structure",
    "leaves",
    "map",
    "all",
    "reduce",
    "ravel",
    "struct",
]
