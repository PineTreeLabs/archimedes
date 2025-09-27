"""Utilities for working with hierarchical tree-structured data."""

from ._flatten_util import ravel_tree as ravel
from ._registry import (
    register_dataclass,
    register_struct,
)
from ._tree_util import is_leaf
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
from ._struct import (
    field,
    struct,
    InitVar,
    is_struct,
    fields,
    replace,
)
from ._config import (
    StructConfig,
    UnionConfig,
)

__all__ = [
    "register_struct",
    "register_dataclass",
    "is_leaf",
    "flatten",
    "unflatten",
    "structure",
    "leaves",
    "map",
    "all",
    "reduce",
    "ravel",
    "struct",
    "field",
    "InitVar",
    "is_struct",
    "fields",
    "replace",
    "StructConfig",
    "UnionConfig",
]
