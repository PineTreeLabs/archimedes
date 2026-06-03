"""Utilities for working with hierarchical tree-structured data."""
# Re-exports from structree
# github.com/pinetreelabs/structree

from structree import (
    InitVar,
    StructConfig,
    UnionConfig,
    all,
    field,
    fields,
    flatten,
    is_leaf,
    is_struct,
    leaves,
    map,
    ravel,
    reduce,
    register_dataclass,
    register_struct,
    replace,
    struct,
    structure,
    unflatten,
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
