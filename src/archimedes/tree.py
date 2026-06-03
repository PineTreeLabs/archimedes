"""Utilities for working with hierarchical tree-structured data."""
# Re-exports from structree
# github.com/pinetreelabs/structree

from structree import (
    StructConfig,
    UnionConfig,
    ravel,
    register_dataclass,
    register_struct,
    InitVar,
    field,
    fields,
    is_struct,
    replace,
    struct,
    is_leaf,
    all,
    flatten,
    leaves,
    map,
    reduce,
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
