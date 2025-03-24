"""
This code modifies code from JAX

Copyright (c) 2021 The JAX Authors
Licensed under Apache License 2.0
https://github.com/jax-ml/jax

Modifications and additions to the original code:
Copyright (c) 2025 Jared Callaham
Licensed under the GNU General Public License v3.0

As a combined work, use of this code requires compliance with the GNU GPL v3.0.
The original license terms are included below for attribution:

=== Apache License 2.0 ===
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations
from typing import Any, Callable, TypeVar, Hashable, Iterable, Iterator, NamedTuple
from functools import reduce, partial
import itertools as it

from .registry import _registry, unzip2


T = TypeVar("T")


class PyTreeDef(NamedTuple):
    node_data: None | tuple[type, Hashable]
    children: tuple[PyTreeDef, ...]
    num_leaves: int

    def unflatten(self, xs: list[Any]) -> Any:
        return tree_unflatten(self, xs)
    
    def __repr__(self) -> str:
        stars = ["*"] * self.num_leaves
        star_tree = self.unflatten(stars)
        return (f"PyTreeDef({star_tree})").replace("'*'", "*")


LEAF = PyTreeDef(None, (), 1)


#
# Flatten/unflatten functions
#
def tree_flatten(
    x: Any, is_leaf: Callable[[Any], bool] | None = None
) -> tuple[list[Any], PyTreeDef]:
    children_iter, treedef = _tree_flatten(x, is_leaf)
    return list(children_iter), treedef


def _tree_flatten(
    x: Any, is_leaf: Callable[[Any], bool] | None
) -> tuple[Iterable, PyTreeDef]:
    if x is None:
        return [], LEAF

    _tree_flatten_leaf = partial(_tree_flatten, is_leaf=is_leaf)

    node_type = type(x)
    # If the node is a namedtuple, use the tuple flatten/unflatten functions
    if isinstance(x, tuple) and hasattr(x, '_fields'):
        node_type = tuple
    if node_type not in _registry or (is_leaf is not None and is_leaf(x)):
        return [x], LEAF

    children, node_metadata = _registry[node_type].to_iter(x)
    children_flat, child_trees = unzip2(map(_tree_flatten_leaf, children))
    flattened = list(it.chain.from_iterable(children_flat))

    node_data = (type(x), node_metadata)
    treedef = PyTreeDef(
        node_data=node_data,
        children=tuple(child_trees),
        num_leaves=len(flattened),
    )
    return flattened, treedef


def tree_unflatten(treedef: PyTreeDef, xs: list[Any]) -> Any:
    return _tree_unflatten(treedef, iter(xs))


def _tree_unflatten(treedef: PyTreeDef, xs: Iterator) -> Any:
    if treedef.node_data is None:
        try:
            return next(xs)
        except StopIteration:
            return None
    else:
        children = (_tree_unflatten(t, xs) for t in treedef.children)
        node_type, node_metadata = treedef.node_data

        # Special logic for NamedTuple classes
        if issubclass(node_type, tuple) and hasattr(node_type, "_fields"):
            return node_type(*children)

        return _registry[node_type].from_iter(node_metadata, children)


#
# Other utility functions
#

def tree_structure(
    tree: Any, is_leaf: Callable[[Any], bool] | None = None
) -> PyTreeDef:
    """
    Returns a PyTreeDef object that describes the structure of the input tree.
    """
    flat, treedef = tree_flatten(tree, is_leaf)
    return treedef


def tree_leaves(
    tree: Any, is_leaf: Callable[[Any], bool] | None = None
) -> list[Any]:
    """
    Returns the leaves in the tree.
    """
    flat, treedef = tree_flatten(tree, is_leaf)
    return flat


def tree_all(
    tree: Any, is_leaf: Callable[[Any], bool] | None = None
) -> bool:
    """
    Returns True if all leaves in the tree are True.
    """
    flat, treedef = tree_flatten(tree, is_leaf)
    print(flat)
    return all(flat)


def tree_map(
    f: Callable,
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None
) -> Any:
    """
    Maps the function f over each leaf in the tree
    """
    flat, treedef = tree_flatten(tree, is_leaf)
    flat = [flat]
    for r in rest:
        r_flat, r_treedef = tree_flatten(r, is_leaf)
        if treedef != r_treedef:
            raise ValueError(
                "Trees must have the same structure but got treedefs: "
                f"{treedef} and {r_treedef}"
            )
        flat.append(r_flat)

    flat = [f(*args) for args in zip(*flat)]
    return tree_unflatten(treedef, flat)


def tree_reduce(
    function: Callable[[T, Any], T],
    tree: Any,
    initializer: T,
    is_leaf: Callable[[Any], bool] | None = None,
) -> T:
    """Reduces the tree using the function f and the initializer."""
    flat, treedef = tree_flatten(tree, is_leaf)
    print(flat)
    return reduce(function, flat, initializer)

