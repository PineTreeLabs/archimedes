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

import itertools as it
from functools import partial, reduce
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    NamedTuple,
    TypeVar,
)

import numpy as np

from ._registry import _registry, unzip2

if TYPE_CHECKING:
    from ..typing import ArrayLike, PyTree

    T = TypeVar("T", bound=PyTree)
    V = TypeVar("V")


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
    x: PyTree, is_leaf: Callable[[Any], bool] | None = None
) -> tuple[list[ArrayLike], PyTreeDef]:
    """
    Flatten a pytree into a list of leaves and a treedef.

    This function recursively traverses the pytree and extracts all leaf values
    while recording the structure. This is useful when you need to apply operations
    to all leaf values or convert structured data to a flat representation.

    Parameters
    ----------
    x : PyTree
        A pytree to be flattened. A pytree is a nested structure of containers
        (lists, tuples, dicts, etc) and leaves (arrays, scalars, objects not registered
        as pytrees).
    is_leaf : callable, optional
        A function that takes a pytree node as input and returns a boolean
        indicating whether it should be considered a leaf. If not provided,
        the default leaf types are used.

    Returns
    -------
    leaves : list
        A list of all leaf values from the pytree.
    treedef : PyTreeDef
        A structure definition that can be used to reconstruct the original
        pytree using `unflatten`.

    Notes
    -----
    A pytree is defined as a nested structure of:
    - Containers: recognized container types like lists, tuples, and dictionaries
    - Leaves: arrays, scalars, or custom objects not recognized as containers

    When to use:
    - When you need to extract all leaf values from a nested structure
    - When you need to convert between structured and flat representations

    When converting to/from a flat vector it will typically be more convenient to
    use `tree.ravel` instead of this function.

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> # Simple pytree with nested containers
    >>> data = {"a": np.array([1.0, 2.0]), "b": {"c": np.array([3.0])}}
    >>>
    >>> # Flatten the pytree
    >>> leaves, treedef = arc.tree.flatten(data)
    >>> print(leaves)
    [array([1., 2.]), array([3.])]
    >>>
    >>> # Use treedef to reconstruct the pytree
    >>> reconstructed = arc.tree.unflatten(treedef, leaves)
    >>> print(reconstructed)
    {'a': array([1., 2.]), 'b': {'c': array([3.])}}

    See Also
    --------
    unflatten : Reconstruct a pytree from leaves and a treedef
    leaves : Extract just the leaf values from a pytree
    structure : Extract just the structure from a pytree
    ravel : Flatten a pytree into a single 1D array
    """
    children_iter, treedef = _tree_flatten(x, is_leaf)
    return list(children_iter), treedef


def _tree_flatten(
    x: PyTree, is_leaf: Callable[[Any], bool] | None
) -> tuple[Iterable, PyTreeDef]:
    if x is None:
        return [], LEAF

    _tree_flatten_leaf = partial(_tree_flatten, is_leaf=is_leaf)

    node_type = type(x)
    # If the node is a namedtuple, use the tuple flatten/unflatten functions
    if isinstance(x, tuple) and hasattr(x, "_fields"):
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


def tree_unflatten(treedef: PyTreeDef, xs: list[ArrayLike]) -> PyTree:
    """
    Reconstruct a pytree from a list of leaves and a treedef.

    This function is the inverse of `flatten`. It takes a list of leaf values
    and a tree definition, and reconstructs the original pytree structure.

    Parameters
    ----------
    treedef : PyTreeDef
        A tree definition, typically produced by `flatten` or `structure`.
    xs : list[ArrayLike]
        A list of leaf values to be placed in the reconstructed pytree.
        The length must match the number of leaves in the treedef.

    Returns
    -------
    pytree : PyTree
        The reconstructed pytree with the same structure as defined by treedef
        and with leaf values from xs.

    Notes
    -----
    When to use:
    - To reconstruct a structured data object after operating on flattened values

    When converting to/from a flat vector it will typically be more convenient to
    use `tree.ravel` instead of this function.

    Raises
    ------
    ValueError
        If the number of leaves in xs doesn't match the expected number in treedef.

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> # Original pytree
    >>> data = {"positions": np.array([1.0, 2.0]), "velocities": np.array([3.0, 4.0])}
    >>>
    >>> # Flatten the pytree
    >>> leaves, treedef = arc.tree.flatten(data)
    >>>
    >>> # Transform the leaves (e.g., multiply by 2)
    >>> new_leaves = [leaf * 2 for leaf in leaves]
    >>>
    >>> # Reconstruct the pytree with the new leaves
    >>> new_data = arc.tree.unflatten(treedef, new_leaves)
    >>> print(new_data)
    {'positions': array([2., 4.]), 'velocities': array([6., 8.])}

    See Also
    --------
    flatten : Flatten a pytree into a list of leaves and a treedef
    structure : Extract just the structure from a pytree
    ravel : Flatten a pytree into a single 1D array
    """
    return _tree_unflatten(treedef, iter(xs))


def _tree_unflatten(treedef: PyTreeDef, xs: Iterator) -> PyTree:
    if treedef.node_data is None:
        try:
            return next(xs)
        except StopIteration:
            return None
    else:
        children = tuple(_tree_unflatten(t, xs) for t in treedef.children)
        node_type, node_metadata = treedef.node_data

        # Special logic for NamedTuple classes
        if issubclass(node_type, tuple) and hasattr(node_type, "_fields"):
            return node_type(*children)

        return _registry[node_type].from_iter(node_metadata, children)


#
# Other utility functions
#


def tree_structure(
    tree: PyTree, is_leaf: Callable[[Any], bool] | None = None
) -> PyTreeDef:
    """
    Extract the structure of a pytree without the leaf values.

    This function returns a PyTreeDef that describes the structure of the pytree,
    which can be used with `unflatten` to reconstruct a pytree with new leaf values.

    Parameters
    ----------
    tree : PyTree
        A pytree whose structure is to be determined.
    is_leaf : callable, optional
        A function that takes a pytree node as input and returns a boolean
        indicating whether it should be considered a leaf. If not provided,
        the default leaf types (arrays and scalars) are used.

    Returns
    -------
    treedef : PyTreeDef
        A tree definition that describes the structure of the input pytree.

    Notes
    -----
    When to use:
    - When you need to extract just the structure of a pytree for later use
    - When you want to create a template structure that can be filled with
      different leaf values
    - When you need to compare the structures of two pytrees

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> # Create a structured state
    >>> state = {"pos": np.array([0.0, 1.0]), "vel": np.array([2.0, 3.0])}
    >>>
    >>> # Extract the structure
    >>> treedef = arc.tree.structure(state)
    >>> print(treedef)
    PyTreeDef({'pos': *, 'vel': *})
    >>>
    >>> # Create a new state with the same structure but different values
    >>> zeros = [np.zeros_like(leaf) for leaf in arc.tree.leaves(state)]
    >>> initial_state = arc.tree.unflatten(treedef, zeros)
    >>> print(initial_state)
    {'pos': array([0., 0.]), 'vel': array([0., 0.])}

    See Also
    --------
    flatten : Flatten a pytree into a list of leaves and a treedef
    unflatten : Reconstruct a pytree from leaves and a treedef
    ravel : Flatten a pytree into a single 1D array
    """
    flat, treedef = tree_flatten(tree, is_leaf)
    return treedef


def tree_leaves(
    tree: PyTree, is_leaf: Callable[[Any], bool] | None = None
) -> list[ArrayLike]:
    """
    Extract all leaf values from a pytree.

    This function traverses the pytree and returns a list of all leaf values
    without the structure information.

    Parameters
    ----------
    tree : PyTree
        A pytree from which to extract leaves. A pytree is a nested structure of
        containers (lists, tuples, dicts, etc) and leaves (arrays, scalars, objects
        not registered as pytrees).
    is_leaf : callable, optional
        A function that takes a pytree node as input and returns a boolean
        indicating whether it should be considered a leaf. If not provided,
        the default leaf types (arrays and scalars) are used.

    Returns
    -------
    leaves : list
        A list of all leaf values from the pytree.

    Notes
    -----
    When to use:
    - For quick access to all data in a nested structure when you don't need
      to reconstruct the data structure later

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> # Create a structured data object
    >>> data = {"params": {"w": np.array([1.0, 2.0]), "b": 0.5},
    ...         "state": np.array([3.0, 4.0])}
    >>>
    >>> # Extract all leaf values
    >>> leaf_values = arc.tree.leaves(data)
    >>> print(leaf_values)
    [0.5, array([1., 2.]), array([3., 4.])]

    See Also
    --------
    flatten : Flatten a pytree into a list of leaves and a treedef
    ravel : Flatten a pytree into a single 1D array
    map : Apply a function to each leaf in a pytree
    """
    flat, treedef = tree_flatten(tree, is_leaf)
    return flat


def tree_all(tree: PyTree, is_leaf: Callable[[Any], bool] | None = None) -> bool:
    """
    Check if all leaves in the pytree evaluate to True.

    This function traverses the pytree and checks if all leaf values are truthy,
    similar to Python's built-in `all()` function but operating on all leaves
    of a pytree.

    Parameters
    ----------
    tree : PyTree
        A pytree to check. A pytree is a nested structure of containers
        (lists, tuples, dicts) and leaves (arrays, scalars, objects
        not registered as pytrees).
    is_leaf : callable, optional
        A function that takes a pytree node as input and returns a boolean
        indicating whether it should be considered a leaf. If not provided,
        the default leaf types (arrays and scalars) are used.

    Returns
    -------
    result : bool
        True if all leaves in the pytree evaluate to True, False otherwise.

    Notes
    -----
    When to use:
    - To check if all values in a structured object meet a certain condition
      (after applying `map` with a condition function)
    - To validate that all components of a model or state are initialized
    - As a convenient way to check properties across nested structures

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> # Check if all values are positive
    >>> data = {"a": np.array([1.0, 2.0]), "b": {"c": np.array([3.0])}}
    >>> is_positive = arc.tree.map(lambda x: x > 0, data)
    >>> all_positive = arc.tree.all(is_positive)
    >>> print(all_positive)
    True
    >>>
    >>> # Check if all values are greater than 1.5
    >>> is_greater = arc.tree.map(lambda x: x > 1.5, data)
    >>> all_greater = arc.tree.all(is_greater)
    >>> print(all_greater)
    False

    See Also
    --------
    map : Apply a function to each leaf in a pytree
    leaves : Extract just the leaf values from a pytree
    """
    flat, treedef = tree_flatten(tree, is_leaf)
    return np.all(map(np.all, flat))  # type: ignore


def tree_map(
    f: Callable,
    tree: T,
    *rest: tuple[T, ...],
    is_leaf: Callable[[Any], bool] | None = None,
) -> T:
    """
    Apply a function to each leaf in a pytree.

    This function traverses the pytree and applies the function `f` to each leaf,
    returning a new pytree with the same structure but transformed leaf values.
    If additional pytrees are provided, the function is applied to corresponding
    leaves from all pytrees.

    Parameters
    ----------
    f : callable
        A function to apply to each leaf. When multiple trees are provided,
        this function should accept as many arguments as there are trees.
    tree : Any
        The main pytree whose structure will be followed.
    *rest : Any
        Additional pytrees with exactly the same structure as the first tree.
    is_leaf : callable, optional
        A function that takes a pytree node as input and returns a boolean
        indicating whether it should be considered a leaf. If not provided,
        the default leaf types (arrays and scalars) are used.

    Returns
    -------
    mapped_tree : Any
        A new pytree with the same structure as `tree` but with leaf values
        transformed by function `f`.

    Notes
    -----
    When to use:
    - To transform data in a structured object without changing its structure
    - To perform element-wise operations on corresponding elements of multiple pytrees
    - As an alternative to manually looping through nested structures
    - To apply the same operation to all arrays in a complex model

    Raises
    ------
    ValueError
        If additional pytrees do not have exactly the same structure as the main tree.

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> # Single pytree example
    >>> state = {"pos": np.array([1.0, 2.0]), "vel": np.array([3.0, 4.0])}
    >>>
    >>> # Double all values
    >>> doubled = arc.tree.map(lambda x: x * 2, state)
    >>> print(doubled)
    {'pos': array([2., 4.]), 'vel': array([6., 8.])}
    >>>
    >>> # Multiple pytrees example
    >>> state1 = {"pos": np.array([1.0, 2.0]), "vel": np.array([3.0, 4.0])}
    >>> state2 = {"pos": np.array([5.0, 6.0]), "vel": np.array([7.0, 8.0])}
    >>>
    >>> # Add corresponding leaves
    >>> combined = arc.tree.map(lambda x, y: x + y, state1, state2)
    >>> print(combined)
    {'pos': array([6., 8.]), 'vel': array([10., 12.])}

    See Also
    --------
    flatten : Flatten a pytree into a list of leaves and a treedef
    leaves : Extract just the leaf values from a pytree
    """
    leaves, treedef = tree_flatten(tree, is_leaf)
    flat = [leaves]
    for r in rest:
        r_flat, r_treedef = tree_flatten(r, is_leaf)
        if treedef != r_treedef:
            raise ValueError(
                "Trees must have the same structure but got treedefs: "
                f"{treedef} and {r_treedef}"
            )
        flat.append(r_flat)

    flat = [f(*args) for args in zip(*flat)]
    tree_out: T = tree_unflatten(treedef, flat)  # type: ignore
    return tree_out


def tree_reduce(
    function: Callable[[V, ArrayLike], V],
    tree: PyTree,
    initializer: V,
    is_leaf: Callable[[Any], bool] | None = None,
) -> V:
    """
    Reduce a pytree to a single value using a function and initializer.

    This function traverses the pytree, applying the reduction function to
    each leaf and an accumulator, similar to Python's built-in `reduce()`
    but operating on all leaves of a pytree.

    Parameters
    ----------
    function : callable
        A function of two arguments: (accumulated_result, leaf_value) that
        returns a new accumulated result. The function should be commutative
        and associative to ensure results are independent of traversal order.
    tree : PyTree
        A pytree to reduce. A pytree is a nested structure of containers
        (lists, tuples, dicts) and leaves (arrays or scalars).
    initializer : Any
        The initial value for the accumulator.
    is_leaf : callable, optional
        A function that takes a pytree node as input and returns a boolean
        indicating whether it should be considered a leaf. If not provided,
        the default leaf types (arrays and scalars) are used.

    Returns
    -------
    result : Any
        The final accumulated value after applying the function to all leaves.

    Notes
    -----
    When to use:
    - To compute aggregate values (sum, product, etc.) across all leaf values
    - To collect statistics from a structured model
    - To implement custom reduction operations on complex data structures

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> # Sum all values in a nested structure
    >>> data = {"a": np.array([1.0, 2.0]), "b": {"c": np.array([3.0])}}
    >>>
    >>> # Compute the sum
    >>> def sum_leaf(acc, leaf):
    ...     return acc + sum(leaf)
    >>>
    >>> total = arc.tree.reduce(sum_leaf, data, 0.0)
    >>> print(total)
    6.0
    >>>
    >>> # Find the maximum value
    >>> def max_leaf(acc, leaf):
    ...     return np.fmax(acc, np.max(leaf))
    >>>
    >>> maximum = arc.tree.reduce(max_leaf, data, -np.inf)
    >>> print(maximum)
    3.0

    See Also
    --------
    map : Apply a function to each leaf in a pytree
    leaves : Extract just the leaf values from a pytree
    """
    flat, _treedef = tree_flatten(tree, is_leaf)
    return reduce(function, flat, initializer)
