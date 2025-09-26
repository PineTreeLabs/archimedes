# This code modifies code from JAX

# Copyright (c) 2021 The JAX Authors
# Licensed under Apache License 2.0
# https://github.com/jax-ml/jax

# Modifications and additions to the original code:
# Copyright (c) 2025 Pine Tree Labs, LLC
# Licensed under the GNU General Public License v3.0

# SPDX-FileCopyrightText: 2021 The JAX Authors
# SPDX-FileCopyrightText: 2025 Pine Tree Labs, LLC
# SPDX-License-Identifier: GPL-3.0-or-later

# As a combined work, use of this code requires compliance with the GNU GPL v3.0.
# The original license terms are included below for attribution:

# === Apache License 2.0 ===
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING

import numpy as np

from archimedes._core._array_impl import _result_type, array
from archimedes.tree._registry import unzip2
from archimedes.tree._tree_util import tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from archimedes.typing import ArrayLike, Tree


# Original: jax._src.util.HashablePartial
class HashablePartial:
    def __init__(self, f, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

        # Create a new call signature that doesn't include any of the provided args
        signature = inspect.signature(f)
        parameters = []
        for i, (name, param) in enumerate(signature.parameters.items()):
            if i < len(args) or name in kwargs:
                continue
            parameters.append(param)

        self.__signature__ = inspect.Signature(
            parameters=parameters,
            return_annotation=signature.return_annotation,
        )
        self._name = f.__name__
        if self._name.startswith("_"):
            self._name = self._name[1:]

    @property
    def __name__(self):
        return self._name

    def __eq__(self, other):
        return (
            type(other) is HashablePartial
            and self.f.__code__ == other.f.__code__
            and self.args == other.args
            and self.kwargs == other.kwargs
        )

    def __hash__(self):
        return hash(
            (
                self.f.__code__,
                self.args,
                tuple(sorted(self.kwargs.items(), key=lambda kv: kv[0])),
            ),
        )

    def __call__(self, *args, **kwargs):
        return self.f(*self.args, *args, **self.kwargs, **kwargs)


def ravel_tree(tree: Tree) -> tuple[ArrayLike, HashablePartial]:
    """
    Flatten a tree to a single 1D array.

    This function flattens a tree into a single 1D array by concatenating
    all leaf values (which must be arrays or scalars), and provides a function
    to reconstruct the original structure.

    Parameters
    ----------
    tree : Any
        A tree of arrays and scalars to flatten. A tree is a nested structure
        of containers (lists, tuples, dicts) and leaves (arrays or scalars).

    Returns
    -------
    flat_array : ndarray
        A 1D array containing all flattened leaf values concatenated together.
        The dtype is determined by promoting the dtypes of all leaf values.
        If the input tree is empty, a 1D empty array of dtype ``np.float32`` is
        returned.
    unravel : callable
        A function that takes a 1D array of the same length as ``flat_array`` and
        returns a tree with the same structure as the input ``tree``, with the
        values from the 1D array reshaped to match the original leaf shapes.

    Notes
    -----
    When to use:

    - When you need to convert structured data to a single flat vector for
      optimization, ODE solving, or other algorithms that work with flat arrays
    - As a more powerful alternative to :py:func:`flatten` when the leaf values
      themselves need to be flattened
    - When interfacing with external libraries that require flat arrays

    The resulting unravel function is specific to the structure of the input tree
    and expects an array of exactly the right length.

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> # Create a structured state
    >>> state = {"pos": np.array([0.0, 1.0, 2.0]), "vel": np.array([3.0, 4.0, 5.0])}
    >>>
    >>> # Flatten to a single vector
    >>> flat_state, unravel = arc.tree.ravel(state)
    >>> print(flat_state)
    [0. 1. 2. 3. 4. 5.]
    >>>
    >>> # Modify the flat array
    >>> flat_state = flat_state * 2
    >>>
    >>> # Reconstruct the original structure with modified values
    >>> new_state = unravel(flat_state)
    >>> print(new_state)
    {'pos': array([0., 2., 4.]), 'vel': array([6., 8., 10.])}
    >>>
    >>> # Use with ODE solvers that expect flat vectors
    >>> @arc.compile
    >>> def ode_rhs(t, state_flat):
    ...     # Unflatten the state vector to our structured state
    ...     state = unravel(state_flat)
    ...
    ...     # Compute state derivatives using structured data
    ...     pos_dot = state["vel"]
    ...     vel_dot = -state["pos"]  # Simple harmonic oscillator
    ...
    ...     # Return flattened derivatives
    ...     state_deriv = {"pos": pos_dot, "vel": vel_dot}
    ...     state_deriv_flat, _ = arc.tree.ravel(state_deriv)
    ...     return state_deriv_flat

    See Also
    --------
    flatten : Flatten a tree into a list of leaves and a treedef
    """
    leaves, treedef = tree_flatten(tree)
    flat, unravel_list = _ravel_list(leaves)
    return flat, HashablePartial(unravel_tree, treedef, unravel_list)


def unravel_tree(treedef, unravel_list, flat):
    return tree_unflatten(treedef, unravel_list(flat))


def _ravel_list(lst):
    if not lst:
        return np.array([], np.float32), lambda _: []
    from_dtypes = tuple(map(_result_type, lst))
    to_dtype = _result_type(*from_dtypes)
    sizes, shapes = unzip2((np.size(x), np.shape(x)) for x in lst)
    indices = tuple(np.cumsum(sizes).astype(int))
    shapes = tuple(shapes)

    # When there is more than one distinct input dtype, we perform type
    # conversions and produce a dtype-specific unravel function.
    def _ravel(e):
        return np.ravel(array(e, dtype=to_dtype))  # type: ignore

    raveled = np.atleast_1d(np.concatenate([_ravel(e) for e in lst]))
    unrav = HashablePartial(_unravel_list, indices, shapes, from_dtypes, to_dtype)
    return raveled, unrav


def _unravel_list(indices, shapes, from_dtypes, to_dtype, arr):
    chunks = np.split(arr, indices[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
        return [
            chunk.reshape(shape).astype(dtype)
            for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
        ]
