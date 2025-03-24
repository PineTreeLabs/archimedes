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

import warnings

import numpy as np


from archimedes._core._array_impl import _result_type, array
from archimedes.tree._registry import unzip2
from archimedes.tree._tree_util import tree_flatten, tree_unflatten


# Original: jax._src.util.HashablePartial
class HashablePartial:
    def __init__(self, f, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __eq__(self, other):
        return (
           type(other) is HashablePartial and
            self.f.__code__ == other.f.__code__ and
            self.args == other.args and self.kwargs == other.kwargs
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


def ravel_pytree(pytree):
    """Ravel (flatten) a pytree of arrays down to a 1D array.

    Args:
        pytree: a pytree of arrays and scalars to ravel.

    Returns:
        A pair where the first element is a 1D array representing the flattened and
        concatenated leaf values, with dtype determined by promoting the dtypes of
        leaf values, and the second element is a callable for unflattening a 1D
        vector of the same length back to a pytree of the same structure as the
        input ``pytree``. If the input pytree is empty (i.e. has no leaves) then as
        a convention a 1D empty array of dtype float32 is returned in the first
        component of the output.

    TODO: Replace this link
    For details on dtype promotion, see
    https://jax.readthedocs.io/en/latest/type_promotion.html.

    """
    leaves, treedef = tree_flatten(pytree)
    flat, unravel_list = _ravel_list(leaves)
    return flat, HashablePartial(unravel_pytree, treedef, unravel_list)


def unravel_pytree(treedef, unravel_list, flat):
  return tree_unflatten(treedef, unravel_list(flat))


def _ravel_list(lst):
    if not lst: return np.array([], np.float32), lambda _: []
    from_dtypes = tuple(map(_result_type, lst))
    to_dtype = _result_type(*from_dtypes)
    sizes, shapes = unzip2((np.size(x), np.shape(x)) for x in lst)
    indices = tuple(np.cumsum(sizes).astype(int))
    shapes = tuple(shapes)

    # When there is more than one distinct input dtype, we perform type
    # conversions and produce a dtype-specific unravel function.
    ravel = lambda e: np.ravel(array(e, dtype=to_dtype))
    raveled = np.atleast_1d(np.concatenate([ravel(e) for e in lst]))
    unrav = HashablePartial(_unravel_list, indices, shapes, from_dtypes, to_dtype)
    return raveled, unrav


def _unravel_list(indices, shapes, from_dtypes, to_dtype, arr):
    chunks = np.split(arr, indices[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
        return [
            np.astype(chunk.reshape(shape), dtype)
            for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
        ]