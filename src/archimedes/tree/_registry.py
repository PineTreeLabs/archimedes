# This code modifies code from JAX
#
# Copyright (c) 2021 The JAX Authors
# Licensed under Apache License 2.0
# https://github.com/jax-ml/jax
#
# Modifications and additions to the original code:
# Copyright (c) 2025 Jared Callaham
# Licensed under the GNU General Public License v3.0
#
# As a combined work, use of this code requires compliance with the GNU GPL v3.0.
# The original license terms are included below for attribution:
#
# === Apache License 2.0 ===
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creation and management of types recognized as pytree nodes"""

from __future__ import annotations
from typing import NamedTuple, Any, TypeVar, Sequence
from collections import OrderedDict
from collections.abc import Callable
import dataclasses


Typ = TypeVar("Typ", bound=type[Any])


class _RegistryEntry(NamedTuple):
    to_iter: Callable[[Any], tuple[tuple[Any, ...], Any]]
    from_iter: Callable[[Any, tuple[Any, ...]], Any]


_registry: dict[type, _RegistryEntry] = {}


def unzip2(pairs):
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2


def register_pytree_node(
    ty: type, to_iter: Callable, from_iter: Callable
) -> None:
    """Register a type as a pytree node

    `to_iter` should accept an instance of the type and return a
    tuple of `(children, aux_data)`, where `aux_data` is any auxiliary
    metadata that is not part of the pytree structure, and `children` is
    an iterable of the children of the pytree node.

    `from_iter` should accept `aux_data` and an iterable of the children
    and return an instance of the type.
    """
    _registry[ty] = _RegistryEntry(to_iter, from_iter)


register_pytree_node(None, lambda x: (None, None), lambda _, xs: None)
register_pytree_node(tuple, lambda t: (t, None), lambda _, xs: tuple(xs))
register_pytree_node(list, lambda l: (l, None), lambda _, xs:  list(xs))

# dict
def _dict_to_iter(d: dict):
    keys, vals = unzip2(sorted(d.items()))
    return map(tuple, (vals, keys))

def _dict_from_iter(keys, vals):
    return dict(zip(keys, vals))

register_pytree_node(dict, _dict_to_iter, _dict_from_iter)

# OrderedDict
def _od_from_iter(keys, vals):
    return OrderedDict(zip(keys, vals))

register_pytree_node(OrderedDict, _dict_to_iter, _od_from_iter)


def register_dataclass(
    nodetype: Typ,
    data_fields: Sequence[str] | None = None,
    meta_fields: Sequence[str] | None = None,
    drop_fields: Sequence[str] = (),
) -> Typ:
    if data_fields is None or meta_fields is None:
        if (data_fields is None) != (meta_fields is None):
            raise TypeError("register_dataclass: data_fields and meta_fields must both be specified"
                            f" when either is specified. Got {data_fields=} {meta_fields=}.")
        if not dataclasses.is_dataclass(nodetype):
            raise TypeError("register_dataclass: data_fields and meta_fields are required when"
                            f" nodetype is not a dataclass. Got {nodetype=}.")
        data_fields = [f.name for f in dataclasses.fields(nodetype)
                    if not f.metadata.get('static', False)]
        meta_fields = [f.name for f in dataclasses.fields(nodetype)
                    if f.metadata.get('static', False)]


    assert meta_fields is not None
    assert data_fields is not None

    # Store inputs as immutable tuples in this scope, because we close over them
    # for later evaluation. This prevents potentially confusing behavior if the
    # caller were to pass in lists that are later mutated.
    meta_fields = tuple(meta_fields)
    data_fields = tuple(data_fields)

    if dataclasses.is_dataclass(nodetype):
        init_fields = {f.name for f in dataclasses.fields(nodetype) if f.init}
        init_fields.difference_update(*drop_fields)
        if {*meta_fields, *data_fields} != init_fields:
            msg = (
                "data_fields and meta_fields must include all dataclass fields with"
                " ``init=True`` and only them."
            )
            if missing := init_fields - {*meta_fields, *data_fields}:
                msg += (
                    f" Missing fields: {missing}. Add them to drop_fields to suppress"
                    " this error."
                )
            if unexpected := {*meta_fields, *data_fields} - init_fields:
                msg += f" Unexpected fields: {unexpected}."
            raise ValueError(msg)

    def unflatten_func(meta, data):
        meta_args = tuple(zip(meta_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)
        return nodetype(**kwargs)

    def flatten_func(x):
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta
    
    register_pytree_node(nodetype, flatten_func, unflatten_func)
    return nodetype

