# This code modifies code from Flax
#
# Copyright (c) 2024 The Flax Authors
# Licensed under Apache License 2.0
# https://github.com/google/flax
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


"""Utilities for defining custom classes that can be used with pytree transformations."""

from collections.abc import Callable
import dataclasses
from dataclasses import InitVar, fields, replace

import functools
from typing import TypeVar
from typing_extensions import dataclass_transform

from ._registry import register_dataclass


__all__ = [
    "field",
    "pytree_node",
    "InitVar",
    "is_pytree_node",
    "fields",
    "replace",
]

_T = TypeVar('_T')


def field(static=False, *, metadata=None, **kwargs):
    """
    Create a field specification with pytree-related metadata.
    
    This function extends dataclasses.field() with additional metadata to control
    how fields are treated in pytree operations. Fields can be marked as static
    (metadata) or dynamic (data).
    Except for the `static` argument, all other arguments are passed directly to
    `dataclasses.field()`; see documentation for the dataclasses module for details.

    Parameters
    ----------
    static : bool, default=False
        If True, the field is treated as static metadata rather than dynamic data.
        Static fields are preserved during pytree transformations but not included
        in the flattened representation.
    metadata : dict, optional
        Additional metadata to include in the field specification. This will be
        merged with the 'static' setting.
    **kwargs : dict
        Additional keyword arguments passed to dataclasses.field().
    
    Returns
    -------
    field_object : dataclasses.Field
        A field specification with the appropriate metadata.

    Notes
    -----
    When to use:
    - To mark configuration parameters that shouldn't change during operations
    - To define default values or constructors for fields

    Static fields are not included when you flatten a pytree or apply transformations
    like `map`, but they are preserved in the structure and included when you
    reconstruct the object.

    Examples
    --------
    >>> import archimedes as arc
    >>> from archimedes import struct
    >>> import numpy as np
    >>> 
    >>> @struct.pytree_node
    >>> class Vehicle:
    ...     # Dynamic state variables (included in flattening)
    ...     position: np.ndarray
    ...     velocity: np.ndarray
    ...     
    ...     # Static configuration parameters (excluded from flattening)
    ...     mass: float = struct.field(static=True, default=1000.0)
    ...     drag_coef: float = struct.field(static=True, default=0.3)
    ...     
    ...     # With additional metadata
    ...     name: str = struct.field(
    ...         static=True,
    ...         default="vehicle",
    ...         metadata={"description": "Vehicle identifier"}
    ...     )
    >>> 
    >>> # Create an instance
    >>> car = Vehicle(
    ...     position=np.array([0.0, 0.0]),
    ...     velocity=np.array([10.0, 0.0]),
    ... )
    >>> 
    >>> # When flattened, only dynamic fields are included
    >>> flat, _ = arc.tree.flatten(car)
    >>> print(len(flat))  # Only position and velocity are included
    2
    
    See Also
    --------
    struct.pytree_node : Decorator for creating pytree-compatible classes
    register_dataclass : Register a dataclass as a pytree node
    """
    return dataclasses.field(
        metadata=(metadata or {}) | {'static': static},
        **kwargs,
    )


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def pytree_node(
    cls: _T | None = None,
    **kwargs,
) -> _T | Callable[[_T], _T]:
    """Construct a frozen dataclass registered as a pytree node"""
    # Support passing arguments to the decorator (e.g. @dataclass(kw_only=True))
    if cls is None:
        return functools.partial(pytree_node, **kwargs)

    # check if already recognized as a pytree node
    if '_arc_dataclass' in cls.__dict__:
        return cls

    if 'frozen' not in kwargs.keys():
        kwargs['frozen'] = True
    data_cls = dataclasses.dataclass(**kwargs)(cls)  # type: ignore
    meta_fields = []
    data_fields = []
    for field_info in dataclasses.fields(data_cls):
        if not field_info.init:
            continue
        is_static = field_info.metadata.get('static', False)
        if not is_static:
            data_fields.append(field_info.name)
        else:
            meta_fields.append(field_info.name)

    def replace(self, **updates) -> _T:
        """Returns a new object replacing the specified fields with new values."""
        return dataclasses.replace(self, **updates)

    data_cls.replace = replace

    register_dataclass(data_cls, data_fields, meta_fields)

    # add a _arc_dataclass flag to distinguish from regular dataclasses
    data_cls._arc_dataclass = True  # type: ignore[attr-defined]

    return data_cls  # type: ignore


def is_pytree_node(obj):
    """Returns True if the object is a pytree node."""
    return hasattr(obj, '_arc_dataclass')