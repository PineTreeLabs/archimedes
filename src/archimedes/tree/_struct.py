# This code modifies code from Flax
#
# Copyright (c) 2024 The Flax Authors
# Licensed under Apache License 2.0
# https://github.com/google/flax
#
# Modifications and additions to the original code:
# Copyright (c) 2025 Pine Tree Labs, LLC
# Licensed under the GNU General Public License v3.0

# SPDX-FileCopyrightText: 2024 The Flax Authors
# SPDX-FileCopyrightText: 2025 Pine Tree Labs, LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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

"""
Utilities for defining custom classes that can be used with tree transformations.

This module provides tools for creating structured data types that work seamlessly
with Archimedes' tree functions. These tools are built on Python's dataclasses
with extensions for tree-specific behavior.

The module re-exports several names from the dataclasses module:

InitVar : Type annotation for init-only variables in dataclasses
    Used to mark fields that should be passed to __post_init__ but not stored.

fields : Function to retrieve fields of a dataclass
    Returns a list of Field objects representing the fields of the dataclass.
    This is useful for introspection and validation of dataclass instances.

replace : Function to create a new dataclass instance with updated fields
    For tree nodes created with @struct, use the .replace() method instead.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import InitVar, fields, replace
from typing import Any, TypeVar

from typing_extensions import dataclass_transform

from ._registry import register_dataclass

__all__ = [
    "field",
    "struct",
    "InitVar",
    "is_struct",
    "fields",
    "replace",
]


T = TypeVar("T")


def field(
    static: bool = False,
    *,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dataclasses.Field:
    """
    Create a field specification with struct-related metadata.

    This function extends :py:func:`dataclasses.field()` with additional metadata to
    control how fields are treated in tree operations. Fields can be marked as static
    (metadata) or dynamic (data). Except for the `static` argument, all other arguments
    are passed directly to :py:func:`dataclasses.field()`; see documentation for the
    :py:mod:`dataclasses` module for details.

    Parameters
    ----------
    static : bool, default=False
        If True, the field is treated as static metadata rather than dynamic data.
        Static fields are preserved during tree transformations but not included
        in the flattened representation.
    metadata : dict, optional
        Additional metadata to include in the field specification. This will be
        merged with the ``static`` setting.
    **kwargs : dict
        Additional keyword arguments passed to :py:func:`dataclasses.field()`.

    Returns
    -------
    field_object : dataclasses.Field
        A field specification with the appropriate metadata.

    Notes
    -----
    When to use:

    - To mark configuration parameters that shouldn't change during operations
    - To define default values or constructors for fields

    Static fields are not included when you flatten tree-structured data or apply
    transformations like ``map``, but they are preserved in the structure and included
    when you reconstruct the object.

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> @arc.struct
    >>> class Vehicle:
    ...     # Dynamic state variables (included in flattening)
    ...     position: np.ndarray
    ...     velocity: np.ndarray
    ...
    ...     # Static configuration parameters (excluded from flattening)
    ...     mass: float = arc.field(static=True, default=1000.0)
    ...     drag_coef: float = arc.field(static=True, default=0.3)
    ...
    ...     # With additional metadata
    ...     name: str = arc.field(
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
    struct : Decorator for creating tree-compatible dataclasses
    register_dataclass : Register a dataclass as compatible with tree operations
    """
    f: dataclasses.Field = dataclasses.field(
        metadata=(metadata or {}) | {"static": static},
        **kwargs,
    )
    return f


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def struct(cls: T | None = None, **kwargs) -> T | Callable:
    """
    Decorator to convert a class into a tree-compatible frozen dataclass.

    This decorator creates a structured data class that can be seamlessly used
    with Archimedes' tree functions. The class will be registered with the tree
    system, allowing its instances to be flattened, mapped over, and transformed
    while preserving its structure.

    Parameters
    ----------
    cls : type, optional
        The class to convert into a tree-compatible dataclass.
    **kwargs : dict
        Additional keyword arguments passed to dataclasses.dataclass().
        By default, ``frozen=True`` is set unless explicitly overridden.

    Returns
    -------
    decorated_class : type
        The decorated class, now a frozen dataclass registered as tree-compatible.

    Notes
    -----

    The "frozen" attribute makes the class immutable, meaning that once an instance
    is created, its fields cannot be modified. This is useful for ensuring that
    the state of the object remains consistent during operations. The ``replace()``
    method allows you to create modified copies of the object with new values for
    specific fields.

    Fields are automatically classified as either "data" (dynamic values that
    change during operations) or "static" (configuration parameters). By default,
    all fields are treated as data unless marked with ``field(static=True)``.

    The decorated class:

    - Is frozen (immutable) by default
    - Has a ``replace()`` method for creating modified copies
    - Will be properly handled by ``tree.flatten()``, ``tree.map()``, etc.
    - Can be nested within other tree nodes (structs, dicts, tuples, etc.)

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> @arc.struct
    >>> class Vehicle:
    ...     # Dynamic state variables (included in transformations)
    ...     position: np.ndarray
    ...     velocity: np.ndarray
    ...
    ...     # Static configuration parameters (preserved during transformations)
    ...     mass: float = arc.field(static=True, default=1000.0)
    ...     drag_coef: float = arc.field(static=True, default=0.3)
    ...
    ...     def kinetic_energy(self):
    ...         return 0.5 * self.mass * np.sum(self.velocity**2)
    >>>
    >>> # Create an instance
    >>> car = Vehicle(
    ...     position=np.array([0.0, 0.0]),
    ...     velocity=np.array([10.0, 0.0]),
    ... )
    >>>
    >>> # Create a modified copy
    >>> car2 = car.replace(position=np.array([5.0, 0.0]))
    >>>
    >>> # Apply a transformation (only to dynamic fields)
    >>> scaled = arc.tree.map(lambda x: x * 2, car)
    >>> print(scaled.position)    # [0. 0.] -> [0. 0.]
    >>> print(scaled.velocity)    # [10. 0.] -> [20. 0.]
    >>> print(scaled.mass)        # 1000.0 (unchanged)
    >>>
    >>> # Nested structs
    >>> @arc.struct
    >>> class System:
    ...     vehicle1: Vehicle
    ...     vehicle2: Vehicle
    ...
    ...     def total_energy(self):
    ...         return self.vehicle1.kinetic_energy() + self.vehicle2.kinetic_energy()
    >>>
    >>> system = System(car, car2)
    >>> # This transformation applies to all dynamic fields in the entire hierarchy
    >>> scaled_system = arc.tree.map(lambda x: x * 0.5, system)

    See Also
    --------
    field : Define fields with tree-specific metadata
    """
    # Support passing arguments to the decorator (e.g. @struct(kw_only=True))
    if cls is None:
        return functools.partial(struct, **kwargs)

    # check if already recognized as a tree node
    if "_arc_struct" in cls.__dict__:
        return cls

    if "frozen" not in kwargs.keys():
        kwargs["frozen"] = True
    data_cls = dataclasses.dataclass(**kwargs)(cls)  # type: ignore
    meta_fields = []
    data_fields = []
    for field_info in dataclasses.fields(data_cls):
        # if not field_info.init:
        #     continue
        is_static = field_info.metadata.get("static", False)
        if not is_static:
            data_fields.append(field_info.name)
        else:
            meta_fields.append(field_info.name)

    def replace(self, **updates) -> T:
        """Returns a new object replacing the specified fields with new values."""
        new: T = dataclasses.replace(self, **updates)
        return new

    data_cls.replace = replace

    register_dataclass(data_cls, data_fields, meta_fields)

    # add a _arc_struct flag to distinguish from regular dataclasses
    data_cls._arc_struct = True  # type: ignore[attr-defined]

    return data_cls  # type: ignore


def is_struct(obj: Any) -> bool:
    """
    Check if an object is a registered struct class.

    This function determines whether an object was created using the
    :py:func:`struct` decorator, which indicates it has special handling
    for tree operations.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    is_node : bool
        ``True`` if the object is a struct created with the decorator,
        ``False`` otherwise.

    Notes
    -----
    When to use:

    - To check if an object will be handled specially by tree operations
    - For conditional logic based on whether an object is a custom struct
    - For debugging tree-related functionality

    This function specifically checks for objects created with the
    :py:func:`struct` decorator, not built-in structured data types like lists,
    tuples, and dictionaries.

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> @arc.struct
    >>> class State:
    ...     x: np.ndarray
    ...     v: np.ndarray
    >>>
    >>> state = State(np.zeros(3), np.ones(3))
    >>> print(arc.tree.is_struct(state))
    True
    >>>
    >>> # Regular dataclass is not a struct
    >>> from dataclasses import dataclass
    >>>
    >>> @dataclass
    >>> class RegularState:
    ...     x: np.ndarray
    ...     v: np.ndarray
    >>>
    >>> regular_state = RegularState(np.zeros(3), np.ones(3))
    >>> print(arc.tree.is_struct(regular_state))
    False
    >>>
    >>> # Built-in containers aren't custom structs
    >>> print(arc.tree.is_struct({"x": np.zeros(3)}))
    False

    See Also
    --------
    struct : Decorator for creating tree-compatible dataclasses
    """
    return hasattr(obj, "_arc_struct")
