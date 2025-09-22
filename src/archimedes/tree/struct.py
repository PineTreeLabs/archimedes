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
Utilities for defining custom classes that can be used with pytree transformations.

This module provides tools for creating structured data types that work seamlessly
with Archimedes' pytree functions. These tools are built on Python's dataclasses
with extensions for pytree-specific behavior.

The module re-exports several names from the dataclasses module:

InitVar : Type annotation for init-only variables in dataclasses
    Used to mark fields that should be passed to __post_init__ but not stored.

fields : Function to retrieve fields of a dataclass
    Returns a list of Field objects representing the fields of the dataclass.
    This is useful for introspection and validation of dataclass instances.

replace : Function to create a new dataclass instance with updated fields
    For pytree nodes created with @pytree_node, use the .replace() method instead.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import InitVar, fields, replace
from typing import Annotated, Any, Literal, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import dataclass_transform

from ._registry import register_dataclass

__all__ = [
    "field",
    "pytree_node",
    "InitVar",
    "is_pytree_node",
    "fields",
    "replace",
    "module",
    "ModuleConfig",
    "UnionConfig",
]


T = TypeVar("T")


def field(
    static: bool = False,
    *,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dataclasses.Field:
    """
    Create a field specification with pytree-related metadata.

    This function extends :py:func:`dataclasses.field()` with additional metadata to
    control how fields are treated in pytree operations. Fields can be marked as static
    (metadata) or dynamic (data). Except for the `static` argument, all other arguments
    are passed directly to :py:func:`dataclasses.field()`; see documentation for the
    :py:mod:`dataclasses` module for details.

    Parameters
    ----------
    static : bool, default=False
        If True, the field is treated as static metadata rather than dynamic data.
        Static fields are preserved during pytree transformations but not included
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

    Static fields are not included when you flatten a pytree or apply transformations
    like ``map``, but they are preserved in the structure and included when you
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
    pytree_node : Decorator for creating pytree-compatible classes
    register_dataclass : Register a dataclass as a pytree node
    """
    f: dataclasses.Field = dataclasses.field(
        metadata=(metadata or {}) | {"static": static},
        **kwargs,
    )
    return f


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def pytree_node(cls: T | None = None, **kwargs) -> T | Callable:
    """
    Decorator to convert a class into a frozen dataclass registered as a pytree node.

    This decorator creates a dataclass that can be seamlessly used with Archimedes'
    pytree functions. The class will be registered with the pytree system, allowing
    its instances to be flattened, mapped over, and transformed while preserving
    its structure.

    Parameters
    ----------
    cls : type, optional
        The class to convert into a pytree
    **kwargs : dict
        Additional keyword arguments passed to dataclasses.dataclass().
        By default, ``frozen=True`` is set unless explicitly overridden.

    Returns
    -------
    decorated_class : type
        The decorated class, now a frozen dataclass registered as a pytree node.

    Notes
    -----
    When to use:

    - To create structured data objects for use in Archimedes models and simulations
    - To define state containers that work with pytree-based transformations
    - To create modular, composable model components with clear interfaces
    - To define parameter structures for optimization problems

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
    - Can be nested within other pytree nodes

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> @arc.struct.pytree_node
    >>> class Vehicle:
    ...     # Dynamic state variables (included in transformations)
    ...     position: np.ndarray
    ...     velocity: np.ndarray
    ...
    ...     # Static configuration parameters (preserved during transformations)
    ...     mass: float = arc.struct.field(static=True, default=1000.0)
    ...     drag_coef: float = arc.struct.field(static=True, default=0.3)
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
    >>> # Nested pytree nodes
    >>> @arc.struct.pytree_node
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
    field : Define fields with pytree-specific metadata
    module: Decorator for creating functional modules
    """
    # Support passing arguments to the decorator (e.g. @pytree_node(kw_only=True))
    if cls is None:
        return functools.partial(pytree_node, **kwargs)

    # check if already recognized as a pytree node
    if "_arc_dataclass" in cls.__dict__:
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

    # add a _arc_dataclass flag to distinguish from regular dataclasses
    data_cls._arc_dataclass = True  # type: ignore[attr-defined]

    return data_cls  # type: ignore


def is_pytree_node(obj: Any) -> bool:
    """
    Check if an object is a registered pytree node.

    This function determines whether an object was created using the
    :py:func:`pytree_node` decorator, which indicates it has special handling
    for pytree operations.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    is_node : bool
        ``True`` if the object is a pytree node created with the decorator,
        ``False`` otherwise.

    Notes
    -----
    When to use:

    - To check if an object will be handled specially by pytree operations
    - For conditional logic based on whether an object is a custom pytree node
    - For debugging pytree-related functionality

    This function specifically checks for objects created with the
    :py:func:`pytree_node` decorator, not built-in pytree containers like lists,
    tuples, and dictionaries.

    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>>
    >>> @arc.struct.pytree_node
    >>> class State:
    ...     x: np.ndarray
    ...     v: np.ndarray
    >>>
    >>> state = State(np.zeros(3), np.ones(3))
    >>> print(arc.struct.is_pytree_node(state))
    True
    >>>
    >>> # Regular dataclass is not a pytree node
    >>> from dataclasses import dataclass
    >>>
    >>> @dataclass
    >>> class RegularState:
    ...     x: np.ndarray
    ...     v: np.ndarray
    >>>
    >>> regular_state = RegularState(np.zeros(3), np.ones(3))
    >>> print(arc.struct.is_pytree_node(regular_state))
    False
    >>>
    >>> # Built-in containers aren't custom pytree nodes
    >>> print(arc.struct.is_pytree_node({"x": np.zeros(3)}))
    False

    See Also
    --------
    pytree_node : Decorator for creating pytree-compatible classes
    """
    return hasattr(obj, "_arc_dataclass")


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def module(cls: T | None = None, **kwargs) -> T | Callable:
    """
    Decorator to convert a class into a dataclass suitable for modular system design.

    This decorator creates a dataclass that can be used to build modular, composable
    systems. Unlike :py:func:`pytree_node`, classes decorated with ``@module`` are not
    automatically registered with the pytree system, making them suitable for
    organizational and structural components that don't need to participate in
    tree transformations.  Common use cases include physics models and modular
    control systems.

    Parameters
    ----------
    cls : type, optional
        The class to convert into a module dataclass
    **kwargs : dict
        Additional keyword arguments passed to dataclasses.dataclass().
        Unlike :py:func:`pytree_node`, ``frozen`` is not set by default.

    Returns
    -------
    decorated_class : type
        The decorated class, now a dataclass marked as a module.

    Notes
    -----
    The key difference from :py:func:`pytree_node` is that ``@module`` classes:

    - Are not automatically registered with the pytree system
    - Are not frozen by default (mutable unless explicitly frozen)
    - Are intended for structural organization rather than data transformation

    Use ``@module`` when you want dataclass convenience without pytree behavior,
    and ``@pytree_node`` when you need the object to participate in tree operations
    like mapping, flattening, and transformations.

    Modules are designed to work well with ModuleConfig classes for managing
    configurations in large, modular physics models or algorithms.  The combination
    of these lets you define clear interfaces, protocols, and configurations for
    complex, hierarchical systems.

    Examples
    --------
    >>> import archimedes as arc
    >>> from archimedes import struct
    >>> from typing import Protocol
    >>> import numpy as np
    >>>
    >>> class ComponentModel(Protocol):
    ...     def __call__(self, x: np.ndarray) -> np.ndarray:
    ...         ...
    >>>
    >>> @struct.module
    >>> class ComponentA:
    ...     a: float = 1.0
    ...
    ...     def __call__(self, x: np.ndarray) -> np.ndarray:
    ...         return x * self.a
    >>>
    >>> @struct.module
    >>> class ComponentB:
    ...     b: float = 2.0
    ...
    ...     def __call__(self, x: np.ndarray) -> np.ndarray:
    ...         return x + self.b
    >>>
    >>> @struct.module
    >>> class System:
    ...     component: ComponentModel
    ...
    >>>

    See Also
    --------
    pytree_node : Decorator for creating pytree-compatible classes
    field : Define fields with specific metadata
    ModuleConfig : Base class for module configuration management
    """
    # Support passing arguments to the decorator (e.g. @module(kw_only=True))
    if cls is None:
        return functools.partial(module, **kwargs)

    # check if already recognized as a module
    if "_arc_module" in cls.__dict__:
        return cls

    data_cls = dataclasses.dataclass(**kwargs)(cls)  # type: ignore
    data_cls._arc_module = True  # type: ignore[attr-defined]
    return data_cls  # type: ignore[return-value]


class ModuleConfig(BaseModel):
    """
    Base class for creating configuration objects with automatic type discrimination.

    This class extends Pydantic's BaseModel to automatically add a ``type`` field
    based on the class name, enabling type-safe configuration systems with
    automatic serialization and validation. Subclasses must specify their type
    using the ``type`` parameter in the class definition.

    Parameters
    ----------
    type : str
        The type identifier for this configuration class, specified in the
        class definition using ``ModuleConfig, type="typename"``.

    Notes
    -----
    The ``type`` field is automatically added to the class and set to the value
    specified in the class definition. This enables automatic discrimination
    when working with unions of different configuration types.

    Subclasses are expected to implement a ``build()`` method that constructs
    the corresponding module instance based on the configuration parameters.
    This may include any "offline" validation, preprocessing, or data loading
    that should occur once at initialization time rather than at runtime.

    Key features:

    - Automatic ``type`` field addition and population
    - Validation and serialization of the fields
    - Designed to work with :py:class:`UnionConfig` for type discrimination

    Examples
    --------
    >>> from typing import Protocol
    >>> import archimedes as arc
    >>> from archimedes import struct
    >>>
    >>> class GravityModel(Protocol):
    ...     def __call__(self, position: np.ndarray) -> np.ndarray:
    ...         ...
    >>>
    >>> @struct.module
    >>> class ConstantGravity:
    ...     g0: float
    ...
    ...     def __call__(self, position: np.ndarray) -> np.ndarray:
    ...         return np.array([0, 0, self.g0])
    >>>
    >>> class ConstantGravityConfig(struct.ModuleConfig, type="constant"):
    ...     g0: float = 9.81
    ...
    ...     def build(self) -> ConstantGravity:
    ...         return ConstantGravity(self.g0)
    >>>
    >>> ConstantGravityConfig(g0=9.81).build()
    ConstantGravity(g0=9.81)
    >>>
    >>> # Another configuration type
    >>> class PointGravityConfig(struct.ModuleConfig, type="point"):
    ...     mu: float = 3.986e14  # m^3/s^2
    ...     RE: float = 6.3781e6  # m
    ...     lat: float = 0.0  # deg
    ...     lon: float = 0.0  # deg
    ...
    ...     def build(self) -> PointGravity:
    ...         # Implementation omitted for brevity
    ...         pass
    >>>
    >>> # Create a discriminated union of configuration types
    >>> GravityConfig = struct.UnionConfig[ConstantGravityConfig, PointGravityConfig]
    >>>
    >>> # Parse a configuration dictionary
    >>> config_dict = {"type": "constant", "g0": 9.81}
    >>> gravity_config = GravityConfig.parse_obj(config_dict)
    >>> print(gravity_config)
    ConstantGravityConfig(g0=9.81)

    See Also
    --------
    UnionConfig : Create discriminated unions of ModuleConfig subclasses
    module : Decorator for creating modular dataclass components
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init_subclass__(cls, type: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if type is not None:
            cls.__annotations__ = {"type": Literal[type], **cls.__annotations__}
            cls.type = type

    # When printing the config, show the class name and fields only but
    # not the type field
    def __repr__(self):
        rep = super().__repr__()
        if hasattr(self, "type"):
            return rep.replace(f"type='{self.type}', ", "")
        return rep

    def build(self):
        raise NotImplementedError("Subclasses must implement the build() method.")


class UnionConfig:
    """
    Discriminated union of ModuleConfig subclasses.

    Usage:
        AnyConfig = UnionConfig[ConfigTypeA, ConfigTypeB]

    Equivalent to:
        AnyConfig = Annotated[
            Union[ConfigTypeA, ConfigTypeB],
            Field(discriminator="type"),
        ]

    See Also
    --------
    ModuleConfig : Base class for module configuration management
    module : Decorator for creating modular system components
    """

    def __class_getitem__(cls, item) -> Type:
        # Handle single type (UnionConfig[OneType])
        if not isinstance(item, tuple):
            item = (item,)

        # Validate that all types inherit from ModuleConfig
        for config_type in item:
            if not (
                isinstance(config_type, type) and issubclass(config_type, ModuleConfig)
            ):
                raise TypeError(
                    f"{config_type} must be a subclass of ModuleConfig. "
                    f"UnionConfig is only for ModuleConfig discriminated unions."
                )

        # Create the discriminated union
        return Annotated[Union[item], Field(discriminator="type")]
