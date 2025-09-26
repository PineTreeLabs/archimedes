from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Annotated, Literal, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import dataclass_transform


__all__ = [
    "module",
    "ModuleConfig",
    "UnionConfig",
]


T = TypeVar("T")


@dataclass_transform(field_specifiers=(dataclasses.field,))  # type: ignore[literal-required]
def module(cls: T | None = None, **kwargs) -> T | Callable:
    """
    Decorator to convert a class into a dataclass suitable for modular system design.

    This decorator creates a dataclass that can be used to build modular, composable
    systems. Unlike :py:func:`struct`, classes decorated with ``@module`` are not
    automatically registered with the tree system, making them suitable for
    organizational and structural components that don't need to participate in
    tree transformations.  Common use cases include physics models and modular
    control systems.

    Parameters
    ----------
    cls : type, optional
        The class to convert into a module dataclass
    **kwargs : dict
        Additional keyword arguments passed to dataclasses.dataclass().
        Unlike :py:func:`struct`, ``frozen`` is not set by default.

    Returns
    -------
    decorated_class : type
        The decorated class, now a dataclass marked as a module.

    Notes
    -----
    The key difference from :py:func:`struct` is that ``@module`` classes:

    - Are not automatically registered with the tree system
    - Are not frozen by default (mutable unless explicitly frozen)
    - Are intended for structural organization rather than data transformation

    Use ``@module`` when you want dataclass convenience without tree behavior,
    and ``@struct`` when you need the object to participate in tree operations
    like mapping, flattening, and transformations.

    Modules are designed to work well with ModuleConfig classes for managing
    configurations in large, modular physics models or algorithms.  The combination
    of these lets you define clear interfaces, protocols, and configurations for
    complex, hierarchical systems.

    Examples
    --------
    >>> import archimedes as arc
    >>> from typing import Protocol
    >>> import numpy as np
    >>>
    >>> class ComponentModel(Protocol):
    ...     def __call__(self, x: np.ndarray) -> np.ndarray:
    ...         ...
    >>>
    >>> @arc.module
    >>> class ComponentA:
    ...     a: float = 1.0
    ...
    ...     def __call__(self, x: np.ndarray) -> np.ndarray:
    ...         return x * self.a
    >>>
    >>> @arc.module
    >>> class ComponentB:
    ...     b: float = 2.0
    ...
    ...     def __call__(self, x: np.ndarray) -> np.ndarray:
    ...         return x + self.b
    >>>
    >>> @arc.module
    >>> class System:
    ...     component: ComponentModel
    ...
    >>>

    See Also
    --------
    struct : Decorator for creating tree-compatible dataclasses
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
    >>>
    >>> class GravityModel(Protocol):
    ...     def __call__(self, position: np.ndarray) -> np.ndarray:
    ...         ...
    >>>
    >>> @arc.module
    >>> class ConstantGravity:
    ...     g0: float
    ...
    ...     def __call__(self, position: np.ndarray) -> np.ndarray:
    ...         return np.array([0, 0, self.g0])
    >>>
    >>> class ConstantGravityConfig(arc.ModuleConfig, type="constant"):
    ...     g0: float = 9.81
    ...
    ...     def build(self) -> ConstantGravity:
    ...         return ConstantGravity(self.g0)
    >>>
    >>> ConstantGravityConfig(g0=9.81).build()
    ConstantGravity(g0=9.81)
    >>>
    >>> # Another configuration type
    >>> class PointGravityConfig(arc.ModuleConfig, type="point"):
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
    >>> GravityConfig = arc.UnionConfig[ConstantGravityConfig, PointGravityConfig]

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
