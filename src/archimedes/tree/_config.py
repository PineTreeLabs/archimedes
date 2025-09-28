from __future__ import annotations

from typing import Annotated, Literal, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "StructConfig",
    "UnionConfig",
]


T = TypeVar("T")


class StructConfig(BaseModel):
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
        class definition using ``StructConfig, type="typename"``.

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
    >>> @arc.struct
    >>> class ConstantGravity:
    ...     g0: float
    ...
    ...     def __call__(self, position: np.ndarray) -> np.ndarray:
    ...         return np.array([0, 0, self.g0])
    >>>
    >>> class ConstantGravityConfig(arc.StructConfig, type="constant"):
    ...     g0: float = 9.81
    ...
    ...     def build(self) -> ConstantGravity:
    ...         return ConstantGravity(self.g0)
    >>>
    >>> ConstantGravityConfig(g0=9.81).build()
    ConstantGravity(g0=9.81)
    >>>
    >>> # Another configuration type
    >>> class PointGravityConfig(arc.StructConfig, type="point"):
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
    UnionConfig : Create discriminated unions of StructConfig subclasses
    module : Decorator for creating modular dataclass components
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init_subclass__(cls, type: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if type is not None:
            cls.__annotations__ = {"type": Literal[type], **cls.__annotations__}
            cls.type = type  # type: ignore

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
    Discriminated union of StructConfig subclasses.

    Usage:
        AnyConfig = UnionConfig[ConfigTypeA, ConfigTypeB]

    Equivalent to:
        AnyConfig = Annotated[
            Union[ConfigTypeA, ConfigTypeB],
            Field(discriminator="type"),
        ]

    See Also
    --------
    StructConfig : Base class for module configuration management
    module : Decorator for creating modular system components
    """

    def __class_getitem__(cls, item) -> Type:
        # Handle single type (UnionConfig[OneType])
        if not isinstance(item, tuple):
            item = (item,)

        # Validate that all types inherit from StructConfig
        for config_type in item:
            if not (
                isinstance(config_type, type) and issubclass(config_type, StructConfig)
            ):
                raise TypeError(
                    f"{config_type} must be a subclass of StructConfig. "
                    f"UnionConfig is only for StructConfig discriminated unions."
                )

        # Create the discriminated union
        return Annotated[Union[item], Field(discriminator="type")]  # type: ignore
