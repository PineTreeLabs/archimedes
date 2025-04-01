"""Base classes for equaation of state models."""

from __future__ import annotations

import abc
from typing import Any

from archimedes import struct

R_IDEAL = 8.31446261815324  # J/mol-K


@struct.pytree_node
class FluidState:
    rho: float  # density [kg/m³]
    T: float  # temperature [K]
    p: float  # pressure [Pa]
    h: float  # specific enthalpy [J/kg]
    cv: float  # specific heat capacity at constant volume [J/(kg-K)]
    cp: float  # specific heat capacity at constant pressure [J/(kg-K)]
    du_drho: float  # derivative of internal energy with respect to density [J/(kg²)]

    @property
    def gamma(self):
        return self.cp / self.cv


class EoSMeta(abc.ABCMeta):
    @staticmethod
    def _validate_attr(
        cls, name: str, namespace, annotations, attr: str, description: str
    ):
        # # Check for property or class variable
        # if not any([
        #     isinstance(getattr(cls, attr, None), property),  # Property
        #     attr in namespace  # Class variable
        # ]):
        #     # Check if it's a pytree struct with the required field
        #     if struct.is_pytree_node(cls):
        #         if not any(field.name == attr for field in struct.fields(cls)):
        #             raise TypeError(
        #                 f"Class {name} must define {description} '{attr}' "
        #                 "as a dataclass field, property, or class variable"
        #             )
        #     else:
        #         raise TypeError(
        #             f"Class {name} must define {description} '{attr}' as either "
        #             " a property or class variable"
        #         )
        # Check for property or class variable
        if not any(
            [
                isinstance(getattr(cls, attr, None), property),  # Property
                attr in namespace,  # Class variable
                attr in annotations,  # Dataclass field
            ]
        ):
            raise TypeError(
                f"Class {name} must define {description} '{attr}' as either "
                "a dataclass field, property, or class variable"
            )

    def __new__(mcs, name: str, bases: tuple, namespace: dict) -> Any:
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip validation for abstract classes
        if hasattr(cls, "__abstractmethods__") and cls.__abstractmethods__:
            return cls

        # Require that the class has a standard enthalpy of formation and
        # molar mass
        required = {
            ("h0_f", "standard enthalpy of formation", float),
            ("M", "molar mass", float),
        }
        if name != "FluidEoS":  # Don't validate the base class itself
            # Collect annotations for dataclass-style classes
            annotations = getattr(cls, "__annotations__", {})
            for base in bases:
                annotations.update(getattr(base, "__annotations__", {}))

            for attr, description, _ in required:
                if not any(
                    [
                        isinstance(getattr(cls, attr, None), property),  # Property
                        attr in namespace,  # Class variable
                        attr in annotations,  # Dataclass field
                    ]
                ):
                    raise TypeError(
                        f"Class {name} must define {description} `{attr}' as either "
                        "a dataclass field, property, or class variable"
                    )

        # Add annotated fields to the class
        for attr, _, T in required:
            cls.__annotations__[attr] = T

        return cls


class FluidEoS(metaclass=EoSMeta):
    """Abstract base class for equation of state classes."""

    # TODO:
    # - Support v, h, s as kwargs
    # - Compute speed of sound
    # - Other thermodynamic properties?

    _supported_args = {"p", "rho", "T"}

    def _validate_args(self, **kwargs):
        if len(kwargs) != 2:
            raise ValueError(
                "Equation of state must be called with exactly two arguments, "
                f"but {len(kwargs)} were given ({kwargs.keys()})"
            )

        if set(kwargs.keys()) - self._supported_args != set():
            raise ValueError(
                f"Equation of state does not support the following arguments: "
                f"{set(kwargs.keys()) - self._supported_args}"
            )

    @abc.abstractmethod
    def _calc_p(self, rho: float, T: float) -> float:
        """Calculate pressure [Pa] from density [kg/m³] and temperature [K]."""

    def p(self, **kwargs) -> float:
        """Calculate pressure in Pa."""
        self._validate_args(**kwargs)
        if "p" in kwargs:
            return kwargs["p"]
        rho = self.rho(**kwargs)
        T = self.T(**kwargs)
        return self._calc_p(rho, T)

    @abc.abstractmethod
    def _calc_rho(self, p: float, T: float) -> float:
        """Calculate density [kg/m³] from pressure [Pa] and temperature [K]."""

    def rho(self, **kwargs) -> float:
        """Calculate density in kg/m³"""
        self._validate_args(**kwargs)
        if "rho" in kwargs:
            return kwargs["rho"]
        p = self.p(**kwargs)
        T = self.T(**kwargs)
        return self._calc_rho(p, T)

    # @abc.abstractmethod
    # def _calc_T(self, p: float, rho: float) -> float:
    #     """Specific enthalpy [J/kg] from pressure [Pa] and density [kg/m³]."""

    def T(self, **kwargs) -> float:
        """Temperature in K"""
        self._validate_args(**kwargs)
        if "T" in kwargs:
            return kwargs["T"]
        raise NotImplementedError("TODO: implement T calculation")

    @abc.abstractmethod
    def _calc_h(self, rho: float, T: float) -> float:
        """Specific enthalpy [J/kg] from density [kg/m³] and temperature [K]."""

    def h(self, **kwargs) -> float:
        """Specific enthalpy in J/kg"""
        self._validate_args(**kwargs)
        if "h" in kwargs:
            return kwargs["h"]
        rho = self.rho(**kwargs)
        T = self.T(**kwargs)
        return self._calc_h(rho, T)

    @abc.abstractmethod
    def _calc_cv(self, rho: float, T: float) -> float:
        """Isochoric specific heat capacity [J/kg-K] from pressure [Pa] and temperature [K]."""

    def cv(self, **kwargs) -> float:
        """Isochoric specific heat capacity in J/kg-K"""
        self._validate_args(**kwargs)
        rho = self.rho(**kwargs)
        T = self.T(**kwargs)
        return self._calc_cv(rho, T)

    @abc.abstractmethod
    def _calc_cp(self, rho: float, T: float) -> float:
        """Isobaric specific heat capacity [J/kg-K] from pressure [Pa] and temperature [K]."""

    def cp(self, **kwargs) -> float:
        """Isobaric specific heat capacity in J/kg-K"""
        self._validate_args(**kwargs)
        rho = self.rho(**kwargs)
        T = self.T(**kwargs)
        return self._calc_cp(rho, T)

    @abc.abstractmethod
    def _calc_du_drho(self, rho: float, T: float) -> float:
        """Compute (∂u/∂ρ)|T [J-m³/kg²] from density [kg/m³] and temperature [K]."""

    def du_drho(self, **kwargs) -> float:
        """(∂u/∂ρ)|T in J-m³/kg²"""
        self._validate_args(**kwargs)
        rho = self.rho(**kwargs)
        T = self.T(**kwargs)
        return self._calc_du_drho(rho, T)

    def __call__(self, **kwargs) -> FluidState:
        """Compute fluid state (p, T, ρ, h, ...)"""
        self._validate_args(**kwargs)
        return FluidState(
            p=self.p(**kwargs),
            rho=self.rho(**kwargs),
            T=self.T(**kwargs),
            h=self.h(**kwargs),
            cv=self.cv(**kwargs),
            cp=self.cp(**kwargs),
            du_drho=self.du_drho(**kwargs),
        )


class ThermalFluidEoS(FluidEoS):
    """Equation of state for a thermal fluid.

    A thermal fluid is defined as one whose internal energy is a function of
    temperature only; specifically, the partial derivative (∂u/∂ρ)|T = 0.

    This is true for ideal gases and incompressible, variable-density fluids,
    for example.
    """

    def _calc_du_drho(self, rho: float, T: float) -> float:
        return 0.0
