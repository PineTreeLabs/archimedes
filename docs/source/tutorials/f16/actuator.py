from __future__ import annotations

import abc

import numpy as np

from archimedes import StructConfig, UnionConfig, field, struct

__all__ = [
    "Actuator",
    "IdealActuator",
    "IdealActuatorConfig",
    "LagActuator",
    "LagActuatorConfig",
    "ActuatorConfig",
]


class Actuator(metaclass=abc.ABCMeta):
    """Abstract base class for SISO actuator components."""

    @struct
    class State:
        pass  # No state by default

    def dynamics(self, t: float, x: State, u: float) -> State:
        """Compute the actuator state derivative."""
        return x

    @abc.abstractmethod
    def output(self, t: float, x: State, u: float) -> float:
        """Compute the actuator output."""
        pass

    def trim(self, command: float) -> State:
        """Return a steady-state actuator state for the given command."""
        return self.State()


@struct
class IdealActuator(Actuator):
    """Ideal actuator with instantaneous linear response."""

    gain: float = 1.0

    def output(self, t: float, x: Actuator.State, u: float) -> float:
        return self.gain * u


class IdealActuatorConfig(StructConfig, type="ideal"):
    gain: float = 1.0

    def build(self) -> IdealActuator:
        return IdealActuator(self.gain)


@struct
class LagActuator(Actuator):
    tau: float
    gain: float = 1.0
    rate_limit: float | None = field(static=True, default=None)
    pos_limits: tuple[float, float] | None = field(static=True, default=None)

    @struct
    class State(Actuator.State):
        position: float = 0.0

    def dynamics(self, t: float, x: State, u: float) -> State:
        # Compute desired rate
        cmd, pos = u, x.position
        rate = (self.gain * cmd - pos) / self.tau

        # Apply rate limit
        if self.rate_limit is not None:
            max_rate = self.rate_limit
            rate = np.clip(rate, -max_rate, max_rate)

        if self.pos_limits is not None:
            min_pos, max_pos = self.pos_limits
            rate = np.where((pos <= min_pos) * (rate < 0.0), 0.0, rate)
            rate = np.where((pos >= max_pos) * (rate > 0.0), 0.0, rate)

        return self.State(rate)

    def output(self, t: float, x: State, u: float) -> float:
        pos = x.position
        if self.pos_limits is not None:
            min_pos, max_pos = self.pos_limits
            pos = np.clip(pos, min_pos, max_pos)
        return pos

    def trim(self, command: float) -> State:
        pos = command
        if self.pos_limits is not None:
            min_pos, max_pos = self.pos_limits
            pos = np.clip(pos, min_pos, max_pos)
        return self.State(position=pos)


class LagActuatorConfig(StructConfig, type="lag"):
    tau: float  # Time constant [sec]
    gain: float = field(default=1.0)
    rate_limit: float | None = field(default=None)  # Rate limit [units/sec]
    pos_limits: tuple[float, float] | None = field(
        default=None
    )  # Position limits [units]

    def build(self) -> LagActuator:
        return LagActuator(
            tau=self.tau,
            gain=self.gain,
            rate_limit=self.rate_limit,
            pos_limits=self.pos_limits,
        )


ActuatorConfig = UnionConfig[
    IdealActuatorConfig,
    LagActuatorConfig,
]
