from __future__ import annotations
import abc

import numpy as np

from archimedes import struct, field, StructConfig, UnionConfig


__all__ = [
    "Actuator",
    "IdealActuator",
    "IdealActuatorConfig",
    "RateLimitedActuator",
    "RateLimitedActuatorConfig",
    "ActuatorConfig",
]


class Actuator(metaclass=abc.ABCMeta):
    """Abstract base class for aircraft actuators with rate and position limits."""

    @struct
    class State:
        pass  # No state by default

    @struct
    class Input:
        command: float

    @struct
    class Output:
        position: float

    def dynamics(self, t: float, x: State, u: Input) -> State:
        """Compute the actuator state derivative."""
        return x
    
    @abc.abstractmethod
    def output(self, t: float, x: State, u: Input) -> Output:
        """Compute the actuator output."""
        pass

    def trim(self, command: float) -> State:
        """Return a steady-state actuator state for the given command."""
        return self.State()


class IdealActuator(Actuator):
    """Ideal actuator with no rate or position limits."""

    def output(
        self, t: float, x: Actuator.State, u: Actuator.Input
    ) -> Actuator.Output:
        return Actuator.Output(position=u.command)
    

class IdealActuatorConfig(StructConfig, type="ideal"):
    def build(self) -> IdealActuator:
        return IdealActuator()
    

@struct
class RateLimitedActuator(Actuator):
    tau: float
    rate_limit: float | None = field(static=True)
    pos_limit: tuple[float, float] | None = field(static=True)

    @struct
    class State(Actuator.State):
        position: float = 0.0

    def dynamics(self, t: float, x: State, u: Actuator.Input) -> State:
        # Compute desired rate
        cmd, pos = u.command, x.position
        rate = (cmd - pos) / self.tau

        # Apply rate limit
        if self.rate_limit is not None:
            max_rate = self.rate_limit
            rate = np.clip(rate, -max_rate, max_rate)

        if self.pos_limit is not None:
            min_pos, max_pos = self.pos_limit
            rate = np.where((pos <= min_pos) * (rate < 0.0), 0.0, rate)
            rate = np.where((pos >= max_pos) * (rate > 0.0), 0.0, rate)

        return self.State(rate)
    
    def output(self, t: float, x: State, u: Actuator.Input) -> Actuator.Output:
        pos = x.position
        if self.pos_limit is not None:
            min_pos, max_pos = self.pos_limit
            pos = np.clip(pos, min_pos, max_pos)
        return Actuator.Output(position=pos)
    
    def trim(self, command: float) -> State:
        pos = command
        if self.pos_limit is not None:
            min_pos, max_pos = self.pos_limit
            pos = np.clip(pos, min_pos, max_pos)
        return self.State(position=pos)
    

class RateLimitedActuatorConfig(StructConfig, type="rate_limited"):
    tau: float  # Time constant [sec]
    rate_limit: float | None = field(default=None)  # Rate limit [units/sec]
    pos_limit: tuple[float, float] | None = field(default=None)  # Position limits [units]

    def build(self) -> RateLimitedActuator:
        return RateLimitedActuator(
            rate_limit=self.rate_limit,
            pos_limit=self.pos_limit,
        )
    
ActuatorConfig = UnionConfig[
    IdealActuatorConfig,
    RateLimitedActuatorConfig,
]