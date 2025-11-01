from __future__ import annotations

import abc

import numpy as np

import archimedes as arc
from archimedes import StructConfig, UnionConfig, struct

__all__ = [
    "F16Engine",
    "F16EngineConfig",
    "TabulatedEngine",
    "TabulatedEngineConfig",
    "NASAEngine",
    "NASAEngineConfig",
]

#
# Engine lookup tables
#
alt_vector = np.array([0.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0])
mach_vector = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

Tidl_data = np.array(
    [
        1060.0,
        670.0,
        880.0,
        1140.0,
        1500.0,
        1860.0,
        635.0,
        425.0,
        690.0,
        1010.0,
        1330.0,
        1700.0,
        60.0,
        25.0,
        345.0,
        755.0,
        1130.0,
        1525.0,
        -1020.0,
        -710.0,
        -300.0,
        350.0,
        910.0,
        1360.0,
        -2700.0,
        -1900.0,
        -1300.0,
        -247.0,
        600.0,
        1100.0,
        -3600.0,
        -1400.0,
        -595.0,
        -342.0,
        -200.0,
        700.0,
    ]
).reshape((6, 6), order="F")

Tmil_data = np.array(
    [
        12680.0,
        9150.0,
        6200.0,
        3950.0,
        2450.0,
        1400.0,
        12680.0,
        9150.0,
        6313.0,
        4040.0,
        2470.0,
        1400.0,
        12610.0,
        9312.0,
        6610.0,
        4290.0,
        2600.0,
        1560.0,
        12640.0,
        9839.0,
        7090.0,
        4660.0,
        2840.0,
        1660.0,
        12390.0,
        10176.0,
        7750.0,
        5320.0,
        3250.0,
        1930.0,
        11680.0,
        9848.0,
        8050.0,
        6100.0,
        3800.0,
        2310.0,
    ]
).reshape((6, 6), order="F")

Tmax_data = np.array(
    [
        20000.0,
        15000.0,
        10800.0,
        7000.0,
        4000.0,
        2500.0,
        21420.0,
        15700.0,
        11225.0,
        7323.0,
        4435.0,
        2600.0,
        22700.0,
        16860.0,
        12250.0,
        8154.0,
        5000.0,
        2835.0,
        24240.0,
        18910.0,
        13760.0,
        9285.0,
        5700.0,
        3215.0,
        26070.0,
        21075.0,
        15975.0,
        11115.0,
        6860.0,
        3950.0,
        28886.0,
        23319.0,
        18300.0,
        13484.0,
        8642.0,
        5057.0,
    ]
).reshape((6, 6), order="F")

Tidl_interpolant = arc.interpolant([alt_vector, mach_vector], Tidl_data)
Tmil_interpolant = arc.interpolant([alt_vector, mach_vector], Tmil_data)
Tmax_interpolant = arc.interpolant([alt_vector, mach_vector], Tmax_data)


@struct
class F16Engine(metaclass=abc.ABCMeta):
    lo_gear: float = 64.94  # Low gear throttle slope
    hi_gear: float = 217.38  # High gear throttle slope
    throttle_breakpoint: float = 0.77  # Switch between linear throttle models

    @struct
    class Input:
        throttle: float  # Throttle position [0-1]
        alt: float  # Altitude [ft]
        mach: float  # Mach number

    @struct
    class Output:
        thrust: np.ndarray  # Thrust magnitude [lbf]

    @struct
    class State:
        pass  # No state by default

    def tgear(self, thtl):
        c_hi = (self.hi_gear - self.lo_gear) * self.throttle_breakpoint
        return np.where(
            thtl <= self.throttle_breakpoint,
            self.lo_gear * thtl,
            self.hi_gear * thtl - c_hi,
        )

    def dynamics(self, t: float, x: State, u: Input) -> State:
        """Time derivative of engine model state"""
        return x  # No dynamics by default

    @abc.abstractmethod
    def output(self, t: float, x: State, u: Input) -> Output:
        """Calculate engine thrust output"""
        pass

    @abc.abstractmethod
    def trim(self, throttle: float) -> State:
        """Calculate trim conditions for the engine model"""
        pass


class EngineConfigBase(StructConfig):
    lo_gear: float = 64.94  # Low gear throttle slope
    hi_gear: float = 217.38  # High gear throttle slope
    throttle_breakpoint: float = 0.77  # Switch between linear throttle models


@struct
class TabulatedEngine(F16Engine):
    """Simple tabulated engine model with zero dynamics"""

    def _calc_thrust(self, power: float, alt: float, mach: float) -> np.ndarray:
        """Calculate body-frame thrust vector from engine power"""
        T_mil = Tmil_interpolant(alt, mach)
        T_idl = Tidl_interpolant(alt, mach)
        T_max = Tmax_interpolant(alt, mach)

        return np.where(
            power < 50.0,
            T_idl + (T_mil - T_idl) * power * 0.02,
            T_mil + (T_max - T_mil) * (power - 50.0) * 0.02,
        )

    def output(
        self, t: float, x: F16Engine.State, u: F16Engine.Input
    ) -> F16Engine.Output:
        power = self.tgear(u.throttle)
        thrust = self._calc_thrust(power, u.alt, u.mach)
        return F16Engine.Output(thrust=thrust)

    def trim(self, throttle: float) -> F16Engine.State:
        return F16Engine.State()  # No state to trim


class TabulatedEngineConfig(EngineConfigBase, type="tabulated"):
    def build(self) -> TabulatedEngine:
        return TabulatedEngine(
            lo_gear=self.lo_gear,
            hi_gear=self.hi_gear,
            throttle_breakpoint=self.throttle_breakpoint,
        )


@struct
class NASAEngine(TabulatedEngine):
    """Tabulated engine model with first-order lag dynamics based on NASA model"""

    rtau_min: float = 0.1  # Minimum inv time constant for engine response [1/s]
    rtau_max: float = 1.0  # Maximum inv time constant for engine response [1/s]

    @struct
    class State(F16Engine.State):
        power: float  # Engine power

    def _rtau(self, dP):
        """Inverse time constant for engine response"""
        return np.where(
            dP <= 25,
            self.rtau_max,
            np.where(dP >= 50, self.rtau_min, 1.9 - 0.036 * dP),
        )

    def dynamics(
        self, t: float, x: NASAEngine.State, u: F16Engine.Input
    ) -> NASAEngine.State:
        """Time derivative of engine model state"""
        P = x.power  # Engine power
        thtl = u.throttle  # Throttle position

        cpow = self.tgear(thtl)  # Command power
        P2 = np.where(
            cpow >= 50.0,
            np.where(P >= 50.0, cpow, 60.0),
            np.where(P >= 50.0, 40.0, cpow),
        )

        # 1/tau
        rtau = np.where(P >= 50.0, 5.0, self._rtau(P2 - P))

        dP_dt = rtau * (P2 - P)
        return NASAEngine.State(power=dP_dt)

    def output(
        self, t: float, x: NASAEngine.State, u: F16Engine.Input
    ) -> F16Engine.Output:
        thrust = self._calc_thrust(x.power, u.alt, u.mach)
        return F16Engine.Output(thrust=thrust)

    def trim(self, throttle: float) -> NASAEngine.State:
        power = self.tgear(throttle)
        return NASAEngine.State(power=power)


class NASAEngineConfig(EngineConfigBase, type="nasa"):
    rtau_min: float = 0.1  # Minimum inv time constant for engine response [1/s]
    rtau_max: float = 1.0  # Maximum inv time constant for engine response [1/s]

    def build(self) -> NASAEngine:
        return NASAEngine(
            lo_gear=self.lo_gear,
            hi_gear=self.hi_gear,
            throttle_breakpoint=self.throttle_breakpoint,
            rtau_min=self.rtau_min,
            rtau_max=self.rtau_max,
        )


F16EngineConfig = UnionConfig[
    TabulatedEngineConfig,
    NASAEngineConfig,
]
