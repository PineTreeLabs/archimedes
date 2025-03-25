from __future__ import annotations
import abc

import numpy as np

from archimedes import struct
from archimedes.experimental import thermo



@struct.pytree_node
class ControlVolume:
    vol: float  # Component volume [m³]

    @struct.pytree_node
    class State:
        rho: float
        T: float


    def dynamics(self, x: thermo.FluidState, dm: float, dh: float) -> State:
        vol = self.vol
        rho, T, p, h = x.rho, x.T, x.p, x.h
        du_drho, cv = x.du_drho, x.cv

        # Equation of state calculations

        rho_t = dm / vol  # [kg/m³-s]
        T_t = ((dh - h * dm) / (rho * vol) + (p / rho ** 2 - du_drho) * rho_t) / cv
        return self.State(rho_t, T_t)


@struct.pytree_node
class FlowPath:
    I: float  # Inertance [m⁻⁴]

    @struct.pytree_node
    class State:
        dm: float

    @abc.abstractmethod
    def dynamics(
        self,
        x: FlowPath.State,
        fluid_up: thermo.FluidState,
        fluid_dn: thermo.FluidState
    ) -> float:
        pass

    def _calc_dynamics(self, dm: float, dp: float, rho: float, CdA: float) -> State:
        return FlowPath.State(dm=(dp - dm ** 2 / (2 * rho * CdA ** 2)) / self.I)


@struct.pytree_node
class Orifice(FlowPath):
    CdA: float  # Discharge coefficient * area [m²]

    def dynamics(
        self,
        x: Orifice.State,
        fluid_up: thermo.FluidState,
        fluid_dn: thermo.FluidState
    ) -> Orifice.State:
        dp = fluid_up.p - fluid_dn.p
        rho = fluid_up.rho
        return self._calc_dynamics(x.dm, dp, rho, self.CdA)


@struct.pytree_node
class LookupTable1D:
    xp: np.array = struct.field(static=True)
    yp: np.array = struct.field(static=True)

    def __call__(self, x: float) -> float:
        return np.interp(x, self.xp, self.yp)


@struct.pytree_node
class Valve(FlowPath):
    CdA: LookupTable1D  # Discharge coefficient * area lookup table [m²]

    def dynamics(
        self,
        x: Valve.State,
        fluid_up: thermo.FluidState,
        fluid_dn: thermo.FluidState,
        pos: float
    ) -> Valve.State:
        CdA = self.CdA(pos)
        dp = fluid_up.p - fluid_dn.p
        rho = fluid_up.rho
        return self._calc_dynamics(x.dm, dp, rho, CdA)
    

@struct.pytree_node
class Nozzle(FlowPath):
    """Choked flow model
    
    
    dm = Cq * Cm * At * sqrt(2 * p * rho)

    with Cm = sqrt(2 * gamma / (gamma + 1)) * (2 / (gamma + 1)) ** (1 / (gamma - 1))
    
    """
    At: float  # Throat area [m²]
    Cq: float  # Flow coefficient [-]

    def dynamics(
        self,
        x: Valve.State,
        fluid_up: thermo.FluidState,
        fluid_dn: thermo.FluidState,
    ) -> Valve.State:
        gamma = fluid_up.gamma
        dp = fluid_up.p  # For choked flow, dp = p_up
        rho = fluid_up.rho
        Cm = np.sqrt(2 * gamma / (gamma + 1)) * (2 / (gamma + 1)) ** (1 / (gamma - 1))
        CdA = self.Cq * Cm * self.At / np.sqrt(2)
        return self._calc_dynamics(x.dm, dp, rho, CdA)

