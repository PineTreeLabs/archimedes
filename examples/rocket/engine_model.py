from __future__ import annotations
from functools import partial

import numpy as np

import archimedes as arc
from archimedes import struct
from archimedes.experimental import thermo
from archimedes.experimental.thermo import ThermalFluidEoS

from archimedes.experimental.fluid import (
    ControlVolume,
    Orifice,
    Valve,
    LookupTable1D,
    Nozzle,
)


class NitrogenTetroxide(ThermalFluidEoS):
    """Dinitrogen tetroxide (N2O4) in liquid phase.

    Uses a lookup table for pressure as a function of density and temperature.
    Enthalpy and entropy are calculated from the Shomate equation with coefficients
    given by NIST WebBook.  Heat capacity is then calculated from the derivative
    of enthalpy with respect to temperature.

    Generally expected to be valid from around 300-500 K at pressures from 5-10 MPa
    
    Thermal expansion data from:
        Properties of the System N2O4 ⇆ 2NO2 ⇆ 2NO + O2
        S. S. T. F and D. M. Mason
        J. Chem. Eng. Data 1962, 7, 2, 183–186
        https://doi.org/10.1021/je60013a007

    Isothermal compressibility from table 2.3.1
        USAF Propellant Handbooks
        Volume II: Nitric Acid/Nitric Tetroxide Oxidizers
        Martin Marietta Corporation, 1971

    Lookup tables from:
        The Thermodynamic Properties of Nitrogen Tetroxide
        R.D. McCarty, H.-U. Steurer, and C.M. Daily
        National Bureau of Standards, 1986

    Enthalpy of formation and Shomate coefficients:
        NIST WebBook, Dinitrogen tetroxide
    """
    M = 92.011  # molar mass [g/mol]

    # Shomate coefficients from NIST WebBook
    A = 89.16313
    B = 178.9141
    C = 0.929459
    D = 0.0
    E = -0.007107
    F = -54.13217
    G = 263.6757
    H = -19.56401

    p_bkpts = np.array([6.895, 10.342]) * 1e6  # Pa
    T_bkpts = np.array([294.26, 310.93, 327.59, 344.26, 360.93, 377.59, 394.26, 410.93])  # K
    rho_data = np.array([
        [1452.2666, 1457.5532],
        [1413.8048, 1420.0849],
        [1373.7974, 1379.7048],
        [1328.2225, 1337.0923],
        [1276.1576, 1287.2479],
        [1214.9109, 1228.1338],
        [1138.8123, 1158.7416],
        [1029.4641, 1064.2805],
    ])

    h0_f = -19.56e3  # enthalpy of formation [J/mol]

    def __init__(self):
        # 2D interpolation for rho(p, T)
        self._rho = arc.interpolant([self.p_bkpts, self.T_bkpts], self.rho_data.T)
    
        # Use root-finding to solve for pressure
        @arc.compile(kind="MX")
        def p_solve(p, rho, T):
            return rho - self.rho(p=p, T=T)
    
        # p(p_guess=1MPa, rho, T)
        self._p_solve = partial(arc.implicit(p_solve), 1e6)

    def _calc_p(self, rho: float, T: float) -> float:
        return self._p_solve(rho, T)
    
    def _calc_rho(self, p: float, T: float) -> float:
        return self._rho(p, T)

    def _calc_h(self, rho: float, T: float) -> float:
        t = T / 1000
        return (
            self.A * t
            + self.B * t**2 / 2
            + self.C * t**3 / 3
            + self.D * t**4 / 4
            - self.E / t
            + self.F
            - self.H
        )
    
    def s(self, rho: float, T: float) -> float:
        t = T / 1000
        return (
            self.A * np.log(t)
            + self.B * t
            + self.C * t**2 / 2
            + self.D * t**3 / 3
            - self.E / (2 * t**2)
            + self.G
        )

    def _calc_cv(self, rho: float, T: float) -> float:
        # cv = T * (∂s/∂T)_ρ
        ds_dT = arc.grad(self.s, 1)(rho, T)
        return T * ds_dT

    def _calc_cp(self, rho: float, T: float) -> float:
        # Not needed for this example
        return 0.0


class MonoMethylHydrazine(ThermalFluidEoS):
    """Mono-methyl hydrazine (MMH/CH6N2)
    
    Implemented with a lookup table for ρ(T) using RocketProp
    (based on published data from Aerojet Rocketdyne), with a
    correction for pressure based on the speed of sound.

    Speed of sound:
        USAF PROPELLANT HANDBOOKS: HYDRAZINE FUELS
        Table 2.4.1
    
    Heat capacity and enthalpy of formation:
        NIST WebBook: Hydrazine, methyl-
    """

    Tc = 585.0  # critical temperature [K]
    pc = 8.237e6  # critical pressure [Pa]
    w = 0.294  # accentric factor [-]
    M = 46.07  # molar mass [g/mol]

    h0_f = 54.14e3  # enthalpy of formation [J/mol]

    T_ref = 298.15  # reference temperature [K]
    p_ref = 1.01325e5  # reference pressure (1 bar) [Pa]
    rho_ref = 875.0  # reference density [kg/m^3]

    cs = 1548.0  # speed of sound [m/s]
    T_bkpts = np.array([300., 325., 350., 375., 400.])
    rho_data = np.array([873.377, 849.244, 824.137, 797.884, 770.255])

    # Reference energy
    u_ref = h0_f - (p_ref / rho_ref)  # [J/kg]

    def __init__(self):
        self._calc_rho = arc.compile(self._calc_rho, kind="MX", name="rho")

        # Use root-finding to solve for pressure
        @arc.compile(kind="MX")
        def p_solve(p, rho, T):
            return rho - self._calc_rho(p=p, T=T)
    
        # p(p_guess=1MPa, rho, T)
        self._p_solve = partial(arc.implicit(p_solve), 1e6)

    def _calc_rho(self, p: float, T: float) -> float:
        rho = np.interp(T, self.T_bkpts, self.rho_data)
        rho = rho + (p - self.p_ref) / self.cs**2  # Pressure correction
        return rho
    
    def _calc_p(self, rho: float, T: float) -> float:
        return self._p_solve(rho, T)

    def _calc_h(self, rho: float, T: float) -> float:
        cv = self._calc_cv(rho, T)
        p = self._calc_p(rho, T)
        u = self.u_ref + cv * (T - self.T_ref)
        return u + (p / rho)
    
    def _calc_cp(self, rho: float, T: float) -> float:
        cp = 134.93  # specific heat capacity [J/mol-K]
        return cp / (1e-3 * self.M)  # [J/kg-K]

    def _calc_cv(self, rho: float, T: float) -> float:
        p = self._calc_p(rho, T)
        k = (1 / rho) * arc.grad(self._calc_rho, 0)(p, T)  # isothermal compressibility [Pa⁻¹]
        alpha = -(1 / rho) * arc.grad(self._calc_rho, 1)(p, T)  # thermal expansion coefficient [K⁻¹]
        cp = self._calc_cp(rho, T)
        return cp - (T / rho) * (alpha ** 2 / k)


#
# Engine model
#

@struct.pytree_node
class Engine:
    fuel: thermo.FluidEoS = struct.field(static=True)
    oxidizer: thermo.FluidEoS = struct.field(static=True)
    exhaust: thermo.PerfectGasMixture = struct.field(static=True)

    fv: Valve  # Fuel valve
    fl: ControlVolume  # Fuel line
    fmi: Orifice  # Fuel manifold inlet
    fm: ControlVolume  # Fuel manifold
    fin: Orifice  # Fuel injectors

    ov: Valve  # Oxidizer valve
    ol: ControlVolume  # Oxidizer line
    omi: Orifice  # Oxidizer manifold inlet
    om: ControlVolume  # Oxidizer manifold
    oin: Orifice  # Oxidizer injectors

    cc: ControlVolume  # Combustion chamber
    noz: Nozzle  # Nozzle

    @struct.pytree_node
    class BoundaryConditions:
        p_ft: float  # Fuel tank pressure [Pa]
        T_ft: float  # Fuel tank temperature [K]
        p_ot: float  # Oxidizer tank pressure [Pa]
        T_ot: float  # Oxidizer tank temperature [K]
        p_amb: float  # Ambient pressure [Pa]
        T_amb: float  # Ambient temperature [K]
        pos_fv: float  # Fuel valve position [0-1]
        pos_ov: float  # Oxidizer valve position [0-1]

    @struct.pytree_node
    class State:
        # Fuel flow path
        fv: Valve.State
        fl: ControlVolume.State
        fmi: Orifice.State
        fm: ControlVolume.State
        fin: Orifice.State

        # Oxidizer flow path
        ov: Valve.State
        ol: ControlVolume.State
        omi: Orifice.State
        om: ControlVolume.State
        oin: Orifice.State

        # Combustion chamber
        cc: ControlVolume.State

        # Nozzle
        noz: Nozzle.State

    def dynamics(self, x: State, u: BoundaryConditions) -> State:
        # Compute the thermodynamic conditions using the EoS
    
        fluid_ft = self.fuel(p=u.p_ft, T=u.T_ft)
        fluid_fl = self.fuel(rho=x.fl.rho, T=x.fl.T)
        fluid_fm = self.fuel(rho=x.fm.rho, T=x.fm.T)
        fluid_ot = self.oxidizer(p=u.p_ot, T=u.T_ot)
        fluid_ol = self.oxidizer(rho=x.ol.rho, T=x.ol.T)
        fluid_om = self.oxidizer(rho=x.om.rho, T=x.om.T)
        fluid_cc = self.exhaust(rho=x.cc.rho, T=x.cc.T)
        fluid_amb = self.exhaust(p=u.p_amb, T=u.T_amb)

        # Compute the mass flows
        dm_fv = x.fv.dm
        dm_fmi = x.fmi.dm
        dm_fin = x.fin.dm
        dm_ov = x.ov.dm
        dm_omi = x.omi.dm
        dm_oin = x.oin.dm
        dm_noz = x.noz.dm

        # Compute enthalpy flow
        dh_fv = dm_fv * fluid_ft.h
        dh_fmi = dm_fmi * fluid_fl.h
        dh_fin = dm_fin * fluid_fm.h
        dh_ov = dm_ov * fluid_ot.h
        dh_omi = dm_omi * fluid_ol.h
        dh_oin = dm_oin * fluid_om.h
        dh_noz = dm_noz * fluid_cc.h

        # Combustion heat release
        dn_f = dm_fin / (self.fuel.M * 1e-3)  # molar flow rate of fuel
        dn_o = dm_oin / (self.oxidizer.M * 1e-3)  # molar flow rate of oxidizer
        dn_e = dm_noz / (self.exhaust.M * 1e-3)  # molar flow rate of exhaust
        dQ_cc = (
            dn_f * self.fuel.h0_f
            + dn_o * self.oxidizer.h0_f
            - dn_e * self.exhaust.h0_f
        )

        dm_fl = dm_fv - dm_fmi
        dh_fl = dh_fv - dh_fmi

        dm_fm = dm_fmi - dm_fin
        dh_fm = dh_fmi - dh_fin

        dm_ol = dm_ov - dm_omi
        dh_ol = dh_ov - dh_omi

        dm_om = dm_omi - dm_oin
        dh_om = dh_omi - dh_oin

        dm_cc = dm_oin + dm_fin - dm_noz
        dh_cc = dh_oin + dh_fin - dh_noz + dQ_cc

        return self.State(
            fv=self.fv.dynamics(x.fv, fluid_ft, fluid_fl, u.pos_fv),
            fl=self.fl.dynamics(fluid_fl, dm_fl, dh_fl),
            fmi=self.fmi.dynamics(x.fmi, fluid_fl, fluid_fm),
            fm=self.fm.dynamics(fluid_fm, dm_fm, dh_fm),
            fin=self.fin.dynamics(x.fin, fluid_fm, fluid_cc),
            ov=self.ov.dynamics(x.ov, fluid_ot, fluid_ol, u.pos_ov),
            ol=self.ol.dynamics(fluid_ol, dm_ol, dh_ol),
            omi=self.omi.dynamics(x.omi, fluid_ol, fluid_om),
            om=self.om.dynamics(fluid_om, dm_om, dh_om),
            oin=self.oin.dynamics(x.oin, fluid_om, fluid_cc),
            cc=self.cc.dynamics(fluid_cc, dm_cc, dh_cc),
            noz=self.noz.dynamics(x.noz, fluid_cc, fluid_amb),
        )

    def thrust(self, x: State, u: BoundaryConditions) -> float:
        p_cc = self.exhaust.p(rho=x.cc.rho, T=x.cc.T)
        T_cc = x.cc.T
        cp = self.exhaust.cp(rho=x.cc.rho, T=x.cc.T)
        cv = self.exhaust.cv(rho=x.cc.rho, T=x.cc.T)
        gamma = cp / cv
        p_amb = u.p_amb

        g0 = 9.81  # m/s^2

        # Exhaust velocity [m/s]
        ve = np.sqrt(2 * cp * T_cc * (1 - (p_amb / p_cc)**((gamma - 1) / gamma)))

        # Exhaust mass flow rate [kg/s]
        dm_e = x.noz.dm

        # Thrust [N]
        return dm_e * ve
    
    @property
    def OF_stoich(self):
        """Stoichiometrix O/F ratio determined by CEA"""
        return 2.49641
    
    def OF(self, x: State) -> float:
        """O/F ratio"""
        return x.oin.dm / x.fin.dm

    def equivalence_ratio(self, x: State) -> float:
        return self.OF_stoich / self.OF(x)