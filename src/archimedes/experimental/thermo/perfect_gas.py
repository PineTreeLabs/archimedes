import abc
import numpy as np

from archimedes import struct
from .eos import ThermalFluidEoS, R_IDEAL



class PerfectGasBase(ThermalFluidEoS):
    """Base class for thermally perfect gases.
    
    A thermally perfect gas is defined as one for which the internal energy,
    enthalpy, and specific heat capacities are functions of temeperature only.
    """

    @property
    def Rs(self) -> float:
        """Specific gas constant [J/kg-K]"""
        return R_IDEAL / (1e-3 * self.M)

    @abc.abstractmethod
    def _calc_cp(self, rho: float, T: float) -> float:
        """Isobaric specific heat capacity [J/kg-K] from pressure [Pa] and temperature [K].
        
        Note that this function accepts density for API compatibility, but
        density is unused for ideal gas heat capacity."""

    def _calc_cv(self, rho: float, T: float) -> float:
        """Isochoric specific heat capacity [J/kg-K] from temperature [K]
        
        Note that this function accepts density for API compatibility, but
        density is unused for ideal gas heat capacity.
        """
        return self._calc_cp(rho, T) - self.Rs

    def _calc_p(self, rho: float, T: float) -> float:
        return rho * self.Rs * T

    def _calc_rho(self, p: float, T: float) -> float:
        return p / (self.Rs * T)

    @abc.abstractmethod
    def _calc_h(self, rho: float, T: float) -> float:
        """Specific enthalpy [J/kg] from temperature [K]

        Note that this function accepts density for API compatibility, but
        density is unused for ideal gas heat enthalpy.
        """


@struct.pytree_node
class PerfectGasEoS(PerfectGasBase):
    M: float = struct.field(static=True)  # molar mass [g/mol]
    h0_f: float = struct.field(static=True)  # standard enthalpy of formation [J/mol]
    info: str = struct.field(static=True)  # gas name and references


@struct.pytree_node
class TabulatedPerfectGas(PerfectGasEoS):
    """Ideal gas with tabulated specific heat capacity and enthalpy."""
    T_data: np.ndarray = struct.field(static=True)  # Lookup table temperature values [K]
    h_data: np.ndarray = struct.field(static=True)  # Lookup table enthalpy values [J/mol]
    cp_data: np.ndarray = struct.field(static=True)  # Lookup table specific heat capacity values [J/mol-K]

    def _calc_cp(self, rho: float, T: float) -> float:
        return np.interp(T, self.T_data, self.cp_data) / (1e-3 * self.M)
    
    def _calc_h(self, rho: float, T: float) -> float:
        return np.interp(T, self.T_data, self.h_data) / (1e-3 * self.M)
    


CO = TabulatedPerfectGas(
    M=28.01,
    h0_f=-110541,
    T_data=np.arange(1000.0, 5000.0, 100.0),
    h_data=np.array([
        21697, 25046, 28440, 31874, 35345, 38847, 42379, 45937, 49517, 53118,
        56737, 60371, 64020, 67682, 71354, 75036, 78727, 82426, 86132, 89844,
        93562, 97287, 101016, 104751, 108490, 112235, 115985, 119739, 123499, 127263,
        131032, 134806, 138585, 142368, 146156, 149948, 153743, 157541, 161342, 165145,
    ]),
    cp_data=np.array([
        33.255, 33.725, 34.148, 34.530, 34.872, 35.178, 35.451, 35.694, 35.910,
        36.101, 36.271, 36.421, 36.553, 36.670, 36.774, 36.867, 36.950, 37.025,
        37.093, 37.155, 37.215, 37.268, 37.321, 37.372, 37.422, 37.471, 37.521,
        37.570, 37.619, 37.667, 37.716, 37.764, 37.810, 37.855, 37.897, 37.936,
        37.970, 37.998, 38.019, 38.031
    ]),
    info="Carbon monoxide\n\nData from Turns & Hayworth Table A.1",
)


CO2 = TabulatedPerfectGas(
    M=44.011,
    h0_f=-393546,
    T_data=np.arange(1000.0, 5000.0, 100.0),
    h_data=np.array([
        33425, 38911, 44488, 50149, 55882, 61681, 67538, 73446, 79399, 85392,
        91420, 97477, 103562, 109670, 115798, 121944, 128107, 134284, 140474,
        146677, 152891, 159116, 165351, 171597, 177853, 184120, 190397, 196685,
        202983, 209293, 215613, 221945, 228287, 234640, 241002, 247373, 253752,
        260138, 266528, 272920
    ]),
    cp_data=np.array([
        54.360, 55.333, 56.205, 56.984, 57.677, 58.292, 58.836, 59.316, 59.738,
        60.108, 60.433, 60.717, 60.966, 61.185, 61.378, 61.548, 61.701, 61.839,
        61.965, 62.083, 62.194, 62.301, 62.406, 62.510, 62.614, 62.716, 62.825,
        62.932, 63.041, 63.151, 63.261, 63.369, 63.474, 63.575, 63.669, 63.753,
        63.825, 63.881, 63.918, 63.932
    ]),
    info="Carbon dioxide\n\nData from Turns & Hayworth Table A.2",
)


H2 = TabulatedPerfectGas(
    M=2.016,
    h0_f=0.0,
    T_data=np.arange(1000.0, 5000.0, 100.0),
    h_data=np.array([
        20664, 23704, 26789, 29919, 33092, 36307, 39562, 42858, 46191, 49562, 52968,
        56408, 59882, 63388, 66925, 70492, 74087, 77710, 81359, 85033, 88733, 92455,
        96201, 99968, 103757, 107566, 111395, 115243, 119109, 122994, 126897, 130817,
        134755, 138710, 142682, 146672, 150679, 154703, 158746, 162806,
    ]),
    cp_data=np.array([
        30.160, 30.625, 31.077, 31.516, 31.943, 32.356, 32.758, 33.146, 33.522, 33.885,
        34.236, 34.575, 34.901, 35.216, 35.519, 35.811, 36.091, 36.361, 36.621, 36.871,
        37.112, 37.343, 37.566, 37.781, 37.989, 38.190, 38.385, 38.574, 38.759, 38.939,
        39.116, 39.291, 39.464, 39.636, 39.808, 39.981, 40.156, 40.334, 40.516, 40.702,
    ]),
    info="Diatomic hydrogen\n\nData from Turns & Hayworth Table A.3",
)


H = TabulatedPerfectGas(
    M=1.008,
    h0_f=217977,
    T_data=np.arange(1000.0, 5000.0, 100.0),
    h_data=np.array([
        14589, 16667, 18746, 20824, 22903, 24982, 27060, 29139, 31217, 33296,
        35375, 37453, 39532, 41610, 43689, 45768, 47846, 49925, 52003, 54082,
        56161, 58239, 60318, 62396, 64475, 66554, 68632, 70711, 72789, 74868,
        76947, 79025, 81104, 83182, 85261, 87340, 89418, 91497, 93575, 95654,
    ]),
    cp_data=np.full(40, 20.786),
    info="Monatomic hydrogen\n\nData from Turns & Hayworth Table A.4",
)


OH = TabulatedPerfectGas(
    M=12.007,
    h0_f=28995,
    T_data=np.arange(1000.0, 5000.0, 100.0),
    h_data=np.array([
        20928, 24022, 27164, 30353, 33586, 36860, 40174, 43524, 46910, 50328,
        53776, 57254, 60759, 64289, 67843, 71420, 75017, 78634, 82269, 85922,
        89590, 93273, 96970, 100681, 104403, 108137, 111882, 115638, 119403, 123178,
        126962, 130754, 134555, 138365, 142182, 146008, 149842, 153685, 157536, 161395,
    ]),
    cp_data=np.array([
        30.682, 31.186, 31.662, 32.114, 32.540, 32.943, 33.323, 33.682, 34.019, 34.337,
        34.635, 34.915, 35.178, 34.425, 35.656, 35.872, 36.074, 36.263, 36.439, 36.604,
        36.759, 36.903, 37.039, 37.166, 37.285, 37.298, 37.504, 37.605, 37.701, 37.793,
        37.882, 37.968, 38.052, 38.135, 38.217, 38.300, 38.382, 38.466, 38.552, 38.640,
    ]),
    info="Hydroxyl radical\n\nData from Turns & Hayworth Table A.5",
)


H2O = TabulatedPerfectGas(
    M=18.016,
    h0_f=-241845,
    T_data=np.arange(1000.0, 5000.0, 100.0),
    h_data=np.array([
        25993, 30191, 34518, 38963, 43520, 48181, 52939, 57786, 62717, 67725, 72805,
        77952, 83160, 88426, 93744, 99112, 104524, 109979, 115472, 121001, 126563,
        132156, 137777, 143426, 149099, 154795, 160514, 166252, 172011, 177787, 183582,
        189392, 195219, 201061, 206918, 212790, 218674, 224573, 230484, 236407,
    ]),
    cp_data=np.array([
        41.315, 42.638, 43.874, 45.027, 46.102, 47.103, 48.035, 48.901, 49.705,
        50.451, 51.143, 51.784, 52.378, 52.927, 53.435, 53.905, 54.340, 54.742,
        55.115, 55.459, 55.779, 56.076, 56.353, 56.610, 56.851, 57.076, 57.288,
        57.488, 57.676, 57.856, 58.026, 58.190, 58.346, 58.496, 58.641, 58.781,
        59.116, 59.047, 59.173, 59.295,
    ]),
    info="Water\n\nData from Turns & Hayworth Table A.6",
)


N2 = TabulatedPerfectGas(
    M=28.013,
    h0_f=0,
    T_data=np.arange(1000.0, 5000.0, 100.0),
    h_data=np.array([
        21468, 24770, 28118, 31510, 34939, 38404, 41899, 45423, 48971, 52541, 56130,
        59738, 63360, 66997, 70645, 74305, 77974, 81652, 85338, 89031, 92730, 96436,
        100148, 103865, 107587, 111315, 115048, 118786, 122528, 126276, 130028, 133786,
        137548, 141314, 145085, 148860, 152639, 156420, 160205, 163991,
    ]),
    cp_data=np.array([
        32.762, 33.258, 33.707, 34.113, 34.477, 34.805, 35.099, 35.361, 35.595, 35.803,
        35.988, 36.152, 36.289, 36.428, 36.543, 36.645, 36.737, 36.820, 36.895, 36.964,
        37.028, 37.088, 37.144, 37.198, 37.251, 37.302, 37.352, 37.402, 37.452, 37.501,
        37.549, 37.597, 37.643, 37.688, 37.730, 37.768, 37.803, 37.832, 37.854, 37.868,
    ]),
    info="Diatomic nitrogen\n\nData from Turns & Hayworth Table A.7",
)


# N = TabulatedPerfectGas(
#     M=14.007,
#     h0_f=472629,
#     T_data=np.arange(1000.0, 5000.0, 100.0),
#     h_data=np.array([
#     ]),
#     cp_data=np.array([
#     ]),
#     info="Monatomic nitrogen\n\nData from Turns & Hayworth Table A.8",
# )


NO = TabulatedPerfectGas(
    M=30.006,
    h0_f=90297,
    T_data=np.arange(1000.0, 5000.0, 100.0),
    h_data=np.array([
        22241, 25669, 29136, 32638, 36171, 39732, 43317, 46925, 50552, 54197, 57857,
        61531, 65216, 68912, 72617, 76331, 80052, 83779, 87513, 91251, 94995, 98744,
        102498, 106255, 110018, 113784, 117555, 121330, 125109, 128893, 132680, 136473,
        140269, 144069, 147873, 151681, 155492, 159305, 163121, 166938,
    ]),
    cp_data=np.array([
        34.076, 34.483, 34.850, 35.180, 35.474, 35.737, 35.972, 36.180, 36.364, 36.527,
        36.671, 36.797, 36.909, 37.008, 37.095, 37.173, 37.242, 37.305, 37.362, 37.415,
        37.464, 37.511, 37.556, 37.600, 37.643, 37.686, 37.729, 37.771, 38.015, 38.058,
        38.900, 38.943, 38.984, 38.023, 38.060, 38.093, 38.122, 38.146, 38.162, 38.171,
    ]),
    info="Nitric oxide\n\nData from Turns & Hayworth Table A.9",
)


# NO2 = TabulatedPerfectGas(
#     M=46.006,
#     h0_f=33098,
#     T_data=np.arange(1000.0, 5000.0, 100.0),
#     h_data=np.array([
#     ]),
#     cp_data=np.array([
#     ]),
#     info="Nitrogen dioxide\n\nData from Turns & Hayworth Table A.10",
# )


O2 = TabulatedPerfectGas(
    M=31.999,
    h0_f=0,
    T_data=np.arange(1000.0, 5000.0, 100.0),
    h_data=np.array([
        22721, 26232, 29775, 33350, 36955, 40590, 44253, 47943, 51660, 55402, 59169,
        62959, 66773, 70609, 74467, 78346, 82245, 86164, 90103, 94060, 98036, 102029,
        106040, 110068, 114112, 118173, 122249, 126341, 130448, 134570, 138705, 142855,
        147019, 151195, 155384, 159586, 163800, 168026, 172262, 176510,
    ]),
    cp_data=np.array([
        34.936, 35.270, 35.593, 35.903, 36.202, 36.490, 36.768, 37.036, 37.296, 37.546,
        37.788, 38.023, 38.250, 38.470, 38.684, 38.891, 39.093, 39.289, 39.480, 39.665,
        39.846, 40.023, 40.195, 40.362, 40.526, 40.686, 40.842, 40.994, 41.143, 41.287,
        41.429, 41.556, 41.700, 41.830, 41.957, 42.079, 42.197, 42.312, 42.421, 42.527,
    ]),
    info="Diatomic oxygen\n\nData from Turns & Hayworth Table A.9=11",
)


O = TabulatedPerfectGas(
    M=16.000,
    h0_f=249197,
    T_data=np.arange(1000.0, 5000.0, 100.0),
    h_data=np.array([
        14861, 16952, 19041, 21128, 23214, 25299, 27383, 29466, 31548, 33630,
        35712, 37794, 39877, 41959, 44043, 46127, 48213, 50300, 52389, 54480,
        56574, 58669, 60768, 62869, 64973, 67081, 69192, 71308, 73427, 75550,
        77678, 79810, 81947, 84088, 86235, 88386, 90543, 92705, 94872, 97045,
    ]),
    cp_data=np.array([
        20.915, 20.898, 20.882, 20.867, 20.854, 20.843, 20.834, 20.827, 20.822, 20.820,
        20.819, 20.821, 20.825, 20.831, 20.840, 20.851, 20.865, 20.881, 20.899, 20.920,
        20.944, 20.970, 20.998, 21.028, 21.061, 21.095, 21.132, 21.171, 21.212, 21.254,
        21.299, 21.345, 21.392, 21.441, 21.490, 21.541, 21.593, 21.646, 12.699, 21.752,
    ]),
    info="Monatomic oxygen\n\nData from Turns & Hayworth Table A.12",
)


PERFECT_GASES = {
    'CO': CO,
    'CO2': CO2,
    'H2': H2,
    'H': H,
    'OH': OH,
    'H2O': H2O,
    'N2': N2,
    # 'N': N,
    'NO': NO,
    # 'NO2': NO2,
    'O2': O2,
    'O': O,
}



@struct.pytree_node
class PerfectGasMixture(PerfectGasBase):
    """Mixture of thermally perfect gases"""

    composition: dict[str, float]  # molar fractions

    # By default, the database is a set of predefined ideal gas models
    # based on tabulated data.  However, this can be overridden with a
    # set of custom gas models.
    database: dict[str, PerfectGasEoS] = struct.field(
        static=True, default_factory=lambda: PERFECT_GASES
    )

    mass_frac: dict[str, float] = struct.field(init=False)  # mass fractions
    M: float = struct.field(init=False)  # molar mass [g/mol]
    h0_f: float = struct.field(init=False)  # standard enthalpy of formation [J/mol]

    @property
    def mole_frac(self) -> dict[str, float]:
        return self.composition
    
    def _mass_avg(self, qty: dict[str, float]) -> float:
        return sum(self.mass_frac[gas] * qty[gas] for gas in self.composition)
        
    def _mole_avg(self, qty: dict[str, float]) -> float:
        return sum(self.mole_frac[gas] * qty[gas] for gas in self.composition)

    def __post_init__(self):
        # Normalize molar fractions in case they don't exactly sum to 1
        den = sum(self.composition.values())
        mole_frac = {
            gas: frac / den for gas, frac in self.composition.items()
        }
        # NOTE: object.__setattr__ is used instead of direclty setting because
        # this is a frozen dataclass
        object.__setattr__(self, 'composition', mole_frac)
    
        M = self._mole_avg({gas: self.database[gas].M for gas in self.composition})
        object.__setattr__(self, 'M', M)
    
        # Calculate mass fractions for each component
        mass_frac = {
            gas: frac * self.database[gas].M / M
            for gas, frac in mole_frac.items()
        }
        object.__setattr__(self, 'mass_frac', mass_frac)

        # Calculate mole-weighted standard enthalpy of formation
        h0_f = self._mole_avg({gas: self.database[gas].h0_f for gas in self.composition})
        object.__setattr__(self, 'h0_f', h0_f)

    def _mass_prop_avg(self, attr: str, **kwargs) -> float:
        # Call the function for each component
        qty = {
            gas: getattr(self.database[gas], attr)(**kwargs)
            for gas in self.composition
        }
        # Average the results by mass fraction
        return self._mass_avg(qty)

    def _calc_cp(self, rho: float, T: float) -> float:
        """Isobaric specific heat capacity [J/kg-K] from pressure [Pa] and temperature [K].
        
        Note that this function accepts density for API compatibility, but
        density is unused for ideal gas heat capacity."""
        return self._mass_prop_avg("cp", rho=rho, T=T)

    def _calc_h(self, rho: float, T: float) -> float:
        """Specific enthalpy [J/kg] from temperature [K]

        Note that this function accepts density for API compatibility, but
        density is unused for ideal gas heat enthalpy.
        """
        return self._mass_prop_avg("h", rho=rho, T=T)