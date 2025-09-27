import numpy as np

import archimedes as arc
from archimedes import struct


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
class F16Engine:
    lo_gear: float = 64.94  # Low gear throttle slope
    hi_gear: float = 217.38  # High gear throttle slope
    throttle_breakpoint: float = 0.77  # Switch between linear throttle models
    rtau_min: float = 0.1  # Minimum inv time constant for engine response [1/s]
    rtau_max: float = 1.0  # Maximum inv time constant for engine response [1/s]

    def _tgear(self, thtl):
        c_hi = (self.hi_gear - self.lo_gear) * self.throttle_breakpoint
        return np.where(
            thtl <= self.throttle_breakpoint,
            self.lo_gear * thtl,
            self.hi_gear * thtl - c_hi,
        )

    def _rtau(self, dP):
        """Inverse time constant for engine response"""
        return np.where(
            dP <= 25,
            self.rtau_max,
            np.where(dP >= 50, self.rtau_min, 1.9 - 0.036 * dP),
        )

    def dynamics(self, t, x, u):
        """Time derivative of engine model (power variable)"""
        P = x  # Engine power
        thtl = u  # Throttle position

        cpow = self._tgear(thtl)  # Command power
        P2 = np.where(
            cpow >= 50.0,
            np.where(P >= 50.0, cpow, 60.0),
            np.where(P >= 50.0, 40.0, cpow),
        )

        # 1/tau
        rtau = np.where(P >= 50.0, 5.0, self._rtau(P2 - P))

        return rtau * (P2 - P)

    def calc_thrust(self, x, alt, rmach):
        P = x  # Engine power

        T_mil = Tmil_interpolant(alt, rmach)
        T_idl = Tidl_interpolant(alt, rmach)
        T_max = Tmax_interpolant(alt, rmach)

        Tx_B = np.where(
            P < 50.0,
            T_idl + (T_mil - T_idl) * P * 0.02,
            T_mil + (T_max - T_mil) * (P - 50.0) * 0.02,
        )

        return np.stack([Tx_B, 0.0, 0.0])
