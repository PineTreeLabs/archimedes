import abc
import numpy as np

import archimedes as arc
from archimedes import struct

from archimedes.experimental.aero import FlightVehicle

g0 = 32.17  # ft/s^2


#
# Engine lookup tables
#
alt_vector = np.array([
    0.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0
])
mach_vector = np.array([
    0.0, 0.2, 0.4, 0.6, 0.8, 1.0
])

Tidl_data = np.array([
    1060.0, 670.0, 880.0, 1140.0, 1500.0, 1860.0,
    635.0, 425.0, 690.0, 1010.0, 1330.0, 1700.0,
    60.0, 25.0, 345.0, 755.0, 1130.0, 1525.0,
    -1020.0, -710.0, -300.0, 350.0, 910.0, 1360.0,
    -2700.0, -1900.0, -1300.0, -247.0, 600.0, 1100.0,
    -3600.0, -1400.0, -595.0, -342.0, -200.0, 700.0
]).reshape((6, 6), order='F')

Tmil_data = np.array([
    12680.0, 9150.0, 6200.0, 3950.0, 2450.0, 1400.0,
    12680.0, 9150.0, 6313.0, 4040.0, 2470.0, 1400.0,
    12610.0, 9312.0, 6610.0, 4290.0, 2600.0, 1560.0,
    12640.0, 9839.0, 7090.0, 4660.0, 2840.0, 1660.0,
    12390.0, 10176.0, 7750.0, 5320.0, 3250.0, 1930.0,
    11680.0, 9848.0, 8050.0, 6100.0, 3800.0, 2310.0,
]).reshape((6, 6), order='F')

Tmax_data = np.array([
    20000.0, 15000.0, 10800.0, 7000.0, 4000.0, 2500.0,
    21420.0, 15700.0, 11225.0, 7323.0, 4435.0, 2600.0,
    22700.0, 16860.0, 12250.0, 8154.0, 5000.0, 2835.0,
    24240.0, 18910.0, 13760.0, 9285.0, 5700.0, 3215.0,
    26070.0, 21075.0, 15975.0, 11115.0, 6860.0, 3950.0,
    28886.0, 23319.0, 18300.0, 13484.0, 8642.0,  5057.0,
]).reshape((6, 6), order='F')

Tidl_interpolant = arc.interpolant([alt_vector, mach_vector], Tidl_data)
Tmil_interpolant = arc.interpolant([alt_vector, mach_vector], Tmil_data)
Tmax_interpolant = arc.interpolant([alt_vector, mach_vector], Tmax_data)


#
# Aerodynamics lookup tables
#

# Angle of attack data for lookup tables
alpha_vector = np.array(
    [-10,  -5,   0,   5,  10,  15,  20,  25,  30,  35,  40,  45]
)

# Sideslip angle data for lookup tables
beta_vector = np.array([ 0,  5, 10, 15, 20, 25, 30])

# Elevator deflection data for lookup tables
ele_vector = np.array([-24, -12,   0,  12,  24])

# Cx(alpha, ele)
cx_data = np.array([
    [-0.099, -0.048, -0.022, -0.04 , -0.083],
    [-0.081, -0.038, -0.02 , -0.038, -0.073],
    [-0.081, -0.04 , -0.021, -0.039, -0.076],
    [-0.063, -0.021, -0.004, -0.025, -0.072],
    [-0.025,  0.016,  0.032,  0.006, -0.046],
    [ 0.044,  0.083,  0.094,  0.062,  0.012],
    [ 0.097,  0.127,  0.128,  0.087,  0.024],
    [ 0.113,  0.137,  0.13 ,  0.085,  0.025],
    [ 0.145,  0.162,  0.154,  0.1  ,  0.043],
    [ 0.167,  0.177,  0.161,  0.11 ,  0.053],
    [ 0.174,  0.179,  0.155,  0.104,  0.047],
    [ 0.166,  0.167,  0.138,  0.091,  0.04 ],
])
_calc_cx = arc.interpolant([alpha_vector, ele_vector], cx_data)

# Cz(alpha)
cz_data = [.770, .241, -.100, -.416, -.731, -1.053, -1.366, -1.646, -1.917, -2.120, -2.248, -2.229]

# Cl(alpha, beta)
# TODO: Should be able to use the textbook numbers here
cl_data = np.array([
    [ 0.   , -0.001, -0.003, -0.001,  0.   ,  0.007,  0.009],
    [ 0.   , -0.004, -0.009, -0.01 , -0.01 , -0.01 , -0.011],
    [ 0.   , -0.008, -0.017, -0.02 , -0.022, -0.023, -0.023],
    [ 0.   , -0.012, -0.024, -0.03 , -0.034, -0.034, -0.037],
    [ 0.   , -0.016, -0.03 , -0.039, -0.047, -0.049, -0.05 ],
    [ 0.   , -0.019, -0.034, -0.044, -0.046, -0.046, -0.047],
    [ 0.   , -0.02 , -0.04 , -0.05 , -0.059, -0.068, -0.074],
    [ 0.   , -0.02 , -0.037, -0.049, -0.061, -0.071, -0.079],
    [ 0.   , -0.015, -0.016, -0.023, -0.033, -0.06 , -0.091],
    [ 0.   , -0.008, -0.002, -0.006, -0.036, -0.058, -0.076],
    [ 0.   , -0.013, -0.01 , -0.014, -0.035, -0.062, -0.077],
    [ 0.   , -0.015, -0.019, -0.027, -0.035, -0.059, -0.076],
])  # Textbook data

# Cm(alpha, ele)
cm_data = np.array([
    [ 0.205,  0.081, -0.046, -0.174, -0.259],
    [ 0.168,  0.077, -0.02 , -0.145, -0.202],
    [ 0.186,  0.107, -0.009, -0.121, -0.184],
    [ 0.196,  0.11 , -0.005, -0.127, -0.193],
    [ 0.213,  0.11 , -0.006, -0.129, -0.199],
    [ 0.251,  0.141,  0.01 , -0.102, -0.15 ],
    [ 0.245,  0.127,  0.006, -0.097, -0.16 ],
    [ 0.238,  0.119, -0.001, -0.113, -0.167],
    [ 0.252,  0.133,  0.014, -0.087, -0.104],
    [ 0.231,  0.108,  0.   , -0.084, -0.076],
    [ 0.198,  0.081, -0.013, -0.069, -0.041],
    [ 0.192,  0.093,  0.032, -0.006, -0.005],
])

# Cn(alpha, beta)
cn_data = np.array([
    [ 0.   ,  0.018,  0.038,  0.056,  0.064,  0.074,  0.079],
    [ 0.   ,  0.019,  0.042,  0.057,  0.077,  0.086,  0.09 ],
    [ 0.   ,  0.018,  0.042,  0.059,  0.076,  0.093,  0.106],
    [ 0.   ,  0.019,  0.042,  0.058,  0.074,  0.089,  0.106],
    [ 0.   ,  0.019,  0.043,  0.058,  0.073,  0.08 ,  0.096],
    [ 0.   ,  0.018,  0.039,  0.053,  0.057,  0.062,  0.08 ],
    [ 0.   ,  0.013,  0.03 ,  0.032,  0.029,  0.049,  0.068],
    [ 0.   ,  0.007,  0.017,  0.012,  0.007,  0.022,  0.03 ],
    [ 0.   ,  0.004,  0.004,  0.002,  0.012,  0.028,  0.064],
    [ 0.   , -0.014, -0.035, -0.046, -0.034, -0.012,  0.015],
    [ 0.   , -0.017, -0.047, -0.071, -0.065, -0.002,  0.011],
    [ 0.   , -0.033, -0.057, -0.073, -0.041, -0.013, -0.001],
])

cx_interpolant = arc.interpolant([alpha_vector, ele_vector], cx_data)
cl_interpolant = arc.interpolant([alpha_vector, beta_vector], cl_data)
_calc_cm = arc.interpolant([alpha_vector, ele_vector], cm_data)
cn_interpolant = arc.interpolant([alpha_vector, beta_vector], cn_data)

#
# Control surfaces coefficients
#
# NOTE: Different beta_vector than for Cl and Cn
beta_vector = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])

dlda_data = np.array([
    [-0.041, -0.041, -0.042, -0.04 , -0.043, -0.044, -0.043],
    [-0.052, -0.053, -0.053, -0.052, -0.049, -0.048, -0.049],
    [-0.053, -0.053, -0.052, -0.051, -0.048, -0.048, -0.047],
    [-0.056, -0.053, -0.051, -0.052, -0.049, -0.047, -0.045],
    [-0.05 , -0.05 , -0.049, -0.048, -0.043, -0.042, -0.042],
    [-0.056, -0.051, -0.049, -0.048, -0.042, -0.041, -0.037],
    [-0.082, -0.066, -0.043, -0.042, -0.042, -0.02 , -0.003],
    [-0.059, -0.043, -0.035, -0.037, -0.036, -0.028, -0.013],
    [-0.042, -0.038, -0.026, -0.031, -0.025, -0.013, -0.01 ],
    [-0.038, -0.027, -0.016, -0.026, -0.021, -0.014, -0.003],
    [-0.027, -0.023, -0.018, -0.017, -0.016, -0.011, -0.007],
    [-0.017, -0.016, -0.014, -0.012, -0.011, -0.01 , -0.008],
])

dldr_data = np.array([
    [ 0.005,  0.007,  0.013,  0.018,  0.015,  0.021,  0.023],
    [ 0.017,  0.016,  0.013,  0.015,  0.014,  0.011,  0.01 ],
    [ 0.014,  0.014,  0.011,  0.015,  0.013,  0.01 ,  0.011],
    [ 0.01 ,  0.014,  0.012,  0.014,  0.013,  0.011,  0.011],
    [-0.005,  0.013,  0.011,  0.014,  0.012,  0.01 ,  0.011],
    [ 0.009,  0.009,  0.009,  0.014,  0.011,  0.009,  0.01 ],
    [ 0.019,  0.012,  0.008,  0.014,  0.011,  0.008,  0.008],
    [ 0.005,  0.005,  0.005,  0.015,  0.01 ,  0.01 ,  0.01 ],
    [ 0.   ,  0.   , -0.002,  0.013,  0.008,  0.006,  0.006],
    [-0.005,  0.004,  0.005,  0.011,  0.008,  0.005,  0.014],
    [-0.011,  0.009,  0.003,  0.006,  0.007,  0.   ,  0.02 ],
    [ 0.008,  0.007,  0.005,  0.001,  0.003,  0.001,  0.   ],
])

dnda_data = np.array([
    [ 0.001,  0.002, -0.006, -0.011, -0.015, -0.024, -0.022],
    [-0.027, -0.014, -0.008, -0.011, -0.015, -0.01 ,  0.002],
    [-0.017, -0.016, -0.006, -0.01 , -0.014, -0.004, -0.003],
    [-0.013, -0.016, -0.006, -0.009, -0.012, -0.002, -0.005],
    [-0.012, -0.014, -0.005, -0.008, -0.011, -0.001, -0.003],
    [-0.016, -0.019, -0.008, -0.006, -0.008,  0.003, -0.001],
    [ 0.001, -0.021, -0.005,  0.   , -0.002,  0.014, -0.009],
    [ 0.017,  0.002,  0.007,  0.004,  0.002,  0.006, -0.009],
    [ 0.011,  0.012,  0.004,  0.007,  0.006, -0.001, -0.001],
    [ 0.017,  0.016,  0.007,  0.01 ,  0.012,  0.004,  0.003],
    [ 0.008,  0.015,  0.006,  0.004,  0.011,  0.004, -0.002],
    [ 0.016,  0.011,  0.006,  0.01 ,  0.011,  0.006,  0.001],
])

dndr_data = np.array([
    [-0.018, -0.028, -0.037, -0.048, -0.043, -0.052, -0.062],
    [-0.052, -0.051, -0.041, -0.045, -0.044, -0.034, -0.034],
    [-0.052, -0.043, -0.038, -0.045, -0.041, -0.036, -0.027],
    [-0.052, -0.046, -0.04 , -0.045, -0.041, -0.036, -0.028],
    [-0.054, -0.045, -0.04 , -0.044, -0.04 , -0.035, -0.027],
    [-0.049, -0.049, -0.038, -0.045, -0.038, -0.028, -0.027],
    [-0.059, -0.057, -0.037, -0.047, -0.034, -0.024, -0.023],
    [-0.051, -0.052, -0.03 , -0.048, -0.035, -0.023, -0.023],
    [-0.03 , -0.03 , -0.027, -0.049, -0.035, -0.02 , -0.019],
    [-0.037, -0.033, -0.024, -0.045, -0.029, -0.016, -0.009],
    [-0.026, -0.03 , -0.019, -0.033, -0.022, -0.01 , -0.025],
    [-0.013, -0.008, -0.013, -0.016, -0.009, -0.014, -0.01 ],
])


dlda = arc.interpolant([alpha_vector, beta_vector], dlda_data)
dldr = arc.interpolant([alpha_vector, beta_vector], dldr_data)
dnda = arc.interpolant([alpha_vector, beta_vector], dnda_data)
dndr = arc.interpolant([alpha_vector, beta_vector], dndr_data)


def _adc(Vt, alt):
    """Standard atmosphere model"""
    R0 = 2.377e-3  # Density scale [slug/ft^3]
    gamma = 1.4  # Adiabatic index for air [-]
    Rs = 1716.3  # Specific gas constant for air [ft·lbf/slug-R]
    Tfac = 1 - 0.703e-5 * alt  # Temperature factor

    T = np.where(alt >= 35000.0, 390.0, 519.0 * Tfac)

    if alt > 35000.0:
        T = 390.0
    else:
        T = 519.0 * Tfac

    rho = R0 * Tfac ** 4.14
    amach = Vt / np.sqrt(gamma * Rs * T)
    qbar = 0.5 * rho * Vt ** 2

    return amach, qbar


def _tgear(thtl):
    return np.where(
        thtl <= .77,
        64.94 * thtl,
        217.38 * thtl - 117.38,
    )


def _rtau(dp):
    """Inverse time constant for engine response"""

    return np.where(
        dp <= 25,
        1.0,
        np.where(dp >= 50, .1, 1.9 - 0.036 * dp),
    )


def _calc_pdot(pow, thtl):
    """Time derivative of engine model (power variable)"""

    cpow = _tgear(thtl)  # Command power

    p2 = np.where(
        cpow >= 50.0,
        np.where(pow >= 50.0, cpow, 60.0),
        np.where(pow >= 50.0, 40.0, cpow),
    )

    # 1/tau
    rtau = np.where(pow >= 50.0, 5.0, _rtau(p2 - pow))

    return rtau * (p2 - pow)


def _calc_thrust(pow, alt, rmach):

    T_mil = Tmil_interpolant(alt, rmach)
    T_idl = Tidl_interpolant(alt, rmach)
    T_max = Tmax_interpolant(alt, rmach)

    Tx_B = np.where(
        pow < 50.0,
        T_idl + (T_mil - T_idl) * pow * 0.02,
        T_mil + (T_max - T_mil) * (pow - 50.0) * 0.02
    )

    return np.stack([Tx_B, 0.0, 0.0])


def _calc_cy(beta, ail, rdr):
    return -.02 * beta + .021 * (ail / 20) + .086 * (rdr / 30)


def _calc_cz(alpha, beta, el):
    cz_lookup = np.interp(alpha, alpha_vector, cz_data)
    return (-0.19 / 25) * el + cz_lookup * (1.0 - (beta / 57.3) ** 2)


def _calc_cl(alpha, beta):
    return np.sign(beta) * cl_interpolant(alpha, np.abs(beta))


def _calc_cn(alpha, beta):
    return np.sign(beta) * cn_interpolant(alpha, np.abs(beta))


def _calc_damp(alpha):
    Cxq_data = np.array([-.267, .110,  .308,  1.34,  2.08,  2.91,  2.76,  2.05,   1.5,  1.49,  1.83,  1.21])
    Cyr_data = np.array([ .882,  .852,  .876,  .958,  .962,  .974,  .819,  .483,  .590,  1.21, -.493, -1.04])
    Cyp_data = np.array([-.108, -.108, -.188,  .110,  .258,  .226,  .344,  .362,  .611,  .529,  .298, -2.27])
    Czq_data = np.array([ -8.8, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2,   -29, -29.8, -38.3, -35.3])

    Clr_data = np.array([-.126, -.026,  .063,  .113,  .208,  .230,  .319,  .437,  .680,    .1,  .447, -.330])
    Clp_data = np.array([ -.36, -.359, -.443,  -.42, -.383, -.375, -.329, -.294,  -.23,  -.21,  -.12,   -.1])
    Cmq_data = np.array([-7.21,  -.54, -5.23, -5.26, -6.11, -6.64, -5.69,    -6,  -6.2,  -6.4,  -6.6,    -6])
    Cnr_data = np.array([ -.38, -.363, -.378, -.386,  -.37, -.453,  -.55, -.582, -.595, -.637, -1.02,  -.84])
    Cnp_data = np.array([ .061,  .052,  .052, -.012, -.013, -.024,   .05,   .15,   .13,  .158,   .24,   .15])

    return np.stack([
        np.interp(alpha, alpha_vector, Cxq_data),
        np.interp(alpha, alpha_vector, Cyr_data),
        np.interp(alpha, alpha_vector, Cyp_data),
        np.interp(alpha, alpha_vector, Czq_data),
        np.interp(alpha, alpha_vector, Clr_data),
        np.interp(alpha, alpha_vector, Clp_data),
        np.interp(alpha, alpha_vector, Cmq_data),
        np.interp(alpha, alpha_vector, Cnr_data),
        np.interp(alpha, alpha_vector, Cnp_data),
    ])


@struct.pytree_node
class SubsonicF16(FlightVehicle):
    xcg: float = 0.35  # CG location (% of cbar)

    S: float = 300.0  # Planform area
    b: float = 30.0  # Span
    cbar: float = 11.32  # Mean aerodynamic chord
    xcgr: float = 0.35  # Reference CG location (% of cbar)
    hx: float = 160.0  # Engine angular momentum (assumed constant)

    def net_forces(self, t, x, u, C_BN):
        """Net forces and moments in body frame B, plus any extra state derivatives
        
        Args:
            t: time
            x: state vector
            u: rotor speeds
            C_BN: rotation matrix from inertial (N) to body (B) frame

        Returns:
            F_B: net forces in body frame B
            M_B: net moments in body frame B
            aux_state_derivs: time derivatives of auxiliary state variables
        """
        weight = self.m * g0

        # Unpack state and controls
        thtl, el, ail, rdr = u
        u, v, w = x.v_B  # Velocity of the center of mass in inertial frame N
        p, q, r = x.w_B  # Angular velocity in body frame (ω_B)
    
        # Dynamic component of engine state (auxiliary state)
        pow = x.aux
        pow_t = _calc_pdot(pow, thtl)

        vt = np.sqrt(u**2 + v**2 + w**2)
        alpha = np.arctan2(w, u)
        beta = np.arcsin(v / vt)

        alpha_deg = np.rad2deg(alpha)
        beta_deg = np.rad2deg(beta)

        # Air data computer and engine model
        alt = x.p_N[2]
        amach, qbar = _adc(vt, alt)
        T_B = _calc_thrust(pow, alt, amach)

        # Lookup tables and component buildup
        cxt = _calc_cx(alpha_deg, el)
        cyt = _calc_cy(beta_deg, ail, rdr)
        czt = _calc_cz(alpha_deg, beta_deg, el)
        dail = ail / 20.0
        drdr = rdr / 30.0
        clt = _calc_cl(alpha_deg, beta_deg) + dlda(alpha_deg, beta_deg) * dail + dldr(alpha_deg, beta_deg) * drdr
        cmt = _calc_cm(alpha_deg, el)
        cnt = _calc_cn(alpha_deg, beta_deg) + dnda(alpha_deg, beta_deg) * dail + dndr(alpha_deg, beta_deg) * drdr

        # Add damping derivatives
        tvt = 0.5 / vt
        b2v = self.b * tvt
        cq = self.cbar * q * tvt
        d = _calc_damp(alpha_deg)
        cxt = cxt + cq * d[0]
        cyt = cyt + b2v * (d[1] * r + d[2] * p)
        czt = czt + cq * d[3]

        clt = clt + b2v * (d[4] * r + d[5] * p)
        cmt = cmt + cq * d[6] + czt * (self.xcgr - self.xcg)
        cnt = cnt + b2v * (d[7] * r + d[8] * p) - cyt * (self.xcgr - self.xcg) * self.cbar / self.b

        Fgrav_N = np.stack([0, 0, weight])
        Faero_B = qbar * self.S * np.stack([cxt, cyt, czt])
        F_B = Faero_B + T_B + C_BN @ Fgrav_N

        # Moments
        Meng_B = self.hx * np.array([0.0, -r, q])
        Maero_B = qbar * self.S * np.array([
            self.b * clt, self.cbar * cmt, self.b * cnt
        ])
        M_B = Meng_B + Maero_B

        aux_state_derivs = np.atleast_1d(pow_t)
        return F_B, M_B, aux_state_derivs

