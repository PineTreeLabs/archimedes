from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

import archimedes as arc
from archimedes import struct

if TYPE_CHECKING:
    from f16_plant import SubsonicF16

#
# Aerodynamics lookup tables
#

# Angle of attack data for lookup tables
alpha_vector = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])

# Sideslip angle data for lookup tables
beta_vector = np.array([0, 5, 10, 15, 20, 25, 30])

# Elevator deflection data for lookup tables
ele_vector = np.array([-24, -12, 0, 12, 24])

# Cx(alpha, ele)
cx_data = np.array(
    [
        [-0.099, -0.048, -0.022, -0.04, -0.083],
        [-0.081, -0.038, -0.02, -0.038, -0.073],
        [-0.081, -0.04, -0.021, -0.039, -0.076],
        [-0.063, -0.021, -0.004, -0.025, -0.072],
        [-0.025, 0.016, 0.032, 0.006, -0.046],
        [0.044, 0.083, 0.094, 0.062, 0.012],
        [0.097, 0.127, 0.128, 0.087, 0.024],
        [0.113, 0.137, 0.13, 0.085, 0.025],
        [0.145, 0.162, 0.154, 0.1, 0.043],
        [0.167, 0.177, 0.161, 0.11, 0.053],
        [0.174, 0.179, 0.155, 0.104, 0.047],
        [0.166, 0.167, 0.138, 0.091, 0.04],
    ]
)

# Cz(alpha)
cz_data = [
    0.770,
    0.241,
    -0.100,
    -0.416,
    -0.731,
    -1.053,
    -1.366,
    -1.646,
    -1.917,
    -2.120,
    -2.248,
    -2.229,
]

# Cl(alpha, beta)
# TODO: Should be able to use the textbook numbers here
cl_data = np.array(
    [
        [0.0, -0.001, -0.003, -0.001, 0.0, 0.007, 0.009],
        [0.0, -0.004, -0.009, -0.01, -0.01, -0.01, -0.011],
        [0.0, -0.008, -0.017, -0.02, -0.022, -0.023, -0.023],
        [0.0, -0.012, -0.024, -0.03, -0.034, -0.034, -0.037],
        [0.0, -0.016, -0.03, -0.039, -0.047, -0.049, -0.05],
        [0.0, -0.019, -0.034, -0.044, -0.046, -0.046, -0.047],
        [0.0, -0.02, -0.04, -0.05, -0.059, -0.068, -0.074],
        [0.0, -0.02, -0.037, -0.049, -0.061, -0.071, -0.079],
        [0.0, -0.015, -0.016, -0.023, -0.033, -0.06, -0.091],
        [0.0, -0.008, -0.002, -0.006, -0.036, -0.058, -0.076],
        [0.0, -0.013, -0.01, -0.014, -0.035, -0.062, -0.077],
        [0.0, -0.015, -0.019, -0.027, -0.035, -0.059, -0.076],
    ]
)  # Textbook data

# Cm(alpha, ele)
cm_data = np.array(
    [
        [0.205, 0.081, -0.046, -0.174, -0.259],
        [0.168, 0.077, -0.02, -0.145, -0.202],
        [0.186, 0.107, -0.009, -0.121, -0.184],
        [0.196, 0.11, -0.005, -0.127, -0.193],
        [0.213, 0.11, -0.006, -0.129, -0.199],
        [0.251, 0.141, 0.01, -0.102, -0.15],
        [0.245, 0.127, 0.006, -0.097, -0.16],
        [0.238, 0.119, -0.001, -0.113, -0.167],
        [0.252, 0.133, 0.014, -0.087, -0.104],
        [0.231, 0.108, 0.0, -0.084, -0.076],
        [0.198, 0.081, -0.013, -0.069, -0.041],
        [0.192, 0.093, 0.032, -0.006, -0.005],
    ]
)

# Cn(alpha, beta)
cn_data = np.array(
    [
        [0.0, 0.018, 0.038, 0.056, 0.064, 0.074, 0.079],
        [0.0, 0.019, 0.042, 0.057, 0.077, 0.086, 0.09],
        [0.0, 0.018, 0.042, 0.059, 0.076, 0.093, 0.106],
        [0.0, 0.019, 0.042, 0.058, 0.074, 0.089, 0.106],
        [0.0, 0.019, 0.043, 0.058, 0.073, 0.08, 0.096],
        [0.0, 0.018, 0.039, 0.053, 0.057, 0.062, 0.08],
        [0.0, 0.013, 0.03, 0.032, 0.029, 0.049, 0.068],
        [0.0, 0.007, 0.017, 0.012, 0.007, 0.022, 0.03],
        [0.0, 0.004, 0.004, 0.002, 0.012, 0.028, 0.064],
        [0.0, -0.014, -0.035, -0.046, -0.034, -0.012, 0.015],
        [0.0, -0.017, -0.047, -0.071, -0.065, -0.002, 0.011],
        [0.0, -0.033, -0.057, -0.073, -0.041, -0.013, -0.001],
    ]
)

cx_interpolant = arc.interpolant([alpha_vector, ele_vector], cx_data)
cl_interpolant = arc.interpolant([alpha_vector, beta_vector], cl_data)
cm_interpolant = arc.interpolant([alpha_vector, ele_vector], cm_data)
cn_interpolant = arc.interpolant([alpha_vector, beta_vector], cn_data)

#
# Control surfaces coefficients
#
# NOTE: Different beta_vector than for Cl and Cn
beta_vector = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])

dlda_data = np.array(
    [
        [-0.041, -0.041, -0.042, -0.04, -0.043, -0.044, -0.043],
        [-0.052, -0.053, -0.053, -0.052, -0.049, -0.048, -0.049],
        [-0.053, -0.053, -0.052, -0.051, -0.048, -0.048, -0.047],
        [-0.056, -0.053, -0.051, -0.052, -0.049, -0.047, -0.045],
        [-0.05, -0.05, -0.049, -0.048, -0.043, -0.042, -0.042],
        [-0.056, -0.051, -0.049, -0.048, -0.042, -0.041, -0.037],
        [-0.082, -0.066, -0.043, -0.042, -0.042, -0.02, -0.003],
        [-0.059, -0.043, -0.035, -0.037, -0.036, -0.028, -0.013],
        [-0.042, -0.038, -0.026, -0.031, -0.025, -0.013, -0.01],
        [-0.038, -0.027, -0.016, -0.026, -0.021, -0.014, -0.003],
        [-0.027, -0.023, -0.018, -0.017, -0.016, -0.011, -0.007],
        [-0.017, -0.016, -0.014, -0.012, -0.011, -0.01, -0.008],
    ]
)

dldr_data = np.array(
    [
        [0.005, 0.007, 0.013, 0.018, 0.015, 0.021, 0.023],
        [0.017, 0.016, 0.013, 0.015, 0.014, 0.011, 0.01],
        [0.014, 0.014, 0.011, 0.015, 0.013, 0.01, 0.011],
        [0.01, 0.014, 0.012, 0.014, 0.013, 0.011, 0.011],
        [-0.005, 0.013, 0.011, 0.014, 0.012, 0.01, 0.011],
        [0.009, 0.009, 0.009, 0.014, 0.011, 0.009, 0.01],
        [0.019, 0.012, 0.008, 0.014, 0.011, 0.008, 0.008],
        [0.005, 0.005, 0.005, 0.015, 0.01, 0.01, 0.01],
        [0.0, 0.0, -0.002, 0.013, 0.008, 0.006, 0.006],
        [-0.005, 0.004, 0.005, 0.011, 0.008, 0.005, 0.014],
        [-0.011, 0.009, 0.003, 0.006, 0.007, 0.0, 0.02],
        [0.008, 0.007, 0.005, 0.001, 0.003, 0.001, 0.0],
    ]
)

dnda_data = np.array(
    [
        [0.001, 0.002, -0.006, -0.011, -0.015, -0.024, -0.022],
        [-0.027, -0.014, -0.008, -0.011, -0.015, -0.01, 0.002],
        [-0.017, -0.016, -0.006, -0.01, -0.014, -0.004, -0.003],
        [-0.013, -0.016, -0.006, -0.009, -0.012, -0.002, -0.005],
        [-0.012, -0.014, -0.005, -0.008, -0.011, -0.001, -0.003],
        [-0.016, -0.019, -0.008, -0.006, -0.008, 0.003, -0.001],
        [0.001, -0.021, -0.005, 0.0, -0.002, 0.014, -0.009],
        [0.017, 0.002, 0.007, 0.004, 0.002, 0.006, -0.009],
        [0.011, 0.012, 0.004, 0.007, 0.006, -0.001, -0.001],
        [0.017, 0.016, 0.007, 0.01, 0.012, 0.004, 0.003],
        [0.008, 0.015, 0.006, 0.004, 0.011, 0.004, -0.002],
        [0.016, 0.011, 0.006, 0.01, 0.011, 0.006, 0.001],
    ]
)

dndr_data = np.array(
    [
        [-0.018, -0.028, -0.037, -0.048, -0.043, -0.052, -0.062],
        [-0.052, -0.051, -0.041, -0.045, -0.044, -0.034, -0.034],
        [-0.052, -0.043, -0.038, -0.045, -0.041, -0.036, -0.027],
        [-0.052, -0.046, -0.04, -0.045, -0.041, -0.036, -0.028],
        [-0.054, -0.045, -0.04, -0.044, -0.04, -0.035, -0.027],
        [-0.049, -0.049, -0.038, -0.045, -0.038, -0.028, -0.027],
        [-0.059, -0.057, -0.037, -0.047, -0.034, -0.024, -0.023],
        [-0.051, -0.052, -0.03, -0.048, -0.035, -0.023, -0.023],
        [-0.03, -0.03, -0.027, -0.049, -0.035, -0.02, -0.019],
        [-0.037, -0.033, -0.024, -0.045, -0.029, -0.016, -0.009],
        [-0.026, -0.03, -0.019, -0.033, -0.022, -0.01, -0.025],
        [-0.013, -0.008, -0.013, -0.016, -0.009, -0.014, -0.01],
    ]
)


dlda = arc.interpolant([alpha_vector, beta_vector], dlda_data)
dldr = arc.interpolant([alpha_vector, beta_vector], dldr_data)
dnda = arc.interpolant([alpha_vector, beta_vector], dnda_data)
dndr = arc.interpolant([alpha_vector, beta_vector], dndr_data)


@struct
class F16Aerodynamics:
    def __call__(self, vt, alpha, beta, w_B, el, ail, rdr, vehicle: SubsonicF16):
        p, q, r = w_B  # Angular velocity in body frame (ω_B)

        # Lookup tables and component buildup
        alpha_deg = np.rad2deg(alpha)
        beta_deg = np.rad2deg(beta)
        cxt = self._calc_cx(alpha_deg, el)
        cyt = self._calc_cy(beta_deg, ail, rdr)
        czt = self._calc_cz(alpha_deg, beta_deg, el)
        dail = ail / 20.0
        drdr = rdr / 30.0
        clt = (
            self._calc_cl(alpha_deg, beta_deg)
            + dlda(alpha_deg, beta_deg) * dail
            + dldr(alpha_deg, beta_deg) * drdr
        )
        cmt = self._calc_cm(alpha_deg, el)
        cnt = (
            self._calc_cn(alpha_deg, beta_deg)
            + dnda(alpha_deg, beta_deg) * dail
            + dndr(alpha_deg, beta_deg) * drdr
        )

        # Add damping derivatives
        tvt = 0.5 / vt
        b2v = vehicle.b * tvt
        cq = vehicle.cbar * q * tvt
        d = self._calc_damp(alpha_deg)
        cxt = cxt + cq * d[0]
        cyt = cyt + b2v * (d[1] * r + d[2] * p)
        czt = czt + cq * d[3]

        clt = clt + b2v * (d[4] * r + d[5] * p)
        cmt = cmt + cq * d[6] + czt * (vehicle.xcgr - vehicle.xcg)
        cnt = (
            cnt
            + b2v * (d[7] * r + d[8] * p)
            - cyt * (vehicle.xcgr - vehicle.xcg) * vehicle.cbar / vehicle.b
        )

        force_coeffs = np.hstack([cxt, cyt, czt])
        moment_coeffs = np.hstack([clt, cmt, cnt])
        return force_coeffs, moment_coeffs

    def _calc_cx(self, alpha, el):
        return cx_interpolant(alpha, el)

    def _calc_cy(self, beta, ail, rdr):
        return -0.02 * beta + 0.021 * (ail / 20) + 0.086 * (rdr / 30)

    def _calc_cz(self, alpha, beta, el):
        cz_lookup = np.interp(alpha, alpha_vector, cz_data)
        return (-0.19 / 25) * el + cz_lookup * (1.0 - (beta / 57.3) ** 2)

    def _calc_cl(self, alpha, beta):
        return np.sign(beta) * cl_interpolant(alpha, np.abs(beta))

    def _calc_cm(self, alpha, el):
        return cm_interpolant(alpha, el)

    def _calc_cn(self, alpha, beta):
        return np.sign(beta) * cn_interpolant(alpha, np.abs(beta))

    def _calc_damp(self, alpha):
        Cxq_data = np.array(
            [-0.267, 0.110, 0.308, 1.34, 2.08, 2.91, 2.76, 2.05, 1.5, 1.49, 1.83, 1.21]
        )
        Cyr_data = np.array(
            [
                0.882,
                0.852,
                0.876,
                0.958,
                0.962,
                0.974,
                0.819,
                0.483,
                0.590,
                1.21,
                -0.493,
                -1.04,
            ]
        )
        Cyp_data = np.array(
            [
                -0.108,
                -0.108,
                -0.188,
                0.110,
                0.258,
                0.226,
                0.344,
                0.362,
                0.611,
                0.529,
                0.298,
                -2.27,
            ]
        )
        Czq_data = np.array(
            [
                -8.8,
                -25.8,
                -28.9,
                -31.4,
                -31.2,
                -30.7,
                -27.7,
                -28.2,
                -29,
                -29.8,
                -38.3,
                -35.3,
            ]
        )

        Clr_data = np.array(
            [
                -0.126,
                -0.026,
                0.063,
                0.113,
                0.208,
                0.230,
                0.319,
                0.437,
                0.680,
                0.1,
                0.447,
                -0.330,
            ]
        )
        Clp_data = np.array(
            [
                -0.36,
                -0.359,
                -0.443,
                -0.42,
                -0.383,
                -0.375,
                -0.329,
                -0.294,
                -0.23,
                -0.21,
                -0.12,
                -0.1,
            ]
        )
        Cmq_data = np.array(
            [-7.21, -0.54, -5.23, -5.26, -6.11, -6.64, -5.69, -6, -6.2, -6.4, -6.6, -6]
        )
        Cnr_data = np.array(
            [
                -0.38,
                -0.363,
                -0.378,
                -0.386,
                -0.37,
                -0.453,
                -0.55,
                -0.582,
                -0.595,
                -0.637,
                -1.02,
                -0.84,
            ]
        )
        Cnp_data = np.array(
            [
                0.061,
                0.052,
                0.052,
                -0.012,
                -0.013,
                -0.024,
                0.05,
                0.15,
                0.13,
                0.158,
                0.24,
                0.15,
            ]
        )

        return np.stack(
            [
                np.interp(alpha, alpha_vector, Cxq_data),
                np.interp(alpha, alpha_vector, Cyr_data),
                np.interp(alpha, alpha_vector, Cyp_data),
                np.interp(alpha, alpha_vector, Czq_data),
                np.interp(alpha, alpha_vector, Clr_data),
                np.interp(alpha, alpha_vector, Clp_data),
                np.interp(alpha, alpha_vector, Cmq_data),
                np.interp(alpha, alpha_vector, Cnr_data),
                np.interp(alpha, alpha_vector, Cnp_data),
            ]
        )
