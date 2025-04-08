import numpy as np
import archimedes as arc

gd = 32.17

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

Tidl_interpolant = arc.interpolant([alt_vector, mach_vector], Tidl_data.ravel("F"))
Tmil_interpolant = arc.interpolant([alt_vector, mach_vector], Tmil_data.ravel("F"))
Tmax_interpolant = arc.interpolant([alt_vector, mach_vector], Tmax_data.ravel("F"))


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
cx = arc.interpolant([alpha_vector, ele_vector], cx_data.ravel("F"))

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

cx_interpolant = arc.interpolant([alpha_vector, ele_vector], cx_data.ravel("F"))
cl_interpolant = arc.interpolant([alpha_vector, beta_vector], cl_data.ravel("F"))
cm = arc.interpolant([alpha_vector, ele_vector], cm_data.ravel("F"))
cn_interpolant = arc.interpolant([alpha_vector, beta_vector], cn_data.ravel("F"))

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


dlda = arc.interpolant([alpha_vector, beta_vector], dlda_data.ravel("F"))
dldr = arc.interpolant([alpha_vector, beta_vector], dldr_data.ravel("F"))
dnda = arc.interpolant([alpha_vector, beta_vector], dnda_data.ravel("F"))
dndr = arc.interpolant([alpha_vector, beta_vector], dndr_data.ravel("F"))


def adc(Vt, alt):
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

    rho = R0 * Tfac**4.14
    amach = Vt / np.sqrt(gamma * Rs * T)
    qbar = 0.5 * rho * Vt**2

    return amach, qbar


def tgear(thtl):
    "tgear function"

    if thtl <= 0.77:
        tg = 64.94 * thtl
    else:
        tg = 217.38 * thtl - 117.38

    return tg


def rtau(dp):
    "rtau function"

    if dp <= 25:
        rt = 1.0
    elif dp >= 50:
        rt = 0.1
    else:
        rt = 1.9 - 0.036 * dp

    return rt


def pdot(p3, p1):
    "pdot function"

    if p1 >= 50:
        if p3 >= 50:
            t = 5
            p2 = p1
        else:
            p2 = 60
            t = rtau(p2 - p3)
    else:
        if p3 >= 50:
            t = 5
            p2 = 40
        else:
            p2 = p1
            t = rtau(p2 - p3)

    pd = t * (p2 - p3)

    return pd


def thrust(pow, alt, rmach):
    T_mil = Tmil_interpolant(alt, rmach)
    T_idl = Tidl_interpolant(alt, rmach)
    T_max = Tmax_interpolant(alt, rmach)

    return np.where(
        pow < 50.0,
        T_idl + (T_mil - T_idl) * pow * 0.02,
        T_mil + (T_max - T_mil) * (pow - 50.0) * 0.02,
    )


def cy(beta, ail, rdr):
    return -0.02 * beta + 0.021 * (ail / 20) + 0.086 * (rdr / 30)


def cz(alpha, beta, el):
    cz_lookup = np.interp(alpha, alpha_vector, cz_data)
    return (-0.19 / 25) * el + cz_lookup * (1.0 - (beta / 57.3) ** 2)


def cl(alpha, beta):
    return np.sign(beta) * cl_interpolant(alpha, np.abs(beta))


def cn(alpha, beta):
    return np.sign(beta) * cn_interpolant(alpha, np.abs(beta))


def damp(alpha):
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


def f16(x, controls, xcg=0.35, weight=20490.4459):
    thtl, el, ail, rdr = controls
    xd = np.zeros_like(x)

    Axx = 9496.0
    Ayy = 55814.0
    Azz = 63100.0
    Axz = 982.0
    Axzs = Axz**2
    xpq = Axz * (Axx - Ayy + Azz)
    gam = Axx * Azz - Axz**2
    xqr = Azz * (Azz - Ayy) + Axzs
    zpq = (Axx - Ayy) * Axx + Axzs
    ypr = Azz - Axx
    # weight = 25000.0  # Appendix A
    # weight = 20490.4459  # From aerobench (matches Table 3.5-2)
    gd = 32.17
    mass = weight / gd

    s = 300.0
    b = 30.0
    cbar = 11.32
    xcgr = 0.35  # Reference CG location
    hx = 160.0
    rtod = 57.29578

    # Assign state & control variables
    u, v, w = x[:3]

    vt = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)
    beta = np.arcsin(v / vt)

    alpha_deg = alpha * rtod
    beta_deg = beta * rtod
    phi = x[3]
    theta = x[4]
    psi = x[5]
    p = x[6]
    q = x[7]
    r = x[8]
    alt = x[11]
    pow = x[12]

    # Air data computer and engine model
    amach, qbar = adc(vt, alt)
    cpow = tgear(thtl)
    xd[12] = pdot(pow, cpow)
    t = thrust(pow, alt, amach)

    # Lookup tables and component buildup
    cxt = cx(alpha_deg, el)
    cyt = cy(beta_deg, ail, rdr)
    czt = cz(alpha_deg, beta_deg, el)
    dail = ail / 20.0
    drdr = rdr / 30.0
    clt = (
        cl(alpha_deg, beta_deg)
        + dlda(alpha_deg, beta_deg) * dail
        + dldr(alpha_deg, beta_deg) * drdr
    )
    cmt = cm(alpha_deg, el)
    cnt = (
        cn(alpha_deg, beta_deg)
        + dnda(alpha_deg, beta_deg) * dail
        + dndr(alpha_deg, beta_deg) * drdr
    )

    # Add damping derivatives
    tvt = 0.5 / vt
    b2v = b * tvt
    cq = cbar * q * tvt
    d = damp(alpha_deg)
    cxt = cxt + cq * d[0]
    cyt = cyt + b2v * (d[1] * r + d[2] * p)
    czt = czt + cq * d[3]

    clt = clt + b2v * (d[4] * r + d[5] * p)
    cmt = cmt + cq * d[6] + czt * (xcgr - xcg)
    cnt = cnt + b2v * (d[7] * r + d[8] * p) - cyt * (xcgr - xcg) * cbar / b

    # Get ready for state equations
    sth = np.sin(theta)
    cth = np.cos(theta)
    sph = np.sin(phi)
    cph = np.cos(phi)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)
    qs = qbar * s
    qsb = qs * b
    rmqs = qs / mass
    gcth = gd * cth
    qsph = q * sph
    ay = rmqs * cyt
    az = rmqs * czt

    # Force equations
    xd[0] = r * v - q * w - gd * sth + (qs * cxt + t) / mass
    xd[1] = p * w - r * u + gcth * sph + ay
    xd[2] = q * u - p * v + gcth * cph + az

    # kinematics
    xd[3] = p + (sth / cth) * (qsph + r * cph)
    xd[4] = q * cph - r * sph
    xd[5] = (qsph + r * cph) / cth

    # Moments
    roll = qsb * clt
    pitch = qs * cbar * cmt
    yaw = qsb * cnt
    pq = p * q
    qr = q * r
    qhx = q * hx
    xd[6] = (xpq * pq - xqr * qr + Azz * roll + Axz * (yaw + qhx)) / gam
    xd[7] = (ypr * p * r - Axz * (p**2 - r**2) + pitch - r * hx) / Ayy
    xd[8] = (zpq * pq - xpq * qr + Axz * roll + Axx * (yaw + qhx)) / gam

    # navigation
    t1 = sph * cpsi
    t2 = cph * sth
    t3 = sph * spsi
    s1 = cth * cpsi
    s2 = cth * spsi
    s3 = t1 * sth - cph * spsi
    s4 = t3 * sth + cph * cpsi
    s5 = sph * cth
    s6 = t2 * cpsi + t3
    s7 = t2 * spsi - t1
    s8 = cph * cth
    xd[9] = u * s1 + v * s3 + w * s6  # north speed
    xd[10] = u * s2 + v * s4 + w * s7  # east speed
    xd[11] = -u * sth + v * s5 + w * s8  # vertical speed

    return xd


if __name__ == "__main__":
    #
    # From Lewis, Johnson, Stevens Table 3.5-2
    #

    u = np.array([0.9, 20.0, -15.0, -20.0])

    # Original state used (Vt, alpha, beta) = (500.0, 0.5, -0.2)
    # New model uses equivalent (u, v, w) = (430.0447, -99.3347, 234.9345)
    #   --> (du, dv, dw) = 100.8536, -218.3080, -437.0399
    x = np.array(
        [
            430.0447,
            -99.3347,
            234.9345,
            -1.0,
            1.0,
            -1.0,
            0.7,
            -0.8,
            0.9,
            1000.0,
            900.0,
            10000.0,
            90.0,
        ]
    )

    # NOTE: There is a typo in the chapter 3 code implementation of the DCM,
    # leading to a sign change for yaw rate xd[11].  Hence, Table 3.5-2 has
    # 248.1241 instead of -248.1241 (the latter is consistent with the SciPy
    # DCM implementation).
    xd_expected = np.array(
        [
            100.8536,
            -218.3080,
            -437.0399,
            2.505734,
            0.3250820,
            2.145926,
            12.62679,
            0.9649671,
            0.5809759,
            342.4439,
            -266.7707,
            -248.1241,
            -58.68999,
        ]
    )

    xd = f16(x, u, xcg=0.4)
    print(f"Test 3.5: {np.allclose(xd, xd_expected, atol=1e-2)}")
    # assert np.allclose(xd, xd_expected, atol=1e-2)

    #
    # Trim conditions (3.6)
    #
    vt = 5.020000e2
    alpha = 2.392628e-1
    beta = 5.061803e-4
    u = vt * np.cos(alpha) * np.cos(beta)
    v = vt * np.sin(beta)
    w = vt * np.sin(alpha) * np.cos(beta)

    thtl = 8.349601e-1
    el = -1.481766e0
    ail = 9.553108e-2
    rdr = -4.118124e-1

    x = np.array(
        [
            u,
            v,
            w,
            1.366289e0,
            5.000808e-2,
            2.340769e-1,
            -1.499617e-2,
            2.933811e-1,
            6.084932e-2,
            0.0,
            0.0,
            0.0,
            6.412363e1,
        ]
    )

    u = np.array([thtl, el, ail, rdr])

    # x[12] = tgear(thtl)
    # x = constr(x)  # Apply rate-of-climb and turn coordination constraints

    xd = f16(x, u, xcg=0.35)
    zero_idx = [0, 1, 2, 6, 7, 8]
    print(xd[zero_idx])
    print(f"\nTest 3.6")
    print(f"\tSteady:\t\t{np.allclose(xd[zero_idx], 0.0, atol=1e-4)}")

    print(f"\tRoll rate:\t{np.allclose(xd[3], 0.0, atol=1e-4)}")
    print(f"\tPitch rate:\t{np.allclose(xd[4], 0.0, atol=1e-4)}")
    print(f"\tTurn rate:\t{np.allclose(xd[5], 0.3, atol=1e-4)}")

    # Turn coordination when flight path angle is zero
    phi, theta = x[3:5]
    G = xd[5] * vt / gd
    tph1 = np.tan(phi)
    tph2 = G * np.cos(beta) / (np.cos(alpha) - G * np.sin(alpha) * np.sin(beta))
    print(f"\tTurn coord\t: {np.allclose(tph1, tph2)}")

    # Trim conditions (3.6-3)
    #
    vt = 5.020000e2
    alpha = 0.03691
    beta = -4e-9
    u = vt * np.cos(alpha) * np.cos(beta)
    v = vt * np.sin(beta)
    w = vt * np.sin(alpha) * np.cos(beta)

    thtl = 0.1385
    el = np.deg2rad(-0.7588)
    ail = np.deg2rad(-1.2e-7)
    rdr = np.deg2rad(-6.2e-7)

    x = np.array(
        [
            u,
            v,
            w,
            0.0,
            0.03691,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            tgear(thtl),
        ]
    )

    u = np.array([thtl, el, ail, rdr])

    xd = f16(x, u, xcg=0.35)
    zero_idx = [0, 1, 2, 6, 7, 8]

    # FIXME!!
    print("\nTest 3.6-3:")
    print(xd[zero_idx])
    print(f"\tSteady:\t{np.allclose(xd[zero_idx], 0.0, atol=1e-4)}")

    # print(xd)

    print("All tests completed!")
