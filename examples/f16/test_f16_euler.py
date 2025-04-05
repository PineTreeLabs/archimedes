import pytest
import numpy as np

# from f16_plant_euler import SubsonicF16
from f16_plant import SubsonicF16


g0 = 32.17

# NOTE: The weight in the textbook is 25,000 lbs, but this
# does not give consistent values - the default value here
# matches the values given in the tables
weight = 20490.4459

Axx = 9496.0
Ayy = 55814.0
Azz = 63100.0
Axz = -982.0
mass = weight / g0

J_B = np.array([
    [Axx, 0.0, Axz],
    [0.0, Ayy, 0.0],
    [Axz, 0.0, Azz],
])

def test_352():
    """Compare to Table 3.5-2 in Lewis, Johnson, Stevens"""

    u = np.array([0.9, 20.0, -15.0, -20.0])

    # Original state used (Vt, alpha, beta) = (500.0, 0.5, -0.2)
    # New model uses equivalent (u, v, w) = (430.0447, -99.3347, 234.9345)
    #   --> (du, dv, dw) = 100.8536, -218.3080, -437.0399
    x = np.array([
        1000.0,  # x (north)
        900.0,  # y (east)
        10000.0,  # z (down)
        -1.0,  # phi (roll)
        1.0,  # theta (pitch)
        -1.0,  # psi (yaw)
        430.0447,  # u
        -99.3347,  # v
        234.9345,  # w
        0.7,  # p
        -0.8,  # q
        0.9,  # r
        90.0,  # engine power
    ])

    # NOTE: There is a typo in the chapter 3 code implementation of the DCM,
    # leading to a sign change for yaw rate xd[11].  Hence, Table 3.5-2 has
    # 248.1241 instead of -248.1241 (the latter is consistent with the SciPy
    # DCM implementation).
    xd_expected = np.array([
        342.4439,  # x (north)
        -266.7707,  # y (east)
        -248.1241,  # z (down)
        2.505734,  # phi (roll)
        0.3250820,  # theta (pitch)
        2.145926,  # psi (yaw)
        100.8536,  # u
        -218.3080,  # v
        -437.0399,  # w
        12.62679,  # p
        0.9649671,  # q
        0.5809759,  # r
        -58.68999,  # engine power
    ])

    model = SubsonicF16(m=mass, J_B=J_B, xcg=0.4, attitude="euler")
    xd = model.dynamics(0.0, x, u)
    assert np.allclose(xd - xd_expected, 0.0, atol=1e-2)


def test_36():
    """Trim conditions (Sec. 3.6 in Lewis, Johnson, Stevens)"""
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

    x = np.array([
        0.0,  # x (north)
        0.0,  # y (east)
        0.0,  # z (down)
        1.366289e0,  # phi (roll)
        5.000808e-2,  # theta (pitch)
        2.340769e-1,  # psi (yaw)
        u,
        v,
        w,
        -1.499617e-2,  # p
        2.933811e-1,  # q
        6.084932e-2,  # r
        6.412363e1,  # engine power
    ])

    u = np.array([thtl, el, ail, rdr])

    model = SubsonicF16(m=mass, J_B=J_B, xcg=0.35, attitude="euler")
    xd = model.dynamics(0.0, x, u)
    zero_idx = [6, 7, 8, 9, 10, 11]
    assert np.allclose(xd[zero_idx], 0.0, atol=1e-4)

    # Check command rates
    assert np.allclose(xd[3], 0.0, atol=1e-4)  # Roll rate
    assert np.allclose(xd[4], 0.0, atol=1e-4)  # Pitch-up rate
    assert np.allclose(xd[5], 0.3, atol=1e-4)  # Turn rate

    # Turn coordination when flight path angle is zero
    phi, theta = x[3:5]
    G = xd[5] * vt / g0
    tph1 = np.tan(phi)
    tph2 = G * np.cos(beta) / (np.cos(alpha) - G * np.sin(alpha) * np.sin(beta))
    assert np.allclose(tph1, tph2)

