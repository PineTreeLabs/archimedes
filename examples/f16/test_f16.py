import numpy as np

from archimedes.experimental.aero import (
    euler_to_quaternion,
    euler_kinematics,
    quaternion_multiply,
    quaternion_inverse,
)

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

J_B = np.array(
    [
        [Axx, 0.0, Axz],
        [0.0, Ayy, 0.0],
        [Axz, 0.0, Azz],
    ]
)


def test_352():
    """Compare to Table 3.5-2 in Lewis, Johnson, Stevens"""

    u = np.array([0.9, 20.0, -15.0, -20.0])

    # Original state used (Vt, alpha, beta) = (500.0, 0.5, -0.2)
    # New model uses equivalent (u, v, w) = (430.0447, -99.3347, 234.9345)
    #   --> (du, dv, dw) = 100.8536, -218.3080, -437.0399
    p_N = np.array([1000.0, 900.0, -10000.0])  # NED-frame position
    rpy = np.array([-1.0, 1.0, -1.0])  # Roll, pitch, yaw
    v_B = np.array([430.0447, -99.3347, 234.9345])  # Velocity in body frame
    w_B = np.array([0.7, -0.8, 0.9])  # Angular velocity in body frame
    q = euler_to_quaternion(rpy)
    pow = 90.0  # Engine power
    x = SubsonicF16.State(p_N, q, v_B, w_B, pow)

    # NOTE: There is a typo in the chapter 3 code implementation of the DCM,
    # leading to a sign change for yaw rate xd[11].  Hence, Table 3.5-2 has
    # 248.1241 instead of -248.1241 (the latter is consistent with the SciPy
    # DCM implementation).
    q_t = 0.5 * quaternion_multiply(q, np.array([0, *w_B]))
    drpy_ex = np.array(
        [
            2.505734,  # phi (roll)
            0.3250820,  # theta (pitch)
            2.145926,  # psi (yaw)
        ]
    )
    dp_N_ex = np.array(
        [
            342.4439,  # x (north)
            -266.7707,  # y (east)
            -248.1241,  # z (down)
        ]
    )
    dv_B_ex = np.array(
        [
            100.8536,  # u
            -218.3080,  # v
            -437.0399,  # w
        ]
    )
    dw_B_ex = np.array(
        [
            12.62679,  # p
            0.9649671,  # q
            0.5809759,  # r
        ]
    )

    model = SubsonicF16(m=mass, J_B=J_B, xcg=0.4)
    x_t = model.dynamics(0.0, x, u)
    assert np.allclose(x_t.p_N, dp_N_ex, atol=1e-2)
    assert np.allclose(x_t.att, q_t, atol=1e-2)
    assert np.allclose(x_t.v_B, dv_B_ex, atol=1e-2)
    assert np.allclose(x_t.w_B, dw_B_ex, atol=1e-2)

    # Check with Euler attitude representation
    model = SubsonicF16(m=mass, J_B=J_B, xcg=0.4, attitude="euler")
    x = SubsonicF16.State(p_N, rpy, v_B, w_B, pow)
    x_t = model.dynamics(0.0, x, u)
    assert np.allclose(x_t.p_N, dp_N_ex, atol=1e-2)
    assert np.allclose(x_t.att, drpy_ex, atol=1e-2)
    assert np.allclose(x_t.v_B, dv_B_ex, atol=1e-2)
    assert np.allclose(x_t.w_B, dw_B_ex, atol=1e-2)


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

    p_N = np.array([0.0, 0.0, 0.0])  # NED-frame position
    rpy = np.array([1.366289e0, 5.000808e-2, 2.340769e-1])  # Roll, pitch, yaw
    v_B = np.array([u, v, w])  # Velocity in body frame
    w_B = np.array(
        [-1.499617e-2, 2.933811e-1, 6.084932e-2]
    )  # Angular velocity in body frame
    q = euler_to_quaternion(rpy)
    pow = 6.412363e1  # Engine power
    x = SubsonicF16.State(p_N, q, v_B, w_B, pow)

    u = np.array([thtl, el, ail, rdr])

    model = SubsonicF16(m=mass, J_B=J_B, xcg=0.35)
    x_t = model.dynamics(0.0, x, u)
    assert np.allclose(x_t.v_B, 0.0, atol=1e-4)
    assert np.allclose(x_t.w_B, 0.0, atol=1e-4)

    # Check that the angle rates are correct
    # First we have to convert the desired angular rates to angular momentum
    rpy_t = np.array([0.0, 0.0, 0.3])  # Roll, pitch-up, turn rates
    H_inv = euler_kinematics(rpy, inverse=True)  # rpy_t -> w_B
    w_B_expected = H_inv @ rpy_t
    # Second, convert the angular momentum to a quaternion rate
    q_t_expected = 0.5 * quaternion_multiply(q, np.array([0, *w_B_expected]))
    assert np.allclose(x_t.att, q_t_expected, atol=1e-4)

    # Turn coordination when flight path angle is zero
    # This verifies equation 3.6-7 in Lewis, Johnson, Stevens
    phi = rpy[0]
    w_B = 2 * quaternion_multiply(quaternion_inverse(q), x_t.att)
    w_B = w_B[1:]  # Discard the scalar component of the angular momentum "quaternion"
    H = euler_kinematics(rpy)  # w_B -> rpy_t
    rpy_t = H @ w_B
    psi_t = rpy_t[2]  # Turn rate

    G = psi_t * vt / g0
    tph1 = np.tan(phi)
    tph2 = G * np.cos(beta) / (np.cos(alpha) - G * np.sin(alpha) * np.sin(beta))
    assert np.allclose(tph1, tph2)

    # Recheck with Euler attitude representation
    model = SubsonicF16(m=mass, J_B=J_B, xcg=0.35, attitude="euler")
    x = SubsonicF16.State(p_N, rpy, v_B, w_B, pow)
    x_t = model.dynamics(0.0, x, u)

    assert np.allclose(x_t.v_B, 0.0, atol=1e-4)
    assert np.allclose(x_t.w_B, 0.0, atol=1e-4)

    # Check command rates
    assert np.allclose(x_t.att[0], 0.0, atol=1e-4)  # Roll rate
    assert np.allclose(x_t.att[1], 0.0, atol=1e-4)  # Pitch-up rate
    assert np.allclose(x_t.att[2], 0.3, atol=1e-4)  # Turn rate

    # Turn coordination when flight path angle is zero
    phi, theta, _psi = x.att
    G = x_t.att[2] * vt / g0
    tph1 = np.tan(phi)
    tph2 = G * np.cos(beta) / (np.cos(alpha) - G * np.sin(alpha) * np.sin(beta))
    assert np.allclose(tph1, tph2)
