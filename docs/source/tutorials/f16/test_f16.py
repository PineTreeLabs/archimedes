import pytest
from pathlib import Path
import yaml

import numpy as np

from archimedes.spatial import Rotation, euler_kinematics

from f16 import SubsonicF16, GRAV_FTS2, trim

CURRENT_PATH = Path(__file__).parent


@pytest.fixture
def trim_cases():
    with open(CURRENT_PATH / "_static" / "trim_cases.yaml", "r") as f:
        trim_cases = yaml.safe_load(f)
    return trim_cases


@pytest.fixture
def f16():
    return SubsonicF16(xcg=0.35)


g0 = GRAV_FTS2  # ft/s^2


def test_352(f16: SubsonicF16):
    """Compare to Table 3.5-2 in Lewis, Johnson, Stevens"""
    f16 = f16.replace(xcg=0.40)

    u = f16.Input(
        throttle=0.9,
        elevator=20.0,
        aileron=-15.0,
        rudder=-20.0,
    )

    # Original state used (Vt, alpha, beta) = (500.0, 0.5, -0.2)
    # New model uses equivalent (u, v, w) = (430.0447, -99.3347, 234.9345)
    #   --> (du, dv, dw) = 100.8536, -218.3080, -437.0399
    p_N = np.array([1000.0, 900.0, -10000.0])  # NED-frame position
    rpy = np.array([-1.0, 1.0, -1.0])  # Roll, pitch, yaw
    v_B = np.array([430.0447, -99.3347, 234.9345])  # Velocity in body frame
    w_B = np.array([0.7, -0.8, 0.9])  # Angular velocity in body frame

    att = Rotation.from_euler("xyz", rpy)
    pow = 90.0  # Engine power
    x = f16.State(p_N, att, v_B, w_B, pow)

    # NOTE: There is a typo in the chapter 3 code implementation of the DCM,
    # leading to a sign change for yaw rate xd[11].  Hence, Table 3.5-2 has
    # 248.1241 instead of -248.1241 (the latter is consistent with the SciPy
    # DCM implementation).
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

    x_t = f16.dynamics(0.0, x, u)

    # Extract body angular velocity from quaternion derivative
    # q_t = 0.5 * q ⊗ ω_B
    # => ω_B = 2 * q⁻¹ ⊗ q_t
    # This gives a check on the quaternion derivative calculation
    # without using the roll-pitch-yaw rates.
    w_B_out = 2 * (att.inv().mul(x_t.att, normalize=False)).as_quat()[1:]

    assert np.allclose(x_t.p_N, dp_N_ex, atol=1e-2)
    assert np.allclose(w_B_out, w_B, atol=1e-2)
    assert np.allclose(x_t.v_B, dv_B_ex, atol=1e-2)
    assert np.allclose(x_t.w_B, dw_B_ex, atol=1e-2)


def test_362(f16: SubsonicF16):
    """Trim conditions (Sec. 3.6-2 in Lewis, Johnson, Stevens)"""

    vt = 5.020000e2
    alpha = 2.392628e-1
    beta = 5.061803e-4
    u = vt * np.cos(alpha) * np.cos(beta)
    v = vt * np.sin(beta)
    w = vt * np.sin(alpha) * np.cos(beta)

    p_N = np.array([0.0, 0.0, 0.0])  # NED-frame position
    rpy = np.array([1.366289e0, 5.000808e-2, 2.340769e-1])  # Roll, pitch, yaw
    v_B = np.array([u, v, w])  # Velocity in body frame
    w_B = np.array(
        [-1.499617e-2, 2.933811e-1, 6.084932e-2]
    )  # Angular velocity in body frame

    att = Rotation.from_euler("xyz", rpy)
    pow = 6.412363e1  # Engine power
    x = f16.State(p_N, att, v_B, w_B, pow)

    u = f16.Input(
        throttle=8.349601e-1,
        elevator=-1.481766e0,
        aileron=9.553108e-2,
        rudder=-4.118124e-1,
    )

    x_t = f16.dynamics(0.0, x, u)
    assert np.allclose(x_t.v_B, 0.0, atol=1e-4)
    assert np.allclose(x_t.w_B, 0.0, atol=1e-4)

    # Check that the angle rates are correct
    # First we have to convert the desired angular rates to angular momentum
    rpy_t = np.array([0.0, 0.0, 0.3])  # Roll, pitch-up, turn rates
    H_inv = euler_kinematics(rpy, inverse=True)  # rpy_t -> w_B
    w_B_expected = H_inv @ rpy_t

    # Second, convert the quaternion derivative to angular momentum
    # See notes above on this conversion
    w_B_out = 2 * (att.inv().mul(x_t.att, normalize=False)).as_quat()[1:]
    assert np.allclose(w_B_out, w_B_expected, atol=1e-4)

    # Turn coordination when flight path angle is zero
    # This verifies equation 3.6-7 in Lewis, Johnson, Stevens
    phi = rpy[0]
    H = euler_kinematics(rpy)  # w_B -> rpy_t
    rpy_t = H @ w_B_out
    psi_t = rpy_t[2]  # Turn rate

    G = psi_t * vt / g0
    tph1 = np.tan(phi)
    tph2 = G * np.cos(beta) / (np.cos(alpha) - G * np.sin(alpha) * np.sin(beta))
    assert np.allclose(tph1, tph2)


def test_363(trim_cases):
    """Trim conditions (Sec. 3.6-3 in Stevens, Lewis, Johnson)"""

    tolerances = {
        "alpha": 1e-3,
        "beta": 1e-3,
        "throttle": 1e-3,
        "elevator": 1e-2,
        "aileron": 1e-2,
        "rudder": 1e-1,
    }

    for case in trim_cases:
        print(case["name"])
        print(case["description"])

        f16 = SubsonicF16(xcg=case["xcg"])
        result = trim(f16, **case["condition"])

        # Check that the residuals (v_B and w_B) are small
        assert np.linalg.norm(result.residuals) < 1e-8

        # Check that the trim point matches Table 3.6-3
        for field, atol in tolerances.items():
            expected = float(case["variables"].get(field))
            actual = getattr(result.variables, field)
            close = np.allclose(expected, actual, atol=atol)
            print("\t", field, close)
            assert close