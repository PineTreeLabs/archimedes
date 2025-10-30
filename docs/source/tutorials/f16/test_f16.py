import pytest
from pathlib import Path
import yaml

import numpy as np
import numpy.testing as npt

import archimedes as arc
from archimedes.spatial import Rotation, RigidBody, euler_kinematics

from f16 import SubsonicF16, GRAV_FTS2, trim
from stability import (
    LongitudinalState,
    LongitudinalInput,
    LateralState,
    LateralInput,
    StabilityState,
    StabilityInput,
)

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
    x_eng = f16.engine.State(90.0)  # Engine power
    x = f16.State(p_N, att, v_B, w_B, x_eng)

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
    x_eng = f16.engine.State(6.412363e1)  # Engine power
    x = f16.State(p_N, att, v_B, w_B, x_eng)

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


def test_trim(trim_cases):
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


def test_linearization(trim_cases):
    """Linearization about a trim point (Sec. 3.7 in Stevens, Lewis, Johnson)"""
    case = next(case for case in trim_cases if case["name"] == "pull_up")
    print(case)

    rigid_body = RigidBody(rpy_attitude=True)
    f16 = SubsonicF16(rigid_body=rigid_body, xcg=case["xcg"])
    result = trim(f16, **case["condition"])
        
    x0 = StabilityState.from_full_state(result.state)
    u0 = StabilityInput.from_full_input(result.inputs)

    x0_flat, unravel_x = arc.tree.ravel(x0)
    u0_flat, unravel_u = arc.tree.ravel(u0)

    def unravel_stab(x_flat, u_flat):
        x_stab = unravel_x(x_flat)
        u_stab = unravel_u(u_flat)
        x_full = x_stab.as_full_state(rpy_attitude=True)
        u_full = u_stab.as_full_input()
        return x_full, u_full

    def dynamics(t, x_flat, u_flat, x0_dot_flat=None):
        x_full, u_full = unravel_stab(x_flat, u_flat)
        x_dot_full = f16.dynamics(t, x_full, u_full)

        x_dot = StabilityState.from_full_derivative(x_full, x_dot_full)

        x_dot_flat, _ = arc.tree.ravel(x_dot)
        if x0_dot_flat is not None:
            x_dot_flat -= x0_dot_flat

        return x_dot_flat


    def output(t, x_flat, u_flat, y0_flat=None):
        # Output normal acceleration, pitch rate, and angle of attack
        x_full, u_full = unravel_stab(x_flat, u_flat)

        x_stab = unravel_x(x_flat)
        q = x_stab.long.q
        alpha = np.rad2deg(x_stab.long.alpha)  # Angle of attack [deg]

        # Normal acceleration [g's]
        F_net_B, _ = f16.net_forces(t, x_full, u_full)
        F_grav_B = f16.calc_gravity(x_full)
        a_n = -(F_net_B[2] - F_grav_B[2]) / (f16.m * GRAV_FTS2)

        y = np.hstack([a_n, q, alpha])

        if y0_flat is not None:
            y -= y0_flat

        return y


    x0_dot_flat = dynamics(0.0, x0_flat, u0_flat)
    x0_dot_expected = StabilityState(
        long=LongitudinalState(
            vt=0.0,
            alpha=0.0,
            theta=case["condition"]["pitch_rate"],
            q=0.0,
            eng=f16.engine.State(0.0),
        ),
        lat=LateralState(
            beta=0.0,
            phi=0.0,
            p=0.0,
            r=0.0,
        ),
    )
    assert np.allclose(
        x0_dot_flat,
        arc.tree.ravel(x0_dot_expected)[0],
        atol=1e-3,
    )

    y0 = output(0.0, x0_flat, u0_flat)
    y0_expected = np.array([5.4267, 0.3, 17.2208])
    assert np.allclose(y0, y0_expected, atol=1e-3)

    A, B = arc.jac(dynamics, argnums=(1, 2))(0.0, x0_flat, u0_flat, x0_dot_flat)
    C, D = arc.jac(output, argnums=(1, 2))(0.0, x0_flat, u0_flat, y0)

    A_long_ex = np.array([
        [-0.127, -235, -32.2, -9.51, 0.314],
        [-7e-4, -0.969, 0, 0.908, -2e-4],
        [0, 0, 0, 1, 0],
        [9e-4, -4.56, 0, -1.58, 0],
        [0, 0, 0, 0, -5]
    ])
    npt.assert_allclose(A[:5, :5], A_long_ex, rtol=1e-2, atol=1e-3)

    A_lat_ex = np.array([
        [-0.322, 0.0612, 0.298, -0.948],
        [0, 0.093, 1.0, 0.310],
        [-62.5, 0, -3.0, 1.99],
        [7.67, 0, -0.262, -0.629],
    ])
    npt.assert_allclose(A[5:, 5:], A_lat_ex, rtol=0.05, atol=1e-3)

    A_mix_ex = np.array([
        [-0.0028, 0.00126, 5e-5, 2e-4],
        [1.5e-5, 0, -4e-5, -1e-5],
        [0, 0, 0, 0],
        [9.2e-5, 0, 0, -0.00287],
        [0, 0, 0, 0],
    ])
    npt.assert_allclose(A[:5, 5:], A_mix_ex, rtol=0.1, atol=1e-2)

    A_mix_ex = np.array([
        [1e-8, 2e-5, 3e-6, 8e-7, -3e-8],
        [0, 0, 0, 0, 0],
        [-3e-7, -0.00248, 0, 3e-4, 0],
        [-3e-6, -0.00188, 0, 0.00254, 0],
    ])
    npt.assert_allclose(A[5:, :5], A_mix_ex, rtol=0.1, atol=1e-2)

    B_ex = np.array([
        [0, -0.244, 6e-6, 2e-5],
        [0, -0.00209, 0, 0],
        [0, 0, 0, 0],
        [0, -0.199, 0, 0],
        [1087, 0, 0, 0],
        [0, 2e-8, 3e-4, 8e-4],
        [0, 0, 0, 0],
        [0, 0, -0.645, 0.126],
        [0, 0, -0.018, -0.0657],
    ])
    npt.assert_allclose(B, B_ex, rtol=1e-3, atol=1e-3)

    C_ex = np.array([
        [0.0208, 15.2, 0, 1.45, 0, -4.5e-4, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 57.3, 0, 0, 0, 0, 0, 0, 0],
    ])
    npt.assert_allclose(C, C_ex, rtol=1e-2, atol=1e-3)

    D_ex = np.array([
        [0, 0.0333, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    npt.assert_allclose(D, D_ex, rtol=1e-3, atol=1e-3)


def test_lon_stability():
    """Test case for longitudinal stability"""
    # Steady pitch-up
    rigid_body = RigidBody(rpy_attitude=True)
    model = SubsonicF16(rigid_body=rigid_body, xcg=0.3)
    result = trim(model, vt=502, pitch_rate=0.0)

    x0 = LongitudinalState.from_full_state(result.state)
    x0_lat = LateralState.from_full_state(result.state)
    u0 = LongitudinalInput.from_full_input(result.inputs)

    x0_flat, unravel_x = arc.tree.ravel(x0)
    u0_flat, unravel_u = arc.tree.ravel(u0)


    def unravel_stab(x_flat, u_flat):
        x_stab = StabilityState(long=unravel_x(x_flat), lat=x0_lat)
        u_stab = unravel_u(u_flat)
        x_full = x_stab.as_full_state(rpy_attitude=True)
        u_full = u_stab.as_full_input()
        return x_full, u_full

    def dynamics(t, x_flat, u_flat, x0_dot_flat=None):
        x_full, u_full = unravel_stab(x_flat, u_flat)
        x_dot_full = model.dynamics(t, x_full, u_full)

        x_dot = StabilityState.from_full_derivative(x_full, x_dot_full)

        x_dot_flat, _ = arc.tree.ravel(x_dot.long)
        if x0_dot_flat is not None:
            x_dot_flat -= x0_dot_flat

        return x_dot_flat


    def output(t, x_flat, u_flat, y0_flat=None):
        # Output normal acceleration, pitch rate, and angle of attack
        x_full, u_full = unravel_stab(x_flat, u_flat)

        x_stab = unravel_x(x_flat)
        q = x_stab.q
        alpha = np.rad2deg(x_stab.alpha)  # Angle of attack [deg]

        # Normal acceleration [g's]
        F_net_B, _ = model.net_forces(t, x_full, u_full)
        F_grav_B = model.calc_gravity(x_full)
        a_n = -(F_net_B[2] - F_grav_B[2]) / (model.m * GRAV_FTS2)

        y = np.hstack([a_n, q, alpha])

        if y0_flat is not None:
            y -= y0_flat

        return y


    x0_dot_flat = dynamics(0.0, x0_flat, u0_flat)
    y0 = output(0.0, x0_flat, u0_flat)

    A = arc.jac(dynamics, argnums=(1,))(0.0, x0_flat, u0_flat, x0_dot_flat)
    evals, evecs = np.linalg.eig(A)

    phugoid_idx = np.argmax(evals.real)
    short_idx = np.argmax(evals.imag)

    # Stevens, Lewis, Johnson values:
    slj_phugoid = -0.0087  + 0.0739j
    slj_short = -1.2039 + 1.4922j

    print(f"Phugoid mode:\t\t{evals[phugoid_idx].real:.4f} ± {evals[phugoid_idx].imag:.4f}j")
    print(f"Short period mode:\t{evals[short_idx].real:.4f} ± {evals[short_idx].imag:.4f}j")

    assert np.allclose(evals[phugoid_idx], slj_phugoid, rtol=1e-3)
    assert np.allclose(evals[short_idx], slj_short, rtol=1e-3)

    v_phugoid = evecs[:, phugoid_idx]
    if np.any(v_phugoid.imag < 0):
        v_phugoid = v_phugoid.conj()  # Make imaginary part positive for comparison
    npt.assert_allclose(
        v_phugoid,
        np.array([1.0, -9.6e-5 + 5e-7j, -3.8e-4 + 2.3e-3j, 1.7e-4 + 8.4e-6j, 0]),
        rtol=1e-2,
        atol=1e-5,
    )

    v_short = evecs[:, short_idx]
    if np.any(v_short.imag < 0):
        v_short = v_short.conj()  # Make imaginary part positive for comparison
    npt.assert_allclose(
        v_short,
        np.array([1.0, 0.09 + 0.017j, 0.059 + 0.054j, 0.0092 - 0.15j, 0]),
        rtol=5e-2,
        atol=1e-5,
    )


def test_lat_stability():
    """Test case for lateral-directional stability"""
    # Steady pitch-up
    rigid_body = RigidBody(rpy_attitude=True)
    model = SubsonicF16(rigid_body=rigid_body, xcg=0.3)
    result = trim(model, vt=502, pitch_rate=0.0)

    vt = result.condition.vt

    x0 = LateralState.from_full_state(result.state)
    x0_long = LongitudinalState.from_full_state(result.state)
    u0 = LateralInput.from_full_input(result.inputs)

    x0_flat, unravel_x = arc.tree.ravel(x0)
    u0_flat, unravel_u = arc.tree.ravel(u0)

    def unravel_stab(x_flat, u_flat):
        x_stab = StabilityState(long=x0_long, lat=unravel_x(x_flat))
        u_stab = unravel_u(u_flat)
        x_full = x_stab.as_full_state(rpy_attitude=True)
        u_full = u_stab.as_full_input()
        return x_full, u_full


    def dynamics(t, x_flat, u_flat, x0_dot_flat=None):
        x_full, u_full = unravel_stab(x_flat, u_flat)
        x_dot_full = model.dynamics(t, x_full, u_full)

        x_dot = StabilityState.from_full_derivative(x_full, x_dot_full)

        x_dot_flat, _ = arc.tree.ravel(x_dot.lat)
        if x0_dot_flat is not None:
            x_dot_flat -= x0_dot_flat

        return x_dot_flat


    x0_dot_flat = dynamics(0.0, x0_flat, u0_flat)
    A = arc.jac(dynamics, argnums=(1))(0.0, x0_flat, u0_flat, x0_dot_flat)

    # Example 3.8-1 in Stevens, Lewis, Johnson
    A_ex = np.array([
        [-0.32200, 0.064032, 0.038904, -0.99156],
        [0.0, 0.0, 1.0, 0.039385],
        [-30.919, 0.0, -3.6730, 0.67425],
        [9.4724, 0, -0.026358, -0.49849],
    ])
    npt.assert_allclose(A, A_ex, rtol=1e-3, atol=1e-4)

    evals, evecs = np.linalg.eig(A)

    dutch_idx = np.where(evals.imag != 0)[0][0]
    spiral_idx = np.argmax(evals.real)
    roll_idx = np.argmin(evals.real)

    # Stevens, Lewis, Johnson values:
    slj_dutch = -0.4399 + 3.220j
    slj_spiral = -0.0128
    slj_roll = -3.601

    print(f"Dutch roll mode:\t{evals[dutch_idx].real:.4f} ± {evals[dutch_idx].imag:.4f}j")
    print(f"Spiral mode:\t\t{evals[spiral_idx].real:.4f}")
    print(f"Roll subsidence mode:\t{evals[roll_idx].real:.4f}")

    assert np.allclose(evals[dutch_idx], slj_dutch, rtol=1e-3)
    assert np.allclose(evals[spiral_idx], slj_spiral, rtol=5e-3)
    assert np.allclose(evals[roll_idx], slj_roll, rtol=1e-3)

    v_spiral = evecs[:, spiral_idx].real
    npt.assert_allclose(
        v_spiral,
        np.array([0.0032, 1, -0.015, 0.063]),
        rtol=2e-2,
    )

    v_roll = evecs[:, roll_idx].real
    npt.assert_allclose(
        v_roll,
        np.array([0.002, 0.28, -1, -0.015]),
        rtol=1e-1,
    )

    v_dutch = evecs[:, dutch_idx]
    if np.any(v_dutch.imag < 0):
        v_dutch = v_dutch.conj()  # Make imaginary part positive for comparison
    npt.assert_allclose(
        v_dutch,
        np.array([-0.11 + 0.097j, -0.037 + 0.3j, 1.0, -0.29 - 0.33j]),
        rtol=2e-1,
        atol=2e-1,
    )