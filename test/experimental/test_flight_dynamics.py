# ruff: noqa: N802
# ruff: noqa: N803
# ruff: noqa: N806

import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial.transform import Rotation

import archimedes as arc
from archimedes.experimental.aero import (
    FlightVehicle,
    dcm_from_euler,
    dcm_from_quaternion,
    euler_to_quaternion,
    quaternion_derivative,
    quaternion_multiply,
)

m = 1.7  # Arbitrary mass
g0 = 9.81
J_B = np.diag([0.1, 0.2, 0.3])  # Arbitrary inertia matrix
J_B_inv = np.linalg.inv(J_B)


class BasicFlightVehicle(FlightVehicle):
    def net_forces(self, t, x, u, C_BN):
        # Zero forces and moments by default
        return np.zeros_like(x.p_N), np.zeros_like(x.p_N), np.array([])


@pytest.fixture
def basic_vehicle() -> FlightVehicle:
    return BasicFlightVehicle(m=m, J_B=J_B)


class TestQuaternionOperations:
    def test_quaternion_multiply(self):
        # Test identity quaternion multiplication
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        result = quaternion_multiply(q1, q2)
        npt.assert_allclose(result, q2)

        # Test arbitrary quaternion multiplication
        q1 = np.array([0.7071, 0.7071, 0, 0])  # 90-degree rotation around x
        q2 = np.array([0.7071, 0, 0.7071, 0])  # 90-degree rotation around y
        result = quaternion_multiply(q1, q2)

        # Compare with scipy's rotation composition
        r1 = Rotation.from_quat([0.7071, 0, 0, 0.7071])  # Note: scipy uses [x,y,z,w]
        r2 = Rotation.from_quat([0, 0.7071, 0, 0.7071])
        r_combined = r1 * r2
        expected = np.roll(r_combined.as_quat(), 1)  # Convert to [w,x,y,z]
        npt.assert_allclose(result, expected, rtol=1e-4)

    def test_quaternion_derivative(self):
        # Test zero angular velocity
        q = np.array([1, 0, 0, 0])
        w = np.zeros(3)
        result = quaternion_derivative(q, w)
        npt.assert_allclose(result, np.zeros(4))

        # Test constant angular velocity
        q = np.array([1, 0, 0, 0])
        w = np.array([1, 0, 0])  # Rotation around x-axis
        result = quaternion_derivative(q, w)
        expected = np.array([0, 0.5, 0, 0])  # Half the angular velocity
        npt.assert_allclose(result, expected)

    def test_dcm_from_quaternion(self):
        test_angles = [
            np.array([0, 0, 0]),  # Identity
            np.array([np.pi / 4, 0, 0]),  # 45 degree roll
            np.array([0, np.pi / 4, 0]),  # 45 degree pitch
            np.array([0, 0, np.pi / 4]),  # 45 degree yaw
            np.array([np.pi / 6, np.pi / 4, np.pi / 3]),  # Arbitrary rotation
        ]

        for angles in test_angles:
            # Convert Euler angles to quaternion
            q = euler_to_quaternion(angles)
            C_BN_custom = dcm_from_quaternion(q)

            # Compare with scipy's rotation
            r = Rotation.from_euler("ZYX", angles[::-1])
            C_BN_scipy = r.as_matrix().T

            npt.assert_allclose(
                C_BN_custom,
                C_BN_scipy,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Mismatch for angles {angles}",
            )

            # Verify orthogonality
            identity = np.eye(3)
            npt.assert_allclose(
                C_BN_custom @ C_BN_custom.T, identity, rtol=1e-6, atol=1e-8
            )

            # Verify determinant is 1
            npt.assert_allclose(np.linalg.det(C_BN_custom), 1.0, rtol=1e-6, atol=1e-8)

    def test_euler_to_quaternion(self):
        test_angles = [
            np.array([0, 0, 0]),
            np.array([np.pi / 4, 0, 0]),
            np.array([0, np.pi / 4, 0]),
            np.array([0, 0, np.pi / 4]),
            np.array([np.pi / 6, np.pi / 4, np.pi / 3]),
        ]

        for angles in test_angles:
            q_custom = euler_to_quaternion(angles)

            # Compare with scipy's conversion
            r = Rotation.from_euler("ZYX", angles[::-1])
            q_scipy = np.roll(r.as_quat(), 1)  # Convert to [w,x,y,z]

            # Note: quaternions q and -q represent the same rotation
            if np.sign(q_custom[0]) != np.sign(q_scipy[0]):
                q_scipy = -q_scipy

            npt.assert_allclose(q_custom, q_scipy, rtol=1e-6)


class TestVehicleDynamics:
    def test_constant_velocity_no_orientation(self, basic_vehicle):
        t = 0
        v_B = np.array([1, 0, 0])  # Constant velocity in x-direction
        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=np.array([1, 0, 0, 0]),  # Unit quaternion (no rotation)
            v_B=v_B,
            w_B=np.zeros(3),
        )
        u = np.zeros(4)

        dynamics = arc.compile(basic_vehicle.dynamics)
        x_dot = dynamics(t, x, u)

        dp_N_ex = np.array([1, 0, 0])  # Velocity in x-direction
        npt.assert_allclose(x_dot.p_N, dp_N_ex, atol=1e-8)
        npt.assert_allclose(x_dot.att, np.zeros(4), atol=1e-8)
        npt.assert_allclose(x_dot.v_B, np.zeros(3), atol=1e-8)
        npt.assert_allclose(x_dot.w_B, np.zeros(3), atol=1e-8)

    def test_constant_velocity_with_orientation(self, basic_vehicle):
        # When the vehicle is not aligned with the world frame, the velocity
        # should be transformed accordingly
        rpy = np.array([0.1, 0.2, 0.3])
        v_B = np.array([1, 2, 3])

        C_BN = dcm_from_euler(rpy)
        v_N = C_BN.T @ v_B

        q = euler_to_quaternion(rpy)

        t = 0
        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=q,
            v_B=v_B,
            w_B=np.zeros(3),
        )
        u = np.zeros(4)

        dynamics = arc.compile(basic_vehicle.dynamics)
        x_dot = dynamics(t, x, u)

        dp_N_ex = v_N
        npt.assert_allclose(x_dot.p_N, dp_N_ex, atol=1e-8)
        npt.assert_allclose(x_dot.att, np.zeros(4), atol=1e-8)
        npt.assert_allclose(x_dot.v_B, np.zeros(3), atol=1e-8)
        npt.assert_allclose(x_dot.w_B, np.zeros(3), atol=1e-8)

    def test_constant_force(self, basic_vehicle):
        # Test that constant acceleration leads to correct velocity changes
        t = 0
        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=np.array([1, 0, 0, 0]),
            v_B=np.zeros(3),
            w_B=np.zeros(3),
        )
        fx = 1.0
        u = np.zeros(4)

        def constant_force(t, x, u, C_BN):
            F_B = np.array([fx, 0, 0])  # Constant force in x-direction
            M_B = np.zeros(3)
            return F_B, M_B, np.array([])

        basic_vehicle.net_forces = constant_force
        dynamics = arc.compile(basic_vehicle.dynamics)
        x_dot = dynamics(t, x, u)

        dv_B_ex = np.array([fx / m, 0, 0])
        npt.assert_allclose(x_dot.v_B, dv_B_ex)
        npt.assert_allclose(x_dot.w_B, np.zeros(3))

    def test_constant_angular_velocity(self, basic_vehicle):
        t = 0
        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=np.array([1, 0, 0, 0]),
            v_B=np.zeros(3),
            w_B=np.array([1, 0, 0]),  # Constant angular velocity around x-axis
        )
        u = np.zeros(4)

        dynamics = arc.compile(basic_vehicle.dynamics)
        x_dot = dynamics(t, x, u)

        # Check quaternion derivative
        expected_qdot = np.array([0, 0.5, 0, 0])  # From quaternion kinematics
        npt.assert_allclose(x_dot.att, expected_qdot)

    def test_constant_moment(self, basic_vehicle):
        # Test that constant moment results in expected angular velocity changes
        t = 0
        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=np.array([1, 0, 0, 0]),
            v_B=np.zeros(3),
            w_B=np.zeros(3),
        )
        u = np.zeros(4)
        mx = 1.0

        def constant_moment(t, x, u, C_BN):
            F_B = np.zeros(3)
            M_B = np.array([mx, 0, 0])  # Constant moment around x-axis
            return F_B, M_B, np.array([])

        basic_vehicle.net_forces = constant_moment
        dynamics = arc.compile(basic_vehicle.dynamics)
        x_dot = dynamics(t, x, u)

        dw_B_ex = np.array([mx * J_B_inv[0, 0], 0, 0])
        npt.assert_allclose(x_dot.w_B, dw_B_ex)

    def test_combined_motion(self, basic_vehicle):
        t = 0
        p_N = np.array([0, 0, 0])
        q = np.array([1, 0, 0, 0])  # Unit quaternion (no rotation)
        v_B = np.array([1, 0, 0])  # Initial velocity in x-direction
        w_B = np.array([0, 0.1, 0])  # Angular velocity around y-axis
        x = basic_vehicle.State(p_N, q, v_B, w_B)
        u = np.zeros(4)

        def constant_force_and_moment(t, x, u, C_BN):
            F_B = np.array([1, 0, 0])  # Constant force in x-direction
            M_B = np.array([0, 0.1, 0])  # Constant moment around y-axis
            return F_B, M_B, np.array([])

        basic_vehicle.net_forces = constant_force_and_moment
        dynamics = arc.compile(basic_vehicle.dynamics)
        x_dot = dynamics(t, x, u)

        # Check linear motion
        npt.assert_allclose(x_dot.p_N[0], 1.0)  # Velocity in x-direction
        npt.assert_allclose(x_dot.v_B[0], 1 / m)  # Acceleration in x-direction

        # Check quaternion derivative
        expected_qdot = quaternion_derivative(x.att, x.w_B)
        npt.assert_allclose(x_dot.att, expected_qdot)

        # Check Coriolis effect
        expected_z_velocity = 0.1  # ω_y * v_x
        npt.assert_allclose(x_dot.v_B[2], expected_z_velocity)

    def test_quaternion_normalization(self, basic_vehicle):
        # Test that quaternion remains normalized under dynamics
        t = 0
        angles = np.array([np.pi / 6, np.pi / 4, np.pi / 3])
        q = euler_to_quaternion(angles)

        x = np.zeros(13)
        p_N = np.array([0, 0, 0])
        v_B = np.array([0, 0, 0])
        w_B = np.array([0.1, 0.2, 0.3])  # Angular velocity
        u = np.zeros(4)
        x = basic_vehicle.State(p_N, q, v_B, w_B)

        dynamics = arc.compile(basic_vehicle.dynamics)
        x_dot = dynamics(t, x, u)

        # Verify that quaternion derivative maintains unit norm
        # q·q̇ should be zero for unit quaternion
        q_dot = x_dot.att
        npt.assert_allclose(np.dot(q, q_dot), 0, atol=1e-10)
