# ruff: noqa: N802, N803, N806

import numpy as np
import numpy.testing as npt
from scipy.spatial.transform import Rotation as ScipyRotation

import archimedes as arc
from archimedes.experimental.aero import (
    RigidBody,
    dcm_from_quaternion,
    euler_to_quaternion,
    quaternion_derivative,
    quaternion_multiply,
)
from archimedes.experimental.spatial import Rotation

m = 1.7  # Arbitrary mass
g0 = 9.81
J_B = np.diag([0.1, 0.2, 0.3])  # Arbitrary inertia matrix
J_B_inv = np.linalg.inv(J_B)


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
        r1 = ScipyRotation.from_quat(
            [0.7071, 0, 0, 0.7071]
        )  # Note: scipy uses [x,y,z,w]
        r2 = ScipyRotation.from_quat([0, 0.7071, 0, 0.7071])
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
            r = ScipyRotation.from_euler("ZYX", angles[::-1])
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
            r = ScipyRotation.from_euler("ZYX", angles[::-1])
            q_scipy = np.roll(r.as_quat(), 1)  # Convert to [w,x,y,z]

            # Note: quaternions q and -q represent the same rotation
            if np.sign(q_custom[0]) != np.sign(q_scipy[0]):
                q_scipy = -q_scipy

            npt.assert_allclose(q_custom, q_scipy, rtol=1e-6)


class TestVehicleDynamics:
    def test_constant_velocity_no_orientation(self):
        rigid_body = RigidBody()
        t = 0
        v_B = np.array([1, 0, 0])  # Constant velocity in x-direction
        att = Rotation.from_quat([1, 0, 0, 0])  # No rotation
        x = rigid_body.State(
            p_N=np.zeros(3),
            att=att,
            v_B=v_B,
            w_B=np.zeros(3),
        )
        u = rigid_body.Input(
            F_B=np.zeros(3),
            M_B=np.zeros(3),
            m=m,
            J_B=J_B,
        )

        dynamics = arc.compile(rigid_body.dynamics)
        x_dot = dynamics(t, x, u)
        q_dot = x_dot.att.as_quat()

        dp_N_ex = np.array([1, 0, 0])  # Velocity in x-direction
        npt.assert_allclose(x_dot.p_N, dp_N_ex, atol=1e-8)
        npt.assert_allclose(q_dot, np.zeros(4), atol=1e-8)
        npt.assert_allclose(x_dot.v_B, np.zeros(3), atol=1e-8)
        npt.assert_allclose(x_dot.w_B, np.zeros(3), atol=1e-8)

    def test_constant_velocity_with_orientation(self):
        rigid_body = RigidBody()

        # When the vehicle is not aligned with the world frame, the velocity
        # should be transformed accordingly
        rpy = np.array([0.1, 0.2, 0.3])
        v_B = np.array([1, 2, 3])

        att = Rotation.from_euler("xyz", rpy)
        v_N = att.apply(v_B)

        t = 0
        x = rigid_body.State(
            p_N=np.zeros(3),
            att=att,
            v_B=v_B,
            w_B=np.zeros(3),
        )
        u = rigid_body.Input(
            F_B=np.zeros(3),
            M_B=np.zeros(3),
            m=m,
            J_B=J_B,
        )

        dynamics = arc.compile(rigid_body.dynamics)
        x_dot = dynamics(t, x, u)

        dp_N_ex = v_N
        npt.assert_allclose(x_dot.p_N, dp_N_ex, atol=1e-8)
        npt.assert_allclose(x_dot.att.as_quat(), np.zeros(4), atol=1e-8)
        npt.assert_allclose(x_dot.v_B, np.zeros(3), atol=1e-8)
        npt.assert_allclose(x_dot.w_B, np.zeros(3), atol=1e-8)

    def test_constant_force(self):
        rigid_body = RigidBody()
        att = Rotation.from_quat([1, 0, 0, 0])  # No rotation

        # Test that constant acceleration leads to correct velocity changes
        t = 0
        x = rigid_body.State(
            p_N=np.zeros(3),
            att=att,
            v_B=np.zeros(3),
            w_B=np.zeros(3),
        )
        fx = 1.0
        u = rigid_body.Input(
            F_B=np.array([fx, 0, 0]),  # Constant force in x-direction
            M_B=np.zeros(3),
            m=m,
            J_B=J_B,
        )

        dynamics = arc.compile(rigid_body.dynamics)
        x_dot = dynamics(t, x, u)

        dv_B_ex = np.array([fx / m, 0, 0])
        npt.assert_allclose(x_dot.v_B, dv_B_ex)
        npt.assert_allclose(x_dot.w_B, np.zeros(3))

    def test_constant_angular_velocity(self):
        rigid_body = RigidBody()

        att = Rotation.from_quat([1, 0, 0, 0])  # No rotation

        t = 0
        x = rigid_body.State(
            p_N=np.zeros(3),
            att=att,
            v_B=np.zeros(3),
            w_B=np.array([1, 0, 0]),  # Constant angular velocity around x-axis
        )
        u = rigid_body.Input(
            F_B=np.zeros(3),
            M_B=np.zeros(3),
            m=m,
            J_B=J_B,
        )

        dynamics = arc.compile(rigid_body.dynamics)
        x_dot = dynamics(t, x, u)

        # Check quaternion derivative
        expected_qdot = np.array([0, 0.5, 0, 0])  # From quaternion kinematics
        npt.assert_allclose(x_dot.att.as_quat(), expected_qdot)

    def test_constant_moment(self):
        rigid_body = RigidBody()
        att = Rotation.from_quat([1, 0, 0, 0])  # No rotation

        # Test that constant moment results in expected angular velocity changes
        t = 0
        x = rigid_body.State(
            p_N=np.zeros(3),
            att=att,
            v_B=np.zeros(3),
            w_B=np.zeros(3),
        )
        mx = 1.0
        u = rigid_body.Input(
            F_B=np.zeros(3),
            M_B=np.array([mx, 0, 0]),
            m=m,
            J_B=J_B,
        )

        dynamics = arc.compile(rigid_body.dynamics)
        x_dot = dynamics(t, x, u)

        dw_B_ex = np.array([mx * J_B_inv[0, 0], 0, 0])
        npt.assert_allclose(x_dot.w_B, dw_B_ex)

    def test_combined_motion(self):
        rigid_body = RigidBody()

        t = 0
        p_N = np.array([0, 0, 0])
        att = Rotation.from_quat([1, 0, 0, 0])  # No rotation
        v_B = np.array([1, 0, 0])  # Initial velocity in x-direction
        w_B = np.array([0, 0.1, 0])  # Angular velocity around y-axis
        x = rigid_body.State(p_N, att, v_B, w_B)
        u = rigid_body.Input(
            F_B=np.array([1, 0, 0]),
            M_B=np.array([0, 0.1, 0]),
            m=m,
            J_B=J_B,
        )

        dynamics = arc.compile(rigid_body.dynamics)
        x_dot = dynamics(t, x, u)

        # Check linear motion
        npt.assert_allclose(x_dot.p_N[0], 1.0)  # Velocity in x-direction
        npt.assert_allclose(x_dot.v_B[0], 1 / m)  # Acceleration in x-direction

        # Check quaternion derivative
        att_deriv = x.att.derivative(x.w_B, baumgarte=rigid_body.baumgarte)
        npt.assert_allclose(x_dot.att.as_quat(), att_deriv.as_quat())

        # Check Coriolis effect
        expected_z_velocity = 0.1  # ω_y * v_x
        npt.assert_allclose(x_dot.v_B[2], expected_z_velocity)

    def test_quaternion_normalization(self):
        rigid_body = RigidBody()

        # Test that quaternion remains normalized under dynamics
        t = 0
        rpy = np.array([np.pi / 6, np.pi / 4, np.pi / 3])
        att = Rotation.from_euler("xyz", rpy)

        x = np.zeros(13)
        p_N = np.array([0, 0, 0])
        v_B = np.array([0, 0, 0])
        w_B = np.array([0.1, 0.2, 0.3])  # Angular velocity
        u = rigid_body.Input(
            F_B=np.zeros(3),
            M_B=np.zeros(3),
            m=m,
            J_B=J_B,
        )
        x = rigid_body.State(p_N, att, v_B, w_B)

        dynamics = arc.compile(rigid_body.dynamics)
        x_dot = dynamics(t, x, u)

        # Verify that quaternion derivative maintains unit norm
        # q·q̇ should be zero for unit quaternion
        q = x.att.as_quat()
        q_dot = x_dot.att.as_quat()
        npt.assert_allclose(np.dot(q, q_dot), 0, atol=1e-10)
