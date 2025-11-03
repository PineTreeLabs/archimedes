# ruff: noqa: N802, N803, N806

import numpy as np
import numpy.testing as npt

import archimedes as arc
from archimedes.spatial import (
    RigidBody,
    RigidBodyConfig,
    Quaternion,
    euler_to_dcm,
    euler_kinematics,
)

m = 1.7  # Arbitrary mass
J_B = np.diag([0.1, 0.2, 0.3])  # Arbitrary inertia matrix
J_B_inv = np.linalg.inv(J_B)


def test_euler_kinematics():
    rpy = np.array([0.1, 0.2, 0.3])

    # Given roll-pitch-yaw rates, compute the body-frame angular velocity
    # using the rotation matrices directly.
    pqr = np.array([0.4, 0.5, 0.6])  # Roll, pitch, yaw rates
    C_roll = Quaternion.from_euler("x", rpy[0]).as_matrix().T  # C_φ
    C_pitch = Quaternion.from_euler("y", rpy[1]).as_matrix().T  # C_θ
    # Successively transform each rate into the body frame
    w_B_ex = np.array([pqr[0], 0.0, 0.0]) + C_roll @ (
        np.array([0.0, pqr[1], 0.0]) + C_pitch @ np.array([0.0, 0.0, pqr[2]])
    )

    # Use the Euler kinematics function to duplicate the transformation
    Hinv = euler_kinematics(rpy, inverse=True)
    w_B = Hinv @ pqr

    npt.assert_allclose(w_B, w_B_ex)

    # Test the forward transformation
    H = euler_kinematics(rpy)
    result = H @ w_B
    npt.assert_allclose(pqr, result)


class TestVehicleDynamics:
    def test_build_from_config(self):
        config = {
            "baumgarte": 0.5,
            "rpy_attitude": False,
        }
        rb = RigidBodyConfig(**config).build()
        assert isinstance(rb, RigidBody)
        assert rb.baumgarte == 0.5
        assert rb.rpy_attitude is False

    def test_constant_velocity_no_orientation(self):
        rigid_body = RigidBody()
        t = 0
        v_B = np.array([1, 0, 0])  # Constant velocity in x-direction
        att = Quaternion([1, 0, 0, 0])  # No rotation
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
        q_dot = x_dot.att

        dp_N_ex = np.array([1, 0, 0])  # Velocity in x-direction
        npt.assert_allclose(x_dot.p_N, dp_N_ex, atol=1e-8)
        npt.assert_allclose(q_dot.array, np.zeros(4), atol=1e-8)
        npt.assert_allclose(x_dot.v_B, np.zeros(3), atol=1e-8)
        npt.assert_allclose(x_dot.w_B, np.zeros(3), atol=1e-8)

    def test_constant_velocity_with_orientation(self):
        rigid_body = RigidBody()

        # When the vehicle is not aligned with the world frame, the velocity
        # should be transformed accordingly
        rpy = np.array([0.1, 0.2, 0.3])
        v_B = np.array([1, 2, 3])

        att = Quaternion.from_euler("xyz", rpy)

        # Could do att.rotate(v_B) but this tests euler_to_dcm
        R_NB = euler_to_dcm(rpy)
        v_N = R_NB @ v_B

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
        npt.assert_allclose(x_dot.att.array, np.zeros(4), atol=1e-8)
        npt.assert_allclose(x_dot.v_B, np.zeros(3), atol=1e-8)
        npt.assert_allclose(x_dot.w_B, np.zeros(3), atol=1e-8)

    def test_constant_force(self):
        rigid_body = RigidBody()
        att = Quaternion([1, 0, 0, 0])  # No rotation

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

        att = Quaternion([1, 0, 0, 0])  # No rotation

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
        npt.assert_allclose(x_dot.att.array, expected_qdot)

    def test_constant_moment(self):
        rigid_body = RigidBody()
        att = Quaternion([1, 0, 0, 0])  # No rotation

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
        att = Quaternion([1, 0, 0, 0])  # No rotation
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
        npt.assert_allclose(x_dot.att.array, att_deriv.array)

        # Check Coriolis effect
        expected_z_velocity = 0.1  # ω_y * v_x
        npt.assert_allclose(x_dot.v_B[2], expected_z_velocity)

    def test_quaternion_normalization(self):
        rigid_body = RigidBody()

        # Test that quaternion remains normalized under dynamics
        t = 0
        rpy = np.array([np.pi / 6, np.pi / 4, np.pi / 3])
        att = Quaternion.from_euler("xyz", rpy)

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
        q = x.att.array
        q_dot = x_dot.att.array
        npt.assert_allclose(np.dot(q, q_dot), 0, atol=1e-10)
