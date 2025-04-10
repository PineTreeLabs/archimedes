from functools import partial
import pytest
import numpy as np
import numpy.testing as npt

from scipy.spatial.transform import Rotation

import archimedes as arc
from archimedes import struct

from archimedes.experimental.aero import (
    FlightVehicle,
    dcm_from_euler as dcm,
    euler_kinematics,
)
from multirotor import (
    RotorGeometry,
    ThinAirfoil,
    BladeElementModel,
    MultiRotorVehicle,
    QuadraticRotorModel,
    QuadraticDragModel,
    ConstantGravityModel,
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
def basic_vehicle():
    return BasicFlightVehicle(m=m, J_B=J_B, attitude="euler")


class TestVehicleKinematics:
    def test_dcm(self):
        # Test cases: [roll, pitch, yaw] in radians
        test_angles = [
            np.array([0, 0, 0]),  # Identity
            np.array([np.pi / 4, 0, 0]),  # 45 degree roll
            np.array([0, np.pi / 4, 0]),  # 45 degree pitch
            np.array([0, 0, np.pi / 4]),  # 45 degree yaw
            np.array([np.pi / 6, np.pi / 4, np.pi / 3]),  # Arbitrary rotation
            np.array([np.pi, np.pi / 2, np.pi / 4]),  # Edge case
        ]

        for angles in test_angles:
            C_BN_custom = dcm(angles)

            # SciPy's rotation (use 'ZYX' for intrinsic rotations, which is
            # equivalent to yaw-pitch-roll).  Note that SciPy expects angles
            # in sequence order, so in this case yaw-pitch-roll and will return
            # the inertial-to-body transformation matrix (the transpose of `dcm`).
            r = Rotation.from_euler("ZYX", angles[::-1])
            C_BN_scipy = r.as_matrix().T

            # Compare the results
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
                C_BN_custom @ C_BN_custom.T,
                identity,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Non-orthogonal matrix for angles {angles}",
            )

            # Verify determinant is 1 (proper rotation)
            npt.assert_allclose(
                np.linalg.det(C_BN_custom),
                1.0,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Determinant not 1 for angles {angles}",
            )

    def test_euler_kinematics(self):
        rpy = np.array([0.1, 0.2, 0.3])

        # Given roll-pitch-yaw rates, compute the body-frame angular velocity
        # using the rotation matrices directly.
        pqr = np.array([0.4, 0.5, 0.6])  # Roll, pitch, yaw rates
        C_roll = dcm(np.array([rpy[0], 0, 0]))  # C_φ
        C_pitch = dcm(np.array([0, rpy[1], 0]))  # C_θ
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

    def test_constant_velocity_no_orientation(self, basic_vehicle):
        # Test that constant velocity results in expected position rate
        # when the body frame is aligned with the world frame
        t = 0
        x = np.zeros(12)
        x = basic_vehicle.State(
            p_N=np.zeros(3),  # Position in the Newtonian frame
            att=np.zeros(3),  # Attitude (roll, pitch, yaw)
            v_B=np.array([1, 0, 0]),  # Velocity in the body frame
            w_B=np.zeros(3),  # Angular velocity in the body frame
        )
        u = np.zeros(4)

        x_t = basic_vehicle.dynamics(t, x, u)

        x_t_ex = basic_vehicle.State(
            p_N=np.array([1, 0, 0]),  # Position in the Newtonian frame
            att=np.zeros(3),  # Attitude (roll, pitch, yaw)
            v_B=np.zeros(3),  # Velocity in the body frame
            w_B=np.zeros(3),  # Angular velocity in the body frame
        )

        npt.assert_allclose(x_t.p_N, x_t_ex.p_N)
        npt.assert_allclose(x_t.att, x_t_ex.att)
        npt.assert_allclose(x_t.v_B, x_t_ex.v_B)
        npt.assert_allclose(x_t.w_B, x_t_ex.w_B)

    def test_constant_velocity_with_orientation(self, basic_vehicle):
        # When the vehicle is not aligned with the world frame, the velocity
        # should be transformed accordingly
        rpy = np.array([0.1, 0.2, 0.3])
        v_B = np.array([1, 2, 3])

        C_BN = dcm(rpy)
        v_N = C_BN.T @ v_B

        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=rpy,  # Attitude (roll, pitch, yaw)
            v_B=v_B,  # Velocity in the body frame
            w_B=np.zeros(3),  # Angular velocity in the body frame
        )
        u = np.zeros(4)

        x_t = basic_vehicle.dynamics(0, x, u)

        x_t_ex = basic_vehicle.State(
            p_N=v_N,  # Position in the Newtonian frame
            att=np.zeros(3),  # Attitude (roll, pitch, yaw)
            v_B=np.zeros(3),  # Velocity in the body frame
            w_B=np.zeros(3),  # Angular velocity in the body frame
        )
        npt.assert_allclose(x_t.p_N, x_t_ex.p_N)
        npt.assert_allclose(x_t.att, x_t_ex.att)
        npt.assert_allclose(x_t.v_B, x_t_ex.v_B)
        npt.assert_allclose(x_t.w_B, x_t_ex.w_B)

    def test_constant_force(self, basic_vehicle):
        # Test that constant acceleration leads to correct velocity changes
        t = 0
        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=np.zeros(3),  # Attitude (roll, pitch, yaw)
            v_B=np.zeros(3),  # Velocity in the body frame
            w_B=np.zeros(3),  # Angular velocity in the body frame
        )
        fx = 1.0
        u = np.zeros(4)

        def constant_force(t, x, u, C_BN):
            F_B = np.array([fx, 0, 0])  # Constant force in x-direction
            M_B = np.zeros(3)
            return F_B, M_B, np.array([])

        basic_vehicle.net_forces = constant_force
        x_t = basic_vehicle.dynamics(t, x, u)

        x_t_ex = basic_vehicle.State(
            p_N=np.zeros(3),  # Position in the Newtonian frame
            att=np.zeros(3),  # Attitude (roll, pitch, yaw)
            v_B=np.array([fx / m, 0, 0]),  # Velocity in the body frame
            w_B=np.zeros(3),  # Angular velocity in the body frame
        )
        npt.assert_allclose(x_t.p_N, x_t_ex.p_N)
        npt.assert_allclose(x_t.att, x_t_ex.att)
        npt.assert_allclose(x_t.v_B, x_t_ex.v_B)
        npt.assert_allclose(x_t.w_B, x_t_ex.w_B)

    def test_constant_angular_velocity(self, basic_vehicle):
        # Test that constant angular velocity results in expected orientation changes
        t = 0

        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=np.zeros(3),
            v_B=np.zeros(3),
            w_B=np.array([1.0, 0, 0]),  # Constant angular velocity around x-axis
        )
        u = np.zeros(4)

        x_t = basic_vehicle.dynamics(t, x, u)

        x_t_ex = basic_vehicle.State(
            p_N=np.zeros(3),
            att=np.array([1, 0, 0]),  # Constant roll rate
            v_B=np.zeros(3),
            w_B=np.zeros(3),
        )
        npt.assert_allclose(x_t.p_N, x_t_ex.p_N)
        npt.assert_allclose(x_t.att, x_t_ex.att)
        npt.assert_allclose(x_t.v_B, x_t_ex.v_B)
        npt.assert_allclose(x_t.w_B, x_t_ex.w_B)

    def test_constant_moment(self, basic_vehicle):
        # Test that constant moment results in expected angular velocity changes
        t = 0

        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=np.zeros(3),
            v_B=np.zeros(3),
            w_B=np.zeros(3),
        )
        u = np.zeros(4)
        mx = 1.0
        npt.assert_allclose(np.linalg.inv(basic_vehicle.J_B), J_B_inv)

        def constant_moment(t, x, u, C_BN):
            F_B = np.zeros(3)
            M_B = np.array([mx, 0, 0])  # Constant moment around x-axis
            return F_B, M_B, np.array([])

        basic_vehicle.net_forces = constant_moment
        x_t = basic_vehicle.dynamics(t, x, u)

        x_t_ex = basic_vehicle.State(
            p_N=np.zeros(3),
            att=np.zeros(3),
            v_B=np.zeros(3),
            w_B=np.array(
                [mx * J_B_inv[0, 0], 0, 0]
            ),  # Angular velocity in the body frame
        )
        npt.assert_allclose(x_t.p_N, x_t_ex.p_N)
        npt.assert_allclose(x_t.att, x_t_ex.att)
        npt.assert_allclose(x_t.v_B, x_t_ex.v_B)
        npt.assert_allclose(x_t.w_B, x_t_ex.w_B)

    def test_euler_kinematic_consistency(self, basic_vehicle):
        # Test consistency between Euler rates and body angular rates
        rpy = np.array([0.1, 0.2, 0.3])
        w_B = np.array([0.1, 0.2, 0.3])

        H = euler_kinematics(rpy)
        euler_rates = H @ w_B

        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=rpy,
            v_B=np.zeros(3),
            w_B=w_B,
        )
        u = np.zeros(4)

        x_t = basic_vehicle.dynamics(0, x, u)

        x_t_ex = basic_vehicle.State(
            p_N=np.zeros(3),
            att=euler_rates,
            v_B=np.zeros(3),
            w_B=-J_B_inv @ np.cross(w_B, J_B @ w_B),
        )
        npt.assert_allclose(x_t.p_N, x_t_ex.p_N)
        npt.assert_allclose(x_t.att, x_t_ex.att)
        npt.assert_allclose(x_t.v_B, x_t_ex.v_B)
        npt.assert_allclose(x_t.w_B, x_t_ex.w_B)

    def test_combined_motion(self, basic_vehicle):
        t = 0
        x = basic_vehicle.State(
            p_N=np.zeros(3),
            att=np.zeros(3),  # Attitude (roll, pitch, yaw)
            v_B=np.array([1, 0, 0]),  # Velocity in the body frame
            w_B=np.array([0, 0.1, 0]),  # Angular velocity in the body frame
        )
        u = np.zeros(4)

        def constant_force_and_moment(t, x, u, C_BN):
            F_B = np.array([1, 0, 0])  # Constant force in x-direction
            M_B = np.array([0, 0.1, 0])  # Constant moment around y-axis
            return F_B, M_B, np.array([])

        basic_vehicle.net_forces = constant_force_and_moment
        x_t = basic_vehicle.dynamics(t, x, u)

        # Check linear motion
        npt.assert_allclose(x_t.p_N, np.array([1, 0, 0]))
        az_B = 0.1  # Coriolis: ω_y * v_x
        npt.assert_allclose(x_t.v_B, np.array([1 / m, 0, az_B]))

        # Check angular motion
        npt.assert_allclose(x_t.att, np.array([0, 0.1, 0]))  # Pitch rate


class TestBladeElementModel:
    def test_hover_zero_inflow(self):
        rho = 1.225  # Air density [kg/m^3]
        c0 = 0.02  # Chord length [m]
        R = 0.15  # Rotor radius [m]
        e = 0.1  # Root cutout ratio

        th0 = np.deg2rad(5.0)  # Constant blade bitch
        h = 0.1  # Airfoil max camber, as fraction of chord length

        def pitch(r_):
            return th0

        def chord(r_):
            return c0

        Cd_0 = 0.02

        # Theoretical predictions for thin airfoil with parabolic camber
        Cl_0 = 4 * np.pi * h
        Cl_alpha = 2 * np.pi
        Cm_0 = -np.pi * h

        airfoil_model = ThinAirfoil(
            Cl_0=Cl_0,
            Cl_alpha=Cl_alpha,
            Cd_0=Cd_0,
            Cm_0=Cm_0,
        )

        # Specify hover conditions
        v_W = np.zeros(3)  # Wind-frame velocity
        w_W = np.zeros(3)  # Wind-frame angular velocity

        Omega = 500.0  # Rotor angular velocity [rad/s]

        # For verification, set lambda=0.0
        inflow_ratio = 0.0

        rotor_model = BladeElementModel(
            n_rad=5,
            n_az=5,
            R=R,
            rho=rho,
            e=e,
            T0=0.0,  # Unused for this test
            airfoil_model=airfoil_model,
            chord=chord,
            blade_pitch=pitch,
        )

        # Check computation of differential blade root shear loads
        dSx, dSz, dMx, dMy, dMz = rotor_model._differential_shear(
            v_W, w_W, Omega, inflow_ratio
        )

        r = rotor_model.nodes_rad
        c = c0
        Cl = Cl_0 + Cl_alpha * th0
        Cd = Cd_0
        Cm = Cm_0

        # Blade root drag shear
        dSx_ex = 0.5 * rho * c * Cd * (Omega * r) ** 2
        npt.assert_allclose(dSx, dSx_ex)

        # Blade root vertical shear
        dSz_ex = 0.5 * rho * c * Cl * (Omega * r) ** 2
        npt.assert_allclose(dSz, dSz_ex)

        # Check bending moments
        npt.assert_allclose(dMx, r * dSz_ex)
        npt.assert_allclose(dMz, -r * dSx_ex)

        dMy_ex = 0.5 * rho * c**2 * Cm * (Omega * r) ** 2
        npt.assert_allclose(dMy, dMy_ex)

        # Check net forces and moments
        geometry = RotorGeometry()  # Default geometry has zero offset
        F_W, M_W, _ = rotor_model._compute_forces_moments(
            0.0, v_W, w_W, None, Omega, inflow_ratio, geometry
        )

        # Everything but thrust and torque average to zero
        Sz = (1 / 6) * rho * c * Cl * Omega**2 * R**3 * (1 - e**3)
        Mz = -(1 / 8) * rho * c * Cd * Omega**2 * R**4 * (1 - e**4)

        Fz_W = -rotor_model.Nb * Sz
        Mz_W = -rotor_model.Nb * Mz

        npt.assert_allclose(F_W, np.array([0, 0, Fz_W]), atol=1e-12)
        npt.assert_allclose(M_W, np.array([0, 0, Mz_W]), atol=1e-12)

    def test_hover_constant_inflow(self):
        # Create an inflow ratio such that the inflow angle is constant
        # along the span of the rotor blade
        rho = 1.225  # Air density [kg/m^3]
        c0 = 0.02  # Chord length [m]
        R = 0.15  # Rotor radius [m]
        e = 0.1  # Root cutout ratio

        th0 = np.deg2rad(5.0)  # Constant blade bitch
        h = 0.1  # Airfoil max camber, as fraction of chord length

        def pitch(r_):
            return th0

        def chord(r_):
            return c0

        Cd_0 = 0.02

        # Theoretical predictions for thin airfoil with parabolic camber
        Cl_0 = 4 * np.pi * h
        Cl_alpha = 2 * np.pi
        Cm_0 = -np.pi * h

        airfoil_model = ThinAirfoil(
            Cl_0=Cl_0,
            Cl_alpha=Cl_alpha,
            Cd_0=Cd_0,
            Cm_0=Cm_0,
        )

        # Specify hover conditions
        v_W = np.zeros(3)  # Wind-frame velocity
        w_W = np.zeros(3)  # Wind-frame angular velocity

        Omega = 500.0  # Rotor angular velocity [rad/s]

        rotor_model = BladeElementModel(
            R=R,
            rho=rho,
            e=e,
            T0=0.0,  # Unused for this test
            airfoil_model=airfoil_model,
            chord=chord,
            blade_pitch=pitch,
        )
        geometry = RotorGeometry()  # Default geometry has zero offset

        # For verification, choose a value of lambda that will lead to
        # constant inflow angle
        a = 0.1  # Equivalent to arctan(phi)
        r = rotor_model.nodes_rad
        inflow_ratio = a * r / R

        # Check computation of differential blade root shear loads
        dSx, dSz, dMx, dMy, dMz = rotor_model._differential_shear(
            v_W, w_W, Omega, inflow_ratio
        )

        c = c0
        Cl = Cl_0 + Cl_alpha * (th0 - np.arctan(a))
        Cd = Cd_0
        Cm = Cm_0

        # Blade root drag shear
        dSx_ex = 0.5 * rho * c * (a * Cl + Cd) * (Omega * r) ** 2 * np.sqrt(1 + a**2)
        npt.assert_allclose(dSx, dSx_ex)

        # Blade root vertical shear
        dSz_ex = 0.5 * rho * c * (Cl - a * Cd) * (Omega * r) ** 2 * np.sqrt(1 + a**2)
        npt.assert_allclose(dSz, dSz_ex)

        # Check bending moments
        npt.assert_allclose(dMx, r * dSz_ex)
        npt.assert_allclose(dMz, -r * dSx_ex)

        dMy_ex = 0.5 * rho * c**2 * Cm * (Omega * r) ** 2 * (1 + a**2)
        npt.assert_allclose(dMy, dMy_ex)

        # Check net force and moments
        F_W, M_W, _ = rotor_model._compute_forces_moments(
            0.0, v_W, w_W, None, Omega, inflow_ratio, geometry
        )

        # Everything but thrust and torque average to zero
        Sz = (
            (1 / 6)
            * rho
            * c
            * (Cl - a * Cd)
            * Omega**2
            * R**3
            * (1 - e**3)
            * np.sqrt(1 + a**2)
        )
        Mz = -(
            (1 / 8)
            * rho
            * c
            * (a * Cl + Cd)
            * Omega**2
            * R**4
            * (1 - e**4)
            * np.sqrt(1 + a**2)
        )

        Fz_W = -rotor_model.Nb * Sz
        Mz_W = -rotor_model.Nb * Mz

        npt.assert_allclose(F_W, np.array([0, 0, Fz_W]), atol=1e-12)
        npt.assert_allclose(M_W, np.array([0, 0, Mz_W]), atol=1e-12)

    def test_zero_blade_speed(self):
        rho = 1.225  # Air density [kg/m^3]
        c0 = 0.02  # Chord length [m]
        R = 0.15  # Rotor radius [m]
        e = 0.1  # Root cutout ratio

        th0 = np.deg2rad(0.0)  # Constant blade bitch
        h = 0.0  # Airfoil max camber, as fraction of chord length

        pitch = np.deg2rad(5.0)  # Pitch angle
        V_inf = 10.0  # Forward flight speed
        Vx = V_inf * np.cos(pitch)
        Vz = -V_inf * np.sin(pitch)

        def pitch(r_):
            return th0

        def chord(r_):
            return c0

        Cd_0 = 0.0

        # Theoretical predictions for thin airfoil with parabolic camber
        Cl_0 = 4 * np.pi * h
        Cl_alpha = 2 * np.pi
        Cm_0 = -np.pi * h

        airfoil_model = ThinAirfoil(
            Cl_0=Cl_0,
            Cl_alpha=Cl_alpha,
            Cd_0=Cd_0,
            Cm_0=Cm_0,
        )

        # Specify flight conditions
        v_W = np.array([Vx, 0.0, Vz])  # Wind-frame velocity
        w_W = np.zeros(3)  # Wind-frame angular velocity

        Omega = 0.0  # Zero rotor angular velocity [rad/s]

        rotor_model = BladeElementModel(
            R=R,
            rho=rho,
            e=e,
            T0=0.0,  # Unused for this test
            airfoil_model=airfoil_model,
            chord=chord,
            blade_pitch=pitch,
        )

        psi = rotor_model.nodes_az
        assert not (np.any(np.isclose(psi % 2 * np.pi, 0.0)))  # Singularity at psi=0

        phi = np.arctan2(-Vz, Vx * np.sin(psi))

        # For verification purposes choose lambda=0
        inflow_ratio = 0.0

        # Check computation of differential blade root shear loads
        dSx, dSz, dMx, dMy, dMz = rotor_model._differential_shear(
            v_W, w_W, Omega, inflow_ratio, force_only=True
        )

        c = c0
        Cl = Cl_0 - Cl_alpha * phi
        U_sq = (Vx * np.sin(psi)) ** 2 + Vz**2

        dSx_ex = 0.5 * rho * c * U_sq * Cl * np.sin(phi)
        npt.assert_allclose(dSx, dSx_ex)

        dSz_ex = 0.5 * rho * c * U_sq * Cl * np.cos(phi)
        npt.assert_allclose(dSz, dSz_ex)

    def test_forward_inflow_consistency(self):
        # Check that the thrust computed by the blade element model is consistent
        # with the momentum theory for forward flight
        rho = 1.225  # Air density [kg/m^3]
        c0 = 0.02  # Chord length [m]
        R = 0.15  # Rotor radius [m]
        e = 0.1  # Root cutout ratio

        th0 = np.deg2rad(5.0)  # Constant blade bitch
        h = 0.1  # Airfoil max camber, as fraction of chord length

        def pitch(r_):
            return th0

        def chord(r_):
            return c0

        Cd_0 = 0.02

        # Theoretical predictions for thin airfoil with parabolic camber
        Cl_0 = 4 * np.pi * h
        Cl_alpha = 2 * np.pi
        Cm_0 = -np.pi * h

        airfoil_model = ThinAirfoil(
            Cl_0=Cl_0,
            Cl_alpha=Cl_alpha,
            Cd_0=Cd_0,
            Cm_0=Cm_0,
        )

        # Specify hover conditions
        v_inf = 10.0
        theta = np.deg2rad(-5.0)
        v_W = v_inf * np.array([np.cos(theta), 0.0, -np.sin(theta)])
        w_W = np.zeros(3)  # Wind-frame angular velocity

        Omega = 500.0  # Rotor angular velocity [rad/s]

        rotor_model = BladeElementModel(
            R=R,
            rho=rho,
            e=e,
            T0=0.0,  # Unused for this test
            airfoil_model=airfoil_model,
            chord=chord,
            blade_pitch=pitch,
        )
        geometry = RotorGeometry()  # Default geometry has zero offset

        lambda_guess = 0.0
        t = 0.0
        x = np.array([])
        lambda_ = rotor_model._lambda_solve(
            lambda_guess, t, v_W, w_W, x, Omega, geometry
        )

        CT_blade_element = rotor_model.thrust_coefficient(
            t, v_W, w_W, x, Omega, lambda_, geometry
        )

        mu_x = -v_W[0] / (Omega * rotor_model.R)
        mu_z = -v_W[2] / (Omega * rotor_model.R)
        CT_momentum_disk = 2 * (lambda_ - mu_z) * np.sqrt(mu_x**2 + lambda_**2)

        npt.assert_allclose(CT_blade_element, CT_momentum_disk, rtol=1e-6, atol=1e-6)


class TestTrimStability:
    """Check trim and linear stability of the multirotor vehicle"""

    def test_hover_stability(self):
        kF = 1.0
        rotor_model = QuadraticRotorModel(
            kF=kF,
            kM=0.01,
        )
        drag_model = QuadraticDragModel(
            rho=1.225,
            Cd=0.0,
            A=1.0,
        )
        gravity_model = ConstantGravityModel(g0=g0)

        L = 0.2  # Arm length
        rotors = []
        theta = np.pi / 4
        ccw = True
        for i in range(4):
            rotors.append(
                RotorGeometry(
                    offset=np.array([L * np.cos(theta), L * np.sin(theta), 0]),
                    ccw=ccw,
                )
            )
            theta += np.pi / 2
            ccw = not ccw

        vehicle = MultiRotorVehicle(
            m=m,
            J_B=J_B,
            rotors=rotors,
            drag_model=drag_model,
            gravity_model=gravity_model,
            rotor_model=rotor_model,
            attitude="euler",
        )

        #
        # Trim the vehicle at hover
        #

        # Target forward velocity in the inertial frame
        v_N = np.array([0.0, 0.0, 0.0])
        p_N = np.zeros(3)  # Arbitrary position
        w_B = np.zeros(3)  # Zero angular velocity

        @arc.compile
        def residual(p):
            phi, theta = p[:2]  # (pitch, roll) angles
            u = p[2:]  # Rotor angular velocities
            rpy = np.array([phi, theta, 0.0], like=p)
            C_BN = dcm(rpy)
            v_B = C_BN @ v_N
            x = vehicle.State(p_N, rpy, v_B, w_B)
            x_t = vehicle.dynamics(0.0, x, u)
            # Residuals of dynamics equations only
            return np.hstack([x_t.v_B, x_t.w_B])

        u0 = 500.0
        p0 = np.array([0.0, 0.0, u0, u0, u0, u0])

        p_trim = arc.root(residual, p0)

        phi_trim = p_trim[0]
        theta_trim = p_trim[1]
        u_trim = p_trim[2:]

        u_trim_ex = np.sqrt(m * g0 / (4 * kF)) * np.ones(4)

        npt.assert_allclose(phi_trim, 0.0, atol=1e-6)
        npt.assert_allclose(theta_trim, 0.0, atol=1e-6)
        npt.assert_allclose(u_trim, u_trim_ex, atol=1e-6)

        rpy_trim = np.array([phi_trim, theta_trim, 0.0])
        C_BN = dcm(rpy_trim)
        v_B_trim = C_BN @ v_N
        x_trim = vehicle.State(p_N, rpy_trim, v_B_trim, w_B)

        #
        # Linear stability analysis for longitudinal motion
        #

        # Longitudinal dynamics include surge (vx), heave (vz),
        # pitch (theta), and pitch rate (q). The other states are
        # assumed to be in trim
        def longitudinal_dofs(x):
            return np.hstack(
                [
                    x.att[1],  # theta
                    x.v_B[0],  # vx
                    x.v_B[2],  # vz
                    x.w_B[1],  # q
                ]
            )

        # Right-hand side function for the lateral dynamics
        @arc.compile
        def f_lon(x_lon, u):
            theta, vx, vz, q = x_lon  # Perturbations
            x = vehicle.State(
                np.zeros(3),
                np.hstack([phi_trim, theta, 0.0]),
                np.hstack([vx, v_B_trim[1], vz]),
                np.hstack([0.0, q, 0.0]),
            )
            x_t = vehicle.dynamics(0.0, x, u)
            return longitudinal_dofs(x_t)

        x_lon_trim = longitudinal_dofs(x_trim)

        # Linearized state-space matrices
        A_lon = arc.jac(f_lon, 0)(x_lon_trim, u_trim)
        B_lon = arc.jac(f_lon, 1)(x_lon_trim, u_trim)

        A_lon_ex = np.array(
            [
                [0, 0, 0, 1],
                [-g0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        npt.assert_allclose(A_lon, A_lon_ex, atol=1e-6)

        # Check that the control matrix is zero for (theta, u)
        npt.assert_allclose(B_lon[:2, :], 0.0, atol=1e-6)

        # Check vertical thrust derivatives are equal and non-zero
        npt.assert_allclose(B_lon[2, :], B_lon[2, 0], atol=1e-6)
        npt.assert_array_less(0, abs(B_lon[2, 0]))

        # Pitch moments have alternating signs but are otherwise equal
        signs = np.array([1, -1, -1, 1])
        npt.assert_allclose(B_lon[3, 0], signs * B_lon[3, :], atol=1e-6)
        npt.assert_array_less(0, abs(B_lon[3, 0]))

        #
        # Lateral-directional stability analysis
        #
        # Lateral-directional dynamics include roll (phi),
        # side-slip (vy), roll rate (p), and yaw rate (r).
        # The other states are assumed to be in trim

        def lateral_dofs(x):
            return np.hstack(
                [
                    x.att[0],  # phi
                    x.v_B[1],  # vy
                    x.w_B[0],  # p
                    x.w_B[2],  # r
                ]
            )

        # Right-hand side function for the lateral dynamics
        @arc.compile
        def f_lat(x_lat, u):
            phi, vy, p, r = x_lat  # Perturbations
            x = vehicle.State(
                np.zeros(3),
                np.hstack([phi, theta_trim, 0.0]),
                np.hstack([v_B_trim[0], vy, v_B_trim[2]]),
                np.hstack([p, 0.0, r]),
            )
            x_t = vehicle.dynamics(0.0, x, u)
            return lateral_dofs(x_t)

        x_lat_trim = lateral_dofs(x_trim)

        # Linearized state-space matrices
        A_lat = arc.jac(f_lat, 0)(x_lat_trim, u_trim)
        B_lat = arc.jac(f_lat, 1)(x_lat_trim, u_trim)

        A_lat_ex = np.array(
            [
                [0, 0, 1, 0],
                [g0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        npt.assert_allclose(A_lat, A_lat_ex, atol=1e-6)

        # Check that the control matrix is zero for (phi, u)
        npt.assert_allclose(B_lat[:2, :], 0.0, atol=1e-6)

        # Roll and yaw moments have alternating signs but are otherwise equal
        p_signs = np.array([1, 1, -1, -1])
        r_signs = np.array([1, -1, 1, -1])
        npt.assert_allclose(B_lat[2, 0], p_signs * B_lat[2, :], atol=1e-6)
        npt.assert_allclose(B_lat[3, 0], r_signs * B_lat[3, :], atol=1e-6)
        # Check that the control derivatives are non-zero
        npt.assert_array_less(0, abs(B_lat[2, 0]))
        npt.assert_array_less(0, abs(B_lat[3, 0]))
