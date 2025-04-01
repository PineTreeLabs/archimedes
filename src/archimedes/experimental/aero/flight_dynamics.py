import abc

import numpy as np

from archimedes import struct


def quaternion_inverse(q):
    """
    Inverse of a quaternion q = [w, x, y, z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], like=q)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions q1 = [w1, x1, y1, z1] and q2 = [w2, x2, y2, z2]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        like=q1,
    )


def quaternion_derivative(q, w, lambda_correction=1.0):
    """Compute quaternion derivative with Baumgarte stabilization on normalization.

    Args:
        q: quaternion [w, x, y, z]
        w: angular velocity [wx, wy, wz]
        lambda_correction: feedback gain for normalization (default: 1.0)
    """
    # Form pure quaternion from angular velocity
    q_w = np.array([0, w[0], w[1], w[2]], like=q)

    # Standard quaternion kinematics
    qdot = 0.5 * quaternion_multiply(q, q_w)

    # Add normalization correction term
    error = np.dot(q, q) - 1.0
    qdot = qdot - lambda_correction * error * q

    return qdot


def dcm_from_quaternion(q):
    """Convert quaternion to direction cosine matrix (rotation matrix)

    q: quaternion [w, x, y, z]
    Returns: 3x3 rotation matrix to transform from inertial to body frame (R_BN)
    """
    w, x, y, z = q

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
            [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        ],
        like=q,
    )


def dcm_from_euler(rpy, transpose=False):
    """Returns matrix to transform from inertial to body frame (R_BN)

    If transpose=True, returns matrix to transform from body to inertial frame (R_NB).
    """
    φ, θ, ψ = rpy[0], rpy[1], rpy[2]

    sφ, cφ = np.sin(φ), np.cos(φ)
    sθ, cθ = np.sin(θ), np.cos(θ)
    sψ, cψ = np.sin(ψ), np.cos(ψ)

    R = np.array(
        [
            [cθ * cψ, cθ * sψ, -sθ],
            [sφ * sθ * cψ - cφ * sψ, sφ * sθ * sψ + cφ * cψ, sφ * cθ],
            [cφ * sθ * cψ + sφ * sψ, cφ * sθ * sψ - sφ * cψ, cφ * cθ],
        ],
        like=rpy,
    )

    if transpose:
        R = R.T

    return R


def euler_to_quaternion(rpy):
    """Convert roll-pitch-yaw Euler angles to quaternion."""
    φ, θ, ψ = rpy[0], rpy[1], rpy[2]

    # Half angles
    c1, s1 = np.cos(φ / 2), np.sin(φ / 2)
    c2, s2 = np.cos(θ / 2), np.sin(θ / 2)
    c3, s3 = np.cos(ψ / 2), np.sin(ψ / 2)

    # Quaternion components
    w = c1 * c2 * c3 + s1 * s2 * s3
    x = s1 * c2 * c3 - c1 * s2 * s3
    y = c1 * s2 * c3 + s1 * c2 * s3
    z = c1 * c2 * s3 - s1 * s2 * c3

    return np.array([w, x, y, z], like=rpy)


def quaternion_to_euler(q):
    """Convert quaternion to roll-pitch-yaw Euler angles."""
    w, x, y, z = q

    φ = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    θ = 2 * np.arctan2(1 + 2 * (w * y - x * z), 1 - 2 * (w * y - x * z)) - np.pi / 2
    ψ = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return np.array([φ, θ, ψ], like=q)


def euler_kinematics(rpy, inverse=False):
    """Euler kinematical equations

    Define 𝚽 = [phi, theta, psi] == Euler angles for roll, pitch, yaw (same in body and inertial frames)

    The kinematics in body and inertial frames are:
            ω = [P, Q, R] == [roll_rate, pitch_rate, yaw_rate] in body frame
            d𝚽/dt = time derivative of Euler angles (inertial frame)

    Returns matrix H(𝚽) such that d𝚽/dt = H(𝚽) * ω
    If inverse=True, returns matrix H(𝚽)^-1 such that ω = H(𝚽)^-1 * d𝚽/dt.
    """

    φ, θ = rpy[0], rpy[1]  # Roll, pitch

    sφ, cφ = np.sin(φ), np.cos(φ)
    sθ, cθ = np.sin(θ), np.cos(θ)
    tθ = np.tan(θ)

    _1 = np.ones_like(φ)
    _0 = np.zeros_like(φ)

    if inverse:
        Hinv = np.array(
            [
                [_1, _0, -sθ],
                [_0, cφ, cθ * sφ],
                [_0, -sφ, cθ * cφ],
            ],
            like=rpy,
        )
        return Hinv

    else:
        H = np.array(
            [
                [_1, sφ * tθ, cφ * tθ],
                [_0, cφ, -sφ],
                [_0, sφ / cθ, cφ / cθ],
            ],
            like=rpy,
        )
        return H


@struct.pytree_node
class FlightVehicle(metaclass=abc.ABCMeta):
    m: float = 1.0  # mass [kg]
    J_B: np.ndarray = struct.field(default_factory=lambda: np.eye(3))  # inertia matrix

    @abc.abstractmethod
    def net_forces(self, t, x, u, C_BN):
        """Net forces and moments in body frame B, plus any extra state derivatives

        Args:
            t: time
            x: state vector
            u: rotor speeds
            C_BN: rotation matrix from inertial (N) to body (B) frame

        Returns:
            F_B: net forces in body frame B
            M_B: net moments in body frame B
            aux_state_derivs: time derivatives of auxiliary state variables
        """

    def dynamics(self, t, x, u):
        """
        Flat-earth 6-dof dynamics for a multirotor vehicle

        Based on equations 1.7-18 from Lewis, Johnson, Stevens

        The input should be a function of time and state: u(t, x) -> u

        Args:
            t: time
            x: state vector
            u: rotor speeds

        Returns:
            xdot: time derivative of the state vector
        """

        # Unpack the state
        p_N = x[0:3]  # Position of the center of mass in the Newtonian frame N
        q = x[3:7]  # Quaternion orientation
        v_B = x[7:10]  # Velocity of the center of mass in body frame B
        w_B = x[10:13]  # Angular velocity in body frame (ω_B)

        # Convert roll-pitch-yaw (rpy) orientation to the direction cosine matrix.
        # C_BN rotates from the Newtonian frame N to the body frame B.
        # C_BN.T = C_NB rotates from the body frame B to the Newtonian frame N.
        C_BN = dcm_from_quaternion(q)

        F_B, M_B, aux_state_derivs = self.net_forces(t, x, u, C_BN)

        # Time derivative of the quaternion
        dq = quaternion_derivative(q, w_B)

        # Velocity in the Newtonian frame
        dp_N = C_BN.T @ v_B

        # Acceleration in body frame
        dv_B = (F_B / self.m) - np.cross(w_B, v_B)

        # Angular acceleration in body frame
        # solve Euler dynamics equation 𝛕 = I α + ω × (I ω)  for α
        # dw_B = np.linalg.inv(self.J_B) @ (M_B - np.cross(w_B, self.J_B @ w_B))
        dw_B = np.linalg.solve(self.J_B, M_B - np.cross(w_B, self.J_B @ w_B))

        # Pack the state derivatives
        return np.hstack([dp_N, dq, dv_B, dw_B, aux_state_derivs])
