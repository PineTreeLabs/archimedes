from __future__ import annotations
from typing import TYPE_CHECKING
import abc

import numpy as np

from archimedes import struct

from .rotations import (
    dcm_from_euler,
    dcm_from_quaternion,
    euler_kinematics,
    quaternion_derivative,
    quaternion_to_euler,
)

if TYPE_CHECKING:
    from archimedes.typing import PyTree


def wind_frame(v_rel_B):
    """Compute total velocity, angle of attack, and sideslip angle

    The input should be the vehicle wind-relative velocity computed in
    body-frame axes.  If the inertial velocity of the vehicle expressed in
    body-frame axes is v_B and the Earth-relative wind velocity is w_N,
    then the relative velocity is v_rel_B = v_B + R_BN @ w_N, where R_BN
    is the rotation matrix from inertial to body frame.

    If there is no wind, then v_rel_B = v_B.
    """
    u, v, w = v_rel_B
    vt = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)
    beta = np.arcsin(v / vt)
    return vt, alpha, beta


@struct.pytree_node
class FlightVehicle(metaclass=abc.ABCMeta):
    m: float = 1.0  # mass [kg]
    J_B: np.ndarray = struct.field(default_factory=lambda: np.eye(3))  # inertia matrix
    attitude: str = struct.field(default="quaternion", static=True)

    @struct.pytree_node
    class State:
        p_N: np.ndarray  # Position of the center of mass in the Newtonian frame N
        att: np.ndarray  # Attitude (orientation) of the vehicle
        v_B: np.ndarray  # Velocity of the center of mass in body frame B
        w_B: np.ndarray  # Angular velocity in body frame (ω_B)
        aux: PyTree = struct.field(default=None)  # Auxiliary state variables

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
        p_N = x.p_N  # Position of the center of mass in the Newtonian frame N
        v_B = x.v_B  # Velocity of the center of mass in body frame B
        w_B = x.w_B  # Angular velocity in body frame (ω_B)

        if self.attitude == "euler":
            rpy = x.att

            # Convert roll-pitch-yaw (rpy) orientation to the direction cosine matrix.
            # C_BN rotates from the Newtonian frame N to the body frame B.
            # C_BN.T = C_NB rotates from the body frame B to the Newtonian frame N.
            C_BN = dcm_from_euler(rpy)

            # Transform roll-pitch-yaw rates in the body frame to time derivatives of Euler angles
            # These are the Euler kinematic equations (1.4-5)
            H = euler_kinematics(rpy)

            # Time derivatives of roll-pitch-yaw (rpy) orientation
            att_deriv = H @ w_B

        elif self.attitude == "quaternion":
            q = x.att

            # Convert roll-pitch-yaw (rpy) orientation to the direction cosine matrix.
            # C_BN rotates from the Newtonian frame N to the body frame B.
            # C_BN.T = C_NB rotates from the body frame B to the Newtonian frame N.
            C_BN = dcm_from_quaternion(q)

            # Time derivative of the quaternion
            att_deriv = quaternion_derivative(q, w_B)

        F_B, M_B, aux_state_derivs = self.net_forces(t, x, u, C_BN)

        # Velocity in the Newtonian frame
        dp_N = C_BN.T @ v_B

        # Acceleration in body frame
        dv_B = (F_B / self.m) - np.cross(w_B, v_B)

        # Angular acceleration in body frame
        # solve Euler dynamics equation 𝛕 = I α + ω × (I ω)  for α
        # dw_B = np.linalg.inv(self.J_B) @ (M_B - np.cross(w_B, self.J_B @ w_B))
        dw_B = np.linalg.solve(self.J_B, M_B - np.cross(w_B, self.J_B @ w_B))

        # Pack the state derivatives
        return self.State(
            p_N=dp_N,
            att=att_deriv,
            v_B=dv_B,
            w_B=dw_B,
            aux=aux_state_derivs,
        )
