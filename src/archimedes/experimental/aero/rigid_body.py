from __future__ import annotations

import numpy as np

from archimedes import struct, field, StructConfig

from .rotations import (
    dcm_from_euler,
    dcm_from_quaternion,
    euler_kinematics,
    quaternion_derivative,
)
from ..spatial import Rotation


@struct
class RigidBody:
    baumgarte: float = 1.0  # Baumgarte stabilization factor for quaternion kinematics

    @struct
    class State:
        p_N: np.ndarray  # Position of the center of mass in the Newtonian frame N
        att: Rotation  # Attitude (orientation) of the vehicle
        v_B: np.ndarray  # Velocity of the center of mass in body frame B
        w_B: np.ndarray  # Angular velocity in body frame (ω_B)

    @struct
    class Input:
        F_B: np.ndarray  # Net forces in body frame B
        M_B: np.ndarray  # Net moments in body frame B
        m: float  # mass [kg]
        J_B: np.ndarray  # inertia matrix [kg·m²]
        dm_dt: float = 0.0  # mass rate of change [kg/s]
        # inertia rate of change [kg·m²/s]
        dJ_dt: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def calc_kinematics(self, x: State):
        # Unpack the state
        v_B = x.v_B  # Velocity of the center of mass in body frame B

        # Velocity in the Newtonian frame
        dp_N = x.att.apply(v_B)

        att_deriv = x.att.derivative(x.w_B, baumgarte=self.baumgarte)

        return dp_N, att_deriv

    def calc_dynamics(self, t, x: State, u: Input):
        # Unpack the state
        v_B = x.v_B  # Velocity of the center of mass in body frame B
        w_B = x.w_B  # Angular velocity in body frame (ω_B)

        # Acceleration in body frame
        dv_B = ((u.F_B - u.dm_dt * v_B) / u.m) - np.cross(w_B, v_B)

        # Angular acceleration in body frame
        # solve Euler dynamics equation 𝛕 = I α + ω × (I ω)  for α
        dw_B = np.linalg.solve(
            u.J_B, u.M_B - u.dJ_dt @ w_B - np.cross(w_B, u.J_B @ w_B)
        )

        return dv_B, dw_B

    def dynamics(self, t, x: State, u: Input) -> State:
        """
        Flat-earth 6-dof dynamics

        Based on equations 1.7-18 from Lewis, Johnson, Stevens

        Args:
            t: time
            x: state vector
            u: input vector containing net forces and moments

        Returns:
            xdot: time derivative of the state vector
        """
        dp_N, att_deriv = self.calc_kinematics(x)
        dv_B, dw_B = self.calc_dynamics(t, x, u)

        # Pack the state derivatives
        return self.State(
            p_N=dp_N,
            att=att_deriv,
            v_B=dv_B,
            w_B=dw_B,
        )


class RigidBodyConfig(StructConfig):
    baumgarte: float = 1.0  # Baumgarte stabilization factor

    def build(self) -> RigidBody:
        """Build and return a RigidBody instance."""
        return RigidBody(baumgarte=self.baumgarte)
