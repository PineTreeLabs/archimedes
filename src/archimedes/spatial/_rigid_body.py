# ruff: noqa: N806, N803, N815
from __future__ import annotations

from typing import cast

import numpy as np

from ..tree import StructConfig, field, struct
from ._attitude import Rotation, euler_to_dcm, euler_kinematics

__all__ = [
    "RigidBody",
    "RigidBodyConfig",
]


@struct
class RigidBody:
    """6-dof rigid body dynamics model

    This class implements 6-dof rigid body dynamics based on reference equations
    from Lewis, Johnson, and Stevens, "Aircraft Control and Simulation" [1]_.

    This implementation is general and does not make any assumptions about the
    forces, moments, or mass properties.  These must be provided as inputs to the
    dynamics function.

    The model assumes a non-inertial body-fixed reference frame B and a Newtonian
    inertial reference frame N.  The body frame is assumed to be located at the
    vehicle's center of mass.

    With these conventions, the state vector is defined as
        ``x = [p_N, q, v_B, w_B]``

    where

    - ``p_N`` = position of the center of mass in the Newtonian frame N
    - ``q`` = attitude (orientation) of the vehicle as a unit quaternion
    - ``v_B`` = velocity of the center of mass in body frame B
    - ``w_B`` = angular velocity in body frame (ω_B)

    Note that the attitude is implemented using the :py:class:`Rotation` class.
    The transformation implemented by ``Rotation.apply`` with this attitude
    represents ``R_NB``, the rotation from the body frame B to the inertial frame N.

    The equations of motion are given by

    .. math::
        \\dot{\\mathbf{p}}_N &= \\mathbf{R}_{BN}^T(\\mathbf{q}) \\mathbf{v}_B \\\\
        \\dot{\\mathbf{q}} &= \\frac{1}{2} \\mathbf{q} \\otimes \\mathbf{\\omega}_B
            \\\\
        \\dot{\\mathbf{v}}_B &= \\frac{1}{m}(\\mathbf{F}_B - \\dot{m} \\mathbf{v}_B)
            - \\mathbf{\\omega}_B \\times \\mathbf{v}_B \\\\
        \\dot{\\mathbf{\\omega}}_B &= \\mathbf{J}_B^{-1}(\\mathbf{M}_B
            - \\dot{\\mathbf{J}}_B \\mathbf{\\omega}_B - \\mathbf{\\omega}_B
            \\times (\\mathbf{J}_B \\mathbf{\\omega}_B))

    where

    - ``R_{BN}(q)`` = direction cosine matrix (DCM)
    - ``m`` = mass of the vehicle
    - ``J_B`` = inertia matrix of the vehicle in body axes
    - ``F_B`` = net forces acting on the vehicle in body frame B
    - ``M_B`` = net moments acting on the vehicle in body frame B

    The inputs to the dynamics function are a ``RigidBody.Input`` struct
    containing the forces, moments, mass, and inertia properties.  By default
    the time derivatives of the mass and inertia are zero unless specified
    in the input struct.

    Parameters
    ----------
    rpy_attitude : bool, optional
        If True, use roll-pitch-yaw angles for attitude representation instead
        of quaternions.  Default is False.  Note that using roll-pitch-yaw angles
        introduces a singularity (gimbal lock) and are not recommended for general use.
    baumgarte : float, optional
        Baumgarte stabilization factor for quaternion kinematics.  Default is 1.0.
        This adds a correction term to the quaternion kinematics to help maintain
        the unit norm constraint.

    Examples
    --------
    >>> import archimedes as arc
    >>> from archimedes.spatial import RigidBody, Rotation
    >>> import numpy as np
    >>> rigid_body = RigidBody()
    >>> t = 0
    >>> v_B = np.array([1, 0, 0])  # Constant velocity in x-direction
    >>> att = Rotation.from_quat([1, 0, 0, 0])  # No rotation
    >>> x = rigid_body.State(
    ...     p_N=np.zeros(3),
    ...     att=att,
    ...     v_B=v_B,
    ...     w_B=np.zeros(3),
    ... )
    >>> u = rigid_body.Input(
    ...     F_B=np.array([0, 0, -9.81]),  # Gravity
    ...     M_B=np.zeros(3),
    ...     m=2.0,
    ...     J_B=np.diag([1.0, 1.0, 1.0]),
    ... )
    >>> rigid_body.dynamics(t, x, u)
    State(p_N=array([1., 0., 0.]),
      att=Rotation(quat=array([0., 0., 0., 0.]), scalar_first=True),
      v_B=array([ 0.   ,  0.   , -4.905]),
      w_B=array([0., 0., 0.]))

    References
    ----------
    .. [1] Lewis, F. L., Johnson, E. N., & Stevens, B. L. (2015).
            Aircraft Control and Simulation. Wiley.
    """

    rpy_attitude: bool = False  # If True, use roll-pitch-yaw for attitude
    baumgarte: float = 1.0  # Baumgarte stabilization factor for quaternion kinematics

    @struct
    class State:
        p_N: np.ndarray  # Position of the center of mass in the Newtonian frame N
        att: Rotation | np.ndarray  # Attitude (orientation) of the vehicle
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
        dJ_dt: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # type: ignore

    def calc_kinematics(self, x: State) -> tuple[np.ndarray, Rotation | np.ndarray]:
        """Calculate kinematics (position and attitude derivatives)

        Parameters
        ----------
        x : RigidBody.State
            Current state of the rigid body.

        Returns
        -------
        dp_N : np.ndarray
            Time derivative of position in Newtonian frame N.
        att_deriv : Rotation or np.ndarray
            Time derivative of attitude (quaternion derivative or roll-pitch-yaw rates).

        Notes
        -----
        This function calculates the kinematics (position and attitude derivatives)
        based on the current state (velocity and angular velocity).

        Typically this does not need to be called directly, but is available
        separately for special analysis or testing.
        """
        if self.rpy_attitude:
            rpy = cast(np.ndarray, x.att)

            # Convert roll-pitch-yaw (rpy) orientation to the direction cosine matrix.
            # R_NB rotates from the body frame B to the Newtonian frame N.
            R_NB = euler_to_dcm(rpy)

            # Transform roll-pitch-yaw rates in the body frame to time derivatives of
            # Euler angles - Euler kinematic equations
            H = euler_kinematics(rpy)

            # Time derivatives of roll-pitch-yaw (rpy) orientation
            att_deriv = H @ x.w_B

            # Time derivative of position in Newtonian frame N
            dp_N = R_NB.T @ x.v_B

        else:
            att = cast(Rotation, x.att)
            dp_N = att.apply(x.v_B)
            att_deriv = att.derivative(x.w_B, baumgarte=self.baumgarte)

        return dp_N, att_deriv

    def calc_dynamics(self, x: State, u: Input) -> tuple[np.ndarray, np.ndarray]:
        """Calculate dynamics (velocity and angular velocity derivatives)

        Parameters
        ----------
        x : RigidBody.State
            Current state of the rigid body.
        u : RigidBody.Input
            Current inputs (forces, moments, mass properties).

        Returns
        -------
        dv_B : np.ndarray
            Time derivative of velocity in body frame B.
        dw_B : np.ndarray
            Time derivative of angular velocity in body frame B.

        Notes
        -----
        This function calculates the dynamics (velocity and angular velocity
        derivatives) based on the current state and inputs (forces, moments,
        mass properties).

        Typically this does not need to be called directly, but is available
        separately for special analysis or testing.
        """
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

    def dynamics(self, t: float, x: State, u: Input) -> State:
        """Calculate 6-dof dynamics

        Args:
            t: time
            x: state vector
            u: input vector containing net forces and moments

        Returns:
            xdot: time derivative of the state vector
        """
        dp_N, att_deriv = self.calc_kinematics(x)
        dv_B, dw_B = self.calc_dynamics(x, u)

        # Pack the state derivatives
        return self.State(
            p_N=dp_N,
            att=att_deriv,
            v_B=dv_B,
            w_B=dw_B,
        )


class RigidBodyConfig(StructConfig):
    """Configuration for ``RigidBody`` model."""

    baumgarte: float = 1.0  # Baumgarte stabilization factor
    rpy_attitude: bool = False  # If True, use roll-pitch-yaw for attitude

    def build(self) -> RigidBody:
        """Build and return a RigidBody instance."""
        return RigidBody(baumgarte=self.baumgarte, rpy_attitude=self.rpy_attitude)
