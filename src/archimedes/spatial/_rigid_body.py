# ruff: noqa: N806, N803, N815
from __future__ import annotations

import numpy as np

from ..tree import StructConfig, field, struct
from ._attitude import Attitude

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
    - ``q`` = attitude (orientation) of the vehicle
    - ``v_B`` = velocity of the center of mass in body frame B
    - ``w_B`` = angular velocity in body frame (ω_B)

    Note that the attitude can be any object implementing the :py:class:`Attitude`
    protocol, commonly :py:class:`Quaternion` or :py:class:`EulerAngles`.
    The transformation implemented by ``rotate`` with this attitude represents
    ``R_NB``, the rotation from the body frame B to the inertial frame N.

    The equations of motion for a quaternion attitude are given by

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


    Examples
    --------
    >>> import archimedes as arc
    >>> from archimedes.spatial import RigidBody, Quaternion
    >>> import numpy as np
    >>> rigid_body = RigidBody()
    >>> t = 0
    >>> v_B = np.array([1, 0, 0])  # Constant velocity in x-direction
    >>> att = Quaternion([1, 0, 0, 0])  # No rotation
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
      att=Quaternion([0., 0., 0., 0.]),
      v_B=array([ 0.   ,  0.   , -4.905]),
      w_B=array([0., 0., 0.]))

    References
    ----------
    .. [1] Lewis, F. L., Johnson, E. N., & Stevens, B. L. (2015).
            Aircraft Control and Simulation. Wiley.
    """

    @struct
    class State:
        p_N: np.ndarray  # Position of the center of mass in the Newtonian frame N
        att: Attitude  # Attitude (orientation) of the vehicle
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

    @classmethod
    def calc_kinematics(cls, x: State) -> tuple[np.ndarray, Attitude]:
        """Calculate kinematics (position and attitude derivatives)

        Parameters
        ----------
        x : RigidBody.State
            Current state of the rigid body.

        Returns
        -------
        dp_N : np.ndarray
            Time derivative of position in Newtonian frame N.
        att_deriv : Attitude
            Time derivative of attitude (e.g. quaternion derivative or Euler rates).

        Notes
        -----
        This function calculates the kinematics (position and attitude derivatives)
        based on the current state (velocity and angular velocity).

        Typically this does not need to be called directly, but is available
        separately for special analysis or testing.
        """

        dp_N = x.att.rotate(x.v_B)
        att_deriv = x.att.kinematics(x.w_B)

        return dp_N, att_deriv

    @classmethod
    def calc_dynamics(cls, x: State, u: Input) -> tuple[np.ndarray, np.ndarray]:
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

    @classmethod
    def dynamics(cls, t: float, x: State, u: Input) -> State:
        """Calculate 6-dof dynamics

        Args:
            t: time
            x: state vector
            u: input vector containing net forces and moments

        Returns:
            xdot: time derivative of the state vector
        """
        dp_N, att_deriv = cls.calc_kinematics(x)
        dv_B, dw_B = cls.calc_dynamics(x, u)

        # Pack the state derivatives
        return cls.State(
            p_N=dp_N,
            att=att_deriv,
            v_B=dv_B,
            w_B=dw_B,
        )


class RigidBodyConfig(StructConfig):
    """Configuration for ``RigidBody`` model."""

    def build(self) -> RigidBody:
        """Build and return a RigidBody instance."""
        return RigidBody()
