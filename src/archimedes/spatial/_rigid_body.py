# ruff: noqa: N806, N803, N815
from __future__ import annotations

from typing import cast

import numpy as np

from ..tree import StructConfig, field, struct
from ._rotation import Rotation

__all__ = [
    "RigidBody",
    "RigidBodyConfig",
    "euler_kinematics",
    "dcm_from_euler",
]


def dcm_from_euler(rpy: np.ndarray, transpose: bool = False) -> np.ndarray:
    """Returns matrix to transform from inertial to body frame (R_BN).

    If transpose=True, returns matrix to transform from body to inertial frame (R_NB).

    This is the direction cosine matrix (DCM) corresponding to the given
    roll-pitch-yaw (rpy) angles.  This follows the standard aerospace
    convention and corresponds to the "xyz" sequence when using the ``Rotation``
    class.

    In general, the ``Rotation`` class should be preferred over Euler representations,
    although Euler angles are used in some special cases (e.g. stability analysis).
    In these cases, this function gives a more direct calculation of the
    transformation matrix without converting to the intermediate quaternion.

    Parameters
    ----------
    rpy : array_like, shape (3,)
        Roll, pitch, yaw angles in radians.
    transpose : bool, optional
        If True, returns the transpose of the DCM.  Default is False.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Direction cosine matrix R_BN (or R_NB if transpose=True).
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


def euler_kinematics(rpy: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Euler kinematical equations

    Defining 𝚽 = [phi, theta, psi] == Euler angles for roll, pitch, yaw
    attitude representation, this function returns a matrix H(𝚽) such
    that
        d𝚽/dt = H(𝚽) * ω.

    If inverse=True, it returns a matrix H(𝚽)^-1 such that
        ω = H(𝚽)^-1 * d𝚽/dt.

    Parameters
    ----------
    rpy : array_like, shape (3,)
        Roll, pitch, yaw angles in radians.
    inverse : bool, optional
        If True, returns the inverse matrix H(𝚽)^-1. Default is False.

    Returns
    -------
    np.ndarray, shape (3, 3)
        The transformation matrix H(𝚽) or its inverse.

    Notes
    -----

    Typical rigid body dynamics calculations provide the body-frame angular velocity
    ω_B, but this is _not_ the time derivative of the Euler angles.  Instead, one
    can define a matrix H(𝚽) such that d𝚽/dt = H(𝚽) * ω_B.

    This matrix H(𝚽) has a singularity at θ = ±π/2 (gimbal lock).

    Note that the ``RigidBody`` class by default uses quaternions (via the
    ``Rotation`` class) for attitude representation.
    In general this is preferred due to the gimbal lock singularity, but
    special cases like stability analysis may use Euler angle kinematics.
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

    The equations of motion are given by

    .. math::
        \\dot{p}_N &= R_{BN}^T(\\mathbf{q}) v_B \\\\
        \\dot{\\mathbf{q}} &= \\frac{1}{2} \\mathbf{\\omega}_B
            \\otimes \\mathbf{q} \\\\
        \\dot{v}_B &= \\frac{1}{m}(\\mathbf{F}_B - \\dot{m} v_B)
            - \\mathbf{\\omega}_B \\times v_B \\\\
        \\dot{\\mathbf{\\omega}}_B &= J_B^{-1}(\\mathbf{M}_B
            - \\dot{J}_B \\mathbf{\\omega}_B - \\mathbf{\\omega}_B
            \\times (J_B \\mathbf{\\omega}_B))

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
            # C_BN rotates from the Newtonian frame N to the body frame B.
            C_BN = dcm_from_euler(rpy)

            # Transform roll-pitch-yaw rates in the body frame to time derivatives of
            # Euler angles - Euler kinematic equations
            H = euler_kinematics(rpy)

            # Time derivatives of roll-pitch-yaw (rpy) orientation
            att_deriv = H @ x.w_B

            # Time derivative of position in Newtonian frame N
            dp_N = C_BN.T @ x.v_B

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
