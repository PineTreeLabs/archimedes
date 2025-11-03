"""Wrapper class for Euler angles"""
# ruff: noqa: N806, N803, N815
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from archimedes import tree

from ._attitude import _rotate
from ._euler import euler_kinematics, euler_to_dcm, _check_angles, _check_seq
from ._quaternion import Quaternion

__all__ = ["EulerAngles"]

class EulerAngles:
    """Euler angle representation of a rotation in 3 dimensions

    Parameters
    ----------
    angles : array_like
        Euler angles in radians
    seq : str, optional
        Sequence of axes for Euler angles (up to length 3).  Each character must be one
        of 'x', 'y', 'z' (extrinsic) or 'X', 'Y', 'Z' (intrinsic).  Default is 'xyz'.

    Attributes
    ----------
    array : np.ndarray
        Underlying array of Euler angles.
    seq : str
        Sequence of axes for Euler angles.

    Methods
    -------
    __len__
    from_quat
    from_euler
    as_quat
    as_matrix
    as_euler
    identity
    rotate
    inv
    kinematics

    Examples
    --------
    >>> from archimedes.spatial import EulerAngles
    >>> import numpy as np

    TODO: Add examples

    See Also
    --------
    Quaternion : Quaternion representation of rotation in 3D
    RigidBody : Rigid body dynamics supporting ``EulerAngles`` attitude representation
    euler_to_dcm : Directly calculate rotation matrix from roll-pitch-yaw angles
    euler_kinematics : Transform roll-pitch-yaw rates to body-frame angular velocity
    """

    def __init__(self, angles: np.ndarray, seq: str = "xyz"):
        angles = np.hstack(angles)
        _check_seq(seq)
        _check_angles(angles, seq)

        self.array = angles.flatten()
        self.seq = seq


    # === Methods for implementing Attitude protocol ===

    def as_matrix(self) -> np.ndarray:
        """Return the corresponding rotation matrix.

        Returns
        -------
        np.ndarray
            The rotation matrix as a 3x3 numpy array.
        """
        return euler_to_dcm(self.array, self.seq)

    def rotate(self, vectors: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Rotate vectors using this Euler angle rotation.

        If the Euler angles represent the attitude of a body B relative to a frame A,
        then this method rotates vectors from frame B to frame A.  Specifically, for a
        vector v_B expressed in frame B, the corresponding vector in frame A is given by
        ``v_A = R_AB @ v_B``, where R_AB is the direction cosine matrix obtained from
        ``self.as_matrix()``.

        Parameters
        ----------
        vectors : np.ndarray
            Vectors to rotate. Shape should be (..., 3).
        inverse : bool, optional
            If True, applies the inverse rotation. Default is False.

        Returns
        -------
        np.ndarray
            Rotated vectors with the same shape as input.
        """
        return _rotate(self, vectors, inverse=inverse)
    
    def inv(self) -> EulerAngles:
        """Return the inverse (conjugate) of this Euler angle rotation.

        Returns
        -------
        EulerAngles
            A new EulerAngles instance representing the inverse rotation.
        """
        angles = -self.array[::-1]
        seq = self.seq[::-1]
        return EulerAngles(angles=angles, seq=seq)
    
    def kinematics(self, w_B: np.ndarray) -> EulerAngles:
        """Compute the time derivative of the Euler angles given angular velocity.

        If the Euler angles represent the attitude of a body B, then w_B should be
        the body relative angular velocity ω_B.  See :py:func:`euler_kinematics` for
        details.

        **CAUTION**: This method returns the time derivative of the Euler angles,
        which is represented with the same data structure for consistency with
        ODE solving - but this return is not itself a valid rotation representation
        until integrated in time.

        Parameters
        ----------
        w_B : np.ndarray
            Angular velocity vector expressed in the body frame B.

        Returns
        -------
        EulerAngles
            Time derivative of the Euler angles.
        """
        H = euler_kinematics(self.array, inverse=False)
        return H @ w_B

    # === Other methods ===

    def __len__(self) -> int:
        """Return the number of Euler angles (length of sequence)."""
        return len(self.seq)

    @classmethod
    def from_quat(cls, quat: Quaternion, seq: str = "xyz") -> EulerAngles:
        """Create EulerAngles from a Quaternion.

        Parameters
        ----------
        quat : Quaternion
            Quaternion representing the rotation.

        Returns
        -------
        EulerAngles
            New EulerAngles instance representing the same rotation.
        """
        angles = quat.as_euler(seq)
        return cls(angles=angles, seq=seq)

    def as_quat(self) -> Quaternion:
        """Return the corresponding Quaternion representation.

        Returns
        -------
        Quaternion
            The equivalent Quaternion representation of this rotation.
        """
        return Quaternion.from_euler(self, self.seq)

    @classmethod
    def from_euler(cls, euler: EulerAngles, seq: str = "xyz") -> EulerAngles:
        """Return an EulerAngles instance from another EulerAngles instance.

        Can be used to change the sequence of axes.
        """
        if seq == euler.seq:
            return cls(angles=euler.array, seq=seq)

        quat = euler.as_quat()
        return quat.as_euler(seq)

    def as_euler(self, seq: str = "xyz") -> EulerAngles:
        """Return the Euler angles in a different sequence of axes.

        If the requested sequence is the same as the current sequence, returns self.
        """
        if seq == self.seq:
            return self

        quat = self.as_quat()
        euler_new = quat.as_euler(seq)
        return euler_new
    
    @classmethod
    def identity(cls, seq: str = "xyz") -> EulerAngles:
        """Return the identity EulerAngles (zero rotation).

        Parameters
        ----------
        seq : str, optional
            Sequence of axes for Euler angles. Default is 'xyz'.

        Returns
        -------
        EulerAngles
            New EulerAngles instance representing the identity rotation.
        """
        num_angles = len(seq)
        angles = np.zeros(num_angles)
        return cls(angles=angles, seq=seq)


# === Struct registration ===
def to_iter(euler: EulerAngles):
    children = (euler.array,)
    aux_data = (euler.seq)
    return children, aux_data


def from_iter(aux_data, children) -> EulerAngles:
    seq, = aux_data
    angles, = children
    return EulerAngles(angles=angles, seq=seq)


tree.register_struct(EulerAngles, to_iter, from_iter)