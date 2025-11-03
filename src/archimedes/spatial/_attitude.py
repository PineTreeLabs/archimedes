"""High-level wrappers for attitude representations."""
from __future__ import annotations
from typing import Protocol, cast

import numpy as np
from archimedes import array, tree

from ._euler import euler_kinematics, euler_to_dcm, _check_angles, _check_seq
from ._quaternion import (
    dcm_to_quaternion,
    euler_to_quaternion,
    quaternion_inverse,
    quaternion_kinematics,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_to_euler,
)

__all__ = [
    "Attitude",
    "EulerAngles",
    "Quaternion",
]

class Attitude(Protocol):
    """Protocol for attitude representations.

    This protocol defines the interface that all attitude representation classes
    must implement. An attitude representation class encapsulates the orientation
    of a rigid body in 3D space and provides methods for common operations such
    as conversion to/from other representations and kinematic models.
    """

    def as_matrix(self) -> np.ndarray:
        """Convert the attitude to a direction cosine matrix (DCM).

        If the attitude represents the orientation of a body B relative to a frame A,
        then this method returns the matrix R_AB that transforms vectors from
        frame B to frame A.  Specifically, for a vector v_B expressed in frame B,
        the corresponding vector in frame A is given by ``v_A = R_AB @ v_B``.

        The inverse transformation can be obtained by transposing this matrix:
        ``R_BA = R_AB.T``.

        Returns
        -------
            A 3x3 numpy array representing the DCM.
        """
        ...

    def rotate(self, vectors: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Rotate vectors by the attitude.

        If the attitude represents the orientation of a body B relative to a frame A,
        this method rotates vectors between the two frames. Specifically, for a vector v_B
        expressed in frame B, the corresponding vector in frame A is given by
        ``v_A = R_AB @ v_B``, where R_AB is the DCM obtained from `as_matrix()`.

        Parameters
        ----------
        vectors : np.ndarray
            A 1D array of shape (3,) or 2D array of shape (3, N) representing
            N vectors to rotate.
        inverse : bool
            If True, rotate the vectors in the opposite direction.

        Returns
        -------
            An array representing the rotated vectors.
        """
        ...

    def inv(self) -> Attitude:
        """Compute the inverse (conjugate) of the attitude.

        Returns
        -------
        Attitude
            A new attitude instance representing the inverse rotation.
        """
        ...

    def kinematics(self, w_B: np.ndarray) -> Attitude:
        """Compute the time derivative of the attitude given angular velocity.

        **CAUTION**: This method returns the time derivative of the attitude,
        which is represented with the same data structure for consistency with
        ODE solving - but this return is not itself a valid rotation representation
        until integrated in time. Hence, the output of ``kinematics`` should never
        be converted to a different attitude representation or rotation matrix.

        Parameters
        ----------
        w_B : np.ndarray
            Angular velocity vector expressed in the body frame B.

        Returns
        -------
        Attitude
            The time derivative of the attitude representation.
        """
        ...


def _rotate(att: Attitude, v: np.ndarray, inverse: bool = False) -> np.ndarray:
    matrix = att.as_matrix()
    if inverse:
        matrix = matrix.T

    v = cast(np.ndarray, array(v))
    if v.ndim == 1:
        if v.shape != (3,):
            raise ValueError("For 1D input, `vectors` must have shape (3,)")
        result = matrix @ v

    else:
        if v.shape[1] != 3:
            raise ValueError("For 2D input, `vectors` must have shape (N, 3)")
        result = v @ matrix.T

    return cast(np.ndarray, result)


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
        _check_seq(seq)
        _check_angles(angles, seq)

        self.array = angles
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

    def __getitem__(self, index: int) -> float:
        return self.array[index]

    def __iter__(self):
        return iter(self.array)

    @classmethod
    def from_quat(cls, quat: Quaternion | np.ndarray, seq: str = "xyz") -> EulerAngles:
        """Create EulerAngles from a Quaternion.

        Parameters
        ----------
        quat : Quaternion or array_like 
            Quaternion representing the rotation.

        Returns
        -------
        EulerAngles
            New EulerAngles instance representing the same rotation.
        """
        if isinstance(quat, Quaternion):
            quat = quat.array
        angles = quaternion_to_euler(quat, seq=seq)
        return cls(angles=angles, seq=seq)

    def as_quat(self) -> Quaternion:
        """Return the corresponding Quaternion representation.

        Returns
        -------
        Quaternion
            The equivalent Quaternion representation of this rotation.
        """
        quat = euler_to_quaternion(self.array, seq=self.seq)
        return Quaternion(quat)

    @classmethod
    def from_euler(cls, euler: EulerAngles, seq: str = "xyz") -> EulerAngles:
        """Return an EulerAngles instance from another EulerAngles instance.

        Can be used to change the sequence of axes.
        """
        if seq == euler.seq:
            return cls(angles=euler.array, seq=seq)

        return euler.as_euler(seq)

    def as_euler(self, seq: str = "xyz") -> EulerAngles:
        """Return the Euler angles in a different sequence of axes.

        If the requested sequence is the same as the current sequence, returns self.
        """
        if seq == self.seq:
            return self

        return self.as_quat().as_euler(seq)
    
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


class Quaternion:
    """Quaternion representation of a rotation in 3 dimensions.

    This class is closely modeled after [scipy.spatial.transform.Rotation](
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
    with a few differences:

    - The quaternion is always represented in scalar-first format
      (i.e. [w, x, y, z]) instead of scalar-last ([x, y, z, w]).
    - This class is designed for symbolic computation, so some checks (e.g. for valid
      rotation matrices) are omitted, since these cannot be done symbolically.
    - The class currently does not support some representations like Rodrigues parameters
      or rotation vectors
    - The class does not support multiple rotations in a single object
    - This implementation supports kinematic calculations

    The following operations on quaternions are supported:

    - Application on vectors (rotations of vectors)
    - Quaternion Composition
    - Quaternion Inversion
    - Kinematic time derivative given angular velocity

    Parameters
    ----------
    quat : array_like, shape (4,)
        Quaternion representing the rotation in scalar-first format (w, x, y, z).

    Attributes
    ----------
    array : np.ndarray, shape (4,)
        Underlying numpy array representing the quaternion.

    Methods
    -------
    __len__
    from_quat
    from_matrix
    from_euler
    as_quat
    as_matrix
    as_euler
    identity
    rotate
    inv
    mul
    kinematics

    Examples
    --------
    >>> from archimedes.spatial import Quaternion
    >>> import numpy as np

    A `Quaternion` instance can be initialized in any of the above formats and
    converted to any of the others. The underlying object is independent of the
    representation used for initialization.

    Consider a counter-clockwise rotation of 90 degrees about the z-axis. This
    corresponds to the following quaternion (in scalar-first format):

    >>> q = Quaternion([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])

    The quaternion can be expressed in any of the other formats:

    >>> q.as_matrix()
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
    [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    >>> np.rad2deg(q.as_euler('zyx'))
    array([90.,  0.,  0.])

    The same quaternion can be initialized using a rotation matrix:

    >>> q = Quaternion.from_matrix([[0, -1, 0],
    ...                    [1, 0, 0],
    ...                    [0, 0, 1]])

    Representation in other formats:

    >>> np.rad2deg(q.as_euler('zyx'))
    array([90.,  0.,  0.])

    The ``from_euler`` method is flexible in the range of input formats
    it supports. Here we initialize a quaternion about a single axis:

    >>> q = Quaternion.from_euler(np.deg2rad(90), 'z')

    The ``rotate`` method can be used to rotate vectors:

    >>> q.rotate([1, 0, 0])
    array([2.22045e-16, 1, 0])

    If the quaternion represents the attitude of a body B relative to a frame A,
    then this method transforms a vector v_A expressed in frame A to the same
    vector expressed in frame B, v_B = R * v_A. If `inverse` is True, the inverse
    rotation is applied, transforming v_B to v_A.

    The ``kinematics`` method can be used to compute the time derivative of the
    quaternion as an attitude representation given the angular velocity in the
    body frame using quaternion kinematics:

    >>> w_B = np.array([0, 0, np.pi/2])  # 90 deg/s about z-axis
    >>> q.kinematics(w_B)
    array([-0.55536037,  0.        ,  0.        ,  0.55536037])

    See Also
    --------
    scipy.spatial.transform.Rotation : Similar class in SciPy
    RigidBody : Rigid body dynamics supporting ``Quaternion`` attitude representation
    euler_to_dcm : Directly calculate rotation matrix from roll-pitch-yaw angles
    euler_kinematics : Transform roll-pitch-yaw rates to body-frame angular velocity
    quaternion_kinematics : Low-level quaternion kinematics function
    """

    def __init__(self, quat: np.ndarray):
        quat = np.hstack(quat)  # type: ignore
        if quat.shape not in [(4,), (1, 4), (4, 1)]:
            raise ValueError("Quaternion must have shape (4,), (1, 4), or (4, 1)")
        quat = quat.flatten()
        self.array = quat

    # === Methods for implementing Attitude protocol ===

    def as_matrix(self) -> np.ndarray:
        """Return the quaternion as a rotation matrix.

        Returns
        -------
        np.ndarray
            The rotation matrix as a 3x3 numpy array.
        """
        return quaternion_to_dcm(self.array)

    def rotate(self, vectors: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Rotate one or more vectors

        If the quaternion represents the attitude of a body B relative to a frame A,
        then this method transforms a vector v_A expressed in frame A to the same
        vector expressed in frame B, v_B = R * v_A. If `inverse` is True, the inverse
        rotation is applied, transforming v_B to v_A.

        This method is computationally and mathematically equivalent to:

        ... code-block:: python
            R_AB = q.as_matrix()
            v_B = R_AB @ v_A

        or, for the inverse:

        ... code-block:: python
            v_A = R_AB.T @ v_B

        The method supports both single vectors of shape (3,) and multiple vectors
        of shape (N, 3).
        """
        return _rotate(self, vectors, inverse=inverse)

    def inv(self) -> Quaternion:
        """Return the inverse of the quaternion"""
        q_inv = quaternion_inverse(self.array)   
        return Quaternion(q_inv)

    def kinematics(self, w: np.ndarray, baumgarte: float | None = None) -> Quaternion:
        """Return the time derivative of the quaternion given angular velocity w.

        If the quaternion represents the attitude of a body B, then w_B should be
        the body relative angular velocity ω_B.

        The derivative is computed using quaternion kinematics:
            dq/dt = 0.5 * q ⊗ [0, ω]
        where ⊗ is the quaternion multiplication operator.

        The method optionally support Baumgarte stabilization to preserve
        unit normalization.  For a stabilization factor λ, the full
        time derivative is:
            dq/dt = 0.5 * q ⊗ [0, ω] - λ * (||q||² - 1) * q

        **CAUTION**: This method returns the time derivative of the quaternion,
        which is represented with the same data structure for consistency with
        ODE solving - but this return is not itself a valid rotation representation
        until integrated in time - in particular, it is not unit norm.  Hence,
        methods such as ``as_euler`` should never be used on the time derivative, 
        since they will not produce meaningful results.

        Parameters
        ----------
        w : array_like, shape (3,)
            Angular velocity vector in the body frame.
        baumgarte : float, optional
            Baumgarte stabilization factor. If > 0, Baumgarte stabilization is
            applied to enforce unit norm constraint. Default is 0 (no stabilization).

        Returns
        -------
        Quaternion
            The time derivative represented as a Quaternion instance.
        """
        q_dot = quaternion_kinematics(self.array, w, baumgarte=baumgarte)
        return Quaternion(q_dot)

    # === Other methods ===

    def __repr__(self):
        return f"Quaternion({self.array})"

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index: int) -> float:
        return self.array[index]

    def __iter__(self):
        return iter(self.array)

    @classmethod
    def from_quat(cls, quat: Quaternion) -> Quaternion:
        """Returns a copy of the Quaternion object - dummy method for API consistency.

        Returns
        -------
        Quaternion
            A copy of the input Quaternion instance.
        """
        return cls(quat=quat.array)

    def as_quat(self) -> Quaternion:
        """Return the same object - dummy method for API consistency.

        Returns
        -------
        Quaternion
            The same Quaternion instance.
        """
        return self

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Quaternion:
        """Create a Quaternion from a rotation matrix.

        Note that for the sake of symbolic computation, this method assumes that
        the input is a valid rotation matrix (orthogonal and determinant +1).

        Parameters
        ----------
        matrix : array_like, shape (3, 3)
            Quaternion matrix.

        Returns
        -------
        Quaternion
            A new Quaternion instance.

        See Also
        --------
        dcm_to_quaternion : Low-level direction cosine matrix to quaternion conversion
        """
        quat = dcm_to_quaternion(matrix)
        return cls(quat=quat)

    @classmethod
    def from_euler(
        cls, euler: EulerAngles | np.ndarray, seq: str | None = None
    ) -> Quaternion:
        """Create a Quaternion from Euler angles.

        Parameters
        ----------
        euler : EulerAngles or array_like
            Euler angles instance or array of Euler angles in radians.
        seq : str, optional
            Sequence of axes for Euler angles (up to length 3).  Each character must be one
            of 'x', 'y', 'z' (extrinsic) or 'X', 'Y', 'Z' (intrinsic).  Default is 'xyz'.
            Should not be specified if `euler` is an EulerAngles instance.

        Returns
        -------
        Quaternion
            A new Quaternion instance.

        See Also
        --------
        euler_to_quaternion : Low-level Euler to quaternion conversion function
        """

        if isinstance(euler, EulerAngles):
            if seq is not None:
                raise ValueError(
                    "If `euler` is an EulerAngles instance, `seq` should not be passed"
                )
        else:
            if seq is None:
                seq = "xyz"
            euler = EulerAngles(angles=euler, seq=seq)

        return euler.as_quat()

    def as_euler(self, seq: str) -> EulerAngles:
        """Return the Euler angles from the quaternion

        This method uses the same notation and conventions as the SciPy Rotation class.
        See the SciPy documentation and :py:meth:``from_euler`` for more details.

        See Also
        --------
        quaternion_to_euler : Low-level quaternion to Euler conversion function
        """
        return EulerAngles.from_quat(self, seq=seq)

    @classmethod
    def identity(cls) -> Quaternion:
        """Return a quaternion representing the identity rotation."""
        return cls(np.array([1.0, 0.0, 0.0, 0.0]))

    def mul(self, other: Quaternion, normalize: bool = False) -> Quaternion:
        """Compose (multiply) this quaternion with another"""
        q1 = self.array
        q2 = other.array
        q = quaternion_multiply(q1, q2)
        if normalize:
            q = q / np.linalg.norm(q)
        return Quaternion(q)

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Compose (multiply) this quaternion with another and normalize the result"""
        return self.mul(other, normalize=True)


# === Struct registration ===

def euler_to_iter(euler: EulerAngles):
    children = (euler.array,)
    aux_data = (euler.seq)
    return children, aux_data


def euler_from_iter(aux_data, children) -> EulerAngles:
    seq, = aux_data
    angles, = children
    return EulerAngles(angles=angles, seq=seq)


tree.register_struct(EulerAngles, euler_to_iter, euler_from_iter)


def quaternion_to_iter(quat: Quaternion):
    children = (quat.array,)
    aux_data = None  # No static metadata
    return children, aux_data


def quaternion_from_iter(aux_data, children) -> Quaternion:
    quat, = children
    return Quaternion(quat=quat)


tree.register_struct(Quaternion, quaternion_to_iter, quaternion_from_iter)

