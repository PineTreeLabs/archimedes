# ruff: noqa: N806, N803
from __future__ import annotations

import re
from typing import cast

import numpy as np

from .. import array, field, struct

__all__ = ["Rotation"]


def _normalize(q):
    return q / np.linalg.norm(q)


def _compose_quat(q1, q2):
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


def _check_seq(seq: str) -> bool:
    # The following checks are verbatim from:
    # https://github.com/scipy/scipy/blob/3ead2b543df7c7c78619e20f0cb6139e344a8866/scipy/spatial/transform/_rotation_cy.pyx#L461-L476  # ruff: noqa: E501
    intrinsic = re.match(r"^[XYZ]{1,3}$", seq) is not None
    extrinsic = re.match(r"^[xyz]{1,3}$", seq) is not None
    if not (intrinsic or extrinsic):
        raise ValueError(
            "Expected axes from `seq` to be from ['x', 'y', "
            "'z'] or ['X', 'Y', 'Z'], got {}".format(seq)
        )

    if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
        raise ValueError(
            "Expected consecutive axes to be different, got {}".format(seq)
        )

    return intrinsic


def _elementary_basis_index(axis: str) -> int:
    return {"x": 1, "y": 2, "z": 3}[axis.lower()]


# See https://github.com/scipy/scipy/blob/3ead2b543df7c7c78619e20f0cb6139e344a8866/scipy/spatial/transform/_rotation_cy.pyx#L358-L372  # ruff: noqa: E501
def _make_elementary_quat(axis: str, angle: float) -> np.ndarray:
    """Create a quaternion representing a rotation about a principal axis."""

    quat = np.hstack([np.cos(angle / 2), np.zeros(3)])
    axis_idx = _elementary_basis_index(axis)
    quat[axis_idx] = np.sin(angle / 2)

    return quat


# See https://github.com/scipy/scipy/blob/3ead2b543df7c7c78619e20f0cb6139e344a8866/scipy/spatial/transform/_rotation_cy.pyx#L376-L391  # ruff: noqa: E501
def _elementary_quat_compose(
    seq: str, angles: np.ndarray, intrinsic: bool
) -> np.ndarray:
    """Create a quaternion from a sequence of elementary rotations."""
    q = _make_elementary_quat(seq[0], angles[0])

    for idx in range(1, len(seq)):
        qi = _make_elementary_quat(seq[idx], angles[idx])
        if intrinsic:
            q = _compose_quat(q, qi)
        else:
            q = _compose_quat(qi, q)

    return q


@struct
class Rotation:
    """Rotation in 3 dimensions.

    This class provides an interface to initialize from and represent rotations
    with:

    - Quaternions
    - Rotation Matrices
    - Euler Angles

    Internally, the rotation is always represented as a unit quaternion, which is a
    minimal and singularity-free representation.

    This class is closely modeled after [scipy.spatial.transform.Rotation](
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
    with a few differences:

    - By default the quaternion is represented in scalar-first format
      (i.e. [w, x, y, z]) instead of scalar-last ([x, y, z, w]).
    - This class is designed for symbolic computation, so some checks (e.g. for valid
      rotation matrices) are omitted, since these cannot be done symbolically.
    - The class currently does not support some representations like Rodrigues parameters
      or rotation vectors
    - The class does not support multiple rotations in a single object
    - This implementation supports kinematic calculations

    The following operations on rotations are supported:

    - Application on vectors (rotations of vectors)
    - Rotation Composition
    - Rotation Inversion
    - Kinematic time derivative given angular velocity

    Parameters
    ----------
    quat : array_like, shape (4,)
        Quaternion representing the rotation. By default, this is in scalar-first
        format (i.e. [w, x, y, z]). See `scalar_first` parameter.  Typically, the
        class should not be constructed directly, but instead initialized with one of
        the class methods like `from_quat`, `from_matrix`, or `from_euler`.
    scalar_first : bool, optional
        If True, the quaternion is in scalar-first format (i.e. [w, x, y, z]).
        If False, the quaternion is in scalar-last format (i.e. [x, y, z, w]).
        Default is True.

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
    apply
    inv
    mul
    derivative

    Examples
    --------
    >>> from archimedes.spatial import Rotation
    >>> import numpy as np

    A `Rotation` instance can be initialized in any of the above formats and
    converted to any of the others. The underlying object is independent of the
    representation used for initialization.

    Consider a counter-clockwise rotation of 90 degrees about the z-axis. This
    corresponds to the following quaternion (in scalar-first format):

    >>> R = Rotation.from_quat([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])

    The rotation can be expressed in any of the other formats:

    >>> R.as_matrix()
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
    [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    >>> R.as_euler('zyx', degrees=True)
    array([90.,  0.,  0.])

    The same rotation can be initialized using a rotation matrix:

    >>> R = Rotation.from_matrix([[0, -1, 0],
    ...                    [1, 0, 0],
    ...                    [0, 0, 1]])

    Representation in other formats:

    >>> R.as_quat()
    array([0.70710678, 0.        , 0.        , 0.70710678])
    >>> R.as_euler('zyx', degrees=True)
    array([90.,  0.,  0.])

    The ``from_euler`` method is quite flexible in the range of input formats
    it supports. Here we initialize a single rotation about a single axis:

    >>> R = Rotation.from_euler('z', 90, degrees=True)

    Again, the object is representation independent and can be converted to any
    other format:

    >>> R.as_quat()
    array([0.70710678, 0.        , 0.        , 0.70710678])
    >>> R.as_matrix()
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    The ``apply`` method can be used to rotate vectors:

    >>> R.apply([1, 0, 0])
    array([2.22045e-16, 1, 0])

    The ``derivative`` method can be used to compute the time derivative of the
    rotation as an attitude representation given the angular velocity in the
    body frame using quaternion kinematics:

    >>> w_B = np.array([0, 0, np.pi/2])  # 90 deg/s about z-axis
    >>> R.derivative(w_B)
    array([-0.55536037,  0.        ,  0.        ,  0.55536037])

    See Also
    --------
    scipy.spatial.transform.Rotation : Similar class in SciPy
    RigidBody : Rigid body dynamics using this Rotation class
    dcm_from_euler : Directly calculate rotation matrix from roll-pitch-yaw angles
    euler_kinematics : Transform roll-pitch-yaw rates to body-frame angular velocity

    """

    quat: np.ndarray
    scalar_first: bool = field(default=True, static=True)  # type: ignore

    def __len__(self):
        return len(self.quat)

    @classmethod
    def from_quat(
        cls, quat: np.ndarray, scalar_first: bool = True, normalize: bool = True
    ) -> Rotation:
        """Create a Rotation from a quaternion.

        Parameters
        ----------
        quat : array_like, shape (4,)
            Quaternion in scalar-first (w, x, y, z) or scalar-last (x, y, z, w) format.
        scalar_first : bool, optional
            If True, the quaternion is assumed to be in scalar-first format.
            If False, scalar-last format is assumed. Default is True.
        normalize : bool, optional
            If True, the quaternion will be normalized to unit length. Default is True.

        Returns
        -------
        Rotation
            A new Rotation instance.
        """
        quat = np.hstack(quat)  # type: ignore
        if quat.shape not in [(4,), (1, 4), (4, 1)]:
            raise ValueError("Quaternion must have shape (4,), (1, 4), or (4, 1)")
        quat = quat.flatten()
        if normalize:
            quat = _normalize(quat)
        return cls(quat=quat, scalar_first=scalar_first)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Rotation:
        """Create a Rotation from a rotation matrix.

        Note that for the sake of symbolic computation, this method assumes that
        the input is a valid rotation matrix (orthogonal and determinant +1).

        Parameters
        ----------
        matrix : array_like, shape (3, 3)
            Rotation matrix.

        Returns
        -------
        Rotation
            A new Rotation instance.

        References
        ----------
        .. [1] F. Landis Markley, "Unit Quaternion from Rotation Matrix",
               Journal of guidance, control, and dynamics vol. 31.2, pp.
               440-442, 2008.
        """
        matrix = cast(np.ndarray, array(matrix))
        if matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        t = np.linalg.trace(matrix)

        # If matrix[0, 0] is the largest diagonal element
        q0 = np.hstack(
            [
                1 - t + 2 * matrix[0, 0],
                matrix[0, 1] + matrix[1, 0],
                matrix[0, 2] + matrix[2, 0],
                matrix[2, 1] - matrix[1, 2],
            ]
        )

        # If matrix[1, 1] is the largest diagonal element
        q1 = np.hstack(
            [
                1 - t + 2 * matrix[1, 1],
                matrix[2, 1] + matrix[1, 2],
                matrix[0, 1] + matrix[1, 0],
                matrix[0, 2] - matrix[2, 0],
            ]
        )

        # If matrix[2, 2] is the largest diagonal element
        q2 = np.hstack(
            [
                1 - t + 2 * matrix[2, 2],
                matrix[0, 2] + matrix[2, 0],
                matrix[2, 1] + matrix[1, 2],
                matrix[1, 0] - matrix[0, 1],
            ]
        )

        # If t is the largest diagonal element
        q3 = np.hstack(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
                1 + t,
            ]
        )

        quat = q0
        max_val = matrix[0, 0]

        quat = np.where(matrix[1, 1] >= max_val, q1, quat)
        max_val = np.where(matrix[1, 1] >= max_val, matrix[1, 1], max_val)

        quat = np.where(matrix[2, 2] >= max_val, q2, quat)
        max_val = np.where(matrix[2, 2] >= max_val, matrix[2, 2], max_val)

        quat = np.where(t >= max_val, q3, quat)
        quat = _normalize(quat)

        quat = np.roll(quat, 1)  # Convert to scalar-first format
        return cls(quat=quat, scalar_first=True)

    @classmethod
    def from_euler(
        cls, seq: str, angles: np.ndarray, degrees: bool = False
    ) -> Rotation:
        """Create a Rotation from Euler angles.

        This method uses the same notation and conventions as the SciPy Rotation class.
        See the SciPy documentation for more details.  Some common examples:

        - 'xyz': Extrinsic rotations about x, then y, then z axes (classical roll,
          pitch, yaw sequence)
        - 'ZXZ': Rotation from perifocal (Ω, i, ω) frame (right ascension of ascending
          node, inclination, argument of perigee) used by Kepler orbital elements to
          ECI (Earth-Centered Inertial) frame

        Parameters
        ----------
        seq : str
            Specifies sequence of axes for rotations. Up to 3 characters belonging to
            the set {'x', 'y', 'z'} or {'X', 'Y', 'Z'}. Lowercase characters
            correspond to extrinsic rotations about the fixed axes, while uppercase
            characters correspond to intrinsic rotations about the rotating axes.
            Examples include 'xyz', 'ZYX', 'xzx', etc.
        angles : array_like, shape (N,) or (1, N) or (N, 1)
            Euler angles in radians (or degrees if `degrees` is True). The number of
            angles must match the length of `seq`.
        degrees : bool, optional
            If True, the angles are assumed to be in degrees. Default is False (radians).

        Returns
        -------
        Rotation
            A new Rotation instance.

        Raises
        ------
        ValueError
            If `seq` is not a valid sequence of axes, or if the shape of `angles` does
            not match the length of `seq`.
        """
        num_axes = len(seq)
        if num_axes < 1 or num_axes > 3:
            raise ValueError(
                "Expected axis specification to be a non-empty "
                "string of upto 3 characters, got {}".format(seq)
            )

        intrinsic = _check_seq(seq)

        if isinstance(angles, (list, tuple)):
            angles = np.hstack(angles)

        angles = np.atleast_1d(angles)
        if angles.shape not in [(num_axes,), (1, num_axes), (num_axes, 1)]:
            raise ValueError(
                f"For {seq} sequence with {num_axes} axes, `angles` must have shape "
                f"({num_axes},), (1, {num_axes}), or ({num_axes}, 1). Got "
                f"{angles.shape}"
            )

        seq = seq.lower()
        angles = angles.flatten()

        if degrees:
            angles = np.deg2rad(angles)

        quat = _elementary_quat_compose(seq, angles, intrinsic=intrinsic)
        return cls(quat=quat, scalar_first=True)

    def as_quat(self, scalar_first: bool = True) -> np.ndarray:
        """Return the quaternion as a numpy array.

        Parameters
        ----------
        scalar_first : bool, optional
            If True, the quaternion is returned in scalar-first format (w, x, y, z).
            If False, scalar-last format (x, y, z, w) is returned. Default is True.

        Returns
        -------
        np.ndarray
            The quaternion as a numpy array.
        """
        if scalar_first:
            return self.quat
        return np.roll(self.quat, -1)

    def as_matrix(self) -> np.ndarray:
        """Return the rotation as a rotation matrix.

        Returns
        -------
        np.ndarray
            The rotation matrix as a 3x3 numpy array.
        """
        w, x, y, z = self.quat
        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w
        xy = x * y
        xz = x * z
        xw = x * w
        yz = y * z
        yw = y * w
        zw = z * w

        return np.array(
            [
                [w2 + x2 - y2 - z2, 2 * (xy - zw), 2 * (xz + yw)],
                [2 * (xy + zw), w2 - x2 + y2 - z2, 2 * (yz - xw)],
                [2 * (xz - yw), 2 * (yz + xw), w2 - x2 - y2 + z2],
            ],
            like=self.quat,
        )

    # See: https://github.com/scipy/scipy/blob/3ead2b543df7c7c78619e20f0cb6139e344a8866/scipy/spatial/transform/_rotation_cy.pyx#L774-L851  # ruff: noqa: E501
    def as_euler(self, seq: str, degrees: bool = False) -> np.ndarray:
        """Return the Euler angles from the rotation

        This method uses the same notation and conventions as the SciPy Rotation class.
        See the SciPy documentation and ``from_euler`` for more details.

        References
        ----------
        .. [1] Bernardes E, Viollet S (2022) Quaternion to Euler angles
               conversion: A direct, general and computationally efficient
               method. PLoS ONE 17(11): e0276302.
               https://doi.org/10.1371/journal.pone.0276302
        """
        if len(seq) != 3:
            raise ValueError("Expected `seq` to be a string of 3 characters")

        intrinsic = _check_seq(seq)
        seq = seq.lower()

        if intrinsic:
            seq = seq[::-1]

        # Note: the sequence is "static" from a symbolic computation point of view,
        # meaning that the indices are known at "compile-time" and all logic on indices
        # will be evaluated in standard Python.
        i, j, k = (_elementary_basis_index(axis) for axis in seq)

        symmetric = i == k
        if symmetric:
            k = 6 - i - j

        # 0. Check if permutation is odd or even
        sign = (i - j) * (j - k) * (k - i) // 2

        # 1. Permute quaternion components
        q = self.as_quat(scalar_first=True)
        if symmetric:
            a, b, c, d = (q[0], q[i], q[j], q[k] * sign)
        else:
            a, b, c, d = (
                q[0] - q[j],
                q[i] + q[k] * sign,
                q[j] + q[0],
                q[k] * sign - q[i],
            )

        # 2. Compute second angle
        angles = np.zeros(3, like=q)
        angles[1] = 2 * np.arctan2(np.hypot(c, d), np.hypot(a, b))

        # 3. Compute first and third angles
        half_sum = np.arctan2(b, a)
        half_diff = np.arctan2(d, c)

        angles[0] = half_sum - half_diff
        angles[2] = half_sum + half_diff

        # Handle singularities
        s_zero = abs(angles[1]) <= 1e-7
        s_pi = abs(angles[1] - np.pi) <= 1e-7

        angles[0] = np.where(s_zero, 2 * half_sum, angles[0])
        angles[2] = np.where(s_zero, 0.0, angles[2])

        angles[0] = np.where(s_pi, -2 * half_diff, angles[0])
        angles[2] = np.where(s_pi, 0.0, angles[2])

        # Tait-Bryan/asymmetric sequences
        if not symmetric:
            angles[2] *= sign
            angles[1] -= np.pi / 2

        if intrinsic:
            angles = angles[::-1]

        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        if degrees:
            angles = np.rad2deg(angles)

        return angles

    @classmethod
    def identity(cls) -> Rotation:
        """Return the identity rotation"""
        return cls.from_quat(np.array([1.0, 0.0, 0.0, 0.0]), scalar_first=True)

    def apply(self, vectors: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Apply the rotation to a set of vectors

        If the rotation represents the attitude of a body B relative to a frame A,
        then this method transforms a vector v_A expressed in frame A to the same
        vector expressed in frame B, v_B = R * v_A. If `inverse` is True, the inverse
        rotation is applied, transforming v_B to v_A.
        """

        matrix = self.as_matrix()
        if inverse:
            matrix = matrix.T

        vectors = cast(np.ndarray, array(vectors))
        if vectors.ndim == 1:
            if vectors.shape != (3,):
                raise ValueError("For 1D input, `vectors` must have shape (3,)")
            result = matrix @ vectors

        else:
            if vectors.shape[1] != 3:
                raise ValueError("For 2D input, `vectors` must have shape (N, 3)")
            result = vectors @ matrix.T

        return cast(np.ndarray, result)

    def inv(self) -> Rotation:
        """Return the inverse rotation"""
        q = self.as_quat(scalar_first=True)
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]], like=q)
        return Rotation.from_quat(q_inv, scalar_first=True)

    def mul(self, other: Rotation, normalize: bool = False) -> Rotation:
        """Compose this rotation with another rotation"""
        q1 = self.as_quat(scalar_first=True)
        q2 = other.as_quat(scalar_first=True)
        q = _compose_quat(q1, q2)
        return Rotation.from_quat(q, scalar_first=True, normalize=normalize)

    def __mul__(self, other: Rotation) -> Rotation:
        """Compose this rotation with another rotation"""
        return self.mul(other, normalize=True)

    def derivative(self, w: np.ndarray, baumgarte: float = 0.0) -> Rotation:
        """Return the time derivative of the rotation given angular velocity w.

        Note that if the rotation represents the attitude of a body B relative to a
        frame A, then w should be the body relative angular velocity, i.e. ω_B.

        The derivative is computed using quaternion kinematics:
            dq/dt = 0.5 * q ⊗ [0, ω]
        where ⊗ is the quaternion multiplication operator.

        The method optionally support Baumgarte stabilization to preserve
        unit normalization.  For a stabilization factor λ, the full
        time derivative is:
            dq/dt = 0.5 * q ⊗ [0, ω] - λ * (||q||² - 1) * q

        **CAUTION**: This method returns the time derivative of the rotation,
        which is represented with the same data structure for consistency with
        ODE solving - but this return is not itself a valid rotation until
        integrated in time.  Hence, methods such as ``as_euler`` should never
        be used on the time derivative, since they will not produce meaningful results.

        Parameters
        ----------
        w : array_like, shape (3,)
            Angular velocity vector in the body frame.
        baumgarte : float, optional
            Baumgarte stabilization factor. If > 0, Baumgarte stabilization is
            applied to enforce unit norm constraint. Default is 0 (no stabilization).

        Returns
        -------
        Rotation
            The time derivative of the rotation represented as a Rotation instance.
        """
        q = self.as_quat(scalar_first=True)
        omega = np.array([0, *w], like=q)
        q_dot = 0.5 * _compose_quat(q, omega)

        # Baumgarte stabilization to enforce unit norm constraint
        if baumgarte > 0:
            q_dot -= baumgarte * (np.dot(q, q) - 1) * q

        return Rotation.from_quat(q_dot, scalar_first=True, normalize=False)
