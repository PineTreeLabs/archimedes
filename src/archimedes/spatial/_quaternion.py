"""Low-level functions for quaternion operations.

These functions are for conversions and kinematics and operate directly on
arrays rather than higher-level wrapper classes.
"""

# ruff: noqa: N806, N803, N815
from __future__ import annotations

from typing import cast

import numpy as np

from archimedes import array

from ._euler import _check_angles, _check_seq

__all__ = [
    "euler_to_quaternion",
    "quaternion_inverse",
    "quaternion_kinematics",
    "quaternion_multiply",
    "quaternion_to_dcm",
    "quaternion_to_euler",
]


def quaternion_inverse(q):
    """
    Inverse of a quaternion q = [w, x, y, z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], like=q)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply (compose) two quaternions

    Parameters
    ----------
    q1 : array_like, shape (4,)
        First quaternion [w1, x1, y1, z1]
    q2 : array_like, shape (4,)
        Second quaternion [w2, x2, y2, z2]

    Returns
    -------
    np.ndarray, shape (4,)
        Resulting quaternion from multiplication q = q1 * q2

    Notes
    -----
    This function uses the scalar-first convention for quaternions, i.e. a quaternion
    is represented as [w, x, y, z], where w is the scalar part.
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
            q = quaternion_multiply(q, qi)
        else:
            q = quaternion_multiply(qi, q)

    return q


def dcm_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Create a Quaternion from a rotation matrix.

    If the rotation matrix is R_BA that represents the attitude of a body B
    relative to a frame A, then the resulting quaternion represents the attitude
    of the body B relative to frame A.

    Note that for the sake of symbolic computation, this method assumes that
    the input is a valid rotation matrix (orthogonal and determinant +1).

    Implementation based on SciPy and reference [1]_.

    Parameters
    ----------
    matrix : array_like, shape (3, 3)
        Quaternion matrix.

    Returns
    -------
    np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z] corresponding to the rotation matrix,
        where w is the scalar part.

    References
    ----------
    .. [1] F. Landis Markley, "Unit Quaternion from Rotation Matrix",
            Journal of guidance, control, and dynamics vol. 31.2, pp.
            440-442, 2008.
    """
    matrix = cast(np.ndarray, array(matrix))
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")

    # Transpose of SciPy convention  (we assume passive rotations for coordinate
    # system transformations rather than active rotations of vectors)
    matrix = matrix.T

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
    quat = quat / np.linalg.norm(quat)

    quat = np.roll(quat, 1)  # Convert to scalar-first format
    return quat


def euler_to_quaternion(angles: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """Convert Euler angles in radians to unit quaternion.

    This method uses the same notation and conventions as the SciPy Rotation class.
    See the SciPy documentation for more details.  Some common examples:

    - 'xyz': Extrinsic rotations about x, then y, then z axes (classical roll,
        pitch, yaw sequence)
    - 'ZXZ': Rotation from perifocal (Ω, i, ω) frame (right ascension of ascending
        node, inclination, argument of perigee) used by Kepler orbital elements to
        ECI (Earth-Centered Inertial) frame

    Parameters
    ----------
    angles : array_like, shape (N,) or (1, N) or (N, 1)
        Euler angles in radians. The number of angles must match the length of `seq`.
    seq : str
        Specifies sequence of axes for rotations. Up to 3 characters belonging to
        the set {'x', 'y', 'z'} or {'X', 'Y', 'Z'}. Lowercase characters
        correspond to extrinsic rotations about the fixed axes, while uppercase
        characters correspond to intrinsic rotations about the rotating axes.
        Examples include 'xyz', 'ZYX', 'xzx', etc.

    Returns
    -------
    np.ndarray, shape (4,)
        Unit quaternion [q0, q1, q2, q3] corresponding to the Euler angles,
        where q0 is the scalar part.

    Raises
    ------
    ValueError
        If `seq` is not a valid sequence of axes, or if the shape of `angles` does
        not match the length of `seq`.
    """
    intrinsic = _check_seq(seq)
    angles = _check_angles(angles, seq)

    seq = seq.lower()
    angles = angles.flatten()

    quat = _elementary_quat_compose(seq, angles, intrinsic=intrinsic)
    return cast(np.ndarray, quat / np.linalg.norm(quat))


def quaternion_to_dcm(quat: np.ndarray) -> np.ndarray:
    """Direction cosine matrix from unit quaternion

    If the quaternion represents the attitude of a body B relative to a frame A,
    then this function returns the matrix R_BA that transforms vectors from
    frame A to frame B.  Specifically, for a vector v_A expressed in frame A,
    the corresponding vector in frame B is given by ``v_B = R_BA @ v_A``.

    The inverse transformation can be obtained by transposing this matrix:
    ``R_AB = R_BA.T``.

    Note that this function assumes the scalar-first convention, and assumes the
    quaternion is normalized.

    Parameters
    ----------
    quat : array_like, shape (4,)
        Unit quaternion representing rotation from frame A to frame B,
        in the format [w, x, y, z] where w is the scalar part.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Direction cosine matrix R_BA that transforms vectors from frame A to frame B.
    """
    w, x, y, z = quat
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
        like=quat,
    ).T


# See: https://github.com/scipy/scipy/blob/3ead2b543df7c7c78619e20f0cb6139e344a8866/scipy/spatial/transform/_rotation_cy.pyx#L774-L851  # ruff: noqa: E501
def quaternion_to_euler(q: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """Convert unit quaternion to roll-pitch-yaw Euler angles.

    This method uses the same notation and conventions as the SciPy Rotation class.
    See the SciPy documentation and :py:func:`euler_to_quaternion` for more details.

    The algorithm is based on the method described in [1]_ as implemented in SciPy.

    Parameters
    ----------
    q : array_like, shape (4,)
        Unit quaternion representing rotation, in the format [q0, q1, q2, q3]
        where q0 is the scalar part.
    seq : str, optional
        Sequence of axes for Euler angles (default is 'xyz').

    Returns
    -------
    np.ndarray, shape (3,)
        Euler angle sequence corresponding to the quaternion.

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

    return angles


def quaternion_kinematics(
    q: np.ndarray, w: np.ndarray, baumgarte: float | None = None
) -> np.ndarray:
    """Quaternion kinematical equations

    If the rotation represents the attitude of a body B relative to a
    frame A, then w should be the body relative angular velocity, i.e. ω_B.

    The derivative is computed using quaternion kinematics:
        dq/dt = 0.5 * q ⊗ [0, ω]
    where ⊗ is the quaternion multiplication operator.

    The method optionally support Baumgarte stabilization to preserve
    unit normalization.  For a stabilization factor λ, the full
    time derivative is:
        dq/dt = 0.5 * q ⊗ [0, ω] - λ * (||q||² - 1) * q

    Parameters
    ----------
    q : array_like, shape (4,)
        Unit quaternion representing rotation, in the format [q0, q1, q2, q3]
        where q0 is the scalar part.
    w : array_like, shape (3,)
        Angular velocity vector in body frame.
    baumgarte : float, optional
        Baumgarte stabilization factor. If not None, Baumgarte stabilization is
        applied to enforce unit norm constraint. Default is None (no stabilization).

    Returns
    -------
    np.ndarray, shape (4,)
        Time derivative of the quaternion.
    """
    omega = np.array([0, *w], like=q)
    q_dot = 0.5 * quaternion_multiply(q, omega)

    # Baumgarte stabilization to enforce unit norm constraint
    if baumgarte is not None:
        q_dot -= baumgarte * (np.dot(q, q) - 1) * q

    return q_dot
