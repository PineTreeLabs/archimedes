"""Low-level functions for quaternion operations.

These functions are for conversions and kinematics and operate directly on
arrays rather than higher-level wrapper classes.
"""
# ruff: noqa: N806, N803, N815
from __future__ import annotations

import numpy as np

from ._euler import _check_seq

__all__ = [
    "quaternion_kinematics",
    "quaternion_multiply",
    "quaternion_to_dcm",
    "quaternion_to_euler",
]


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
        Euler angles in radians (or degrees if `degrees` is True). The number of
        angles must match the length of `seq`.
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

    quat = _elementary_quat_compose(seq, angles, intrinsic=intrinsic)
    return quat / np.linalg.norm(quat)


def quaternion_to_dcm(quat: np.ndarray) -> np.ndarray:
    """Direction cosine matrix from unit quaternion

    If the quaternion represents the attitude of a body B relative to a frame A,
    then this function returns the matrix R_AB that transforms vectors from
    frame B to frame A.  Specifically, for a vector v_B expressed in frame B,
    the corresponding vector in frame A is given by ``v_A = R_AB @ v_B``.

    The inverse transformation can be obtained by transposing this matrix:
    ``R_BA = R_AB.T``.

    Note that this function assumes the scalar-first convention, and assumes the
    quaternion is normalized.

    Parameters
    ----------
    quat : array_like, shape (4,)
        Unit quaternion representing rotation from frame A to frame B,
        in the format [q0, q1, q2, q3] where q0 is the scalar part.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Direction cosine matrix R_AB that transforms vectors from frame B to frame A.
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
    )


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


def quaternion_kinematics(quat: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion kinematical equations

    Parameters
    ----------
    quat : array_like, shape (4,)
        Unit quaternion representing rotation, in the format [q0, q1, q2, q3]
        where q0 is the scalar part.
    omega : array_like, shape (3,)
        Angular velocity vector [P, Q, R] in body frame.

    Returns
    -------
    np.ndarray, shape (4,)
        Time derivative of the quaternion.
    """
    raise NotImplementedError("quat_kinematics is not yet implemented.")