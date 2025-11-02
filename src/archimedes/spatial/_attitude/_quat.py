"""Low-level utilities for rotation representations.

These functions are for conversions and kinematics and operate directly on
arrays rather than higher-level wrapper classes.
"""
# ruff: noqa: N806, N803, N815
from __future__ import annotations

import numpy as np
from archimedes import array

__all__ = [
    "quat_to_dcm",
    "quat_to_euler",
    "quat_kinematics",
]


def quat_to_dcm(quat: np.ndarray) -> np.ndarray:
    """Direction cosine matrix from unit quaternion

    If the quaternion represents the attitude of a body B relative to a frame A,
    then this function returns the matrix R_AB that transforms vectors from
    frame B to frame A.  Specifically, for a vector v_B expressed in frame B,
    the corresponding vector in frame A is given by ``v_A = R_AB @ v_B``.

    The inverse transformation can be obtained by transposing this matrix:
    ``R_BA = R_AB.T``.

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
    q0, q1, q2, q3 = quat

    R = array(
        [
            [
                1 - 2 * (q2**2 + q3**2),
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * (q1**2 + q3**2),
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                1 - 2 * (q1**2 + q2**2),
            ],
        ]
    )
    return R


def quat_to_euler(quat: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """Convert unit quaternion to roll-pitch-yaw Euler angles.

    Parameters
    ----------
    quat : array_like, shape (4,)
        Unit quaternion representing rotation, in the format [q0, q1, q2, q3]
        where q0 is the scalar part.
    seq : str, optional
        Sequence of axes for Euler angles (default is 'xyz').

    Returns
    -------
    np.ndarray, shape (3,)
        Roll-pitch-yaw Euler angles corresponding to the quaternion.
    """
    raise NotImplementedError("quat_to_euler is not yet implemented.")


def quat_kinematics(quat: np.ndarray, omega: np.ndarray) -> np.ndarray:
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