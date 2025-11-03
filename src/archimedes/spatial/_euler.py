"""Low-level functions for Euler angles.

These functions are for conversions and kinematics and operate directly on
arrays rather than higher-level wrapper classes.
"""
# ruff: noqa: N806, N803, N815
from __future__ import annotations

import re

import numpy as np
from archimedes import array

__all__ = [
    "euler_kinematics",
    "euler_to_dcm",
]

def _check_seq(seq: str) -> bool:
    num_axes = len(seq)
    if num_axes < 1 or num_axes > 3:
        raise ValueError(
            "Expected axis specification to be a non-empty "
            "string of upto 3 characters, got {}".format(seq)
        )
    
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


def _check_angles(angles: np.ndarray, seq: str) -> np.ndarray:
    num_axes = len(seq)

    if isinstance(angles, (list, tuple)):
        angles = np.hstack(angles)

    angles = np.atleast_1d(angles)
    if angles.shape not in [(num_axes,), (1, num_axes), (num_axes, 1)]:
        raise ValueError(
            f"For {seq} sequence with {num_axes} axes, `angles` must have shape "
            f"({num_axes},), (1, {num_axes}), or ({num_axes}, 1). Got "
            f"{angles.shape}"
        )
    return angles


def _rot_x(angle: float) -> np.ndarray:
    """Rotation about x-axis by given angle (radians)."""
    c = np.cos(angle)
    s = np.sin(angle)

    R = array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )
    return R


def _rot_y(angle: float) -> np.ndarray:
    """Rotation about y-axis by given angle (radians)."""
    c = np.cos(angle)
    s = np.sin(angle)

    R = array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )
    return R


def _rot_z(angle: float) -> np.ndarray:
    """Rotation about z-axis by given angle (radians)."""
    c = np.cos(angle)
    s = np.sin(angle)

    R = array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return R


def euler_to_dcm(angles: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """Direction cosine matrix from Euler angles

    If the Euler angles represent the attitude of a body B relative to a frame A,
    then this function returns the matrix R_AB that transforms vectors from
    frame B to frame A.  Specifically, for a vector v_B expressed in frame B,
    the corresponding vector in frame A is given by ``v_A = R_AB @ v_B``.
    
    The inverse transformation can be obtained by transposing this matrix:
    ``R_BA = R_AB.T``.

    By default, the Euler angle sequence is assumed to follow the standard aerospace
    convention of an extrinsic roll-pitch-yaw sequence ("xyz").  However, it supports
    arbitrary sequences of non-repeating axes up to length 3. Both intrinsic
    (uppercase letters) and extrinsic (lowercase letters) sequences are supported.
    
    Specifically, the following will generate equivalent DCMs:

    .. code-block:: python

        R_AB = euler_to_dcm(rpy)
        R_AB = Quaternion.from_euler(rpy, 'xyz').as_matrix()

    In general, the ``Quaternion`` class should be preferred over Euler representations,
    although Euler angles are used in some special cases (e.g. stability analysis).
    In these cases, this function gives a more direct calculation of the
    transformation matrix without converting to the intermediate quaternion.

    Parameters
    ----------
    angles : array_like
        Euler angles in radians. Shape must match the length of ``seq``.
    seq : str, optional
        Sequence of axes for Euler angles (up to length 3).  Each character must be one
        of 'x', 'y', 'z' (extrinsic) or 'X', 'Y', 'Z' (intrinsic).  Default is 'xyz'.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Direction cosine matrix R_AB that transforms vectors from frame B to frame A.
    """

    # Validate angle sequence
    intrinsic = _check_seq(seq)
    angles = _check_angles(angles, seq)

    seq = seq.lower()
    if intrinsic:
        seq = seq[::-1]  # Reverse for intrinsic rotations
        angles = angles[::-1]

    # Note that this approach of building the DCM by composing
    # elemental rotations is usually slower than the direct formula,
    # since it involves multiple matrix multiplications. However, with
    # symbolic arrays there is no difference in speed because the
    # multiplications are not actually carried out until after the
    # full matrix is built.
    R = np.eye(3, like=angles)
    for (char, angle) in zip(seq, angles):
        match char:
            case "x":
                R = _rot_x(angle) @ R
            case "y":
                R = _rot_y(angle) @ R
            case "z":
                R = _rot_z(angle) @ R

    return R


def euler_kinematics(rpy: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Euler kinematical equations

    Defining 𝚽 = [phi, theta, psi] == Euler angles for roll, pitch, yaw
    attitude representation, this function returns a matrix H(𝚽) such
    that
        d𝚽/dt = H(𝚽) * ω.

    If inverse=True, it returns a matrix H(𝚽)^-1 such that
        ω = H(𝚽)^-1 * d𝚽/dt.

    This function supports _only_ the extrinsic roll-pitch-yaw sequence ("xyz").

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
    ``Quaternion`` class) for attitude representation.
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
