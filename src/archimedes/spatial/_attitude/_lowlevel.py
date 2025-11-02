"""Low-level utilities for rotation representations.

These functions are for conversions and kinematics and operate directly on
arrays rather than higher-level wrapper classes.
"""
# ruff: noqa: N806, N803, N815
from __future__ import annotations

import re

import numpy as np
from archimedes import array


def _check_euler_seq(seq: str) -> bool:
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


def _rot_x(angle: float) -> np.ndarray:
    """Rotation about x-axis by given angle (radians)."""
    c = np.cos(angle)
    s = np.sin(angle)

    R = array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ]
    )
    return R


def _rot_y(angle: float) -> np.ndarray:
    """Rotation about y-axis by given angle (radians)."""
    c = np.cos(angle)
    s = np.sin(angle)

    R = array(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ]
    )
    return R


def _rot_z(angle: float) -> np.ndarray:
    """Rotation about z-axis by given angle (radians)."""
    c = np.cos(angle)
    s = np.sin(angle)

    R = array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return R


def euler_to_dcm(
    rpy: np.ndarray, seq: str = "xyz", transpose: bool = False
) -> np.ndarray:
    """Returns matrix to transform from inertial to body frame (R_BN).

    If transpose=True, returns matrix to transform from body to inertial frame (R_NB).

    This is the direction cosine matrix (DCM) corresponding to the given
    roll-pitch-yaw (rpy) angles.  This follows the standard aerospace
    convention and corresponds to the "xyz" sequence when using the
    :py:class:`Rotation` class.  However, by default this function returns the inverse
    of the rotation implemented by :py:meth:`Rotation.apply`.  Specifically, the
    following will generate equivalent DCMs:

    .. code-block:: python

        R_BN = euler_to_dcm(rpy)
        R_BN = Rotation.from_euler('xyz', rpy).inv().as_matrix()

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
    r, p, y = rpy[0], rpy[1], rpy[2]

    # Validate angle sequence
    intrinsic = _check_euler_seq(seq)
    seq = seq.lower()

    if intrinsic:
        seq = seq[::-1]  # Reverse for intrinsic rotations

    # Note that this approach of building the DCM by composing
    # elemental rotations is usually slower than the direct formula,
    # since it involves multiple matrix multiplications. However, with
    # symbolic arrays there is no difference in speed because the
    # multiplications are not actually carried out until after the
    # full matrix is built.
    R = np.eye(3, like=rpy)
    for char in seq:
        match char:
            case "x":
                R = R @ _rot_x(r)
            case "y":
                R = R @ _rot_y(p)
            case "z":
                R = R @ _rot_z(y)

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