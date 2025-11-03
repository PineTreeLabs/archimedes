"""Protocol for attitude representations."""
from __future__ import annotations
from typing import Protocol, cast

import numpy as np

from archimedes import array


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