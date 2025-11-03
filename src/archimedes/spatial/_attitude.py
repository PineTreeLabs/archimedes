"""Protocol for attitude representations."""
from __future__ import annotations
from typing import Protocol

import numpy as np


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