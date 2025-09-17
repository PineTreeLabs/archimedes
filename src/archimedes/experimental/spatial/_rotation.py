from __future__ import annotations
import numpy as np
from archimedes import struct

__all__ = ["Rotation"]



@struct.pytree_node
class Rotation:
    quat: np.ndarray
    scalar_first: bool = struct.field(default=False, static=True)

    def __len__(self):
        return len(self.quat)

    @classmethod
    def from_quat(cls, quat: np.ndarray, scalar_first: bool = False) -> Rotation:
        """Create a Rotation from a quaternion."""
        if quat.ndim == 0:
            raise ValueError("Quaternion must be at least 1D array")
        if quat.shape not in [(4,), (1, 4), (4, 1)]:
            raise ValueError("Quaternion must have shape (4,), (1, 4), or (4, 1)")
        quat = quat.flatten()
        return cls(quat=quat)
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Rotation:
        """Create a Rotation from a rotation matrix.
        
        Note that for the sake of symbolic computation, this method assumes that
        the input is a valid rotation matrix (orthogonal and determinant +1).

        References
        ----------
        .. [1] F. Landis Markley, "Unit Quaternion from Rotation Matrix",
               Journal of guidance, control, and dynamics vol. 31.2, pp.
               440-442, 2008.
        """
        if matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        
        det = np.linalg.det(matrix)

        decision = np.zeros(4, like=matrix)
        quat = np.zeros(4, like=matrix)

        t = np.linalg.trace(matrix)

        decision = np.hstack([np.diag(matrix), t])

        # If matrix[0, 0] is the largest diagonal element
        q0 = np.hstack([
            1 - t + 2 * matrix[0, 0],
            matrix[0, 1] + matrix[1, 0],
            matrix[0, 2] + matrix[2, 0],
            matrix[2, 1] - matrix[1, 2],
        ])

        # If matrix[1, 1] is the largest diagonal element
        q1 = np.hstack([
            1 - t + 2 * matrix[1, 1],
            matrix[2, 1] + matrix[1, 2],
            matrix[0, 1] + matrix[1, 0],
            matrix[0, 2] - matrix[2, 0],
        ])

        # If matrix[2, 2] is the largest diagonal element
        q2 = np.hstack([
            1 - t + 2 * matrix[2, 2],
            matrix[0, 2] + matrix[2, 0],
            matrix[2, 1] + matrix[1, 2],
            matrix[1, 0] - matrix[0, 1],
        ])

        # If t is the largest diagonal element
        # q3 = np.hstack([
            

        choice = np.argmax(decision)

        # Convert rotation matrix to quaternion
        m = matrix
        t = np.trace(m)

