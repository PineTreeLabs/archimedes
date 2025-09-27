from __future__ import annotations
import re
import numpy as np
from archimedes import struct, array, field

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
    # https://github.com/scipy/scipy/blob/3ead2b543df7c7c78619e20f0cb6139e344a8866/scipy/spatial/transform/_rotation_cy.pyx#L461-L476  ruff: noqa: E501
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


# See https://github.com/scipy/scipy/blob/3ead2b543df7c7c78619e20f0cb6139e344a8866/scipy/spatial/transform/_rotation_cy.pyx#L358-L372  ruff: noqa: E501
def _make_elementary_quat(axis: str, angle: float) -> np.ndarray:
    """Create a quaternion representing a rotation about a principal axis."""

    quat = np.hstack([np.cos(angle / 2), np.zeros(3)])
    axis_idx = _elementary_basis_index(axis)
    quat[axis_idx] = np.sin(angle / 2)

    return quat


# See https://github.com/scipy/scipy/blob/3ead2b543df7c7c78619e20f0cb6139e344a8866/scipy/spatial/transform/_rotation_cy.pyx#L376-L391  ruff: noqa: E501
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
    quat: np.ndarray
    scalar_first: bool = field(default=True, static=True)

    def __len__(self):
        return len(self.quat)

    @classmethod
    def from_quat(
        cls, quat: np.ndarray, scalar_first: bool = True, normalize: bool = True
    ) -> Rotation:
        """Create a Rotation from a quaternion."""
        quat = array(quat)
        if quat.ndim == 0:
            raise ValueError("Quaternion must be at least 1D array")
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

        References
        ----------
        .. [1] F. Landis Markley, "Unit Quaternion from Rotation Matrix",
               Journal of guidance, control, and dynamics vol. 31.2, pp.
               440-442, 2008.
        """
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

        quat = np.roll(quat, 1)  # Convert to scalar-first format
        return cls(quat=quat, scalar_first=True)

    @classmethod
    def from_euler(
        cls, seq: str, angles: np.ndarray, degrees: bool = False
    ) -> Rotation:
        """Create a Rotation from Euler angles."""
        num_axes = len(seq)
        if num_axes < 1 or num_axes > 3:
            raise ValueError(
                "Expected axis specification to be a non-empty "
                "string of upto 3 characters, got {}".format(seq)
            )

        intrinsic = _check_seq(seq)

        angles = np.atleast_1d(array(angles))
        if angles.shape not in [(num_axes,), (1, num_axes), (num_axes, 1)]:
            raise ValueError(
                f"For {seq} sequence with {num_axes} axes, `angles` must have shape "
                f"({num_axes},), (1, {num_axes}), or ({num_axes}, 1)"
            )

        seq = seq.lower()
        angles = angles.flatten()

        if degrees:
            angles = np.deg2rad(angles)

        quat = _elementary_quat_compose(seq, angles, intrinsic=intrinsic)
        return cls(quat=quat, scalar_first=True)

    def as_quat(self, scalar_first: bool = True) -> np.ndarray:
        """Return the quaternion as a numpy array."""
        if scalar_first:
            return self.quat
        return np.roll(self.quat, -1)

    def as_matrix(self) -> np.ndarray:
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

    # See: https://github.com/scipy/scipy/blob/3ead2b543df7c7c78619e20f0cb6139e344a8866/scipy/spatial/transform/_rotation_cy.pyx#L774-L851  ruff: noqa: E501
    def as_euler(self, seq: str, degrees: bool = False) -> np.ndarray:
        """Return the Euler angles from the rotation

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
                q[k] - q[i] * sign,
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
        """Apply the rotation to a set of vectors"""

        matrix = self.as_matrix()
        if inverse:
            matrix = matrix.T

        vectors = array(vectors)
        if vectors.ndim == 1:
            if vectors.shape != (3,):
                raise ValueError("For 1D input, `vectors` must have shape (3,)")
            return matrix @ vectors

        elif vectors.ndim == 2:
            if vectors.shape[1] != 3:
                raise ValueError("For 2D input, `vectors` must have shape (N, 3)")
            return vectors @ matrix.T

    def inv(self) -> Rotation:
        """Return the inverse rotation"""
        q = self.as_quat(scalar_first=True)
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]], like=q)
        return Rotation.from_quat(q_inv, scalar_first=True)

    def __mul__(self, other: Rotation) -> Rotation:
        """Compose this rotation with another rotation"""
        q1 = self.as_quat(scalar_first=True)
        q2 = other.as_quat(scalar_first=True)
        q = _compose_quat(q1, q2)
        return Rotation.from_quat(q, scalar_first=True)

    def derivative(self, w: np.ndarray, baumgarte: float = 0.0) -> Rotation:
        """Return the time derivative of the rotation given angular velocity w"""
        q = self.as_quat(scalar_first=True)
        omega = np.array([0, *w], like=q)
        q_dot = 0.5 * _compose_quat(q, omega)

        # Baumgarte stabilization to enforce unit norm constraint
        if baumgarte > 0:
            q_dot -= baumgarte * (np.dot(q, q) - 1) * q

        return Rotation.from_quat(q_dot, scalar_first=True, normalize=False)
