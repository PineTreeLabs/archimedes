# ruff: noqa: N802, N803, N806

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from archimedes.spatial import (
    quaternion_to_euler,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_dcm,
)

np.random.seed(0)


def random_quat():
    """Generate a random rotation"""
    rand_quat = np.random.randn(4)
    return rand_quat / np.linalg.norm(rand_quat)


class TestQuaternionLowLevel:
    def test_multiplication(self):
        R1, R2 = random_quat(), random_quat()
        q1 = quaternion_multiply(R1, R2)
        R1_scipy = ScipyRotation.from_quat(R1, scalar_first=True)
        R2_scipy = ScipyRotation.from_quat(R2, scalar_first=True)
        q2 = (R1_scipy * R2_scipy).as_quat(scalar_first=True)
        assert np.allclose(q1, q2)

    def test_inverse(self):
        q = random_quat()
        q_inv = quaternion_inverse(q)
        q_inv_scipy = ScipyRotation.from_quat(
            q, scalar_first=True
        ).inv().as_quat(scalar_first=True)
        assert np.allclose(q_inv, q_inv_scipy)

    def test_to_dcm(self):
        quat = random_quat()
        R1 = quaternion_to_dcm(quat)
        R2 = ScipyRotation.from_quat(quat, scalar_first=True).as_matrix()
        np.testing.assert_allclose(R1, R2)

    def test_to_euler(self):
        q = random_quat()
        for seq in ("XYZ", "xyz", "ZYX", "zyx", "XYX", "zxz"):
            euler1 = quaternion_to_euler(q, seq=seq)
            euler2 = ScipyRotation.from_quat(q, scalar_first=True).as_euler(seq)
            np.testing.assert_allclose(euler1, euler2)



