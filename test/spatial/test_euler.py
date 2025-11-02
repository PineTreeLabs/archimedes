# ruff: noqa: N802, N803, N806

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

from archimedes.spatial import (
    euler_to_dcm,
    euler_to_quaternion,
)

np.random.seed(0)


def random_euler():
    """Generate a random rotation"""
    return np.random.randn(3)


class TestEulerLowLevel:
    def test_euler_to_quaternion(self):
        angles = random_euler()
        for seq in ("XYZ", "xyz", "ZYX", "zyx", "XYX", "zxz"):
            q1 = euler_to_quaternion(angles, seq=seq)
            q2 = ScipyRotation.from_euler(seq, angles).as_quat(scalar_first=True)
            np.testing.assert_allclose(q1, q2)

    def test_euler_to_dcm(self):
        angles = random_euler()
        for seq in ("XYZ", "xyz", "ZYX", "zyx", "XYX", "zxz"):
            print(seq)
        R1 = euler_to_dcm(angles, seq=seq)
        R2 = ScipyRotation.from_euler(seq, angles).as_matrix()

        np.testing.assert_allclose(R1, R2)

    def test_error_handling(self):
        # Invalid axis sequence
        with pytest.raises(ValueError, match="Expected axes from `seq`"):
            euler_to_dcm([0.1, 0.2, 0.3], "abc")

        # Invalid euler shape
        with pytest.raises(ValueError, match="Expected axis specification to be"):
            euler_to_dcm([0.1, 0.2, 0.3], "xyzyz")

        # Angles shape doesn't match sequence
        with pytest.raises(ValueError, match="For xyz sequence with 3 axes, `angles`"):
            euler_to_dcm([0.1, 0.2], "xyz")

        # Repeated axis in sequence
        with pytest.raises(ValueError, match="Expected consecutive axes"):
            euler_to_dcm([0.1, 0.2, 0.3], "xxz")
