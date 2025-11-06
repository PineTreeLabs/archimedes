# ruff: noqa: N802, N803, N806

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

import archimedes as arc
from archimedes.spatial import (
    EulerAngles,
    euler_kinematics,
    euler_to_dcm,
    euler_to_quaternion,
)

np.random.seed(0)


TEST_SEQ = ["XYZ", "xyz", "ZYX", "zyx", "XYX", "zxz"]


def random_euler(wrapper=False, seq="xyz"):
    """Generate a random rotation"""
    angles = np.random.randn(3)
    if wrapper:
        angles = EulerAngles(angles, seq=seq)
    return angles


def _test_kinematics(wrapper):
    rpy = np.array([0.1, 0.2, 0.3])

    # Given roll-pitch-yaw rates, compute the body-frame angular velocity
    # using the rotation matrices directly.
    pqr = np.array([0.4, 0.5, 0.6])  # Roll, pitch, yaw rates
    C_roll = EulerAngles(rpy[0], "x").as_matrix()  # C_φ
    C_pitch = EulerAngles(rpy[1], "y").as_matrix()  # C_θ
    # Successively transform each rate into the body frame
    w_B_ex = np.array([pqr[0], 0.0, 0.0]) + C_roll @ (
        np.array([0.0, pqr[1], 0.0]) + C_pitch @ np.array([0.0, 0.0, pqr[2]])
    )

    # Use the Euler kinematics function to duplicate the transformation
    Hinv = euler_kinematics(rpy, inverse=True)
    w_B = Hinv @ pqr

    np.testing.assert_allclose(w_B, w_B_ex)

    if wrapper:
        rpy = EulerAngles(rpy)
        result = rpy.kinematics(w_B).array

    else:
        # Test the forward transformation
        H = euler_kinematics(rpy)
        result = H @ w_B

    np.testing.assert_allclose(pqr, result)


class TestEulerLowLevel:
    def test_euler_to_quaternion(self):
        angles = random_euler()
        for seq in TEST_SEQ:
            q1 = euler_to_quaternion(angles, seq=seq)
            q2 = ScipyRotation.from_euler(seq, angles).as_quat(scalar_first=True)
            np.testing.assert_allclose(q1, q2)

    def test_euler_to_dcm(self):
        angles = random_euler()
        for seq in TEST_SEQ:
            print(seq)
            R1 = euler_to_dcm(angles, seq=seq)
            R2 = ScipyRotation.from_euler(seq, angles).as_matrix()

            # NOTE: SciPy's rotation uses an "active" convention (rotating vectors
            # in a single frame), whereas our DCMs are "passive" (rotating between
            # frames).  Thus the matrices are transposes of each other.
            np.testing.assert_allclose(R1, R2.T)

    def test_kinematics(self):
        _test_kinematics(wrapper=False)

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


class TestEulerWrapper:
    def test_ops(self):
        angles = random_euler(wrapper=True)
        assert len(angles) == 3

        # __getitem__ and __len__
        short_angles = EulerAngles(angles[:2], seq=angles.seq[:2])
        assert len(short_angles) == 2

        # __iter__
        r, p, y = angles
        assert np.allclose(np.array([r, p, y]), angles.array)

    def test_identity(self):
        angles = EulerAngles.identity(seq="xyz")
        R = angles.as_matrix()
        np.testing.assert_allclose(R, np.eye(3))

    def test_rotate(self):
        seq = "xyz"
        angles = random_euler(wrapper=True, seq=seq)

        v = np.array([0.1, 0.2, 0.3])
        w = angles.rotate(v)

        R_scipy = ScipyRotation.from_euler(seq, angles.array)
        w_scipy = R_scipy.apply(v, inverse=True)

        assert np.allclose(w, w_scipy)

        # Inverse apply
        w = angles.rotate(v, inverse=True)
        w_scipy = R_scipy.apply(v, inverse=False)
        assert np.allclose(w, w_scipy)

    def test_from_euler(self):
        angles = random_euler(wrapper=True, seq="xyz")

        # Test getting the same rotation back
        np.testing.assert_allclose(
            EulerAngles.from_euler(angles, seq=angles.seq).array, angles.array
        )

        R_scipy = ScipyRotation.from_euler(angles.seq, angles.array)

        for seq in TEST_SEQ:
            angles_out1 = EulerAngles.from_euler(angles, seq=seq).array
            angles_out2 = R_scipy.as_euler(seq=seq)
            np.testing.assert_allclose(angles_out1, angles_out2)

    def test_as_euler(self):
        angles = random_euler(wrapper=True, seq="xyz")

        # Test getting the same rotation back
        assert angles.as_euler(seq=angles.seq) is angles

        R_scipy = ScipyRotation.from_euler(angles.seq, angles.array)

        for seq in TEST_SEQ:
            angles_out1 = angles.as_euler(seq=seq).array
            angles_out2 = R_scipy.as_euler(seq=seq)
            np.testing.assert_allclose(angles_out1, angles_out2)

    def test_as_matrix(self):
        for seq in TEST_SEQ:
            angles = random_euler(wrapper=True, seq=seq)
            R1 = angles.as_matrix()
            R2 = ScipyRotation.from_euler(seq, angles.array).as_matrix()
            # NOTE: SciPy's rotation uses an "active" convention (rotating vectors
            # in a single frame), whereas our DCMs are "passive" (rotating between
            # frames).  Thus the matrices are transposes of each other.
            np.testing.assert_allclose(R1, R2.T)

    def test_inv(self):
        seq = "xyz"
        angles = random_euler(wrapper=True, seq=seq)

        angles_inv = angles.inv()
        R1 = angles.as_matrix()
        R2 = angles_inv.as_matrix()

        np.testing.assert_allclose(R1.T, R2)

    def test_kinematics(self):
        _test_kinematics(wrapper=True)

    def test_tree_ops(self):
        euler = EulerAngles([0.1, 0.2, 0.3], "xyz")
        flat, unflatten = arc.tree.ravel(euler)
        euler_restored = unflatten(flat)

        # Should preserve rotation behavior (here to a sequence of vectors)
        v = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(euler.rotate(v), euler_restored.rotate(v))
