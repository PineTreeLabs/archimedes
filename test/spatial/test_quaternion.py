# ruff: noqa: N802, N803, N806

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

import archimedes as arc
from archimedes.spatial import (
    EulerAngles,
    Quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_to_euler,
)

np.random.seed(0)


def random_quat(wrapper=False):
    """Generate a random rotation"""
    rand_quat = np.random.randn(4)
    rand_quat /= np.linalg.norm(rand_quat)
    if wrapper:
        rand_quat = Quaternion(rand_quat)
    return rand_quat


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
        q_inv_scipy = (
            ScipyRotation.from_quat(q, scalar_first=True)
            .inv()
            .as_quat(scalar_first=True)
        )
        assert np.allclose(q_inv, q_inv_scipy)

    def test_to_dcm(self):
        quat = random_quat()
        R1 = quaternion_to_dcm(quat)
        R2 = ScipyRotation.from_quat(quat, scalar_first=True).as_matrix()
        # NOTE: SciPy's rotation uses an "active" convention (rotating vectors
        # in a single frame), whereas our DCMs are "passive" (rotating between
        # frames).  Thus the matrices are transposes of each other.
        np.testing.assert_allclose(R1, R2.T)

    def test_to_euler(self):
        q = random_quat()
        for seq in ("XYZ", "xyz", "ZYX", "zyx", "XYX", "zxz"):
            euler1 = quaternion_to_euler(q, seq=seq)
            euler2 = ScipyRotation.from_quat(q, scalar_first=True).as_euler(seq)
            np.testing.assert_allclose(euler1, euler2)


class TestQuaternionWrapper:
    def test_ops(self):
        q = Quaternion.identity()
        assert len(q) == 4

        # __getitem__x
        assert q[0] == 1.0

        # __iter__
        w, x, y, z = q
        assert np.allclose(np.array([w, x, y, z]), np.array([1.0, 0.0, 0.0, 0.0]))

        assert q.as_quat() is q
        np.testing.assert_allclose(Quaternion.from_quat(q).array, q.array)

    def test_identity(self):
        q = Quaternion.identity()
        v = np.array([1, 2, 3])
        assert np.allclose(q.rotate(v), v)

    def test_rotate(self):
        q = random_quat(wrapper=True)
        v = np.array([0.1, 0.2, 0.3])
        w = q.rotate(v)

        R_scipy = ScipyRotation.from_quat(q.array, scalar_first=True)
        w_scipy = R_scipy.apply(v, inverse=True)

        assert np.allclose(w, w_scipy)

        # Inverse apply
        w = q.rotate(v, inverse=True)
        w_scipy = R_scipy.apply(v, inverse=False)
        assert np.allclose(w, w_scipy)

    def test_multiplication(self):
        q1, q2 = random_quat(True), random_quat(True)
        q3 = q1 * q2
        R1_scipy = ScipyRotation.from_quat(q1.array, scalar_first=True)
        R2_scipy = ScipyRotation.from_quat(q2.array, scalar_first=True)
        q3_scipy = (R1_scipy * R2_scipy).as_quat(scalar_first=True)
        assert np.allclose(q3.array, q3_scipy)

    def test_composition_associativity(self):
        R1, R2, R3 = [random_quat(True) for _ in range(3)]
        q1 = (R1 * R2) * R3
        q2 = R1 * (R2 * R3)
        assert np.allclose(q1.array, q2.array)

    def test_inverse(self):
        q = random_quat(True)
        q0 = Quaternion.identity()
        q1 = q * q.inv()
        q2 = q.inv() * q
        assert np.allclose(q1.array, q0.array)
        assert np.allclose(q2.array, q0.array)

    def test_kinematics(self):
        q = random_quat(wrapper=True)
        w_B = np.array([0.01, -0.02, 0.03])
        q_t1 = q.kinematics(w_B, baumgarte=1.0)
        q_t2 = 0.5 * quaternion_multiply(q.array, np.array([0, *w_B]))
        assert np.allclose(q_t1.array, q_t2, atol=1e-6)

    def _quat_roundtrip(self, euler_orig, seq, debug=False):
        q = Quaternion.from_euler(euler_orig, seq)
        assert len(q) == 4
        euler2 = q.as_euler(seq)

        if debug:
            R2 = ScipyRotation.from_euler(seq, euler_orig)
            print(f"quat:       {q.array}")
            print(f"SciPy quat: {R2.as_quat(scalar_first=True)}")

            print(f"euler:       {euler2}")
            print(f"SciPy euler: {R2.as_euler(seq)}")

        assert np.allclose(euler2.array, euler_orig)

    def _dcm_roundtrip(self, euler_orig, seq, debug=False):
        # Euler -> matrix -> quat -> matrix -> euler
        q1 = Quaternion.from_euler(euler_orig, seq)
        q2 = Quaternion.from_matrix(q1.as_matrix())
        euler2 = q2.as_euler(seq)

        if debug:
            R1_scipy = ScipyRotation.from_euler(seq, euler_orig)
            R2_scipy = ScipyRotation.from_matrix(q1.as_matrix())
            print(f"quat:       {q1.array}")
            print(f"SciPy quat: {R1_scipy.as_quat(scalar_first=True)}")

            print(f"euler:       {euler2}")
            print(f"SciPy euler: {R2_scipy.as_euler(seq)}")

        assert np.allclose(euler2.array, euler_orig)

    @pytest.mark.parametrize(
        "seq",
        [
            "xyz",  # Standard roll-pitch-yaw
            "zyx",  # Standard yaw-pitch-roll
            "zxz",  # Symmetric sequence
            "ZYX",  # Intrinsic sequence
            "XZX",  # Symmetric intrinsic sequence
        ],
    )
    def test_roundtrip(self, seq):
        euler_orig = np.array([0.1, 0.2, 0.3])
        self._quat_roundtrip(euler_orig, seq)
        self._dcm_roundtrip(euler_orig, seq)

    @pytest.mark.parametrize(
        "angles",
        [
            [0, 0, 0],  # Identity
            [np.pi / 2, 0, 0],  # 90° roll
            [0.1, 0.2, 0.3],  # Small angles
            [np.pi - 0.1, 0.1, 0.1],  # Near-singularity
        ],
    )
    def test_with_scipy(self, angles):
        seq = "xyz"

        # Both libraries should give same results
        R_scipy = ScipyRotation.from_euler(seq, angles)
        q = Quaternion.from_euler(angles, seq)

        v = np.array([1, 2, 3])
        assert np.allclose(q.rotate(v), R_scipy.apply(v, inverse=True))

    def test_compile(self):
        @arc.compile
        def rotate_vector(q, v):
            return q.rotate(v)

        q = Quaternion.from_euler([0.1, 0.2, 0.3], "xyz")
        v = np.array([1, 2, 3])

        result = rotate_vector(q, v)
        assert result.shape == (3,)
        assert np.allclose(result, q.rotate(v))

    def test_tree_ops(self):
        q = Quaternion.from_euler([0.1, 0.2, 0.3], "xyz")
        flat, unflatten = arc.tree.ravel(q)
        q_restored = unflatten(flat)

        # Should preserve rotation behavior (here to a sequence of vectors)
        v = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(q.rotate(v), q_restored.rotate(v))

    def test_errors(self):
        # Invalid output sequence
        with pytest.raises(ValueError, match="Expected `seq` to be a string"):
            Quaternion.identity().as_euler("xz")

        # Pass sequence and Euler angles
        with pytest.raises(ValueError, match="If `euler` is an EulerAngles"):
            angles = EulerAngles([0.1, 0.2, 0.3], seq="xyz")
            Quaternion.from_euler(angles, seq="zyx")

        # Invalid quat shape
        with pytest.raises(ValueError, match="Quaternion must have shape"):
            Quaternion(np.array([1.0, 2.0, 3.0]))

        # Invalid matrix shape
        with pytest.raises(ValueError, match="Rotation matrix must be 3x3"):
            Quaternion.from_matrix(np.eye(4))

        # Invalid rotate input shape
        with pytest.raises(ValueError, match="For 1D input, `vectors` must have"):
            Quaternion.identity().rotate(np.array([1.0, 2.0]))

        with pytest.raises(ValueError, match="For 2D input, `vectors` must have"):
            Quaternion.identity().rotate(np.zeros((1, 2)))
