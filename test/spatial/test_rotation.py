# ruff: noqa: N802, N803, N806

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

import archimedes as arc
from archimedes.spatial import Rotation

np.random.seed(0)


def random_rotation():
    """Generate a random rotation"""
    rand_quat = np.random.randn(4)
    rand_quat /= np.linalg.norm(rand_quat)
    return Rotation.from_quat(rand_quat, scalar_first=True)


class TestRotation:
    def test_identity(self):
        R = Rotation.identity()
        v = np.array([1, 2, 3])
        assert np.allclose(R.apply(v), v)

    def test_multiplication(self):
        R1, R2 = random_rotation(), random_rotation()
        q1 = (R1 * R2).as_quat()
        R1_scipy = ScipyRotation.from_quat(R1.as_quat(), scalar_first=True)
        R2_scipy = ScipyRotation.from_quat(R2.as_quat(), scalar_first=True)
        q2 = (R1_scipy * R2_scipy).as_quat(scalar_first=True)
        assert np.allclose(q1, q2)

    def test_composition_associativity(self):
        R1, R2, R3 = [random_rotation() for _ in range(3)]
        q1 = ((R1 * R2) * R3).as_quat()
        q2 = (R1 * (R2 * R3)).as_quat()
        assert np.allclose(q1, q2)

    def test_inverse(self):
        R = random_rotation()
        q0 = Rotation.identity().as_quat()
        q1 = (R * R.inv()).as_quat()
        q2 = (R.inv() * R).as_quat()
        assert np.allclose(q1, q0)
        assert np.allclose(q2, q0)

        # Check the roll if scalar_first=False
        q0_nsf = Rotation.identity().as_quat(scalar_first=False)
        assert np.allclose(q0_nsf, np.array([0, 0, 0, 1]))

    def _quat_roundtrip(self, euler_orig, seq, debug=False):
        R = Rotation.from_euler(seq, euler_orig)
        assert len(R) == 4
        euler2 = R.as_euler(seq)

        if debug:
            R2 = ScipyRotation.from_euler(seq, euler_orig)
            print(f"quat:       {R.as_quat(scalar_first=True)}")
            print(f"SciPy quat: {R2.as_quat(scalar_first=True)}")

            print(f"euler:       {euler2}")
            print(f"SciPy euler: {R2.as_euler(seq)}")

        assert np.allclose(euler2, euler_orig)

    def _dcm_roundtrip(self, euler_orig, seq, debug=False):
        # Euler -> matrix -> quat -> matrix -> euler
        R1 = Rotation.from_euler(seq, euler_orig)
        R2 = Rotation.from_matrix(R1.as_matrix())
        euler2 = R2.as_euler(seq, degrees=True)
        euler2 = np.deg2rad(euler2)

        if debug:
            R1_scipy = ScipyRotation.from_euler(seq, euler_orig)
            R2_scipy = ScipyRotation.from_matrix(R1.as_matrix())
            print(f"quat:       {R1.as_quat(scalar_first=True)}")
            print(f"SciPy quat: {R1_scipy.as_quat(scalar_first=True)}")

            print(f"euler:       {euler2}")
            print(f"SciPy euler: {R2_scipy.as_euler(seq)}")

        assert np.allclose(euler2, euler_orig)

    def _mixed_roundtrip(self, euler_orig, seq, debug=False):
        # Euler -> matrix -> quat -> euler

        R1 = Rotation.from_euler(seq, euler_orig)
        q = R1.as_quat(scalar_first=True)
        R3 = Rotation.from_quat(q, scalar_first=True)
        euler2 = R3.as_euler(seq)

        assert np.allclose(euler2, euler_orig)

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
        self._mixed_roundtrip(euler_orig, seq)
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
        R = Rotation.from_euler(seq, angles)

        test_vector = np.array([1, 2, 3])
        assert np.allclose(R.apply(test_vector), R_scipy.apply(test_vector))

    def test_compile(self):
        @arc.compile
        def rotate_vector(R, v):
            return R.apply(v)

        R = Rotation.from_euler("xyz", [0.1, 0.2, 0.3], degrees=True)
        v = np.array([1, 2, 3])

        result = rotate_vector(R, v)
        assert result.shape == (3,)
        assert np.allclose(result, R.apply(v))

    def test_tree_ops(self):
        R = Rotation.from_euler("xyz", [0.1, 0.2, 0.3])
        flat, unflatten = arc.tree.ravel(R)
        R_restored = unflatten(flat)

        # Should preserve rotation behavior (here to a sequence of vectors)
        v = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(R.apply(v), R_restored.apply(v))

    def test_errors(self):
        # Invalid axis sequence
        with pytest.raises(ValueError, match="Expected axes from `seq`"):
            Rotation.from_euler("abc", [0.1, 0.2, 0.3])

        # Invalid euler shape
        with pytest.raises(ValueError, match="Expected axis specification to be"):
            Rotation.from_euler("xyzyz", [])

        # Angles shape doesn't match sequence
        with pytest.raises(ValueError, match="For xyz sequence with 3 axes, `angles`"):
            Rotation.from_euler("xyz", [0.1, 0.2])

        # Repeated axis in sequence
        with pytest.raises(ValueError, match="Expected consecutive axes"):
            Rotation.from_euler("xxz", [0.1, 0.2, 0.3])

        # Invalid output sequence
        with pytest.raises(ValueError, match="Expected `seq` to be a string"):
            Rotation.identity().as_euler("xz")

        # Invalid quat shape
        with pytest.raises(ValueError, match="Quaternion must have shape"):
            Rotation.from_quat(np.array([1.0, 2.0, 3.0]))

        # Invalid matrix shape
        with pytest.raises(ValueError, match="Rotation matrix must be 3x3"):
            Rotation.from_matrix(np.eye(4))

        # Invalid apply input shape
        with pytest.raises(ValueError, match="For 1D input, `vectors` must have"):
            Rotation.identity().apply(np.array([1.0, 2.0]))

        with pytest.raises(ValueError, match="For 2D input, `vectors` must have"):
            Rotation.identity().apply(np.zeros((1, 2)))
