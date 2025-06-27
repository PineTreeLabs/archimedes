# ruff: noqa: N802, N803, N806, E741

import numpy as np
import pytest
from scipy.signal import tf2ss as scipy_tf2ss

import archimedes as arc
from archimedes._core import SymbolicArray
from archimedes.experimental.signal import tf2ss


class TestTf2ss:
    """Test basic functionality against SciPy reference"""

    def test_second_order_system(self):
        """Test classic second-order system from SciPy docs"""
        num = [1, 3, 3]
        den = [1, 2, 1]

        A, B, C, D = tf2ss(num, den)
        A_ref, B_ref, C_ref, D_ref = scipy_tf2ss(num, den)

        np.testing.assert_allclose(A, A_ref, rtol=1e-12)
        np.testing.assert_allclose(B, B_ref, rtol=1e-12)
        np.testing.assert_allclose(C, C_ref, rtol=1e-12)
        np.testing.assert_allclose(D, D_ref, rtol=1e-12)

    def test_pure_integrator(self):
        """Test pure integrator: H(s) = 1/s"""
        num = [1]
        den = [1, 0]

        A, B, C, D = tf2ss(num, den)
        A_ref, B_ref, C_ref, D_ref = scipy_tf2ss(num, den)

        np.testing.assert_allclose(A, A_ref, rtol=1e-12)
        np.testing.assert_allclose(B, B_ref, rtol=1e-12)
        np.testing.assert_allclose(C, C_ref, rtol=1e-12)
        np.testing.assert_allclose(D, D_ref, rtol=1e-12)

    def test_constant_gain(self):
        """Test pure gain: H(s) = K"""
        num = [5]
        den = [1]

        A, B, C, D = tf2ss(num, den)
        A_ref, B_ref, C_ref, D_ref = scipy_tf2ss(num, den)

        np.testing.assert_allclose(A, A_ref, rtol=1e-12)
        np.testing.assert_allclose(B, B_ref, rtol=1e-12)
        np.testing.assert_allclose(C, C_ref, rtol=1e-12)
        np.testing.assert_allclose(D, D_ref, rtol=1e-12)

    def test_unity_transfer_function(self):
        """Test H(s) = 1"""
        num = [1]
        den = [1]

        A, B, C, D = tf2ss(num, den)
        A_ref, B_ref, C_ref, D_ref = scipy_tf2ss(num, den)

        np.testing.assert_allclose(A, A_ref, rtol=1e-12)
        np.testing.assert_allclose(B, B_ref, rtol=1e-12)
        np.testing.assert_allclose(C, C_ref, rtol=1e-12)
        np.testing.assert_allclose(D, D_ref, rtol=1e-12)

    def test_different_coefficient_scales(self):
        """Test with very large and very small coefficients"""
        num = [1e-6, 2e-6]
        den = [1e3, 2e3, 1e3]

        A, B, C, D = tf2ss(num, den)
        A_ref, B_ref, C_ref, D_ref = scipy_tf2ss(num, den)

        # Use looser tolerance for scaled coefficients
        np.testing.assert_allclose(A, A_ref, rtol=1e-10)
        np.testing.assert_allclose(B, B_ref, rtol=1e-10)
        np.testing.assert_allclose(C, C_ref, rtol=1e-10)
        np.testing.assert_allclose(D, D_ref, rtol=1e-10)

    def test_improper_transfer_function_raises(self):
        """Test that improper TF (deg(num) > deg(den)) raises error"""
        num = [1, 2, 3, 4]  # degree 3
        den = [1, 2]  # degree 1

        with pytest.raises(ValueError, match="Improper transfer function"):
            tf2ss(num, den)

    def test_output_shapes(self):
        """Verify output matrix dimensions for second-order system"""
        num = [1, 2]
        den = [1, 3, 2]  # n=2 states

        A, B, C, D = tf2ss(num, den)

        assert A.shape == (2, 2), f"A should be (2,2), got {A.shape}"
        assert B.shape == (2, 1), f"B should be (2,1), got {B.shape}"
        assert C.shape == (1, 2), f"C should be (1,2), got {C.shape}"
        assert D.shape == (1, 1), f"D should be (1,1), got {D.shape}"

    def test_controller_canonical(self):
        """Verify matrices have controller canonical structure"""
        num = [1, 2]
        den = [1, 5, 6]  # s^2 + 5s + 6

        A, B, C, D = tf2ss(num, den)

        # Controller canonical form: A = [[-a1, -a0], [1, 0]]
        # where den = [1, a1, a0] after normalization
        expected_A = np.array([[-5, -6], [1, 0]])

        np.testing.assert_allclose(A, expected_A, rtol=1e-12)

        # Controller canonical form: B = [[1], [0]]
        expected_B = np.array([[1], [0]])

        np.testing.assert_allclose(B, expected_B, rtol=1e-12)

    def test_with_symbolic_arrays(self):
        """Test that function works with symbolic arrays"""
        num_sym = arc.sym("num", (2,))
        den_sym = arc.sym("den", (3,))

        # This should not raise an error
        A, B, C, D = tf2ss(num_sym, den_sym)

        # Check that outputs are symbolic arrays
        assert isinstance(A, SymbolicArray)
        assert isinstance(B, SymbolicArray)
        assert isinstance(C, SymbolicArray)
        assert isinstance(D, SymbolicArray)

    def test_compilation_compatibility(self):
        """Test that tf2ss can be used in compiled functions"""

        @arc.compile
        def convert_tf(num, den):
            return tf2ss(num, den)

        num = np.array([1, 2])
        den = np.array([1, 3, 2])

        # This should not raise an error
        A, B, C, D = convert_tf(num, den)

        # Verify against reference
        A_ref, B_ref, C_ref, D_ref = scipy_tf2ss(num, den)
        np.testing.assert_allclose(A, A_ref, rtol=1e-12)
        np.testing.assert_allclose(B, B_ref, rtol=1e-12)
        np.testing.assert_allclose(C, C_ref, rtol=1e-12)
        np.testing.assert_allclose(D, D_ref, rtol=1e-12)

    def test_near_zero_coefficients(self):
        """Test behavior with very small coefficients"""
        num = [1e-15, 1]
        den = [1, 2, 1]

        A, B, C, D = tf2ss(num, den)
        A_ref, B_ref, C_ref, D_ref = scipy_tf2ss(num, den)

        np.testing.assert_allclose(A, A_ref, atol=1e-10)
        np.testing.assert_allclose(B, B_ref, atol=1e-10)
        np.testing.assert_allclose(C, C_ref, atol=1e-10)
        np.testing.assert_allclose(D, D_ref, atol=1e-10)

    def test_ill_conditioned_system(self):
        """Test system with closely spaced poles"""
        # Transfer function with poles at -1, -1.001
        num = [1]
        den = [1, 2.001, 1.001]

        A, B, C, D = tf2ss(num, den)
        A_ref, B_ref, C_ref, D_ref = scipy_tf2ss(num, den)

        np.testing.assert_allclose(A, A_ref, atol=1e-10)
        np.testing.assert_allclose(B, B_ref, atol=1e-10)
        np.testing.assert_allclose(C, C_ref, atol=1e-10)
        np.testing.assert_allclose(D, D_ref, atol=1e-10)


# @pytest.mark.parametrize("num,den", [
#     ([1], [1, 2]),                    # First order
#     ([1, 2], [1, 3, 2]),             # Second order
#     ([2, 3], [1, 4, 5, 6]),          # Third order
#     ([1, 0, 1], [1, 2, 3, 4]),       # Zeros in numerator
#     ([5], [1]),                      # Pure gain
#     ([1], [1, 0]),                   # Pure integrator
# ])
# def test_tf2ss_parametrized(num, den):
#     """Parametrized test across multiple transfer functions"""
#     A, B, C, D = tf2ss(num, den)
#     A_ref, B_ref, C_ref, D_ref = scipy_tf2ss(num, den)

#     np.testing.assert_allclose(A, A_ref, rtol=1e-12)
#     np.testing.assert_allclose(B, B_ref, rtol=1e-12)
#     np.testing.assert_allclose(C, C_ref, rtol=1e-12)
#     np.testing.assert_allclose(D, D_ref, rtol=1e-12)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
