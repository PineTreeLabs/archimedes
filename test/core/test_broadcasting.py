# ruff: noqa: N802, N803, N806
"""
Exhaustive tests for broadcasting rules in binary operations.

This module verifies that archimedes broadcasting produces the same results
as numpy by creating CasADi functions and evaluating them with concrete values.
"""

import casadi as ca
import numpy as np
import pytest

import archimedes as arc
from archimedes._core import sym as _sym


# Override the default symbolic kind to use SX for easier testing
def sym(*args, kind="SX", **kwargs):
    return _sym(*args, kind=kind, **kwargs)


# Binary operations to test
# Note: mod is excluded because CasADi handles it differently
BINARY_OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("truediv", lambda a, b: a / b),
    ("pow", lambda a, b: a**b),
    ("floordiv", lambda a, b: a // b),
]

# Shape combinations for broadcasting tests
# Format: (shape_a, shape_b, expected_shape or None if incompatible)
BROADCAST_SHAPES = [
    # Scalar cases
    ((), (), ()),  # scalar x scalar
    ((), (3,), (3,)),  # scalar x vector
    ((3,), (), (3,)),  # vector x scalar
    ((), (2, 3), (2, 3)),  # scalar x matrix
    ((2, 3), (), (2, 3)),  # matrix x scalar
    # Vector cases
    ((3,), (3,), (3,)),  # vector x vector (same size)
    ((1,), (3,), (3,)),  # (1,) x (3,) -> (3,)
    ((3,), (1,), (3,)),  # (3,) x (1,) -> (3,)
    # Vector x Matrix
    ((3,), (2, 3), (2, 3)),  # (3,) x (2, 3) -> (2, 3)
    ((2, 3), (3,), (2, 3)),  # (2, 3) x (3,) -> (2, 3)
    ((3,), (1, 3), (1, 3)),  # (3,) x (1, 3) -> (1, 3)
    ((1, 3), (3,), (1, 3)),  # (1, 3) x (3,) -> (1, 3)
    ((3,), (3, 1), (3, 3)),  # (3,) x (3, 1) -> (3, 3)
    ((3, 1), (3,), (3, 3)),  # (3, 1) x (3,) -> (3, 3)
    # Matrix cases
    ((2, 3), (2, 3), (2, 3)),  # matrix x matrix (same size)
    ((1, 3), (2, 3), (2, 3)),  # (1, 3) x (2, 3) -> (2, 3)
    ((2, 3), (1, 3), (2, 3)),  # (2, 3) x (1, 3) -> (2, 3)
    ((2, 1), (2, 3), (2, 3)),  # (2, 1) x (2, 3) -> (2, 3)
    ((2, 3), (2, 1), (2, 3)),  # (2, 3) x (2, 1) -> (2, 3)
    ((1, 3), (2, 1), (2, 3)),  # (1, 3) x (2, 1) -> (2, 3)
    ((2, 1), (1, 3), (2, 3)),  # (2, 1) x (1, 3) -> (2, 3)
    ((1, 1), (2, 3), (2, 3)),  # (1, 1) x (2, 3) -> (2, 3)
    ((2, 3), (1, 1), (2, 3)),  # (2, 3) x (1, 1) -> (2, 3)
]

# Incompatible shapes that should raise errors
INCOMPATIBLE_SHAPES = [
    ((2,), (3,)),  # different vector sizes
    ((2, 3), (2, 4)),  # different column sizes
    ((2, 3), (4, 3)),  # different row sizes
    ((2, 3), (3, 2)),  # completely different
]


def create_test_values(shape: tuple, seed: int = 42) -> np.ndarray:
    """Create test values with positive values to avoid issues with pow/div."""
    rng = np.random.default_rng(seed)
    if shape == ():
        return rng.uniform(0.5, 2.0)
    return rng.uniform(0.5, 2.0, size=shape)


def to_numpy_value(cs_dm, arc_shape):
    """
    Convert a CasADi DM result to a numpy array with the correct shape.

    Args:
        cs_dm: CasADi DM object returned from function evaluation
        arc_shape: The expected archimedes shape for the result

    Returns:
        numpy array or scalar with the correct shape
    """
    # Convert CasADi DM to numpy using __array__()
    cs_result_array = cs_dm.__array__()

    # Reshape result back to expected numpy shape
    if arc_shape == ():
        return float(cs_result_array)
    elif len(arc_shape) == 1:
        return cs_result_array.flatten()
    else:
        # CasADi uses column-major (Fortran) order
        return cs_result_array.reshape(arc_shape, order="F")


def to_casadi_input(val, shape, cs_shape):
    """
    Convert a numpy value to CasADi input format.

    Args:
        val: numpy array or scalar
        shape: archimedes shape
        cs_shape: CasADi symbolic shape

    Returns:
        Value formatted for CasADi function input
    """
    if shape == ():
        return val
    return np.atleast_1d(val).flatten("F").reshape(cs_shape, order="F")


def evaluate_arc_operation(op_func, shape_a, shape_b, val_a, val_b):
    """
    Evaluate an archimedes operation by creating symbolic arrays,
    applying the operation, then building a CasADi function to evaluate.

    This follows the approach from manual_casadi_compilation in personal_tests.py.
    """
    # Create symbolic arrays
    sym_a = sym("a", shape_a, dtype=np.float64)
    sym_b = sym("b", shape_b, dtype=np.float64)

    # Apply the operation
    result = op_func(sym_a, sym_b)

    # Build CasADi function
    result_func = ca.Function(
        "test_op",
        [sym_a._sym, sym_b._sym],
        [result._sym],
        ["a", "b"],
        ["result"],
    )

    # Convert inputs to CasADi format
    val_a_cs = to_casadi_input(val_a, shape_a, sym_a._sym.shape)
    val_b_cs = to_casadi_input(val_b, shape_b, sym_b._sym.shape)

    # Evaluate and get DM result
    cs_result = result_func.call({"a": val_a_cs, "b": val_b_cs})

    # Convert DM to numpy using __array__()
    return to_numpy_value(cs_result["result"], result.shape)


class TestBroadcastingBinaryOps:
    """Test broadcasting for all binary operations."""

    @pytest.mark.parametrize("op_name,op_func", BINARY_OPS)
    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_broadcasting_binary_op(self, op_name, op_func, shape_a, shape_b, expected_shape):
        """Test that archimedes broadcasting matches numpy for binary operations."""
        # Create test values
        val_a = create_test_values(shape_a, seed=42)
        val_b = create_test_values(shape_b, seed=123)

        # Get numpy result
        np_result = op_func(val_a, val_b)

        # Get archimedes result via CasADi evaluation
        arc_result = evaluate_arc_operation(op_func, shape_a, shape_b, val_a, val_b)

        # Compare results
        np.testing.assert_allclose(
            arc_result,
            np_result,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Mismatch for {op_name} with shapes {shape_a} x {shape_b}",
        )

    @pytest.mark.parametrize("op_name,op_func", BINARY_OPS)
    @pytest.mark.parametrize("shape_a,shape_b", INCOMPATIBLE_SHAPES)
    def test_incompatible_shapes_raise_error(self, op_name, op_func, shape_a, shape_b):
        """Test that incompatible shapes raise ValueError."""
        sym_a = sym("a", shape_a, dtype=np.float64)
        sym_b = sym("b", shape_b, dtype=np.float64)

        with pytest.raises(ValueError):
            op_func(sym_a, sym_b)


class TestBroadcastingWithMixedTypes:
    """Test broadcasting when mixing symbolic arrays with numpy arrays and scalars."""

    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_symbolic_with_numpy(self, shape_a, shape_b, expected_shape):
        """Test broadcasting between symbolic array and numpy array."""
        val_a = create_test_values(shape_a, seed=42)
        val_b = create_test_values(shape_b, seed=123)

        # Create symbolic array for a only
        sym_a = sym("a", shape_a, dtype=np.float64)

        # Apply operation with numpy array
        result = sym_a + val_b

        # Build CasADi function
        result_func = ca.Function(
            "test_mixed",
            [sym_a._sym],
            [result._sym],
            ["a"],
            ["result"],
        )

        # Evaluate
        val_a_cs = to_casadi_input(val_a, shape_a, sym_a._sym.shape)
        cs_result = result_func.call({"a": val_a_cs})

        # Convert DM to numpy using __array__()
        arc_result = to_numpy_value(cs_result["result"], result.shape)

        # Get numpy result
        np_result = val_a + val_b

        np.testing.assert_allclose(
            arc_result,
            np_result,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Mismatch for shapes {shape_a} x {shape_b}",
        )

    @pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
    def test_symbolic_with_scalar(self, shape):
        """Test broadcasting between symbolic array and Python scalar."""
        val = create_test_values(shape, seed=42)
        scalar = 2.5

        # Create symbolic array
        sym_arr = sym("x", shape, dtype=np.float64)

        # Test addition with scalar
        result = sym_arr + scalar

        # Build CasADi function
        result_func = ca.Function(
            "test_scalar",
            [sym_arr._sym],
            [result._sym],
            ["x"],
            ["result"],
        )

        # Evaluate
        val_cs = to_casadi_input(val, shape, sym_arr._sym.shape)
        cs_result = result_func.call({"x": val_cs})

        # Convert DM to numpy using __array__()
        arc_result = to_numpy_value(cs_result["result"], result.shape)

        # Get numpy result
        np_result = val + scalar

        np.testing.assert_allclose(
            arc_result,
            np_result,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Mismatch for shape {shape} + scalar",
        )


class TestBroadcastingReversedOperands:
    """Test that reversed operand order produces correct results (rmul, radd, etc.)."""

    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_reversed_add(self, shape_a, shape_b, expected_shape):
        """Test that a + b == b + a for commutative operations."""
        val_a = create_test_values(shape_a, seed=42)
        val_b = create_test_values(shape_b, seed=123)

        # Forward: symbolic_a + symbolic_b
        arc_forward = evaluate_arc_operation(lambda a, b: a + b, shape_a, shape_b, val_a, val_b)

        # Reverse: symbolic_b + symbolic_a
        arc_reverse = evaluate_arc_operation(lambda a, b: a + b, shape_b, shape_a, val_b, val_a)

        # Both should equal numpy result
        np_result = val_a + val_b

        np.testing.assert_allclose(arc_forward, np_result, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(arc_reverse, np_result, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_reversed_mul(self, shape_a, shape_b, expected_shape):
        """Test that a * b == b * a for commutative operations."""
        val_a = create_test_values(shape_a, seed=42)
        val_b = create_test_values(shape_b, seed=123)

        # Forward: symbolic_a * symbolic_b
        arc_forward = evaluate_arc_operation(lambda a, b: a * b, shape_a, shape_b, val_a, val_b)

        # Reverse: symbolic_b * symbolic_a
        arc_reverse = evaluate_arc_operation(lambda a, b: a * b, shape_b, shape_a, val_b, val_a)

        # Both should equal numpy result
        np_result = val_a * val_b

        np.testing.assert_allclose(arc_forward, np_result, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(arc_reverse, np_result, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_sub_and_rsub(self, shape_a, shape_b, expected_shape):
        """Test subtraction and reverse subtraction."""
        val_a = create_test_values(shape_a, seed=42)
        val_b = create_test_values(shape_b, seed=123)

        # a - b
        arc_result = evaluate_arc_operation(lambda a, b: a - b, shape_a, shape_b, val_a, val_b)
        np_result = val_a - val_b
        np.testing.assert_allclose(arc_result, np_result, rtol=1e-10, atol=1e-10)

        # b - a (reverse)
        arc_result_rev = evaluate_arc_operation(lambda a, b: a - b, shape_b, shape_a, val_b, val_a)
        np_result_rev = val_b - val_a
        np.testing.assert_allclose(arc_result_rev, np_result_rev, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_div_and_rdiv(self, shape_a, shape_b, expected_shape):
        """Test division and reverse division."""
        val_a = create_test_values(shape_a, seed=42)
        val_b = create_test_values(shape_b, seed=123)

        # a / b
        arc_result = evaluate_arc_operation(lambda a, b: a / b, shape_a, shape_b, val_a, val_b)
        np_result = val_a / val_b
        np.testing.assert_allclose(arc_result, np_result, rtol=1e-10, atol=1e-10)

        # b / a (reverse)
        arc_result_rev = evaluate_arc_operation(lambda a, b: a / b, shape_b, shape_a, val_b, val_a)
        np_result_rev = val_b / val_a
        np.testing.assert_allclose(arc_result_rev, np_result_rev, rtol=1e-10, atol=1e-10)


class TestBroadcastingEdgeCases:
    """Test edge cases in broadcasting."""

    def test_zero_dimensional_result(self):
        """Test operations that produce scalar results."""
        val_a = 2.5
        val_b = 3.5

        for op_name, op_func in BINARY_OPS:
            arc_result = evaluate_arc_operation(op_func, (), (), val_a, val_b)
            np_result = op_func(val_a, val_b)
            np.testing.assert_allclose(
                arc_result,
                np_result,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Scalar {op_name} mismatch",
            )

    def test_single_element_arrays(self):
        """Test broadcasting with single-element arrays of various shapes."""
        shapes = [(), (1,), (1, 1)]
        val = 2.5

        for shape_a in shapes:
            for shape_b in shapes:
                val_a = np.full(shape_a, val) if shape_a != () else val
                val_b = np.full(shape_b, val + 1) if shape_b != () else val + 1

                arc_result = evaluate_arc_operation(
                    lambda a, b: a + b, shape_a, shape_b, val_a, val_b
                )
                np_result = val_a + val_b

                # Flatten for comparison since shapes might differ
                np.testing.assert_allclose(
                    np.atleast_1d(arc_result).flatten(),
                    np.atleast_1d(np_result).flatten(),
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"Mismatch for shapes {shape_a} x {shape_b}",
                )

    def test_large_dimension_difference(self):
        """Test broadcasting between scalar and larger matrix."""
        shape_a = ()
        shape_b = (5, 7)
        val_a = 2.5
        val_b = create_test_values(shape_b, seed=42)

        arc_result = evaluate_arc_operation(lambda a, b: a * b, shape_a, shape_b, val_a, val_b)
        np_result = val_a * val_b

        np.testing.assert_allclose(arc_result, np_result, rtol=1e-10, atol=1e-10)


class TestBroadcastingNumpyUfuncs:
    """Test broadcasting through numpy ufuncs."""

    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_np_add(self, shape_a, shape_b, expected_shape):
        """Test numpy add ufunc with broadcasting."""
        val_a = create_test_values(shape_a, seed=42)
        val_b = create_test_values(shape_b, seed=123)

        sym_a = sym("a", shape_a, dtype=np.float64)
        sym_b = sym("b", shape_b, dtype=np.float64)

        # Use numpy ufunc
        result = np.add(sym_a, sym_b)

        # Build CasADi function
        result_func = ca.Function(
            "test_np_add",
            [sym_a._sym, sym_b._sym],
            [result._sym],
        )

        # Evaluate
        val_a_cs = to_casadi_input(val_a, shape_a, sym_a._sym.shape)
        val_b_cs = to_casadi_input(val_b, shape_b, sym_b._sym.shape)

        cs_result = result_func(val_a_cs, val_b_cs)

        # Convert DM to numpy using __array__()
        arc_result = to_numpy_value(cs_result, result.shape)

        np_result = np.add(val_a, val_b)

        np.testing.assert_allclose(arc_result, np_result, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_np_multiply(self, shape_a, shape_b, expected_shape):
        """Test numpy multiply ufunc with broadcasting."""
        val_a = create_test_values(shape_a, seed=42)
        val_b = create_test_values(shape_b, seed=123)

        sym_a = sym("a", shape_a, dtype=np.float64)
        sym_b = sym("b", shape_b, dtype=np.float64)

        # Use numpy ufunc
        result = np.multiply(sym_a, sym_b)

        # Build CasADi function
        result_func = ca.Function(
            "test_np_mul",
            [sym_a._sym, sym_b._sym],
            [result._sym],
        )

        # Evaluate
        val_a_cs = to_casadi_input(val_a, shape_a, sym_a._sym.shape)
        val_b_cs = to_casadi_input(val_b, shape_b, sym_b._sym.shape)

        cs_result = result_func(val_a_cs, val_b_cs)

        # Convert DM to numpy using __array__()
        arc_result = to_numpy_value(cs_result, result.shape)

        np_result = np.multiply(val_a, val_b)

        np.testing.assert_allclose(arc_result, np_result, rtol=1e-10, atol=1e-10)


class TestBroadcastingResultShapes:
    """Verify that result shapes match numpy broadcasting rules."""

    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_result_shape_matches_numpy(self, shape_a, shape_b, expected_shape):
        """Verify that the result shape matches numpy's broadcast_shapes."""
        sym_a = sym("a", shape_a, dtype=np.float64)
        sym_b = sym("b", shape_b, dtype=np.float64)

        result = sym_a + sym_b

        # Get expected shape from numpy
        np_expected_shape = np.broadcast_shapes(shape_a, shape_b)

        assert result.shape == np_expected_shape, (
            f"Shape mismatch: got {result.shape}, expected {np_expected_shape} "
            f"for inputs {shape_a} and {shape_b}"
        )
