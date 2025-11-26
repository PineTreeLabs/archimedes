# ruff: noqa: N802, N803, N806
"""
Exhaustive tests for broadcasting rules in binary operations.

This module verifies that archimedes broadcasting produces the same results
as numpy by creating symbolic functions and evaluating them with concrete values.
"""

import numpy as np
import pytest

import archimedes as arc
from archimedes._core import sym as _sym

rng = np.random.default_rng(0)


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
    # ufuncs
    ("np.add", np.add),
    ("np.subtract", np.subtract),
    ("np.multiply", np.multiply),
    ("np.divide", np.divide),
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


def create_test_values(shape: tuple) -> np.ndarray:
    """Create test values with positive values to avoid issues with pow/div."""
    if shape == ():
        return rng.uniform(0.5, 2.0)
    return rng.uniform(0.5, 2.0, size=shape)


def _test_broadcast(op_name, op_func, val_a, val_b):
    """
    Evaluate an archimedes operation by creating symbolic arrays,
    applying the operation, then building a CasADi function to evaluate.
    """
    np_result = np.asarray(op_func(val_a, val_b))  # Convert scalar -> () if needed

    @arc.compile
    def func(a, b):
        return op_func(a, b)

    arc_result = func(val_a, val_b)

    # Compare results
    shape_a = val_a.shape if isinstance(val_a, np.ndarray) else ()
    shape_b = val_b.shape if isinstance(val_b, np.ndarray) else ()
    np.testing.assert_allclose(
        arc_result,
        np_result,
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"Mismatch for {op_name} with shapes {shape_a} x {shape_b}",
    )

    assert np_result.shape == arc_result.shape, (
        f"Shape mismatch: got {arc_result.shape}, expected {np_result.shape} "
        f"for inputs {shape_a} and {shape_b}"
    )

    return arc_result


class TestBroadcastingBinaryOps:
    """Test broadcasting for all binary operations."""

    @pytest.mark.parametrize("op_name,op_func", BINARY_OPS)
    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_broadcasting_binary_op(
        self, op_name, op_func, shape_a, shape_b, expected_shape
    ):
        """Test that archimedes broadcasting matches numpy for binary operations."""
        # Create test values
        val_a = create_test_values(shape_a)
        val_b = create_test_values(shape_b)

        _test_broadcast(op_name, op_func, val_a, val_b)

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
        val_a = create_test_values(shape_a)
        val_b = create_test_values(shape_b)

        @arc.compile
        def func(a):
            return a + val_b

        arc_result = func(val_a)
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
        val = create_test_values(shape)
        scalar = 2.5

        @arc.compile
        def func(a):
            return a + scalar

        arc_result = func(val)
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
    def test_sub_and_rsub(self, shape_a, shape_b, expected_shape):
        """Test subtraction and reverse subtraction."""
        val_a = create_test_values(shape_a)
        val_b = create_test_values(shape_b)

        # a - b
        arc_result = _test_broadcast("sub", lambda a, b: a - b, val_a, val_b)
        np_result = val_a - val_b
        np.testing.assert_allclose(arc_result, np_result, rtol=1e-10, atol=1e-10)

        # b - a (reverse)
        arc_result_rev = _test_broadcast("sub", lambda a, b: a - b, val_b, val_a)
        np_result_rev = val_b - val_a
        np.testing.assert_allclose(
            arc_result_rev, np_result_rev, rtol=1e-10, atol=1e-10
        )

    @pytest.mark.parametrize("shape_a,shape_b,expected_shape", BROADCAST_SHAPES)
    def test_div_and_rdiv(self, shape_a, shape_b, expected_shape):
        """Test division and reverse division."""
        val_a = create_test_values(shape_a)
        val_b = create_test_values(shape_b)

        # a / b
        arc_result = _test_broadcast("div", lambda a, b: a / b, val_a, val_b)
        np_result = val_a / val_b
        np.testing.assert_allclose(arc_result, np_result, rtol=1e-10, atol=1e-10)

        # b / a (reverse)
        arc_result_rev = _test_broadcast("div", lambda a, b: a / b, val_b, val_a)
        np_result_rev = val_b / val_a
        np.testing.assert_allclose(
            arc_result_rev, np_result_rev, rtol=1e-10, atol=1e-10
        )
