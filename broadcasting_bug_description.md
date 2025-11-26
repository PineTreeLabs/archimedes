# Broadcasting Bug Fix for Non-Commutative Operations

## Summary

Fixed a bug in `_broadcast_binary_operation` where matrix-vector broadcasting produced incorrect results for non-commutative operations (`-`, `/`, `**`, `//`).

## The Bug

When broadcasting a matrix `(p, q)` with a vector `(n,)` where `n == q` (Case 6), the original code swapped the arguments and recursively called the function:

```python
# Original buggy code in _array_ops.py (lines 218-222)
if len(shape1) == 2 and len(shape2) == 1:
    return _broadcast_binary_operation(
        operation, arr2, arr1, shape2, shape1, common_shape
    )
```

This works for commutative operations (`+`, `*`) but fails for non-commutative ones.

### Example

```python
import archimedes as arc
import numpy as np

A = arc.sym("A", (2, 3))  # Matrix
b = arc.sym("b", (3,))     # Vector

result = A - b  # Should subtract b from each row of A
```

**Expected** (numpy behavior):
```
A - b = [[A[0,0]-b[0], A[0,1]-b[1], A[0,2]-b[2]],
         [A[1,0]-b[0], A[1,1]-b[1], A[1,2]-b[2]]]
```

**Actual** (buggy behavior):
```
# Computed b - A instead of A - b
b - A = [[b[0]-A[0,0], b[1]-A[0,1], b[2]-A[0,2]],
         [b[0]-A[1,0], b[1]-A[1,1], b[2]-A[1,2]]]
```

## Affected Operations

| Operation | Commutative | Affected |
|-----------|-------------|----------|
| `+`       | Yes         | No       |
| `*`       | Yes         | No       |
| `-`       | No          | **Yes**  |
| `/`       | No          | **Yes**  |
| `**`      | No          | **Yes**  |
| `//`      | No          | **Yes**  |
| `@`       | No          | No (separate implementation) |

## The Fix

Split Case 6 into two sub-cases without swapping arguments.

### Explanation

**Case 6a** (`n == q`, vector matches matrix columns):
- Use the transpose trick: `op(A.T, b).T`
- This works because CasADi can natively broadcast `(q, p)` with `(q, 1)`
- Example: `(2,3) - (3,)` → transpose to `(3,2)`, broadcast with `(3,1)`, transpose result back

**Case 6b** (`q == 1`, column singleton):
- Use `cs.mtimes` with ones vectors to expand both arrays to the common shape
- `arr1 @ ones(1,n)` expands `(p,1)` to `(p,n)` by replicating the column
- `ones(p,1) @ arr2.T` expands `(n,1)` to `(p,n)` by replicating the row
- Example: `(4,1) - (3,)` → expand both to `(4,3)`, then subtract

## Test Results

All 592 core tests pass after the fix:
```
====================== 592 passed, 25 warnings in 2.97s =======================
```

---

# FILE 1: src/archimedes/_core/_array_ops/_array_ops.py

Replace the `_broadcast_binary_operation` function (starting around line 170) with this complete implementation:

```python
def _broadcast_binary_operation(operation, arr1, arr2, shape1, shape2, common_shape):
    #
    # NOTE: For broadcasting purposes NumPy treats vectors (n,) as (1, n).  However,
    # trying to broadcast a (n,) NumPy array with a (1, n) CasADi array will return
    # NotImplemented.  This function tries to handle this case by reshaping all (n,)
    # vectors to (n, 1).

    # Ideally enumerating the cases would not be necessary, but since CasADi
    # arrays are always 2D, we need some special logic to make sure that the
    # broadcasting rules are applied correctly.
    #
    # shape1 and shape2 are the shapes of the SymbolicArrays, which may be different
    # from the shapes of the underlying SX expressions. For example, a vector (n,)
    # is represented as a 2D matrix (n, 1) in CasADi.

    # Cases:
    #   1. Both arrays are scalars: shape1 = shape2 = ()
    #   2. First array is a scalar: shape1 = () and shape2 = (n,)
    #   3. Second array is a scalar: shape1 = (n,) and shape2 = ()
    #   4. Both arrays are vectors: shape1 = (n,) and shape2 = (p,)
    #      a. If n == p, no broadcasting necessary
    #      b. raise error
    #   5. First array is a vector: shape1 = (n,) and shape2 = (p, q)
    #      a. If n == q or q == 1, broadcast to (p, n)
    #      b. raise error
    #   6. Second array is a vector: shape1 = (n, m) and shape2 = (p,)
    #      ==> Same as case 5 with arguments reversed
    #   7. Both arrays are matrices: shape1 = (n, m) and shape2 = (p, q)
    #      a. If shapes are equal (n=m, p=q), no broadcasting necessary
    #      b. Some combination of n, m, p, q are equal to 1
    #      c. raise error
    #   8. Matrix, scalar: shape1 = (n, m) and shape2 = ()
    #      ==> broadcast to (n, m)
    #   9. Scalar, matrix: shape1 = () and shape2 = (n, m)
    #      ==> broadcast to (n, m)

    # Based on the previous analysis, assume that the error cases have been handled.
    # That would be 4b, 5b, 6b, 7d.

    # shape1, shape2 = map(np.shape, (arr1, arr2))
    # Cases 1, 4a, 7a: no broadcasting necessary
    if len(shape1) == len(shape2) and all(s1 == s2 for s1, s2 in zip(shape1, shape2)):
        return operation(arr1, arr2)

    # Cases 2, 3, 8, 9: broadcast scalar to vector/matrix - handled automatically
    if len(shape1) == 0 or len(shape2) == 0 or shape1 == (1,) or shape2 == (1,):
        return operation(arr1, arr2)

    # Case 5. First array is a vector: shape1 = (n,) and shape2 = (p, q) with n == q
    #       ==> broadcast to (p, n)
    if len(shape1) == 1 and len(shape2) == 2 and shape2[1] in {1, shape1[0]}:
        # Expand to a consistent shape (note this is not how it's done in numpy,
        # but it makes sense for the scalar symbolics in CasADi)

        # At this point CasADi's "always 2D" representation becomes tricky.
        # If arr1 is actually a NumPy vector (n,) then arr1.shape == shape1
        # and we can just expand it to (n, q) with repmat(arr1, (p, 1)).
        # On the other hand, if arr1 is a CasADi vector (n, 1) then we have
        # to reshape to (n, q) with reshape(arr1, (n, q)).  This function
        # dispatches to NumPy or CasADi depending on the type of arr1 and
        # supports both row-major and column-major ordering (NumPy uses row-major).
        arr1 = _repmat(arr1, (shape2[0], 1))
        arr1 = _cs_reshape(arr1, common_shape)
        return operation(arr1, arr2)

    # Case 6. First array is a matrix, second is a vector: shape1 = (p, q) and shape2 = (n,)
    if len(shape1) == 2 and len(shape2) == 1 and shape1[1] in {1, shape2[0]}:
        # Case 6a: n == q (vector matches matrix columns)
        # Use transpose trick: op(A, b) = op(A.T, b).T
        # CasADi can broadcast (n, 1) with (n, p) natively
        if shape2[0] == shape1[1]:
            return operation(arr1.T, arr2).T
        # Case 6b: q == 1 (matrix column singleton, vector matches rows)
        # (p, 1) op (n,) -> (p, n) where each needs expansion
        # Use mtimes with ones vectors to expand: A @ ones(1,n) and ones(p,1) @ b.T
        p, n = shape1[0], shape2[0]
        # Determine CasADi type (one of them may be a numpy array)
        cs_type = None
        if isinstance(arr1, cs.SX) or isinstance(arr2, cs.SX):
            cs_type = cs.SX
        elif isinstance(arr1, cs.MX) or isinstance(arr2, cs.MX):
            cs_type = cs.MX
        if cs_type is not None:
            ones_row = cs_type.ones(1, n)
            ones_col = cs_type.ones(p, 1)
            arr1 = cs.mtimes(cs_type(arr1), ones_row)
            arr2 = cs.mtimes(ones_col, cs_type(arr2).T)
        else:
            arr1 = np.tile(arr1, (1, n))
            arr2 = np.tile(arr2, (p, 1))
        return operation(arr1, arr2)

    # Case 7b. Both arrays are matrices with some combination of singleton dimensions
    #
    # At this point both arrays are 2D, so the CasADi shape should be the same as the
    # SymbolicArray shape, so we should just be able to expand singleton dimensions
    # using `repmat` until both arrays have the shape of `common_shape`.
    if len(shape1) == 2 and len(shape2) == 2:
        if shape1[0] == 1:
            arr1 = _repmat(arr1, (shape2[0], 1))
        if shape2[1] == 1:
            arr2 = _repmat(arr2, (1, shape1[1]))
        if shape1[1] == 1:
            arr1 = _repmat(arr1, (1, shape2[1]))
        if shape2[0] == 1:
            arr2 = _repmat(arr2, (shape1[0], 1))
        return operation(arr1, arr2)

    raise NotImplementedError(
        f"Unhandled broadcasting case with shapes {shape1} and {shape2}"
    )
```

---

# FILE 2: test/core/test_broadcasting.py (NEW FILE)

Create this new test file:

```python
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
```

---

# FILE 3: examples/broadcasting_bug_example.py (NEW FILE)

Create this example file:

```python
"""
Example verifying broadcasting correctness in archimedes.

This example demonstrates that broadcasting between matrix (2, 3) and vector (3,)
works correctly for non-commutative operations like subtraction.
"""

import numpy as np
import casadi as ca
import archimedes as arc


def demonstrate_broadcasting_bug():
    """Show the broadcasting bug with a simple example."""

    # Create test values
    # Matrix A of shape (2, 3)
    A_val = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    # Vector b of shape (3,)
    b_val = np.array([0.1, 0.2, 0.3])

    print("=" * 60)
    print("Broadcasting Bug Example: Matrix (2,3) - Vector (3,)")
    print("=" * 60)
    print()
    print(f"Matrix A (shape {A_val.shape}):")
    print(A_val)
    print()
    print(f"Vector b (shape {b_val.shape}):")
    print(b_val)
    print()

    # Expected result with numpy
    np_result = A_val - b_val
    print("Expected result (numpy: A - b):")
    print(np_result)
    print()

    # Create symbolic arrays with archimedes
    A_sym = arc.sym("A", (2, 3), kind="SX")
    b_sym = arc.sym("b", (3,), kind="SX")

    # Perform the subtraction
    result_sym = A_sym - b_sym

    print(f"Archimedes result shape: {result_sym.shape}")
    print()

    # Build CasADi function to evaluate
    result_func = ca.Function(
        "broadcast_sub",
        [A_sym._sym, b_sym._sym],
        [result_sym._sym],
        ["A", "b"],
        ["result"]
    )

    # Evaluate with concrete values
    cs_result = result_func.call({"A": A_val, "b": b_val})
    arc_result = cs_result["result"].__array__()

    print("Actual result (archimedes):")
    print(arc_result)
    print()

    # Verify results match
    print("-" * 60)
    print("VERIFICATION:")
    print("-" * 60)
    print()

    if np.allclose(arc_result, np_result):
        print("[OK] Archimedes result matches numpy!")
    else:
        print("[FAIL] Results don't match")
        print("Difference:")
        print(arc_result - np_result)


if __name__ == "__main__":
    demonstrate_broadcasting_bug()
```
