# ruff: noqa: N802, N803, N806

import re

import casadi as cs
import numpy as np
import pytest
from numpy import exceptions as npex

from archimedes._core import SymbolicArray, compile
from archimedes._core import sym as _sym
from archimedes.error import ShapeDtypeError

# NOTE: Most tests here use SX instead of the default MX, since the is_equal
# tests struggle with the array-valued MX type.  This doesn't indicate an error
# in the MX representation, just a difficulty of checking for equality between
# array-valued symbolic expressions


# Override the default symbolic kind to use SX
def sym(*args, kind="SX", **kwargs):
    return _sym(*args, kind=kind, **kwargs)


@pytest.fixture
def array():
    # Set dtype as int to make sure it gets promoted
    return sym("x", (3,), dtype=np.int32)


class TestSymbolicArrayUFuncs:
    def test_sin(self, array):
        result = np.sin(array)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, cs.sin(array._sym), 1)

    def test_cos(self, array):
        result = np.cos(array)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, cs.cos(array._sym), 1)

    def test_tan(self, array):
        result = np.tan(array)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, cs.tan(array._sym), 1)

    def test_exp(self, array):
        result = np.exp(array)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, cs.exp(array._sym), 1)

    def test_log(self, array):
        result = np.log(array)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, cs.log(array._sym), 1)

    def test_sqrt(self, array):
        result = np.sqrt(array)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, cs.sqrt(array._sym), 1)

    def test_fabs(self, array):
        result = np.abs(array)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, cs.fabs(array._sym), 1)

    def test_arctan2(self):
        x = sym("x", (3,), dtype=np.float64)
        y = sym("y", (3,), dtype=np.float64)
        result = np.arctan2(x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, cs.atan2(x._sym, y._sym), 1)

    def test_hypot(self):
        x = sym("x", (3,), dtype=np.float64)
        y = sym("y", (3,), dtype=np.float64)
        result = np.hypot(x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, cs.sqrt(x._sym**2 + y._sym**2), 3)

    def test_angle_conversions(self):
        x = sym("x", (3,), dtype=np.float64)
        result = np.radians(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, x._sym * (np.pi / 180.0), 1)

        result = np.degrees(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, x._sym * (180.0 / np.pi), 1)


class TestSymbolicArrayFunctions:
    def test_broadcast_to(self):
        # Test 0D array broadcast to 0D (no change)
        x = sym("x", shape=(), dtype=np.int32)
        result = np.broadcast_to(x, ())
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == ()
        assert cs.is_equal(result._sym, x._sym, 1)

        # Test 0D array broadcast to 1D
        x = sym("x", shape=(), dtype=np.int32)
        result = np.broadcast_to(x, (3,))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)

        # Test 0D array broadcast to 2D
        x = sym("x", shape=(), dtype=np.int32)
        result = np.broadcast_to(x, (2, 3))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 3)

        # Test 1D array broadcast to 1D (no change due to same shape)
        x = sym("x", shape=(3,), dtype=np.int32)
        result = np.broadcast_to(x, (3,))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, x._sym, 1)

        # Test 1D array broadcast to 2D
        x = sym("x", shape=(3,), dtype=np.int32)
        result = np.broadcast_to(x, (2, 3))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 3)

        # Test 2D array broadcast to 2D (expand dimension with size 1)
        x = sym("x", shape=(1, 3), dtype=np.int32)
        result = np.broadcast_to(x, (2, 3))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 3)

        # Test error cases

        # Input has more dimensions than broadcast shape
        x = sym("x", shape=(2, 3), dtype=np.int32)
        msg = re.escape(
            "input operand with shape (2, 3) has more dimensions than "
            "the broadcast shape (3,)"
        )
        with pytest.raises(ValueError, match=msg):
            np.broadcast_to(x, (3,))

        # Cannot broadcast non-scalar to scalar
        x = sym("x", shape=(3,), dtype=np.int32)
        with pytest.raises(
            ValueError, match="cannot broadcast a non-scalar to a scalar array"
        ):
            np.broadcast_to(x, ())

        # Negative size in shape
        x = sym("x", shape=(3,), dtype=np.int32)
        with pytest.raises(
            ValueError, match="all elements of broadcast shape must be non-negative"
        ):
            np.broadcast_to(x, (3, -1))

        # More than 2D not supported
        x = sym("x", shape=(2, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="Only 0-2D arrays are supported"):
            np.broadcast_to(x, (2, 3, 4))

        # Incompatible shapes
        x = sym("x", shape=(2, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="Cannot broadcast"):
            np.broadcast_to(x, (2, 4))

    def _dot_test(self, shape1, shape2, result_shape):
        x = sym("x", shape=shape1, dtype=np.int32)

        # Check two symbolic arrays
        y = sym("y", shape=shape2, dtype=np.int32)
        result = np.dot(x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == result_shape
        cs_dot = cs.dot if x._sym.shape == y._sym.shape else cs.mtimes
        assert cs.is_equal(result._sym, cs_dot(x._sym, y._sym), 3)

        # Check symbolic array and numpy array
        y = np.zeros(shape2, dtype=np.int32)
        result = np.dot(x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == result_shape
        assert cs.is_equal(result._sym, cs_dot(x._sym, y), 2)

        # Reverse order to check first argument as ndarray
        result = np.dot(y.T, x.T).T
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == result_shape
        assert cs.is_equal(result._sym, cs_dot(x._sym, y), 2)

        # Check scalar, if applicable
        if shape2 == ():
            y = 1
            result = np.dot(x, y)
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == result_shape
            assert cs.is_equal(result._sym, cs_dot(x._sym, y), 2)

        if shape1 == ():
            x = 1
            y = sym("y", shape=shape2, dtype=np.int32)
            result = np.dot(x, y)
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == result_shape
            assert cs.is_equal(result._sym, cs_dot(x, y._sym), 2)

    def test_dot(self):
        # Cases:
        # 1. () * () => ()
        # 2. (n,) * () => (n,)
        # 3. () * (n,) => (n,)
        # 4. (n,) * (n,) => ()
        # 5. (n,) * (m,) => raise error
        # 6. (n,) * (m, p) => raise error
        # 7. (n, m) * (m,) => (n,)
        # 8. (n, m) * (m, p) => (n, p)
        # 9. (n, m) * (p, q) => raise error
        # 10. (n, m) * () => (n, m)
        # 11. () * (n, m) => (n, m)

        n, m, p, q = 2, 3, 4, 5

        # Case 1
        self._dot_test((), (), ())

        # Case 2
        self._dot_test((n,), (), (n,))

        # Case 3
        self._dot_test((), (n,), (n,))

        # Case 4
        self._dot_test((n,), (n,), ())

        # Case 5
        with pytest.raises(ValueError):
            self._dot_test((n,), (m,), None)

        # Case 6
        with pytest.raises(ValueError):
            self._dot_test((n,), (m, p), None)

        # Case 7
        self._dot_test((n, m), (m,), (n,))

        # Case 8
        self._dot_test((n, m), (m, p), (n, p))

        # Case 9
        with pytest.raises(ValueError):
            self._dot_test((n, m), (p, q), None)

        # Case 10
        self._dot_test((n, m), (), (n, m))

        # Case 11
        self._dot_test((), (n, m), (n, m))

        # Check consistent vector handling - both of these CasADi arrays should
        # be transposed in order for the dot product to be valid, but this needs
        # to happen automatically
        x_sym = cs.SX.sym("x", 3, 1)
        y_sym = cs.SX.sym("y", 1, 3)
        x = SymbolicArray(x_sym, shape=(3,))
        y = SymbolicArray(y_sym, shape=(3,))
        result = np.dot(x, y)
        assert isinstance(result, SymbolicArray)
        assert result.shape == ()
        assert cs.is_equal(result._sym, cs.mtimes(x_sym.T, y_sym.T), 3)

        # Test product with scalar
        x = sym("x", (3, 1))
        y = sym("y", ())
        result = np.dot(x, y)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (x.shape[0], 1)

        y = sym("y", (1,))
        result = np.dot(x, y)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (x.shape[0],)
        assert cs.is_equal(result._sym, x._sym @ y._sym, 2)

        y = sym("y", (1, 1))
        result = x @ y
        assert isinstance(result, SymbolicArray)
        assert result.shape == (x.shape[0], 1)
        assert cs.is_equal(result._sym, x._sym @ y._sym, 2)

    def test_outer(self):
        # 1D arrays
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(2,), dtype=np.int32)
        result = np.outer(x, y)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (3, 2)
        assert cs.is_equal(result._sym, cs.mtimes(x._sym, y._sym.T), 2)

    def test_sum(self):
        # 0D array
        x = sym("x", shape=(), dtype=np.int32)
        result = np.sum(x)
        assert isinstance(result, SymbolicArray)
        assert result.shape == ()
        assert cs.is_equal(result._sym, x._sym, 1)

        # 1D array
        x = sym("x", shape=(3,), dtype=np.int32)
        result = np.sum(x)
        assert isinstance(result, SymbolicArray)
        assert result.shape == ()
        assert cs.is_equal(result._sym, cs.sum1(x._sym), 2)

        with pytest.raises(npex.AxisError):
            np.sum(x, axis=1)

        # 2D array
        x = sym("x", shape=(3, 2), dtype=np.int32)
        result = np.sum(x)  # Default: flatten and sum
        assert isinstance(result, SymbolicArray)
        assert result.shape == ()
        result_cs = cs.sum1(cs.reshape(x._sym, (6, 1)))
        assert cs.is_equal(result._sym, result_cs, 5)

        # Sum along rows
        result = np.sum(x, axis=0)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2,)
        assert cs.is_equal(result._sym, cs.sum1(x._sym).T, 2)

        # Sum along columns
        result = np.sum(x, axis=1)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, cs.sum2(x._sym), 1)

        # Invalid axis
        with pytest.raises(npex.AxisError):
            np.sum(x, axis=2)

    def test_flatten(self):
        x = sym("x", shape=(2, 3), dtype=np.int32)
        result = x.flatten()
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (6,)
        # Reverse the order to be consistent with CasADi column-major
        assert cs.is_equal(result._sym, x._sym.T.reshape((1, 6)).T, 1)

        # Check ravel
        result2 = x.ravel()
        assert cs.is_equal(result._sym, result2._sym, 1)

        result = x.flatten("F")
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (6,)
        assert cs.is_equal(result._sym, x._sym.reshape((6, 1)), 1)

        # Check scalar case (note NumPy does convert to a 1D array)
        x = sym("x", shape=(), dtype=np.int32)
        result = x.flatten()
        assert result.shape == (1,)
        assert cs.is_equal(result._sym, x._sym, 1)

    def test_reshape(self):
        def sym_reshape(x, shape, order="C"):
            return np.reshape(x, shape, order=order)

        _reshape = compile(sym_reshape, static_argnames=("shape", "order"))

        x = np.arange(0, 6)
        assert np.allclose(
            _reshape(x, (3, 2)),
            np.reshape(x, (3, 2)),
            atol=1e-5,
        )
        assert np.allclose(
            _reshape(x, (3, 2), order="F"),
            np.reshape(x, (3, 2), order="F"),
            atol=1e-5,
        )

        x = sym("x", shape=(2, 3), dtype=np.int32)

        # Test C ordering
        result = np.reshape(x, (3, 2))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 2)
        assert cs.is_equal(result._sym, x._sym.T.reshape((2, 3)).T, 1)

        # Test Fortran ordering
        result = np.reshape(x, (3, 2), order="F")
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 2)
        result_ex = cs.reshape(x._sym, (3, 2))
        assert cs.is_equal(result._sym, result_ex, 1)

        # Test invalid ordering
        with pytest.raises(ValueError):
            np.reshape(x, (3, 2), order="A")

        # Test some edge cases by calling directly
        from archimedes._core._array_ops._array_ops import _cs_reshape

        x = cs.SX.sym("x", 6, 1)
        y = _cs_reshape(x, (3, 2), order="F")
        assert y.shape == (3, 2)
        assert isinstance(y, cs.SX)
        assert cs.is_equal(y, x.reshape((3, 2)), 1)

    def test_squeeze(self):
        x = sym("x", shape=(1, 3), dtype=np.int32)
        y = x.squeeze()

        assert isinstance(y, SymbolicArray)
        assert cs.is_equal(x._sym, y._sym.T)
        assert y.shape == (3,)

    def test_roll(self):
        # Test scalar
        x = sym("x", shape=(), dtype=np.int32)
        result = np.roll(x, 1)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == ()
        assert cs.is_equal(result._sym, x._sym, 1)

        # Test 1D array
        x = sym("x", shape=(5,), dtype=np.int32)
        result = np.roll(x, 2)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (5,)
        expected = cs.vertcat(x._sym[-2:], x._sym[:-2])
        assert cs.is_equal(result._sym, expected, 1)

        # Test negative shift
        result = np.roll(x, -1)
        expected = cs.vertcat(x._sym[1:], x._sym[:1])
        assert cs.is_equal(result._sym, expected, 1)

        # Test large shift
        result = np.roll(x, 7)  # Equivalent to shift of 2
        expected = cs.vertcat(x._sym[-2:], x._sym[:-2])
        assert cs.is_equal(result._sym, expected, 1)

        # Test zero shift
        result = np.roll(x, 0)
        assert cs.is_equal(result._sym, x._sym, 1)

        # Test 2D array, axis=0
        x = sym("x", shape=(2, 3), dtype=np.int32)
        result = np.roll(x, 1, axis=0)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 3)
        expected = cs.vertcat(x._sym[-1, :], x._sym[:-1, :])
        assert cs.is_equal(result._sym, expected, 1)

        # Test 2D array, axis=1
        result = np.roll(x, -1, axis=1)
        expected = cs.horzcat(x._sym[:, 1:], x._sym[:, :1])
        assert cs.is_equal(result._sym, expected, 1)

    def test_append(self):
        x = sym("x", shape=(2, 3), dtype=np.int32)
        y = sym("y", shape=(2, 3), dtype=np.int32)

        # Test axis=0
        result = np.append(x, y, axis=0)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (4, 3)
        assert cs.is_equal(result._sym, cs.vertcat(x._sym, y._sym), 1)

        # Test axis=1
        result = np.append(x, y, axis=1)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 6)
        assert cs.is_equal(result._sym, cs.horzcat(x._sym, y._sym), 1)

        # Test axis None (flatten)
        result = np.append(x, y, axis=None)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (12,)
        # CasADi reshapes with column-major order ("F"), so to create
        # the expected result we have to force it to use row-major reshaping
        result_ex = cs.vertcat(
            cs.reshape(x._sym.T, (1, 6)).T,
            cs.reshape(y._sym.T, (1, 6)).T,
        )
        assert cs.is_equal(result._sym, result_ex, 1)

        # Test invalid axis
        with pytest.raises(ValueError):
            np.append(x, y, axis=2)

    def test_hstack(self):
        # Test 2D arrays
        x = sym("x", shape=(2, 3), dtype=np.int32)
        y = sym("y", shape=(2, 4), dtype=np.int32)
        result = np.hstack((x, y))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 7)
        assert cs.is_equal(result._sym, cs.horzcat(x._sym, y._sym), 1)

        # Test 1D arrays
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(4,), dtype=np.float32)
        result = np.hstack((x, y))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert result.shape == (7,)

    def test_vstack(self):
        # Test 2D arrays
        x = sym("x", shape=(2, 3), dtype=np.int32)
        y = sym("y", shape=(4, 3), dtype=np.int32)
        result = np.vstack((x, y))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (6, 3)
        assert cs.is_equal(result._sym, cs.vertcat(x._sym, y._sym), 1)

        # Test 1D arrays
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(3,), dtype=np.float32)
        result = np.vstack((x, y))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert result.shape == (2, 3)

    def test_stack(self):
        # Test 0D arrays
        x = sym("x", shape=(), dtype=np.int32)
        y = sym("y", shape=(), dtype=np.int32)
        result = np.stack((x, y))  # Default axis = 0
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2,)
        assert cs.is_equal(result._sym, cs.vertcat(x._sym, y._sym), 1)

        result2 = np.stack((x, y), axis=-1)
        assert isinstance(result2, SymbolicArray)
        assert cs.is_equal(result2._sym, result._sym, 1)

        with pytest.raises(npex.AxisError):
            np.stack((x, y), axis=1)

        # Test 1D arrays
        n = 3
        x = sym("x", shape=(n,), dtype=np.int32)
        y = sym("y", shape=(n,), dtype=np.int32)
        result = np.stack((x, y))  # Default axis = 0
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, n)
        assert cs.is_equal(result._sym, cs.horzcat(x._sym, y._sym).T, 1)

        result = np.stack((x, y), axis=1)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (n, 2)
        assert cs.is_equal(result._sym, cs.horzcat(x._sym, y._sym), 1)

        result2 = np.stack((x, y), axis=-1)
        assert isinstance(result2, SymbolicArray)
        assert result2.dtype == np.int32
        assert result2.shape == (n, 2)
        assert cs.is_equal(result2._sym, result._sym, 1)

        with pytest.raises(npex.AxisError):
            np.stack((x, y), axis=2)

        with pytest.raises(TypeError):
            np.stack((x, y), axis="abc")

        # Test 2D arrays (error)
        x = sym("x", shape=(2, 3), dtype=np.int32)
        y = sym("y", shape=(2, 4), dtype=np.int32)
        with pytest.raises(ValueError):
            np.stack((x, y))

    def test_split(self):
        dtype = np.int32

        # Test split 0D (error)
        x = sym("x", shape=())
        with pytest.raises(npex.AxisError):
            np.split(x, [0])

        # Test split 1D with sections
        x = sym("x", shape=(6,), dtype=dtype)
        x_split = np.split(x, 2)
        assert all(arr.dtype == dtype for arr in x_split)
        assert cs.is_equal(x_split[0]._sym, x._sym[:3], 1)
        assert cs.is_equal(x_split[1]._sym, x._sym[3:], 1)

        # Test split 1D with indices
        ix = [0, 2, 6]
        x_split = np.split(x, ix)
        assert all(arr.dtype == dtype for arr in x_split)
        assert cs.is_equal(x_split[0]._sym, x._sym[:0], 1)  # Empty section
        assert cs.is_equal(x_split[1]._sym, x._sym[: ix[1]], 1)
        assert cs.is_equal(x_split[2]._sym, x._sym[ix[1] :], 1)
        assert cs.is_equal(x_split[3]._sym, x._sym[:0], 1)  # Empty section

        # Test row split 2D with sections
        x = sym("x", shape=(6, 2), dtype=dtype)
        x_split = np.split(x, 2, axis=0)
        assert all(arr.dtype == dtype for arr in x_split)
        assert cs.is_equal(x_split[0]._sym, x._sym[:3, :], 1)
        assert cs.is_equal(x_split[1]._sym, x._sym[3:, :], 1)

        # Check vsplit
        x_vsplit = np.vsplit(x, 2)
        for xs, xv in zip(x_split, x_vsplit):
            assert cs.is_equal(xs._sym, xv._sym, 1)

        # Test row split 2D with indices
        ix = [0, 2, 6]
        x_split = np.split(x, ix, axis=0)
        assert all(arr.dtype == dtype for arr in x_split)
        assert cs.is_equal(x_split[0]._sym, x._sym[:0, :], 1)  # Empty section
        assert cs.is_equal(x_split[1]._sym, x._sym[: ix[1], :], 1)
        assert cs.is_equal(x_split[2]._sym, x._sym[ix[1] :, :], 1)
        assert cs.is_equal(x_split[3]._sym, x._sym[:0, :], 1)  # Empty section

        # Check vsplit
        x_vsplit = np.vsplit(x, ix)
        for xs, xv in zip(x_split, x_vsplit):
            assert cs.is_equal(xs._sym, xv._sym, 1)

        # Test col split 2D with sections
        x = sym("x", shape=(2, 6), dtype=dtype)
        x_split = np.split(x, 2, axis=1)
        assert all(arr.dtype == dtype for arr in x_split)
        assert cs.is_equal(x_split[0]._sym, x._sym[:, :3], 1)
        assert cs.is_equal(x_split[1]._sym, x._sym[:, 3:], 1)

        # Test invalid sections (not divisible by ncol)
        with pytest.raises(ValueError):
            np.split(x, 5, axis=1)

        # Check hsplit
        x_vsplit = np.hsplit(x, 2)
        for xs, xv in zip(x_split, x_vsplit):
            assert cs.is_equal(xs._sym, xv._sym, 1)

        # Test col split 2D with indices
        ix = [0, 2, 6]
        x_split = np.split(x, ix, axis=1)
        assert all(arr.dtype == dtype for arr in x_split)
        assert cs.is_equal(x_split[0]._sym, x._sym[:, :0], 1)  # Empty section
        assert cs.is_equal(x_split[1]._sym, x._sym[:, : ix[1]], 1)
        assert cs.is_equal(x_split[2]._sym, x._sym[:, ix[1] :], 1)
        assert cs.is_equal(x_split[3]._sym, x._sym[:, :0], 1)  # Empty section

        # Check hsplit
        x_vsplit = np.hsplit(x, ix)
        for xs, xv in zip(x_split, x_vsplit):
            assert cs.is_equal(xs._sym, xv._sym, 1)

        # Error for decreasing indices
        with pytest.raises(IndexError):
            np.split(x, [2, 1])

    def test_tile(self):
        # Test 0D array tiling
        x = sym("x", shape=(), dtype=np.int32)
        result = np.tile(x, 3)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)

        # Test 1D array with scalar reps
        x = sym("x", shape=(3,), dtype=np.int32)
        result = np.tile(x, 2)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (6,)

        # Test 1D array with 1D reps
        x = sym("x", shape=(3,), dtype=np.int32)
        result = np.tile(x, (2,))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (6,)

        # Test 1D array with 2D reps
        x = sym("x", shape=(3,), dtype=np.int32)
        result = np.tile(x, (2, 3))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 9)

        # Test 2D array with scalar reps
        x = sym("x", shape=(2, 3), dtype=np.int32)
        result = np.tile(x, 2)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 6)

        # Test 2D array with 1D reps
        x = sym("x", shape=(2, 3), dtype=np.int32)
        result = np.tile(x, (2,))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 6)

        # Test 2D array with 2D reps
        x = sym("x", shape=(2, 3), dtype=np.int32)
        result = np.tile(x, (2, 2))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (4, 6)

        # Test with reps=1
        x = sym("x", shape=(2, 3), dtype=np.int32)
        result = np.tile(x, 1)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 3)
        assert cs.is_equal(result._sym, x._sym, 1)

        # Test with empty reps tuple
        x = sym("x", shape=(2, 3), dtype=np.int32)
        result = np.tile(x, ())
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 3)
        assert cs.is_equal(result._sym, x._sym, 1)

        # Test error case: more than 2D tiling
        x = sym("x", shape=(2, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            np.tile(x, (2, 2, 2))

    def test_atleast_1d(self):
        # Test 0D array
        x = sym("x", shape=(), dtype=np.int32)
        result = np.atleast_1d(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (1,)
        assert x.ndim == 0
        assert result.ndim == 1
        assert cs.is_equal(result._sym, x._sym, 1)

        # Tuple of 0D arrays
        y = sym("y", shape=(), dtype=np.int32)
        result_x, result_y = np.atleast_1d(x, y)
        for result in (result_x, result_y):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (1,)
            assert result.ndim == 1

        # Test 1D array
        x = sym("x", shape=(3,), dtype=np.int32)
        result = np.atleast_1d(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)
        assert result.ndim == 1
        assert cs.is_equal(result._sym, x._sym, 1)

    def test_atleast_2d(self):
        # Test 0D array
        x = sym("x", shape=(), dtype=np.int32)
        result = np.atleast_2d(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (1, 1)
        assert cs.is_equal(result._sym, x._sym, 1)

        # Tuple of 0D arrays
        y = sym("y", shape=(), dtype=np.int32)
        result_x, result_y = np.atleast_2d(x, y)
        for result in (result_x, result_y):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (1, 1)

        # Test 1D array
        x = sym("x", shape=(3,), dtype=np.int32)
        result = np.atleast_2d(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (1, 3)
        assert cs.is_equal(result._sym, x._sym.T, 1)

        # 0D and 1D array
        result_x, result_y = np.atleast_2d(x, y)
        assert result_x.shape == (1, 3)
        assert result_y.shape == (1, 1)
        for result in (result_x, result_y):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32

        # 2D array
        x = sym("x", shape=(3, 2), dtype=np.int32)
        result = np.atleast_2d(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 2)
        assert cs.is_equal(result._sym, x._sym, 1)

    def test_concatenate(self):
        # Test 0D arrays (not supported by numpy)
        x = sym("x", shape=(), dtype=np.int32)
        with pytest.raises(ValueError):
            np.concatenate((x, x))

        # Test 1D arrays
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(4,), dtype=np.int32)
        result = np.concatenate((x, y))
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (7,)
        assert cs.is_equal(result._sym, cs.vertcat(x._sym, y._sym), 1)

        # Test mixed dimensions (error)
        x = sym("x", shape=(2, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            np.concatenate((x, y))

        # Test 2D arrays (axis=0)
        y = sym("y", shape=(4, 3), dtype=np.int32)
        result = np.concatenate((x, y), axis=0)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (6, 3)
        assert cs.is_equal(result._sym, cs.vertcat(x._sym, y._sym), 1)

        # Test invalid shapes
        with pytest.raises(ValueError):
            np.concatenate((x, y), axis=1)

        result = np.concatenate((x.T, y.T), axis=1)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 6)
        assert cs.is_equal(result._sym, cs.horzcat(x._sym.T, y._sym.T), 1)

    def test_where(self):
        # Basic case - all arrays same shape
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(3,), dtype=np.int32)
        c = sym("c", shape=(3,), dtype=bool)
        result = np.where(c, x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, cs.if_else(c._sym, x._sym, y._sym), 3)

        # Scalar condition with array operands
        condition = sym("condition", shape=(), dtype=bool)
        result = np.where(condition, x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, cs.if_else(condition._sym, x._sym, y._sym), 3)

        # Test broadcasting between different shapes
        x = sym("x", shape=(1, 3), dtype=np.int32)
        y = sym("y", shape=(3,), dtype=np.int32)
        condition = sym("condition", shape=(), dtype=bool)
        result = np.where(condition, x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (1, 3)  # Result should have broadcasted shape

        # Test with all different shapes that can broadcast together
        x = sym("x", shape=(2, 1), dtype=np.int32)
        y = sym("y", shape=(1, 3), dtype=np.int32)
        condition = sym("condition", shape=(1,), dtype=bool)
        result = np.where(condition, x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 3)  # Result should be broadcast to (2,3)

        # Test with scalar operands
        x = sym("x", shape=(), dtype=np.int32)
        y = sym("y", shape=(2, 3), dtype=np.int32)
        condition = sym("condition", shape=(2, 3), dtype=bool)
        result = np.where(condition, x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 3)

        # Test with all scalar inputs
        x = sym("x", shape=(), dtype=np.int32)
        y = sym("y", shape=(), dtype=float)
        condition = sym("condition", shape=(), dtype=bool)
        result = np.where(condition, x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64  # Type promotion to float
        assert result.shape == ()

        # Test data type promotion
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(3,), dtype=np.float64)
        condition = sym("condition", shape=(3,), dtype=bool)
        result = np.where(condition, x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64  # Should be promoted to float64
        assert result.shape == (3,)

        # Test incompatible shapes (should raise error)
        x = sym("x", shape=(2, 3), dtype=np.int32)
        y = sym("y", shape=(4, 3), dtype=np.int32)
        condition = sym("condition", shape=(2, 3), dtype=bool)
        with pytest.raises(ValueError):
            np.where(condition, x, y)

        # TODO: Update when where with one arg is supported
        with pytest.raises(ValueError):
            np.where(x)

    def test_cross(self):
        # Test 1D arrays of length 3
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(3,), dtype=np.int32)
        result = np.cross(x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, cs.cross(x._sym, y._sym), 2)

        # Test error for invalid shapes
        with pytest.raises(ValueError):
            y = sym("y", shape=(2,), dtype=np.int32)
            np.cross(x, y)

        # Test 2D arrays with default args (axis=-1)
        x = sym("x", shape=(4, 3), dtype=np.int32)
        y = sym("y", shape=(4, 3), dtype=np.int32)
        result = np.cross(x, y)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (4, 3)
        assert cs.is_equal(result._sym, cs.cross(x._sym, y._sym, 2), 2)

        # Test 2D arrays with axis specification
        result = np.cross(x, y, axis=1)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (4, 3)
        assert cs.is_equal(result._sym, cs.cross(x._sym, y._sym, 2), 2)

        result = np.cross(x.T, y.T, axis=0)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 4)
        assert cs.is_equal(result._sym, cs.cross(x._sym, y._sym, 2).T, 2)

        # Test error for invalid axes
        with pytest.raises(npex.AxisError):
            np.cross(x, y, axis=2)

        with pytest.raises(NotImplementedError):
            np.cross(x, y.T, axisa=1, axisb=0)

        # Test axisc argument
        result = np.cross(x, y, axisc=0)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 4)
        assert cs.is_equal(result._sym, cs.cross(x._sym, y._sym, 2).T, 2)

    def test_clip(self):
        # Test 1D array
        x = sym("x", shape=(3,))
        result = np.clip(x, 0, 1)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, cs.fmin(cs.fmax(x._sym, 0), 1), 2)

        # Test 2D array
        x = sym("x", shape=(2, 3))
        result = np.clip(x, 0, 1)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2, 3)
        assert cs.is_equal(result._sym, cs.fmin(cs.fmax(x._sym, 0), 1), 2)

        # Test symbolic clip
        x_max = sym("x_max", shape=())
        result = np.clip(x, 0, x_max)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2, 3)
        assert cs.is_equal(result._sym, cs.fmin(cs.fmax(x._sym, 0), x_max._sym), 2)

        # Test no lower bound
        result = np.clip(x, None, 1)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2, 3)
        assert cs.is_equal(result._sym, cs.fmin(x._sym, 1), 2)

        # Test no upper bound
        result = np.clip(x, 0, None)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2, 3)
        assert cs.is_equal(result._sym, cs.fmax(x._sym, 0), 2)

        # Test no bounds
        result = np.clip(x, None, None)
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2, 3)
        assert cs.is_equal(result._sym, x._sym, 2)

    def test_norm(self):
        # Vector norm
        x = sym("x", shape=(3,), dtype=np.int32)
        result = np.linalg.norm(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64  # Note cast
        assert result.shape == ()
        assert cs.is_equal(result._sym, cs.norm_2(x._sym), 4)

        result = np.linalg.norm(x, ord=1)
        assert cs.is_equal(result._sym, cs.norm_1(x._sym), 4)

        result = np.linalg.norm(x, ord=np.inf)
        assert cs.is_equal(result._sym, cs.norm_inf(x._sym), 4)

        with pytest.raises(ValueError):
            np.linalg.norm(x, ord=0)

        with pytest.raises(ValueError):
            np.linalg.norm(x, ord="fro")

        # Unsupported args
        with pytest.raises(NotImplementedError):
            np.linalg.norm(x, axis=0)

        with pytest.raises(NotImplementedError):
            np.linalg.norm(x, keepdims=True)

        # Matrix norm
        x = sym("x", shape=(2, 2), dtype=np.int32)
        result = np.linalg.norm(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64  # Note cast
        assert result.shape == ()
        assert cs.is_equal(result._sym, cs.norm_fro(x._sym), 5)

        result = np.linalg.norm(x, ord=1)
        assert cs.is_equal(result._sym, cs.norm_1(x._sym), 4)

        result = np.linalg.norm(x, ord=np.inf)
        assert cs.is_equal(result._sym, cs.norm_inf(x._sym), 5)

    def test_solve(self):
        A = sym("A", shape=(2, 2), dtype=np.float64)
        b = sym("b", shape=(2,), dtype=np.float64)

        x = np.linalg.solve(A, b)
        assert isinstance(x, SymbolicArray)
        assert x.shape == (2,)
        assert x.dtype == np.float64
        assert cs.is_equal(x._sym, cs.solve(A._sym, b._sym), 5)

        # Error handling
        with pytest.raises(ShapeDtypeError, match=r".*not aligned.*"):
            b = sym("b", shape=(3,))
            np.linalg.solve(A, b)

        with pytest.raises(ShapeDtypeError, match=r".*not a vector.*"):
            b = sym("b", shape=(2, 3))
            np.linalg.solve(A, b)
