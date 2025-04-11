# ruff: noqa: N802
# ruff: noqa: N803
# ruff: noqa: N806

import casadi as cs
import numpy as np
import pytest

import archimedes as arc
from archimedes._core import SymbolicArray
from archimedes._core import sym as _sym
from archimedes.error import ShapeDtypeError

# NOTE: Most tests here use SX instead of the default MX, since the is_equal
# tests struggle with the array-valued MX type.  This doesn't indicate an error
# in the MX representation, just a difficulty of checking for equality between
# array-valued symbolic expressions

# TODO:
# - Split this file up


# Override the default symbolic kind to use SX
def sym(*args, kind="SX", **kwargs):
    return _sym(*args, kind=kind, **kwargs)


@pytest.fixture
def array():
    # Set dtype as int to make sure it gets promoted
    return sym("x", (3,), dtype=np.int32)


class TestSymbolicArrayCreate:
    @pytest.mark.parametrize("shape", (0, 2, (), 3, (3, 1), (3, 2)))
    @pytest.mark.parametrize("dtype", (bool, np.int32, np.float32, np.float64))
    def test_create(self, shape, dtype):
        result = sym("x", shape, dtype)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == dtype
        assert result.shape == (shape if isinstance(shape, tuple) else (shape,))
        assert isinstance(result._sym, cs.SX)

        result2 = arc.array(result)
        assert isinstance(result2, SymbolicArray)

        # Cast dtype
        result3 = arc.array(result2, dtype=np.float64)
        assert isinstance(result3, SymbolicArray)
        assert result3.dtype == np.float64

    def test_from_sx(self):
        x_sym = cs.SX.sym("x", 2, 2)
        x = SymbolicArray(x_sym)
        assert x.shape == x_sym.size()

    def test_from_seq(self):
        # Empty sequence
        z = arc.array([])
        assert isinstance(z, np.ndarray)
        assert z.shape == ((0,))

        # Single sequence
        x1 = sym("x", ())
        x2 = 3.0
        z = arc.array([x1, x2])
        assert isinstance(z, SymbolicArray)
        assert z.shape == (2,)
        assert cs.is_equal(z._sym[0], x1._sym, 1)
        assert cs.is_equal(z._sym[1], x2, 1)

        # Nested sequences
        y = sym("y", (2,))
        z = arc.array([[x1, x2], y])
        assert isinstance(z, SymbolicArray)
        assert z.shape == (2, 2)
        assert cs.is_equal(z._sym[0, 0], x1._sym, 1)
        assert cs.is_equal(z._sym[0, 1], x2, 1)
        assert cs.is_equal(z._sym[1, :].T, y._sym, 1)

        # Invalid input
        with pytest.raises(NotImplementedError):
            arc.array({"x": 2})

    @pytest.mark.parametrize("shape", (0, 2, (), 3, (3, 1), (3, 2)))
    @pytest.mark.parametrize("dtype", (bool, np.int32, np.float32, np.float64))
    @pytest.mark.parametrize("sparse", [True, False])
    def test_zeros(self, shape, dtype, sparse):
        x = arc.zeros(shape, sparse=sparse, dtype=dtype, kind="SX")
        assert x.shape == (shape if isinstance(shape, tuple) else (shape,))
        assert x.dtype == dtype
        if sparse:
            assert x._sym.nnz() == 0
        else:
            assert x._sym.nnz() == np.prod(shape)
        assert cs.is_equal(x._sym, np.zeros(shape, dtype=dtype), 1)

    @pytest.mark.parametrize("shape", (0, 2, (), 3, (3, 1), (3, 2)))
    @pytest.mark.parametrize("dtype", (bool, np.int32, np.float32, np.float64))
    def test_ones(self, shape, dtype):
        x = arc.ones(shape, dtype=dtype, kind="SX")
        assert x.shape == (shape if isinstance(shape, tuple) else (shape,))
        assert x.dtype == dtype
        assert cs.is_equal(x._sym, np.ones(shape, dtype=dtype), 1)

    @pytest.mark.parametrize("dtype", (bool, np.int32, np.float32, np.float64))
    def test_zeros_like(self, dtype):
        shape = (3, 2)

        for x in [
            np.ones(shape, dtype=dtype),
            sym("x", shape=shape, dtype=dtype),
            [[0, 0], [0, 0], [0, 0]],
        ]:
            # Test numpy array
            y = arc.zeros_like(x, dtype=dtype, kind="SX")
            assert y.shape == shape
            assert y.dtype == dtype
            assert cs.is_equal(y._sym, np.zeros(shape), 1)

        # Test default symbolic kind
        y = arc.zeros_like(np.ones(shape))
        assert y.shape == shape
        assert y.kind == arc._core._array_impl.DEFAULT_SYM_NAME

    @pytest.mark.parametrize("dtype", (bool, np.int32, np.float32, np.float64))
    def test_ones_like(self, dtype):
        shape = (3, 2)

        for x in [
            np.zeros(shape, dtype=dtype),
            sym("x", shape=shape, dtype=dtype),
            [[0, 0], [0, 0], [0, 0]],
        ]:
            # Test numpy array
            y = arc.ones_like(x, dtype=dtype)
            assert y.shape == shape
            assert y.dtype == dtype
            assert cs.is_equal(y._sym, np.ones(shape), 1)

            # Test changing the symbolic kind
            y = arc.ones_like(x, kind="SX")
            assert y.shape == shape
            assert y.kind == "SX"
            assert cs.is_equal(y._sym, np.ones(shape), 1)

    def test_error_handling(self):
        with pytest.raises(ValueError, match=r"Unknown symbolic kind.*"):
            sym("x", kind="abc")

        with pytest.raises(ValueError, match=r"Shape must be an int or tuple of ints"):
            sym("x", shape="invalid")

        with pytest.raises(ValueError, match=r"Only scalars, vectors.*"):
            sym("x", shape=(3, 3, 3))

        with pytest.raises(ValueError):
            arc.zeros(shape=(2, 3, 4))

        with pytest.raises(ValueError):
            arc.zeros(shape=(5.4, 1))

        with pytest.raises(ValueError):
            arc.zeros(shape="abc")

        # Triply-nested list
        with pytest.raises(ValueError):
            arc.array([[[1, 2], [3, 4]], [5, 6], [7, 8]])

        # Inconsistent lengths
        with pytest.raises(ValueError):
            arc.array([[1, 2], [3]])

        # Inconsistent types
        with pytest.raises(ValueError):
            arc.array([[1, 2], 3.0])

        with pytest.raises(ValueError):
            SymbolicArray(NotImplemented)

    def test_eye(self):
        x = arc.eye(3, kind="SX")
        assert x.shape == (3, 3)
        assert x.dtype == np.float64
        assert cs.is_equal(x._sym, np.eye(3), 1)

    def test_sym_like(self):
        x = [1, 2, 3]
        y = arc.sym_like(x, name="y")
        assert isinstance(y, SymbolicArray)
        assert y.shape == (3,)
        assert y.dtype == int


class TestSymbolicArrayNotImplemented:
    def test_unsupported_ufunc(self):
        x = sym("x")
        with pytest.raises(TypeError):
            np.isnat(x)

        with pytest.raises(NotImplementedError):
            x.__array_ufunc__("sin", "something_besides_call", x)

    def test_unsupported_array_func(self):
        x = sym("x")
        assert x.__array_function__(np.digitize, None, None, None) is NotImplemented

    def test_invalid_repmat(self):
        x = "abc"
        with pytest.raises(NotImplementedError):
            arc._core._array_ops._array_ops._repmat(x, (3, 2))


class TestSymbolicArrayBroadcasting:
    # See broadcast_binary_op for details on the various "cases"

    def test_case1(self):
        "Both arrays are scalars"
        # Symbolic
        x = sym("x", shape=(), dtype=np.int32)
        y = sym("y", shape=(), dtype=np.int32)

        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == ()
            assert isinstance(result._sym, cs.SX)
            assert cs.is_equal(result._sym, x._sym * y._sym, 1)

        # Built-in int
        y = 2
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == x.dtype
            assert result.shape == ()
            assert isinstance(result._sym, cs.SX)
            assert cs.is_equal(result._sym, x._sym * y, 1)

        # Built-in float
        y = 2.0
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.float64
            assert result.shape == ()
            assert isinstance(result._sym, cs.SX)
            assert cs.is_equal(result._sym, x._sym * y, 1)

        # Numpy array
        y = np.array(2, dtype=np.int32)
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == ()
            assert isinstance(result._sym, cs.SX)
            assert cs.is_equal(result._sym, x._sym * y, 1)

    def test_case2(self):
        "Scalar * vector: () x (n,) => (n,)"
        # NOTE: Reversing order also tests Case 3: (n,) x () => (n,)

        # Symbolic
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(), dtype=np.int32)
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (3,)
            assert isinstance(result._sym, cs.SX)
            assert cs.is_equal(result._sym, x._sym * y._sym, 1)

        # Built-in int
        y = 2
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (3,)
            assert isinstance(result._sym, cs.SX)
            assert cs.is_equal(result._sym, x._sym * y, 1)

        # Built-in float
        y = 2.0
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.float64
            assert result.shape == (3,)
            assert isinstance(result._sym, cs.SX)
            assert cs.is_equal(result._sym, x._sym * y, 1)

        # Numpy array
        x = sym("x", shape=(), dtype=np.int32)
        y = np.array([1, 2, 3], dtype=np.int32)
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (3,)
            assert isinstance(result._sym, cs.SX)
            assert cs.is_equal(result._sym, x._sym * y, 1)

    def test_case4a(self):
        "Vector * vector: (n,) x (n,) => (n,)"
        # Symbolic
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(3,), dtype=np.int32)
        result = x * y
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)
        assert isinstance(result._sym, cs.SX)
        assert cs.is_equal(result._sym, x._sym * y._sym, 1)

        y = np.array([1, 2, 3], dtype=np.int32)
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (3,)
            assert isinstance(result._sym, cs.SX)
            assert cs.is_equal(result._sym, x._sym * y, 1)

    def test_case4b(self):
        "Vector * vector: (n,) x (m,) => raise error"
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(2,), dtype=np.int32)
        with pytest.raises(ValueError):
            x * y

    # Also tests case6_mat by virtue of rmul
    def test_case5a_mat(self):
        "Vector * matrix: (n,) x (p, q) => (p, n) with n == q"
        # Symbolic
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(2, 3), dtype=np.int32)
        result = x * y
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (2, 3)
        assert isinstance(result._sym, cs.SX)

        for i in range(y.shape[0]):
            result_row = x._sym * y._sym[i, :].T
            assert cs.is_equal(result._sym[i, :].T, result_row, 1)

        y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (2, 3)
            assert isinstance(result._sym, cs.SX)

            for i in range(y.shape[0]):
                result_row = x._sym * y[i, :].T
                assert cs.is_equal(result._sym[i, :].T, result_row, 1)

    # Also tests case5b_vec by virtue of mul
    def test_case5a_vec(self, array):
        "Vector * matrix: (n,) x (p, q) => (n, q) with q == 1"
        # Test by broadcasting column vector: (3,) * (4, 1) => (3, 4)"

        # Numpy array
        x = np.array([[1], [2], [3], [4]], dtype=np.int64)
        result = x * array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert result.shape == result._sym.shape
        assert result.shape == (4, 3)

        for i in range(len(x)):
            result_row = x[i] * array._sym
            assert cs.is_equal(result._sym[i, :].T, result_row, 1)

        # Reverse order
        x = np.array([[1], [2], [3], [4]], dtype=np.int64)
        result = array * x
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert result.shape == result._sym.shape
        assert result.shape == (4, 3)
        for i in range(len(x)):
            result_row = x[i] * array._sym
            assert cs.is_equal(result._sym[i, :].T, result_row, 1)

        # Symbolic arrays
        array2 = sym("y", (4, 1), dtype=np.int64)
        result = array2 * array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert result.shape == result._sym.shape
        assert result.shape == (4, 3)
        for i in range(array2.shape[0]):
            result_row = array2._sym[i] * array._sym
            assert cs.is_equal(result._sym[i, :].T, result_row, 1)

        # Reverse order
        array2 = sym("y", (4, 1), dtype=np.int64)
        result = array * array2
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert result.shape == result._sym.shape
        assert result.shape == (4, 3)
        for i in range(array2.shape[0]):
            result_row = array2._sym[i] * array._sym
            assert cs.is_equal(result._sym[i, :].T, result_row, 1)

    def test_case5b_mat(self):
        "Vector * matrix: (n,) x (p, q) => raise error with n != q"
        # Symbolic
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(2, 4), dtype=np.int32)
        with pytest.raises(ValueError):
            x * y

    def test_case6b_vec(self, array):
        "Vector * matrix: (n, m) x (p) => (n, p) with m == 1"
        # Test by broadcasting row vector: (1, 3) * (3,) => (1, 3)"
        # Numpy array
        result = np.array([[1, 2, 3]], dtype=np.int64) * array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert result.shape == result._sym.shape
        assert result.shape == (1, 3)
        assert cs.is_equal(result._sym, np.array([[1, 2, 3]]) * array._sym.T, 1)

        # Symbolic array
        array2 = sym("y", (1, 3), dtype=np.int64)
        result = array2 * array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert result.shape == result._sym.shape
        assert result.shape == (1, 3)
        assert cs.is_equal(result._sym, array2._sym * array._sym.T, 1)

    def test_case7a(self):
        "Matrix * matrix: (n, m) x (p, q) => (n, m) with n == p and m == q"
        # Symbolic
        x = sym("x", shape=(3, 2), dtype=np.int32)
        y = sym("y", shape=(3, 2), dtype=np.int32)
        result = x * y
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 2)
        assert isinstance(result._sym, cs.SX)
        assert cs.is_equal(result._sym, x._sym * y._sym, 1)

        # Numpy array
        y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (3, 2)
            assert isinstance(result._sym, cs.SX)
            assert cs.is_equal(result._sym, x._sym * y, 1)

    def test_case7b(self):
        "Matrix * matrix: (n, m) x (p, q) => (p, m) with n == 1 and q == m"
        # Symbolic
        x = sym("x", shape=(1, 2), dtype=np.int32)
        y = sym("y", shape=(3, 2), dtype=np.int32)
        result = y * x
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 2)
        assert isinstance(result._sym, cs.SX)

        for i in range(y.shape[0]):
            result_row = y._sym[i, :] * x._sym
            assert cs.is_equal(result._sym[i, :], result_row, 1)

        # Numpy array
        x = np.array([[1, 2]], dtype=np.int32)
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (3, 2)
            assert isinstance(result._sym, cs.SX)

            for i in range(y.shape[0]):
                result_row = y._sym[i, :] * x
                assert cs.is_equal(result._sym[i, :], result_row, 1)

        "Matrix * matrix: (n, m) x (p, q) => (p, m) with n == 1 and q == 1"
        # Symbolic
        x = sym("x", shape=(1, 2), dtype=np.int32)
        y = sym("y", shape=(3, 1), dtype=np.int32)
        result = y * x
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 2)
        assert isinstance(result._sym, cs.SX)

        for i in range(y.shape[0]):
            result_row = y._sym[i, :] * x._sym
            assert cs.is_equal(result._sym[i, :], result_row, 1)

        # Numpy array
        x = np.array([[1, 2]], dtype=np.int32)
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (3, 2)
            assert isinstance(result._sym, cs.SX)

            for i in range(y.shape[0]):
                result_row = y._sym[i, :] * x
                assert cs.is_equal(result._sym[i, :], result_row, 1)

        "Matrix * matrix: (n, m) x (p, q) => (n, q) with m == 1 and p == n"
        # Symbolic
        x = sym("x", shape=(3, 1), dtype=np.int32)
        y = sym("y", shape=(3, 2), dtype=np.int32)
        result = x * y
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 2)
        assert isinstance(result._sym, cs.SX)

        for i in range(y.shape[1]):
            result_col = x._sym * y._sym[:, i]
            assert cs.is_equal(result._sym[:, i], result_col, 1)

        # Numpy array
        y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        for result in (x * y, y * x):
            assert isinstance(result, SymbolicArray)
            assert result.dtype == np.int32
            assert result.shape == (3, 2)
            assert isinstance(result._sym, cs.SX)

            for i in range(y.shape[1]):
                result_col = x._sym * y[:, i]
                assert cs.is_equal(result._sym[:, i], result_col, 1)

    def test_case7d(self):
        "Matrix * matrix: (n, m) x (p, q) => raise error with n != p and m != q"
        # Symbolic
        x = sym("x", shape=(3, 2), dtype=np.int32)
        y = sym("y", shape=(4, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            x * y

    # Also covers Case 9 by virtue of rmul
    def test_case8(self):
        "Matrix * scalar: (n, m) x () => (n, m)"
        # Symbolic
        x = sym("x", shape=(3, 2), dtype=np.int32)
        y = sym("y", shape=(), dtype=np.int32)
        result = x * y
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 2)
        assert isinstance(result._sym, cs.SX)
        assert cs.is_equal(result._sym, x._sym * y._sym, 1)

    def test_invalid(self):
        x = sym("x", shape=(3, 2), dtype=np.int32)
        y = np.random.randn(4, 3, 2)
        with pytest.raises(NotImplementedError):
            x * y


class TestSymbolicArrayIndexing:
    def test_vec_index(self):
        # Test get
        x = sym("x", shape=(3,), dtype=np.int32)
        result = x[1]
        assert isinstance(result, SymbolicArray)
        assert result.shape == ()
        assert result.dtype == x.dtype
        assert cs.is_equal(result._sym, x._sym[1], 1)

        # Test set symbolic
        x = sym("x", shape=(3,), dtype=np.int32)
        y = sym("y", shape=(), dtype=np.int32)
        x[1] = y
        assert cs.is_equal(x._sym[1], y._sym, 1)

        # Test set numeric
        x = sym("x", shape=(3,), dtype=np.int32)
        x[1] = 2
        assert cs.is_equal(x._sym[1], 2, 1)

    def test_vec_slice(self):
        # Test get
        x = sym("x", shape=(3,), dtype=np.int32)
        result = x[:2]
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2,)
        assert result.dtype == x.dtype
        assert cs.is_equal(result._sym, x._sym[:2], 1)

        # Test set symbolic
        y = sym("y", shape=(2,), dtype=np.int32)
        x[:2] = y
        assert cs.is_equal(x._sym[:2], y._sym, 1)

        # Test set numeric
        y = np.array([1, 2], dtype=np.int32)
        x[:2] = y
        assert cs.is_equal(x._sym[:2], y, 1)

    def test_vec_slice_and_index(self):
        # The underlying CasADi array has shape (3, 1), so slicing it
        # this way is valid.  However, since the symbolic wrapper has
        # shape (3,), we expect the shape inference to fail.
        x = sym("x", shape=(3,), dtype=np.int32)
        with pytest.raises(IndexError):
            x[:2, 0]

    def test_vec_expand_dims(self):
        x = sym("x", shape=(), dtype=np.int32)
        result = x[None]
        assert isinstance(result, SymbolicArray)
        assert result.shape == (1,)
        assert result.dtype == x.dtype
        assert cs.is_equal(result._sym, x._sym, 1)

        x = sym("x", shape=(3,), dtype=np.int32)
        result = x[:2, None]
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2, 1)
        assert result.dtype == x.dtype
        assert cs.is_equal(result._sym, x._sym[:2, 0], 1)

        result = x[None, :2]
        assert isinstance(result, SymbolicArray)
        assert result.shape == (1, 2)
        assert result.dtype == x.dtype
        assert cs.is_equal(result._sym, x._sym[:2, 0].T, 1)

    def test_mat_index(self):
        # Test get
        x = sym("x", shape=(3, 2), dtype=np.int32)
        result = x[1, 1]
        assert isinstance(result, SymbolicArray)
        assert result.shape == ()
        assert result.dtype == x.dtype
        assert cs.is_equal(result._sym, x._sym[1, 1], 1)

        # Test set symbolic
        y = sym("y", shape=(), dtype=np.int32)
        x[1, 1] = y
        assert cs.is_equal(x._sym[1, 1], y._sym, 1)

        # Test set numeric
        x = sym("x", shape=(3, 2), dtype=np.int32)
        x[1, 1] = 2
        assert cs.is_equal(x._sym[1, 1], 2, 1)

    def test_mat_implicit_slice(self):
        # Test get
        x = sym("x", shape=(3, 2), dtype=np.int32)
        result = x[1]
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2,)
        assert result.dtype == x.dtype
        assert cs.is_equal(result._sym, x._sym[1, :].T, 1)

        # Test set symbolic
        x = sym("x", shape=(3, 2), dtype=np.int32)
        y = sym("y", shape=(2,), dtype=np.int32)
        x[1] = y
        assert cs.is_equal(x._sym[1, :], y._sym.T, 1)

        # Test set numeric
        x = sym("x", shape=(3, 2), dtype=np.int32)
        x[1] = np.array([1, 2], dtype=np.int32)
        assert cs.is_equal(x._sym[1, :], np.array([[1, 2]]), 1)

    def test_mat_slice(self):
        # Test get
        x = sym("x", shape=(3, 2), dtype=np.int32)
        result = x[:2, :]  # TODO: x[:2] will fail here - fix implicit slice
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2, 2)
        assert result._sym.shape == (2, 2)
        assert result.dtype == x.dtype
        assert cs.is_equal(result._sym, x._sym[:2, :], 1)

        # Test set symbolic
        y = sym("y", shape=(2, 2), dtype=np.int32)
        x[:2, :] = y
        assert cs.is_equal(x._sym[:2, :], y._sym, 1)

        # Test set numeric
        y = np.array([[1, 2], [3, 4]], dtype=np.int32)
        x[:2, :] = y
        assert cs.is_equal(x._sym[:2, :], y, 1)

    def test_mat_slice_and_index(self):
        # Test get
        x = sym("x", shape=(3, 2), dtype=np.int32)
        result = x[:2, 1]
        assert isinstance(result, SymbolicArray)
        assert result.shape == (2,)
        assert result.dtype == x.dtype
        assert cs.is_equal(result._sym, x._sym[:2, 1], 1)

        # Test set symbolic
        y = sym("y", shape=(2,), dtype=np.int32)
        x[:2, 1] = y
        assert cs.is_equal(x._sym[:2, 1], y._sym, 1)

        # Test set numeric
        y = np.array([1, 2], dtype=np.int32)
        x[:2, 1] = y
        assert cs.is_equal(x._sym[:2, 1], y, 1)


class TestSymbolicArrayIterator:
    def test_vec_iter(self, array):
        x0, x1, x2 = array
        for i, x in enumerate(array):
            assert cs.is_equal(x._sym, array[i]._sym)


class TestSymbolicArrayArithmetic:
    def test_rmul(self, array):
        # Scalar int
        result = 2 * array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, 2 * array._sym, 1)

        # Scalar float
        result = 2.0 * array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, 2.0 * array._sym, 1)

        # Array float
        result = np.array([1, 2, 3], dtype=np.int64) * array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, np.array([1, 2, 3]) * array._sym, 1)

        # Symbolic array
        result = array * array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, array._sym * array._sym, 1)

    def test_mul(self, array):
        # Scalar int
        result = array * 2
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, array._sym * 2, 1)

        # Scalar float
        result = array * 2.0
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, array._sym * 2.0, 1)

        # Array float
        result = array * np.array([1, 2, 3], dtype=np.int64)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert cs.is_equal(result._sym, array._sym * np.array([1, 2, 3]), 1)

        # Symbolic array
        result = array * array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, array._sym * array._sym, 1)

    def test_add(self, array):
        # Scalar int
        result = array + 2
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, array._sym + 2, 1)

        # Simplification
        result = array + 0
        result.simplify()
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, array._sym, 1)

        # Scalar float
        result = array + 2.0
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, array._sym + 2.0, 1)

        # Array float
        result = array + np.array([1, 2, 3], dtype=np.int64)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert cs.is_equal(result._sym, array._sym + np.array([1, 2, 3]), 1)

        # Symbolic array
        result = array + array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, array._sym + array._sym, 1)

    def test_radd(self, array):
        # Scalar int
        result = 2 + array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, 2 + array._sym, 1)

        # Scalar float
        result = 2.0 + array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, 2.0 + array._sym, 1)

        # Array float
        result = np.array([1, 2, 3], dtype=np.int64) + array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert cs.is_equal(result._sym, np.array([1, 2, 3]) + array._sym, 1)

        # Symbolic array
        result = array + array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, array._sym + array._sym, 1)

    def test_sub(self, array):
        # Scalar int
        result = array - 2
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, array._sym - 2, 1)

        # Scalar float
        result = array - 2.0
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, array._sym - 2.0, 1)

        # Array float
        result = array - np.array([1, 2, 3], dtype=np.int64)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert cs.is_equal(result._sym, array._sym - np.array([1, 2, 3]), 1)

        # Symbolic array
        result = array - array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, array._sym - array._sym, 1)

    def test_rsub(self, array):
        # Scalar int
        result = 2 - array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, 2 - array._sym, 1)

        # Scalar float
        result = 2.0 - array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, 2.0 - array._sym, 1)

        # Array float
        result = np.array([1, 2, 3], dtype=np.int64) - array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert cs.is_equal(result._sym, np.array([1, 2, 3]) - array._sym, 1)

        # Symbolic array
        result = array - array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, array._sym - array._sym, 1)

    def test_truediv(self, array):
        # Scalar int
        result = array / 2
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64  # Promoted to float
        assert cs.is_equal(result._sym, array._sym / 2, 1)

        # Scalar float
        result = array / 2.0
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, array._sym / 2.0, 1)

        # Array float
        result = array / np.array([1, 2, 3], dtype=np.int64)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, array._sym / np.array([1, 2, 3]), 1)

        # Symbolic array
        result = array / array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, array._sym / array._sym, 1)

    def test_rtruediv(self, array):
        # Scalar int
        result = 2 / array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64  # Promoted to float
        assert cs.is_equal(result._sym, 2 / array._sym, 1)

        # Scalar float
        result = 2.0 / array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, 2.0 / array._sym, 1)

        # Array float
        result = np.array([1, 2, 3], dtype=np.int64) / array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, np.array([1, 2, 3]) / array._sym, 1)

    def test_neg(self, array):
        result = -array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, -array._sym, 1)

    def test_abs(self, array):
        result = abs(array)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, cs.fabs(array._sym), 1)

    def test_pow(self, array):
        # Integer power
        result = array**2
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, array._sym**2, 1)

        # Float power
        result = array**2.0
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float64
        assert cs.is_equal(result._sym, array._sym**2.0, 1)

        # Array power
        result = array ** np.array([1, 2, 3], dtype=np.int64)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert cs.is_equal(result._sym, array._sym ** np.array([1, 2, 3]), 2)

    def test_rpow(self, array):
        result = 2**array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert cs.is_equal(result._sym, 2**array._sym, 1)

    def test_mod(self, array):
        # Direct call function with type cast
        result = np.mod(array, 2, dtype=np.float32)  # Would normally be int32
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float32
        assert cs.is_equal(result._sym, cs.remainder(array._sym, 2), 2)

        # Modulo operator with int
        result2 = array % 2
        assert isinstance(result2, SymbolicArray)
        assert result2.dtype == array.dtype
        assert cs.is_equal(result2._sym, result._sym, 2)

        # Modulo operator with float (should promote)
        result2 = array % 2.0
        assert isinstance(result2, SymbolicArray)
        assert result2.dtype == np.float64
        assert cs.is_equal(result2._sym, result._sym, 2)

        # Modulo operator with array
        x = np.array([1, 2, 3], dtype=np.int64)
        result = array % x
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert cs.is_equal(result._sym, cs.remainder(array._sym, x), 2)

    def test_rmod(self, array):
        # Direct call function
        result = np.mod(2, array, dtype=np.float32)  # Would normally be int32
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float32
        assert cs.is_equal(result._sym, cs.remainder(2, array._sym), 2)

        # Modulo operator with int
        result2 = 2 % array
        assert isinstance(result2, SymbolicArray)
        assert result2.dtype == array.dtype
        assert cs.is_equal(result2._sym, result._sym, 2)

        # Modulo operator with float (should promote)
        result2 = 2.0 % array
        assert isinstance(result2, SymbolicArray)
        assert result2.dtype == np.float64
        assert cs.is_equal(result2._sym, result._sym, 2)

        # Modulo operator with array
        x = np.array([1, 2, 3], dtype=np.int64)
        result = x % array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert cs.is_equal(result._sym, cs.remainder(x, array._sym), 2)

    def test_floordiv(self, array):
        # Direct call function
        result = np.floor_divide(array, 2, dtype=np.float32)  # Would normally be int32
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float32
        assert cs.is_equal(result._sym, cs.floor(array._sym / 2), 2)

        # Floor division operator with int
        result2 = array // 2
        assert isinstance(result2, SymbolicArray)
        assert result2.dtype == array.dtype
        assert cs.is_equal(result2._sym, result._sym, 2)

        # Floor division operator with float (should promote)
        result2 = array // 2.0
        assert isinstance(result2, SymbolicArray)
        assert result2.dtype == np.float64
        assert cs.is_equal(result2._sym, result._sym, 2)

        # Floor division operator with array
        x = np.array([1, 2, 3], dtype=np.int64)
        result = array // x
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert cs.is_equal(result._sym, cs.floor(array._sym / x), 2)

    def test_rfloordiv(self, array):
        # Direct call function
        result = np.floor_divide(2, array, dtype=np.float32)  # Would normally be int32
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.float32
        assert cs.is_equal(result._sym, cs.floor(2 / array._sym), 2)

        # Floor division operator with int
        result2 = 2 // array
        assert isinstance(result2, SymbolicArray)
        assert result2.dtype == array.dtype
        assert cs.is_equal(result2._sym, result._sym, 2)

        # Floor division operator with float (should promote)
        result2 = 2.0 // array
        assert isinstance(result2, SymbolicArray)
        assert result2.dtype == np.float64
        assert cs.is_equal(result2._sym, result._sym, 2)

        # Floor division operator with array
        x = np.array([1, 2, 3], dtype=np.int64)
        result = x // array
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int64
        assert cs.is_equal(result._sym, cs.floor(x / array._sym), 2)

    def test_divmod(self, array):
        x = 2
        result1, result2 = divmod(array, x)
        assert isinstance(result1, SymbolicArray)
        assert isinstance(result2, SymbolicArray)
        assert cs.is_equal(result1._sym, cs.floor(array._sym / x), 2)
        assert cs.is_equal(result2._sym, cs.remainder(array._sym, x), 2)

    def test_rdivmod(self, array):
        x = 2
        result1, result2 = divmod(x, array)
        assert isinstance(result1, SymbolicArray)
        assert isinstance(result2, SymbolicArray)
        assert cs.is_equal(result1._sym, cs.floor(x / array._sym), 2)
        assert cs.is_equal(result2._sym, cs.remainder(x, array._sym), 2)

    def test_comparison(self, array):
        x = 2

        # greater
        result = array > x
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.gt(array._sym, x), 2)
        assert result.dtype == bool

        # greater_equal
        result = array >= x
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.ge(array._sym, x), 2)
        assert result.dtype == bool

        # less
        result = array < x
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.lt(array._sym, x), 2)
        assert result.dtype == bool

        # less_equal
        result = array <= x
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.le(array._sym, x), 2)
        assert result.dtype == bool

        # equal
        result = array == x
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.eq(array._sym, x), 2)
        assert result.dtype == bool

        # not_equal
        result = array != x
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.ne(array._sym, x), 2)
        assert result.dtype == bool

    def test_logic(self):
        a, b = sym("a"), True

        # and
        result = a & b
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.logic_and(a._sym, b), 2)
        assert result.dtype == bool

        # rand
        result = b & a
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.logic_and(a._sym, b), 2)
        assert result.dtype == bool

        # or
        result = a | b
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.logic_or(a._sym, b), 2)
        assert result.dtype == bool

        # ror
        result = b | a
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.logic_or(a._sym, b), 2)
        assert result.dtype == bool

        # xor
        result = a ^ b
        # (a and not b) or (not a and b)
        expected = cs.logic_or(
            cs.logic_and(a._sym, cs.logic_not(b)), cs.logic_and(cs.logic_not(a._sym), b)
        )
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, expected, 3)
        assert result.dtype == bool

        # rxor
        result = b ^ a
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, expected, 3)
        assert result.dtype == bool

        # not
        result = ~a
        assert isinstance(result, SymbolicArray)
        assert cs.is_equal(result._sym, cs.logic_not(a._sym), 2)
        assert result.dtype == bool

    @pytest.mark.parametrize("dtype", [np.float64, np.int32])
    def test_matmul(self, dtype):
        # Numeric-symbolic product
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype)
        y = sym("y", (2,), dtype=np.int32)
        result = x @ y
        assert isinstance(result, SymbolicArray)
        assert result.dtype == dtype
        assert result.shape == (x.shape[0],)
        assert cs.is_equal(result._sym, x @ y._sym, 2)

        # Symbolic-numeric product
        x = sym("x", (3, 2), dtype=dtype)
        y = np.array([1, 2], dtype=np.int32)
        result = x @ y
        assert isinstance(result, SymbolicArray)
        assert result.dtype == dtype
        assert result.shape == (x.shape[0],)
        assert cs.is_equal(result._sym, x._sym @ y, 2)

        # Symbolic-symbolic product
        x = sym("x", (3, 2), dtype=np.int32)
        y = sym("y", (2,), dtype=dtype)
        result = x @ y
        assert isinstance(result, SymbolicArray)
        assert result.dtype == dtype
        assert result.shape == (x.shape[0],)
        assert cs.is_equal(result._sym, x._sym @ y._sym, 2)

        # Invalid shapes
        with pytest.raises(ValueError):
            result = x.T @ y

        # Test product with scalar
        x = sym("x", (3, 1), dtype=np.int32)
        y = sym("y", (1,), dtype=dtype)
        result = x @ y
        assert isinstance(result, SymbolicArray)
        assert result.dtype == dtype
        assert result.shape == (x.shape[0],)
        assert cs.is_equal(result._sym, x._sym @ y._sym, 2)

        y = sym("y", (1, 1), dtype=dtype)
        result = x @ y
        assert isinstance(result, SymbolicArray)
        assert result.dtype == dtype
        assert result.shape == (x.shape[0], 1)
        assert cs.is_equal(result._sym, x._sym @ y._sym, 2)

        y = sym("y", (), dtype=dtype)
        with pytest.raises(ValueError):
            result = x @ y

        with pytest.raises(ValueError):
            result = y @ x

    def test_max(self):
        x = sym("x", (3,), dtype=np.int32)
        result = np.max(x)
        assert isinstance(result, SymbolicArray)
        assert result.shape == ()
        assert cs.is_equal(result._sym, cs.mmax(x._sym), 3)

        # Unsupported axis specification
        with pytest.raises(NotImplementedError):
            np.max(x, axis=0)

    def test_min(self):
        x = sym("x", (3,), dtype=np.int32)
        result = np.min(x)
        assert isinstance(result, SymbolicArray)
        assert result.shape == ()
        assert cs.is_equal(result._sym, cs.mmin(x._sym), 3)

        # Unsupported axis specification
        with pytest.raises(NotImplementedError):
            np.min(x, axis=0)

    def test_diag(self):
        # Vector to diagonal
        x = sym("x", (3,), dtype=np.int32)
        result = np.diag(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3, 3)
        assert cs.is_equal(result._sym, cs.diag(x._sym), 1)

        # Matrix to diagonal
        x = sym("x", (3, 3), dtype=np.int32)
        result = np.diag(x)
        assert isinstance(result, SymbolicArray)
        assert result.dtype == np.int32
        assert result.shape == (3,)
        assert cs.is_equal(result._sym, cs.diag(x._sym), 1)


class TestSymbolicArrayDifferentiation:
    def test_grad(self):
        for shape in [(), (3,), (3, 1)]:
            x = sym("x", shape=shape, dtype=np.float64)
            f = 0.5 * np.dot(x.T, x)
            df = f.grad(x)
            assert isinstance(df, SymbolicArray)
            assert df.dtype == x.dtype
            assert df.shape == x.shape
            assert cs.is_equal(cs.simplify(df._sym), x._sym, 2)

            # Test with different vectors: d/dx (y^T x) = y
            y = sym("y", shape=shape, dtype=np.float64)
            f = np.dot(y.T, x)
            df = f.grad(x)
            assert cs.is_equal(df._sym, y._sym, 2)

        # Invalid shape of x
        x = sym("x", shape=(3,))
        with pytest.raises(ShapeDtypeError):
            x.grad(x)

    def test_jac(self):
        dtype = np.float64
        m, n = 3, 2

        A_sym = sym("A", shape=(m, n), dtype=dtype)
        A_np = np.random.rand(m, n)
        x = sym("x", shape=(n,), dtype=dtype)
        for A in (A_sym, A_np):
            f = A @ x
            J = f.jac(x)
            assert isinstance(J, SymbolicArray)
            assert J.dtype == dtype
            assert J.shape == A.shape

            if isinstance(A, SymbolicArray):
                assert cs.is_equal(J._sym, A._sym, 1)
            else:
                assert cs.is_equal(J._sym, A, 1)

        # Error handling
        with pytest.raises(ShapeDtypeError):
            A_sym.jac(x)

        with pytest.raises(ShapeDtypeError):
            x.jac(A_sym)

        # Test edge case for shape inference: vector expression, scalar argument
        f = A_sym @ x
        df = f.jac(x[0])
        assert isinstance(df, SymbolicArray)
        assert df.shape == (m,)
        assert cs.is_equal(df._sym, A_sym._sym[:, 0], 1)

    def test_hess(self):
        dtype = np.float64
        n = 3
        Q_sym = sym("Q", shape=(n, n), dtype=dtype)
        Q_np = np.random.rand(n, n)
        x = sym("x", shape=(n,), dtype=dtype)

        for Q in (Q_sym, Q_np):
            Q = 0.5 + (Q.T + Q)  # Symmetrize
            f = 0.5 * x.T @ Q @ x
            H = f.hess(x)
            if isinstance(Q, SymbolicArray):
                assert cs.is_equal(
                    cs.simplify(H._sym),
                    cs.simplify(Q._sym),
                    2,
                )
            else:
                assert cs.is_equal(H._sym, Q, 1)

        # Error handling for incorrect shapes
        with pytest.raises(ShapeDtypeError):
            Q_sym.hess(x)

        f = 0.5 * x.T @ Q_sym @ x
        with pytest.raises(ShapeDtypeError):
            f.hess(Q_sym)

    def test_jvp(self):
        # For f(x) = A @ x, f'(x) v = A @ v
        dtype = np.float64
        n, m = 3, 2
        A = sym("A", shape=(m, n), dtype=dtype)
        x = sym("x", shape=(n,), dtype=dtype)
        v = sym("v", shape=(n,), dtype=dtype)
        f = A @ x
        Jv = f.jvp(x, v)
        assert isinstance(Jv, SymbolicArray)
        assert Jv.dtype == dtype
        assert Jv.shape == (m,)
        assert cs.is_equal(
            cs.simplify(Jv._sym),
            cs.simplify(A._sym @ v._sym),
            3,
        )

        # Error handling for incorrect shapes
        with pytest.raises(ShapeDtypeError):
            f.jvp(A, v)

        with pytest.raises(ShapeDtypeError):
            A.jvp(x, v)

    def test_vjp(self):
        # For f(x) = A @ x, f'(x)^T w = A^T w
        dtype = np.float64
        n, m = 3, 2
        A = sym("A", shape=(m, n), dtype=dtype)
        x = sym("x", shape=(n,), dtype=dtype)
        w = sym("w", shape=(m,), dtype=dtype)
        f = A @ x
        JTw = f.vjp(x, w)
        assert isinstance(JTw, SymbolicArray)
        assert JTw.dtype == dtype
        assert JTw.shape == (n,)
        assert cs.is_equal(
            cs.simplify(JTw._sym),
            cs.simplify(A.T._sym @ w._sym),
            3,
        )

        # Error handling for incorrect shapes
        with pytest.raises(ShapeDtypeError):
            f.vjp(A, w)

        with pytest.raises(ShapeDtypeError):
            A.vjp(x, w)
