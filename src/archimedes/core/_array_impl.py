from __future__ import annotations
import numpy as np
import casadi as cs


DEFAULT_FLOAT = np.float64

from ._type_inference import shape_inference, type_inference


SYM_KINDS = {"MX": cs.MX, "SX": cs.SX, "DM": cs.DM, "ndarray": np.ndarray}
SYM_NAMES = {cs.MX: "MX", cs.SX: "SX", cs.DM: "DM", np.ndarray: "ndarray"}


def casadi_array(x):
    """Convert to a CasADi type (SX, MX, or ndarray)"""
    if isinstance(x, SymbolicArray):
        return x._sym
    # Commented out for codecov... can add back in if these turn out to be necessary
    # elif isinstance(x, (cs.SX, cs.MX)):
    #     return x
    # elif isinstance(x, (tuple, list)):  # Functions with sequence inputs
    #     return [casadi_array(xi) for xi in x]
    else:
        return np.asarray(x)


class SymbolicArray:
    def __init__(self, sx: cs.SX | cs.MX, dtype=None, shape=None):
        # Occasionally CasADi operations will return NotImplemented instead
        # of throwing an error. Ideally we would be able to provide a more
        # helpful error message, but at the very least this should be caught
        # immediately.
        if sx is NotImplemented:
            raise ValueError("SymbolicArray cannot be initialized with NotImplemented")

        if dtype is None:
            dtype = DEFAULT_FLOAT
        self.dtype = np.dtype(dtype)

        if shape is None:
            shape = sx.shape
        self.shape = shape

        # Consistent handling of vector shapes
        if len(shape) == 1 and sx.shape[0] == 1:
            sx = sx.T
        elif len(shape) == 2:
            sx = cs.reshape(sx, *shape)

        self._sym = sx
        self.kind = SYM_NAMES[type(sx)]

    def __repr__(self):
        return f"{self._sym}"

    def __iter__(self):
        if len(self.shape) == 0:
            yield self

        else:
            for i in range(self.shape[0]):
                yield self[i]

    def __getitem__(self, index):
        # This relies on using numpy's indexing and slicing machinery to do
        # shape inference and then applying the same indexing and slicing to
        # the symbolic array.  Probably there will be edge cases where some
        # preprocessing needs to be done on the index before passing it to
        # the symbolic array.  Known issues:
        # - CasADi won't recognize the idiom x[:, None] as a way of adding a
        #   new dimension to x.

        # If a 2D array is indexed with only one index, by default CasADi assumes
        # that the missing index should be 0, whereas NumPy assumes it should be
        # slice(None)
        if len(self.shape) == 2 and not isinstance(index, tuple):
            index = (index, slice(None))

        # Do this before handling the expand_dims cases, because it will make
        # the indices correspond to CasADi rather than numpy.
        result_shape = np.empty(self.shape)[index].shape

        # Since all CasADi SX objects are 2D, CasADi doesn't recognize the idioms
        # x[:, None] or x[None, :] as a way of adding a new dimension to x, so we
        # need to do it manually.
        if index is None or (isinstance(index, tuple) and None in index):
            # Cases:
            # - self.shape = () and index = (None,): result_shape = (1,)
            # - self.shape = () and index = (None, None): result_shape = (1, 1)
            # - self.shape = (n,) and index = (None, idx): result_shape = (1, n)
            # - self.shape = (n,) and index = (idx, None): result_shape = (n, 1)
            if self.shape == ():
                index = (slice(None), slice(None))
            elif len(self.shape) == 1:
                # Indexing a (n,) array with (None,) will return a (1, n) array
                if index == (None,):
                    index = (slice(None), slice(None))
                elif index[0] is None:
                    # The underlying symbolic array will have shape (self.shape[0], 1)
                    # so the index needs to be transposed.
                    index = (index[1], slice(None))
                elif index[1] is None:
                    index = (index[0], slice(None))

        return SymbolicArray(self._sym[index], dtype=self.dtype, shape=result_shape)

    def __setitem__(self, index, value):
        # This assumes that whatever is passed in as `index` will work with
        # CasADi's indexing and slicing machinery.  Probably there will be
        # edge cases where some preprocessing needs to be done on the index
        # before passing it to the underlying symbolic array.
        value = casadi_array(value)  # Make sure it's either SX or ndarray

        # If a 2D array is indexed with only one index, by default CasADi assumes
        # that the missing index should be 0, whereas NumPy assumes it should be
        # slice(None, None, None)
        if len(self.shape) == 2 and not isinstance(index, tuple):
            index = (index, slice(None))

            # Now the problem is that if `value`` is a CasADi SX, then its shape will
            # be (:, index), while `self._sym[index]` will be (index, :).
            value = value.reshape(self._sym[index].shape)

        self._sym[index] = value

    def __len__(self):
        if len(self.shape) == 0:
            return 0
        return self.shape[0]
    
    @property
    def ndim(self):
        return len(self.shape)
    
    @property
    def size(self):
        return int(np.prod(self.shape))

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return np.divide(self, other)
    
    def __rtruediv__(self, other):
        return np.divide(other, self)

    def __pow__(self, other):
        return np.power(self, other)

    def __rpow__(self, other):
        return np.power(other, self)

    def __mod__(self, other):
        return np.mod(self, other)

    def __rmod__(self, other):
        return np.mod(other, self)

    def __floordiv__(self, other):
        return np.floor_divide(self, other)
    
    def __rfloordiv__(self, other):
        return np.floor_divide(other, self)
    
    def __divmod__(self, other):
        return np.divmod(self, other)
    
    def __rdivmod__(self, other):
        return np.divmod(other, self)

    def __neg__(self):
        return np.negative(self)

    def __abs__(self):
        return np.fabs(self)

    def __matmul__(self, other):
        return np.matmul(self, other)
    
    def __gt__(self, other):
        return np.greater(self, other)
    
    def __ge__(self, other):
        return np.greater_equal(self, other)
    
    def __lt__(self, other):
        return np.less(self, other)
    
    def __le__(self, other):
        return np.less_equal(self, other)
    
    def __eq__(self, other):
        return np.equal(self, other)
    
    def __ne__(self, other):
        return np.not_equal(self, other)
    
    def __and__(self, other):
        return np.logical_and(self, other)
    
    def __rand__(self, other):
        return np.logical_and(other, self)
    
    def __or__(self, other):
        return np.logical_or(self, other)
    
    def __ror__(self, other):
        return np.logical_or(other, self)
    
    def __xor__(self, other):
        return np.logical_xor(self, other)
    
    def __rxor__(self, other):
        return np.logical_xor(other, self)
    
    def __invert__(self):
        return np.logical_not(self)

    @property
    def T(self):
        return np.transpose(self)

    def simplify(self):
        return SymbolicArray(cs.simplify(self._sym), dtype=self.dtype, shape=self.shape)
    
    #
    # Other common NumPy array methods
    #
    def flatten(self, order='C'):
        return np.ravel(self, order=order)

    def ravel(self, order='C'):
        return np.ravel(self, order=order)

    def squeeze(self, axis=None):
        return np.squeeze(self, axis=axis)

    def reshape(self, shape, order='C'):
        return np.reshape(self, shape, order=order)
    
    def astype(self, dtype):
        return np.astype(self, dtype)

    #
    # Autodiff operations not supported by NumPy
    #
    def grad(self, x: SymbolicArray):
        dtype = type_inference("default", self, x)
        shape = shape_inference("gradient", self, x)
        return SymbolicArray(cs.gradient(self._sym, x._sym), dtype=dtype, shape=shape)

    def jac(self, x: SymbolicArray):
        dtype = type_inference("default", self, x)
        shape = shape_inference("jacobian", self, x)
        return SymbolicArray(cs.jacobian(self._sym, x._sym), dtype=dtype, shape=shape)


    def hess(self, x: SymbolicArray):
        dtype = type_inference("default", self, x)
        shape = shape_inference("hessian", self, x)
        H, _g = cs.hessian(self._sym, x._sym)
        return SymbolicArray(H, dtype=dtype, shape=shape)

    def jvp(self, x: SymbolicArray, v: SymbolicArray):
        dtype = type_inference("default", self, x, v)
        shape = shape_inference("jvp", self, x, v)
        return SymbolicArray(cs.jtimes(self._sym, x._sym, v._sym), dtype=dtype, shape=shape)

    def vjp(self, x: SymbolicArray, v: SymbolicArray):
        dtype = type_inference("default", self, x, v)
        shape = shape_inference("vjp", self, x, v)
        return SymbolicArray(cs.jtimes(self._sym, x._sym, v._sym, True), dtype=dtype, shape=shape)


ArrayLike = np.ndarray | SymbolicArray