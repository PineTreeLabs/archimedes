from __future__ import annotations
import numpy as np
import casadi as cs


DEFAULT_FLOAT = np.float64

from ._type_inference import shape_inference, type_inference


SYM_KINDS = {"MX": cs.MX, "SX": cs.SX, "DM": cs.DM, "ndarray": np.ndarray}
SYM_NAMES = {cs.MX: "MX", cs.SX: "SX", cs.DM: "DM", np.ndarray: "ndarray"}


def _as_casadi_array(x):
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
        value = _as_casadi_array(value)  # Make sure it's either SX or ndarray

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

#
# Factory functions for constructing SymbolicArrays
#

def sym(name, shape=None, dtype=np.float64, kind="SX") -> SymbolicArray:
    """Create a symbolic array.
    
    Parameters
    ----------
    name : str
        Name of the symbolic variable.
    shape : tuple of int, optional
        Shape of the array. Default is ().
    dtype : numpy.dtype, optional
        Data type of the array. Default is np.float64.
    kind : str, optional
        Kind of the symbolic variable (`"SX"` or `"MX"`). Default is `"SX"`.
        See CasADi documentation for details on the differences between the two.

    Returns
    -------
    SymbolicArray
        Symbolic array with the given name, shape, dtype, and kind.
    """
    # TODO: Use `scalar: bool` instead of `kind: str`
    if shape is None:
        shape = ()
    if isinstance(shape, int):
        shape = (shape,)
    assert (
        isinstance(shape, tuple) and len(shape) >= 0
    ), "shape must be a tuple of length >= 0"
    assert all(isinstance(s, int) for s in shape), "shape must be a tuple of ints"
    assert len(shape) <= 2, "Only scalars, vectors, and matrices are supported for now"
    # Note that CasADi creates variables in column-major order, so to be consistent
    # with NumPy we have to reverse the shape and transpose the result.  This only
    # applies to SX, since for MX it's just a single symbol anyway.
    if kind == "SX":
        _sym = cs.SX.sym(name, *shape[::-1]).T
    elif kind == "MX":
        _sym = cs.MX.sym(name, *shape)
    else:
        raise ValueError(f"Unknown symbolic kind {kind}")
    return SymbolicArray(_sym, dtype=dtype, shape=shape)


def sym_like(x, name, dtype=None, kind="SX") -> SymbolicArray:
    """Create a symbolic array similar to an existing array
    
    Parameters
    ----------
    x : array_like
        Array to copy shape and dtype from.
    name : str
        Name of the symbolic variable.
    dtype : numpy.dtype, optional
        Data type of the array. Default is the dtype of `x`.
    kind : str, optional
        Kind of the symbolic variable (`"SX"` or `"MX"`). Default is `"SX"`.
        See CasADi documentation for details on the differences between the two.

    Returns
    -------
    SymbolicArray
        Symbolic array with the given name and kind, and with the shape of `x`.
    """
    if not isinstance(x, (np.ndarray, SymbolicArray)):
        x = np.asarray(x)
    return sym(name, x.shape, dtype=dtype or x.dtype, kind=kind)



def array(x, dtype=None) -> ArrayLike:
    """Create an array.
    
    Parameters
    ----------
    x : array_like
        An array-like object, either a NumPy `ndarray`, a `SymbolicArray`,
        a CasADi symbolic type (not recommended to use directly), or a list or
        list-of-lists of scalars.
    dtype : str, optional
        The dtype of the array. If not specified, the dtype is inferred from `x`.

    Returns
    -------
    array : numpy.ndarray | SymbolicArray

    Notes
    -----
    Array creation using the NumPy dispatch mechanism (`np.array(..., like=...)`) is
    recommended over calling `array(...)` directly.  This supports a wider range of
    input types and should better support numerical input types.
    """

    # Case 1. x is already an array
    if isinstance(x, (SymbolicArray, np.ndarray)):
        if dtype is not None:
            x = x.astype(dtype)
        return x
    
    if isinstance(x, (cs.SX, cs.MX)):
        return SymbolicArray(x, dtype=dtype)

    if (np.isscalar(x) and np.isreal(x)) or isinstance(x, cs.DM):
        return np.array(x, dtype=dtype)

    return _dispatch_array(x, dtype=dtype)


def _empty_like(x):
    if isinstance(x, SymbolicArray):
        return np.empty(x.shape, dtype=x.dtype)
    else:
        return x


def _result_type(*inputs):
    np_inputs = tuple(map(_empty_like, inputs))
    return np.result_type(*np_inputs)


def _is_list_of_scalars(x):
    return all(isinstance(xi, (int, float)) or len(xi) <= 1 for xi in x)



def _dispatch_array(x, dtype=None) -> ArrayLike:
    """`array` function dispatched from np.array(..., like=[SymbolicArray])"""
    # For now, support three cases:
    # 1. x is already an array (do nothing)
    # 2. x is a list of scalars (convert to an (n,) array)
    # 3. x is a list of lists (convert to an (n, m) array)

    # Case 1: x is already an array
    if isinstance(x, SymbolicArray):
        if dtype is not None:
            x = x.astype(dtype)
        return x

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.array(x, dtype=dtype)

        # Case 2. x is a list of scalars
        if not all(isinstance(xi, (list, tuple, np.ndarray, SymbolicArray)) for xi in x):
            # Check that everything is a scalar
            if not _is_list_of_scalars(x):
                raise ValueError(f"Creating array with inconsistent data: {x}")
            result_shape = (len(x),)
            result_dtype = dtype or _result_type(*x)
            cs_x = list(map(_as_casadi_array, x))
            arr = cs.vcat(cs_x)
            if isinstance(arr, cs.DM):
                return np.array(arr, dtype=result_dtype).reshape(result_shape)
            return SymbolicArray(arr, dtype=result_dtype, shape=result_shape)

        # Case 3. x is a list of lists, arrays, etc
        # check that all lists are only scalars
        if not all(
            _is_list_of_scalars(xi) for xi in x
        ):
            raise ValueError(
                "Can only create array from list of scalars or list of "
                "lists of scalars"
            )

        # Check that the lengths are consistent
        len_set = set(map(len, x))
        if len(len_set) != 1:
            raise ValueError(f"Inconsistent lengths in list of lists: {len_set}")

        result_shape = (len(x),) if len(x[0]) == 0 else (len(x), len(x[0]))
        result_dtype = dtype or _result_type(*[_result_type(*xi) for xi in x])
        cs_x = [list(map(_as_casadi_array, xi)) for xi in x]

        arr = cs.vcat([cs.hcat(xi) for xi in cs_x])
        if isinstance(arr, cs.DM):
            return np.array(arr, dtype=result_dtype).reshape(result_shape)
        return SymbolicArray(
            arr,
            dtype=result_dtype,
            shape=result_shape,
        )

    raise NotImplementedError(f"Converting {x} (type={type(x)}) to array is not supported")


def _np_shape(shape):
    # Check that the shape is valid and return a tuple of ints.
    if isinstance(shape, int):
        return (shape,)
    if not isinstance(shape, tuple):
        raise ValueError("shape must be an int or a tuple of ints")
    if not all(isinstance(s, int) for s in shape):
        raise ValueError("shape must be a tuple of ints")
    return shape



# zeros, ones, zeros_like, eye, diag
def _cs_shape(shape):
    # The shape of the CasADi object is always 2D, so we need to handle the
    # cases where the specified shape is () or (n,) separately.
    cs_shape = _np_shape(shape)

    if len(cs_shape) > 2:
        raise ValueError("Only scalars, vectors, and matrices are supported for now")

    if len(cs_shape) == 0:
        cs_shape = (1, 1)

    elif len(cs_shape) == 1:
        cs_shape = (cs_shape[0], 1)

    return cs_shape


def zeros(shape, dtype=np.float64, sparse=True, kind="SX") -> SymbolicArray:
    """Construct an array of zeros with the given shape and dtype.
    
    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the array.
    dtype : numpy.dtype, optional
        Data type of the array. Default is np.float64.
    sparse : bool, optional
        If `True`, then the array will be a sparse array with "structural"
        zeros.  Otherwise it will be "dense" and full of numerical zeros.
    kind : str, optional
        Kind of the symbolic variable (`"SX"` or `"MX"`). Default is `"SX"`.
        See CasADi documentation for details on the differences between the two.

    Returns
    -------
    SymbolicArray
        Array of zeros with the given shape, dtype, and symbolic kind.

    Notes
    -----
    Prefer using `np.zeros_like` or `np.zeros(..., like=SymbolicArray)` to directly calling
    this function where possible, since this may handle dispatch to numeric arrays slightly
    better.
    """
    X = SYM_KINDS[kind]
    _zeros = X if sparse else X.zeros
    return SymbolicArray(_zeros(*_cs_shape(shape)), dtype=dtype, shape=_np_shape(shape))


def ones(shape, dtype=np.float64, kind="SX"):
    """Construct an array of ones with the given shape and dtype.
    
    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the array.
    dtype : numpy.dtype, optional
        Data type of the array. Default is np.float64.
    kind : str, optional
        Kind of the symbolic variable (`"SX"` or `"MX"`). Default is `"SX"`.
        See CasADi documentation for details on the differences between the two.

    Returns
    -------
    SymbolicArray
        Array of ones with the given shape, dtype, and symbolic kind.

    Notes
    -----
    Prefer using `np.ones_like` or `np.ones(..., like=SymbolicArray)` to directly calling
    this function where possible, since this may handle dispatch to numeric arrays slightly
    better.
    """
    X = SYM_KINDS[kind]
    return SymbolicArray(X.ones(*_cs_shape(shape)), dtype=dtype, shape=_np_shape(shape))


def zeros_like(x, dtype=None, sparse=True, kind=None):
    """Construct an array of zeros with the same shape and dtype as `x`.

    Parameters
    ----------
    dtype : numpy.dtype, optional
        Data type of the array. Default is np.float64.
    sparse : bool, optional
        If `True`, then the array will be a sparse array with "structural"
        zeros.  Otherwise it will be "dense" and full of numerical zeros.
    kind : str, optional
        Kind of the symbolic variable (`"SX"` or `"MX"`). Default is `"SX"`.
        See CasADi documentation for details on the differences between the two.

    Returns
    -------
    SymbolicArray
        Array of zeros with the given dtype, and symbolic kind, and with shape of `x`.
    """
    x = array(x)  # Should be SymbolicArray or ndarray
    if kind is None and isinstance(x, SymbolicArray):
        kind = x.kind
    else:
        kind = "SX"
    return zeros(x.shape, dtype=dtype or x.dtype, sparse=sparse, kind=kind)


def ones_like(x, dtype=None, kind=None):
    """Construct an array of ones with the same shape and dtype as `x`.
    
    Parameters
    ----------
    dtype : numpy.dtype, optional
        Data type of the array. Default is np.float64.
    kind : str, optional
        Kind of the symbolic variable (`"SX"` or `"MX"`). Default is `"SX"`.
        See CasADi documentation for details on the differences between the two.

    Returns
    -------
    SymbolicArray
        Array of ones with the given dtype and symbolic kind, and with shape of `x`.
    """
    x = array(x)  # Should be SymbolicArray or ndarray
    if kind is None and isinstance(x, SymbolicArray):
        kind = x.kind
    else:
        kind = "SX"
    return ones(x.shape, dtype=dtype or x.dtype, kind=kind)


def eye(n, dtype=np.float64, kind="SX"):
    """Construct an identity matrix of size `n` with the given dtype.
    
    Parameters
    ----------
    n : int
        Size of the identity matrix.
    dtype : numpy.dtype, optional
        Data type of the array. Default is np.float64.
    kind : str, optional
        Kind of the symbolic variable (`"SX"` or `"MX"`). Default is `"SX"`.
        See CasADi documentation for details on the differences between the two.

    Returns
    -------
    SymbolicArray
        Identity matrix of size `n` with the given dtype and symbolic kind.

    Notes
    -----
    Prefer using `np.eye` or `np.eye(..., like=SymbolicArray)` to directly calling
    this function where possible, since this may handle dispatch to numeric arrays slightly
    better.
    """
    X = SYM_KINDS[kind]
    return SymbolicArray(X.eye(n), dtype=dtype, shape=(n, n))
