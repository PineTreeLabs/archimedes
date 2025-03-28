from __future__ import annotations
import numpy as np
import casadi as cs


DEFAULT_FLOAT = np.float64

from ._type_inference import shape_inference, type_inference


SYM_KINDS = {"MX": cs.MX, "SX": cs.SX, "DM": cs.DM, "ndarray": np.ndarray}
SYM_NAMES = {cs.MX: "MX", cs.SX: "SX", cs.DM: "DM", np.ndarray: "ndarray"}


__all__ = [
    "array",
    "sym",
    "sym_like",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
]


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
    """
    Create a symbolic array for use in symbolic computations.
    
    This function creates symbolic variables that serve as the foundation for building 
    computational graphs in Archimedes. Symbolic arrays can be manipulated with NumPy-like 
    operations and transformed into efficient computational graphs.
    
    Parameters
    ----------
    name : str
        Name of the symbolic variable. This name is used for display purposes and 
        debugging, and appears in symbolic expression representations.
    shape : int or tuple of int, optional
        Shape of the array. Default is (), which creates a scalar.
        A single integer n creates a vector of length n.
        A tuple (m, n) creates an m×n matrix.
    dtype : numpy.dtype, optional
        Data type of the array. Default is np.float64. This is mostly unused for now
        and all "compiled" functions use float64. It is included for consistency with
        NumPy and will be used for better type control in C code generation in the
        future.
    kind : {"SX", "MX"}, optional
        Kind of symbolic variable to create. Default is "SX".
        - "SX": Scalar-based symbolic type. Each element of the array has its own 
          symbolic representation. Generally more efficient for element-wise operations
          and when computing gradients of scalar functions.
        - "MX": Matrix-based symbolic type. The entire array is represented by a single 
          symbolic object. Supports a broader range of operations but may be less 
          efficient for some applications.
        
    Returns
    -------
    SymbolicArray
        Symbolic array with the given name, shape, dtype, and kind. This object can be
        used in NumPy operations and Archimedes function transformations.
    
    Notes
    -----
    Symbolic arrays are the building blocks of symbolic computation in Archimedes. They 
    represent variables whose values are not specified until later computation, allowing 
    for construction of computational graphs that can be efficiently evaluated, differentiated, 
    and optimized.
    
    The difference between SX and MX symbolic types is important:
    - SX types create element-wise symbolic variables and are more efficient for scalar 
      operations and gradients of scalar functions.
    - MX types represent entire matrices as single symbolic objects and support a wider range
      of operations, including those that cannot be easily represented element-wise.
    
    Current limitations:
    - Only supports up to 2D arrays (scalars, vectors, and matrices)
    - Some operations may behave differently between SX and MX types
    
    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>> 
    >>> # Create a scalar symbolic variable
    >>> x = arc.sym("x")
    >>> print(x)
    x
    >>> 
    >>> # Create a vector (1D array)
    >>> v = arc.sym("v", shape=3)
    >>> print(v)
    [v[0], v[1], v[2]]
    >>> 
    >>> # Create a matrix (2D array)
    >>> M = arc.sym("M", shape=(2, 2))
    >>> print(M)
    [[M[0,0], M[0,1]], [M[1,0], M[1,1]]]
    >>> 
    >>> # Use in symbolic computations
    >>> f = np.sin(x) + np.cos(x)
    >>> print(f)
    sin(x) + cos(x)
    >>> 
    >>> # Create an MX-type symbolic variable
    >>> y = arc.sym("y", shape=2, kind="MX")
    >>> print(y)
    [y[0], y[1]]
    
    See Also
    --------
    archimedes.array : Create a regular or symbolic array from data
    archimedes.compile : Create a symbolic function from a Python function
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
    """
    Create a symbolic array with the same shape and dtype as an existing array.

    Parameters
    ----------
    x : array_like
        Array to copy shape and dtype from. Can be a NumPy ndarray, SymbolicArray, 
        or any array-like object that can be converted to an ndarray.
    name : str
        Name of the symbolic variable. This name is used for display purposes and 
        debugging, and appears in symbolic expression representations.
    dtype : numpy.dtype, optional
        Data type of the array. If None (default), uses the dtype of `x`.
    kind : {"SX", "MX"}, optional
        Kind of symbolic variable to create. Default is "SX".
        - "SX": Scalar-based symbolic type. Each element has its own symbolic
          representation. Generally more efficient for element-wise operations.
        - "MX": Matrix-based symbolic type. The entire array is represented by 
          a single symbolic object. Supports a broader range of operations.

    Returns
    -------
    SymbolicArray
        Symbolic array with the given name and kind, matching the shape and 
        (optionally) dtype of the input array `x`.
    
    Notes
    -----
    This function is particularly useful when:
    - Creating symbolic representations of existing numeric data
    - Building symbolic functions that need to match the structure of numeric inputs
    - Prototyping symbolic algorithms using existing arrays as templates

    The function automatically converts non-array inputs to NumPy arrays using 
    `np.asarray()` before extracting their shape information.

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>> 
    >>> # Create a symbolic variable with the same shape as a numeric array
    >>> data = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> M_sym = arc.sym_like(data, "M")
    >>> print(M_sym.shape)
    (2, 2)
    >>> 
    >>> # Use with a vector
    >>> v = np.ones(5)
    >>> v_sym = arc.sym_like(v, "v")
    >>> print(v_sym.shape)
    (5,)
    >>> 
    >>> # Works with scalar inputs too
    >>> scalar = 42.0
    >>> x = arc.sym_like(scalar, "x")
    >>> print(x.shape)
    ()
    >>> 
    >>> # Create a symbolic matrix based on another symbolic array
    >>> original_sym = arc.sym("y", shape=(3, 3))
    >>> another_sym = arc.sym_like(original_sym, "z", kind="MX")
    >>> print(another_sym.shape)
    (3, 3)
    
    See Also
    --------
    archimedes.sym : Create a symbolic array with explicit shape
    archimedes.array : Create a regular or symbolic array from data
    archimedes.zeros_like : Create an array of zeros with shape and dtype of an input array
    archimedes.ones_like : Create an array of ones with shape and dtype of an input array
    """
    if not isinstance(x, (np.ndarray, SymbolicArray)):
        x = np.asarray(x)
    return sym(name, x.shape, dtype=dtype or x.dtype, kind=kind)


def array(x, dtype=None) -> ArrayLike:
    """
    Create an array from various input types, supporting both numeric and symbolic computation.
    
    This function serves as Archimedes' array creation mechanism, handling both numeric and 
    symbolic inputs. It creates the appropriate array type (NumPy or SymbolicArray) based on 
    the input data.
    
    Parameters
    ----------
    x : array_like
        An array-like object, which can be:
        - NumPy ndarray
        - SymbolicArray
        - CasADi symbolic type (SX or MX)
        - List or nested list of scalars
        - Scalar value
    dtype : str or numpy.dtype, optional
        The desired data type for the array. If not specified, the dtype is inferred from `x`.
    
    Returns
    -------
    array : numpy.ndarray or SymbolicArray
        If the input contains symbolic elements, returns a SymbolicArray.
        Otherwise, returns a NumPy ndarray.
    
    Notes
    -----
    Array creation using the NumPy dispatch mechanism (`np.array(..., like=...)`) is
    recommended over calling `array(...)` directly. The dispatch mechanism supports a wider 
    range of input types and better handles numeric input types.
    
    When working with symbolic computation, this function ensures that array creation follows
    the same patterns as NumPy, allowing for seamless transitions between symbolic and numeric
    computation.
    
    This function currently supports:
    - Creating arrays from existing arrays (preserving type)
    - Creating arrays from lists of scalars (1D arrays)
    - Creating arrays from lists of lists (2D arrays)
    
    Limitations and edge cases:
    - Higher-dimensional arrays (>2D) must be created incrementally
    - Creating arrays with inconsistent dimensions will raise a ValueError
    - Complex-valued arrays may have limited symbolic operation support
    
    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>> 
    >>> # Basic numeric array creation
    >>> arc.array([1.0, 2.0, 3.0])
    array([1., 2., 3.])
    >>> 
    >>> # Creating a symbolic array
    >>> x = arc.sym("x", 3)  # Create a symbolic variable
    >>> arc.array([x[0], 2.0, x[2]])  # Mixed symbolic/numeric content
    [x_0, 2, x_2]
    >>> 
    >>> # 2D array from nested lists
    >>> arc.array([[1.0, 2.0], [3.0, 4.0]])
    array([[1., 2.],
           [3., 4.]])
    >>>
    >>> # Using NumPy dispatch (recommended approach)
    >>> np.array([x[0], 2.0, x[2]], like=x_sym)  # Creates a SymbolicArray
    [x_0, 2, x_2]
    
    See Also
    --------
    numpy.array : The NumPy array creation function
    archimedes.zeros : Create an array of zeros
    archimedes.ones : Create an array of ones
    archimedes.sym : Create a symbolic variable
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

    """
    Construct a symbolic array of zeros with the given shape and dtype.
    
    This function creates an array filled with zeros, either as "structural" 
    (sparse) zeros that are symbolic placeholders, or as actual numerical 
    zero values.
    
    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the array. A single integer n creates a vector of length n.
        A tuple (m, n) creates an m×n matrix.
    dtype : numpy.dtype, optional
        Data type of the array. Default is np.float64.
    sparse : bool, optional
        If True (default), the array will contain "structural" zeros, which are
        symbolic placeholders rather than actual numeric values. These are more
        efficient in memory and computation.
        If False, the array will contain actual numeric zero values.
    kind : {"SX", "MX"}, optional
        Kind of symbolic variable to create. Default is "SX".
        - "SX": Scalar-based symbolic type. Each element has its own symbolic
          representation. Generally more efficient for element-wise operations.
        - "MX": Matrix-based symbolic type. The entire array is represented by 
          a single symbolic object. Supports a broader range of operations.
        
    Returns
    -------
    SymbolicArray
        Symbolic array of zeros with the given shape, dtype, and symbolic kind.
    
    Notes
    -----
    The distinction between sparse (structural) and dense (numerical) zeros is important:
    
    - Sparse/structural zeros (sparse=True) are efficient for building computational
      graphs where many elements are zero, especially in matrix operations.
    
    - Dense/numerical zeros (sparse=False) actually contain the value 0 and behave
      more like traditional NumPy arrays filled with zeros.
    
    When working within a function that will be executed with both symbolic and
    numeric arrays, prefer using `np.zeros_like` or `np.zeros(..., like=x)` where
    `x` is either a SymbolicArray or NumPy array. This provides better compatibility
    across both numeric and symbolic execution paths.
    
    The exception is when you specifically need sparse/structural zeros, which are only
    available through this direct function call with sparse=True.
    
    Examples
    --------
    >>> import archimedes as arc
    >>> import numpy as np
    >>> 
    >>> # Create a vector of structural zeros (sparse representation)
    >>> z1 = arc.zeros(5)
    >>> print(z1)
    [00, 00, 00, 00, 00]
    >>> 
    >>> # Create a matrix of numerical zeros (dense representation)
    >>> z2 = arc.zeros(5, sparse=False)
    >>> print(z2)
    @1=0, [@1, @1, @1, @1, @1]
    >>> 
    >>> # Create MX-type zeros 
    >>> z3 = arc.zeros(4, kind="MX")
    
    See Also
    --------
    numpy.zeros : NumPy's array of zeros function
    archimedes.zeros_like : Create an array of zeros with shape and dtype of an input array
    archimedes.ones : Create an array of ones
    archimedes.array : Create an array from data
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
    """
    Create a symbolic array of zeros with the same shape and dtype as an input array.

    This function constructs a symbolic array filled with zeros, matching the dimensions
    and data type of an existing array. It's particularly useful for creating compatible
    zero arrays in functions that need to work with both symbolic and numeric inputs.

    Parameters
    ----------
    x : array_like
        The array whose shape and dtype will be used. Can be a NumPy ndarray, 
        SymbolicArray, or any array-like object that can be converted to an array.
    dtype : numpy.dtype, optional
        Data type of the new array. If None (default), uses the dtype of `x`.
    sparse : bool, optional
        If True (default), creates "structural" zeros, which are symbolic
        placeholders rather than actual numeric values. These are more efficient
        for building computational graphs.
        If False, creates numerical zero values.
    kind : {"SX", "MX"} or None, optional
        Kind of symbolic variable to create. If None (default), uses the kind
        of `x` if it's a SymbolicArray, otherwise uses "SX".
        - "SX": Scalar-based symbolic type. Each element has its own symbolic
          representation. Generally more efficient for element-wise operations.
        - "MX": Matrix-based symbolic type. The entire array is represented by 
          a single symbolic object. Supports a broader range of operations.
        
    Returns
    -------
    SymbolicArray
        Symbolic array of zeros with the same shape as `x`, and with the
        specified dtype and symbolic kind.
    
    Notes
    -----
    This function is the symbolic counterpart to NumPy's `zeros_like`. While creating
    a standard NumPy array of zeros requires numeric inputs, this function works with
    both symbolic and numeric arrays, preserving the symbolic nature when needed.

    When used inside a function decorated with `compile`, this function helps create
    temporary arrays that match the input's shape, which is particularly important for
    maintaining compatibility between symbolic and numeric execution paths.

    The `sparse` parameter determines whether the zeros are "structural" (symbolic
    placeholders) or actual numeric zeros, which can affect memory usage and computational
    efficiency.

    When the array is not sparse, it is preferred to use `np.zeros_like` or
    `np.zeros(..., like=x)`, where `x` is either a SymbolicArray or NumPy array. This can
    provide better compatibility across both numeric and symbolic execution paths.
    The exception is when you specifically need sparse/structural zeros, which are only
    available through this direct function call with `sparse=True`.

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>> 
    >>> # With a symbolic input array
    >>> x_sym = arc.sym("x", shape=(2, 3))
    >>> z_sym = arc.zeros_like(x_sym)
    >>> print(z_sym.shape)
    (2, 3)
    >>> 
    >>> # With a numeric input array
    >>> x_num = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> z_num = arc.zeros_like(x_num)
    >>> print(z_num.shape)
    (3, 2)
    >>> 
    >>> # Creating numerical zeros instead of structural zeros
    >>> z_dense = arc.zeros_like(x_sym, sparse=False)
    >>> 
    >>> # Changing the kind of symbolic variable
    >>> z_mx = arc.zeros_like(x_sym, kind="MX")
    >>> 
    >>> # In a function that will be traced symbolically:
    >>> @arc.compile
    >>> def process_array(x):
    >>>     # Create a result array with same shape as input
    >>>     # Dispatches to this function when x is a SymbolicArray
    >>>     result = np.zeros_like(x)
    >>>     for i in range(x.shape[0]):
    >>>         result[i] = x[i] ** 2
    >>>     return result
    
    See Also
    --------
    numpy.zeros_like : NumPy's equivalent function for numeric arrays
    archimedes.zeros : Create a symbolic array of zeros with specified shape
    archimedes.ones_like : Create a symbolic array of ones with same shape as input
    archimedes.array : Create an array from data
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
