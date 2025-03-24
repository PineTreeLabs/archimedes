from __future__ import annotations
from typing import TYPE_CHECKING

from functools import wraps
import numpy as np
from numpy import exceptions as npex
import casadi as cs

from .._array_impl import casadi_array, SymbolicArray
from .._type_inference import type_inference, shape_inference

if TYPE_CHECKING:
    from .._array_impl import ArrayLike


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
            cs_x = list(map(casadi_array, x))
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
        cs_x = [list(map(casadi_array, xi)) for xi in x]

        arr = cs.vcat([cs.hcat(xi) for xi in cs_x])
        if isinstance(arr, cs.DM):
            return np.array(arr, dtype=result_dtype).reshape(result_shape)
        return SymbolicArray(
            arr,
            dtype=result_dtype,
            shape=result_shape,
        )

    raise NotImplementedError(f"Converting {x} (type={type(x)}) to array is not supported")


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

def normalize_axis_index(axis, ndim, msg_prefix="axis"):
    if not isinstance(axis, int):
        raise TypeError(f"integer argument expected, got {type(axis)}")
    if axis >= ndim or axis < -ndim:
        raise npex.AxisError(
            f"{msg_prefix} {axis} is out of bounds for array of dimension {ndim}"
        )
    if axis < 0: axis += ndim
    return axis


def unary_op(
    function=None,
    result_type=None,
    shape_inference=None,
):
    if shape_inference is None:
        shape_inference = "unary"
    return SymbolicOp(
        function=function,
        result_type=result_type,
        shape_inference=shape_inference,
    )


def binary_op(function=None, result_type=None, shape_inference="broadcast"):
    if shape_inference == "broadcast":
        return BroadcastOp(function, result_type=result_type)

    return SymbolicOp(
        function=function, result_type=result_type, shape_inference=shape_inference
    )


class SymbolicOp:
    shape_inference_rule = NotImplemented
    type_inference_rule = "default"

    def __init__(
        self,
        function=None,
        result_type=None,
        shape_inference=None,
    ):
        if shape_inference is not None:
            self.shape_inference_rule = shape_inference
        self._func = self._wrap(func=function, result_type=result_type)

    def _compute_result(self, op, inputs, result_shape):
        cs_inputs = map(casadi_array, inputs)
        return op(*cs_inputs)

    def _wrap(self, func, result_type=None):
        def actual_decorator(op):
            @wraps(op)
            def wrapper(*inputs, **kwargs):
                dtype = kwargs.get("dtype", None)
                if dtype is None:
                    dtype = type_inference(self.type_inference_rule, *inputs)
                inputs = tuple(array(x, dtype=dtype) for x in inputs)
                result_shape = shape_inference(self.shape_inference_rule, *inputs)
                cs_result = self._compute_result(op, inputs, result_shape)
                if result_type is not None:
                    dtype = result_type
                return SymbolicArray(cs_result, dtype=dtype, shape=result_shape)

            return wrapper

        if func:
            return actual_decorator(func)
        # return actual_decorator  # Never reached?

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class BroadcastOp(SymbolicOp):
    shape_inference_rule = "broadcast"

    def _compute_result(self, op, inputs, result_shape):
        cs_inputs = map(casadi_array, inputs)
        shapes = map(np.shape, inputs)  # Original shapes
        return broadcast_binary_operation(op, *cs_inputs, *shapes, result_shape)


def _repmat(x, reps):
    # There is no NumPy function that does this - should be renamed to `tile`.
    # For now it is only used internally.
    if isinstance(x, np.ndarray):
        return np.tile(x, reps)
    elif isinstance(x, (cs.SX, cs.MX)):
        return cs.repmat(x, *reps)
    else:
        raise NotImplementedError(f"_repmat not implemented for {type(x)}")


def _cs_reshape(x, shape, order="C"):
    """Reshape CasADi types consistent with NumPy API"""
    
    # Ensure that the "shape" argument to casadi.reshape has exactly two
    # elements, since CasADi always uses 2D arrays.
    if len(shape) == 0:
        shape = (1, 1)
    elif len(shape) == 1:
        shape = (shape[0], 1)
    # CasADi uses column-major ordering ("F") by default, so we need to reverse
    # the shape and transpose the result to have equivalent results
    if order == "C":
        return cs.reshape(x.T, shape[::-1]).T
    return cs.reshape(x, shape)


def broadcast_binary_operation(operation, arr1, arr2, shape1, shape2, common_shape):
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
    if len(shape1) == 0 or len(shape2) == 0:
        return operation(arr1, arr2)

    # Case 6 is a duplicate of case 5 with arguments reversed
    if len(shape1) == 2 and len(shape2) == 1:
        return broadcast_binary_operation(
            operation, arr2, arr1, shape2, shape1, common_shape
        )

    # Case 5a. First array is a vector: shape1 = (n,) and shape2 = (p, q) with n == q
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

