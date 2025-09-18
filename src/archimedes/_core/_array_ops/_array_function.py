# This code modifies code from NumPy.

# Copyright (c) 2005-2024, NumPy Developers.
# Licensed under NumPy license
# https://numpy.org/doc/stable/license.html

# Modifications and additions to the original code:
# Copyright (c) 2025 Pine Tree Labs, LLC
# Licensed under the GNU General Public License v3.0

# As a combined work, use of this code requires compliance with the GNU GPL v3.0.
# The original license terms are included below for attribution:

# === NumPy License ===

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
import numpy.exceptions as npex

from .._array_impl import (
    SymbolicArray,
    _as_casadi_array,
    _dispatch_array,
    _empty_like,
    _result_type,
    array,
    eye,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from .._type_inference import shape_inference, type_inference
from ._array_ops import (
    _cs_reshape,
    binary_op,
    normalize_axis_index,
    unary_op,
)
from ._array_ufunc import SUPPORTED_UFUNCS, _dot

if TYPE_CHECKING:
    from ...typing import ShapeLike


# Do not call directly - this should be called by NumPy's array
# dispatch mechanism when x is a SymbolicArray
def _reshape(x, shape, order="C"):
    if order not in {"C", "F"}:
        raise ValueError(f"Invalid order: {order}")

    new_sym = _cs_reshape(x._sym, shape, order)
    return SymbolicArray(new_sym, dtype=x.dtype, shape=shape)


def _ravel(x, order="C"):
    return np.reshape(x, (x.size,), order=order)


def _tile(x, reps):
    # This is the dispatch function for `np.tile`, so we can
    # assume that x is a SymbolicArray.

    # From NumPy docs:
    # If reps has length d, the result will have dimension of max(d, A.ndim).
    # If A.ndim < d, A is promoted to be d-dimensional by prepending new axes. So a
    # shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3)
    # for 3-D replication. If this is not the desired behavior, promote A to
    # d-dimensions manually before calling this function.

    # If A.ndim > d, reps is promoted to A.ndim by prepending 1’s to it. Thus for
    # an A of shape (2, 3, 4, 5), a reps of (2, 2) is treated as (1, 1, 2, 2).
    reps = tuple(reps) if np.iterable(reps) else (reps,)

    if len(reps) > 2:
        raise ValueError("Only 1D and 2D tiling is supported")

    if len(reps) == 0:
        return x

    if all(rep == 1 for rep in reps):
        return x

    d = len(reps)
    if d < x.ndim:
        # Prepend 1's to reps
        reps = (1,) * (x.ndim - d) + reps

    elif d > x.ndim:
        # Prepend 1's to x.shape
        x = np.reshape(x, (1,) * (d - x.ndim) + x.shape)

    ret_shape = tuple(int(x.shape[i] * reps[i]) for i in range(len(reps)))

    x_cs = _as_casadi_array(x)
    # CasADi arrays are always 2D, so we need might need to adjust the
    # reps before calling cs.repmat
    cs_reps = reps
    if len(cs_reps) < 2:
        cs_reps = (1,) * (2 - len(cs_reps)) + cs_reps

    ret_cs = cs.repmat(x_cs, *cs_reps)
    return SymbolicArray(ret_cs, dtype=x.dtype, shape=ret_shape)


def _broadcast_to(x, shape):
    shape = tuple(shape) if np.iterable(shape) else (shape,)

    if not shape and x.shape:
        raise ValueError("cannot broadcast a non-scalar to a scalar array")

    if any(size < 0 for size in shape):
        raise ValueError("all elements of broadcast shape must be non-negative")

    if len(shape) > 2:
        raise ValueError("Only 0-2D arrays are supported")

    if len(shape) < x.ndim:
        raise ValueError(
            f"input operand with shape {x.shape} has more dimensions than the "
            f"broadcast shape {shape}"
        )

    # From NumPy docs:
    # NumPy compares their shapes element-wise. It starts with the trailing
    # (i.e. rightmost) dimension and works its way left. Two dimensions are
    # compatible when
    # 1. they are equal, or
    # 2. one of them is 1.

    # If these conditions are not met, a ValueError: operands could not be
    # broadcast together exception is thrown, indicating that the arrays
    # have incompatible shapes.

    # Prepend ones to the shape of x if necessary
    x_shape = x.shape
    if len(shape) > len(x_shape):
        x_shape = (1,) * (len(shape) - len(x_shape)) + x_shape

    # Uses `tile` for broadcasting (this is not how NumPy works, but it's fine
    # with symbolic arrays).
    reps = []
    for i in range(len(shape)):
        if x_shape[i] == shape[i]:
            reps.append(1)
        elif x_shape[i] == 1:
            reps.append(shape[i])  # Repeat to match the target shape
        else:
            raise ValueError(f"Cannot broadcast {x_shape} to {shape}")
    return _tile(x, tuple(reps))


def _transpose(x):
    dtype = x.dtype
    shape = shape_inference("transpose", x)
    return SymbolicArray(cs.transpose(x._sym), dtype=dtype, shape=shape)


def _mmax(x, axis=None):
    if axis is None:
        # Max over all elements
        return unary_op(cs.mmax, shape_inference="unary_to_scalar")(x)
    else:
        raise NotImplementedError("max with axis not yet supported")


def _mmin(x, axis=None):
    if axis is None:
        # Min over all elements
        return unary_op(cs.mmin, shape_inference="unary_to_scalar")(x)
    else:
        raise NotImplementedError("min with axis not yet supported")


def _diag(x):
    if len(x.shape) == 1:
        # Vector to diagonal matrix
        return SymbolicArray(cs.diag(x._sym), dtype=x.dtype, shape=(len(x), len(x)))
    else:
        # Matrix to vector
        return SymbolicArray(cs.diag(x._sym), dtype=x.dtype, shape=(len(x),))


def _append(arr, values, axis=None):
    # If axis is None, both `arr` and `values` are flattened
    if axis not in {None, 0, 1}:
        raise ValueError("Only 2D arrays are supported")

    arr, values = map(array, (arr, values))
    shape = None

    if axis is None:
        arr = arr.flatten()
        values = values.flatten()
        axis = 0
        shape = (len(arr) + len(values),)

    arr_ = arr if not isinstance(arr, SymbolicArray) else arr._sym
    values_ = values if not isinstance(values, SymbolicArray) else values._sym
    dtype = _result_type(arr, values)

    _cs_append = cs.vertcat if axis == 0 else cs.horzcat
    return SymbolicArray(_cs_append(arr_, values_), dtype=dtype, shape=shape)


def _astype(arr, dtype):
    return SymbolicArray(arr._sym, dtype=dtype, shape=arr.shape)


def _atleast_1d(*arrays):
    # This function only does anything if any of the arrays are 0D (scalars)
    # The underlying CasADi arrays always have 2D shape, so we just need
    # to superficially change the SymbolicArray shape.

    if len(arrays) == 1:
        arr = arrays[0]
        if arr.ndim > 0:
            return arr
        return SymbolicArray(arr._sym, dtype=arr.dtype, shape=(1,))

    # Call NumPy for the individual arrays, since some may be numeric
    # and not SymbolicArray types
    return tuple(np.atleast_1d(arr) for arr in arrays)


def _atleast_2d(*arrays):
    # This function will do nothing unless any of the arrays are 0D or 1D
    # in which case it will add a leading dimension(s) of size 1.
    # The underlying CasADi arrays always have 2D shape, so we just need
    # to superficially change the SymbolicArray shape.

    if len(arrays) == 1:
        arr = arrays[0]
        if arr.ndim > 1:
            return arr
        if arr.ndim == 0:
            shape = (1, 1)
        elif arr.ndim == 1:
            shape = (1, arr.size)
        return SymbolicArray(arr._sym, dtype=arr.dtype, shape=shape)

    # Call NumPy for the individual arrays, since some may be numeric
    # and not SymbolicArray types
    return tuple(np.atleast_2d(arr) for arr in arrays)


def _concatenate(arrs, axis=0, dtype=None):
    if dtype is not None:
        raise NotImplementedError("dtype argument not yet supported for concatenate")
    dtype = _result_type(*arrs)

    # All arrays must have the same number of dimensions
    ndim = arrs[0].ndim
    if ndim == 0:
        raise ValueError("Zero-dimensional arrays cannot be concatenated")
    if not all(arr.ndim == ndim for arr in arrs):
        # Find the differing array
        for i, arr in enumerate(arrs):
            if arr.ndim != ndim:
                raise ValueError(
                    "All arrays must have the same number of dimensions, but "
                    f"the array at index 0 has {ndim} dimensions and the array "
                    f"at index {i} has {arr.ndim} dimensions."
                )

    # Check that the axis is in bounds
    arrs = np.atleast_1d(*arrs)
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    axis = normalize_axis_index(axis, ndim, "axis")

    # The arrays must have the same shape except for the `axis` dimension
    arr0_shape = arrs[0].shape
    match_axes = [i for i in range(ndim) if i != axis]
    for j, arr in enumerate(arrs[1:]):
        if not all(arr.shape[i] == arr0_shape[i] for i in match_axes):
            # Find the differing array
            for i in match_axes:
                if arr.shape[i] != arr0_shape[i]:
                    raise ValueError(
                        "All the input array dimensions except for the concatenation "
                        f"axis must match exactly, but along dimension {i} the array "
                        f"at index 0 has size {arr0_shape[i]} and the array at index "
                        f"{j} has size {arr.shape[i]}."
                    )

    args = [_as_casadi_array(arr) for arr in arrs]
    shape: ShapeLike = ()

    # If all arrays are 1D, then use cs.vcat
    if ndim == 1:
        shape = (sum([arr.shape[axis] for arr in arrs]),)
        result = cs.vcat(args)

    # If all arrays are 2D, then use cs.hcat or cs.vcat, depending on the axis
    elif axis == 0:
        shape = (sum([arr.shape[0] for arr in arrs]), arrs[0].shape[1])
        result = cs.vcat(args)

    elif axis == 1:
        shape = (arrs[0].shape[0], sum([arr.shape[1] for arr in arrs]))
        result = cs.hcat(args)

    return SymbolicArray(result, dtype=dtype, shape=shape)


def _hstack(arrs, dtype=None):
    # From NumPy docs:
    # "This is equivalent to concatenation along the second axis, except for 1-D
    # arrays where it concatenates along the first axis.."
    # https://github.com/numpy/numpy/blob/v2.1.0/numpy/_core/shape_base.py#L294-L363
    arrs = np.atleast_1d(*arrs)
    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs and arrs[0].ndim == 1:
        return np.concatenate(arrs, 0, dtype=dtype)
    else:
        return np.concatenate(arrs, 1, dtype=dtype)


def _vstack(arrs, dtype=None):
    # From NumPy docs:
    # "This is equivalent to concatenation along the first axis after 1-D arrays of
    # shape (N,) have been reshaped to (1,N)."
    # https://github.com/numpy/numpy/blob/v2.1.0/numpy/_core/shape_base.py#L287C1-L290C66
    arrs = np.atleast_2d(*arrs)
    return np.concatenate(arrs, 0, dtype=dtype)


def _stack(arrays, axis=0, dtype=None):
    # From NumPy docs:
    # Join a sequence of arrays along a new axis.
    # The axis parameter specifies the index of the new axis in the dimensions of the
    # result. For example, if axis=0 it will be the first dimension and if axis=-1 it
    # will be the last dimension.
    # https://github.com/numpy/numpy/blob/v2.1.0/numpy/_core/shape_base.py#L377-L465

    arrays = [array(arr) for arr in arrays]
    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError("All input arrays must have the same shape")

    result_ndim = arrays[0].ndim + 1
    axis = normalize_axis_index(axis, result_ndim, "axis")
    sl = (slice(None),) * axis + (None,)
    expanded_arrays = [arr[sl] for arr in arrays]
    return np.concatenate(expanded_arrays, axis=axis, dtype=dtype)


def _array_split(arr, indices_or_sections, axis=0):
    arr = array(arr)
    dtype = arr.dtype

    try:
        len(indices_or_sections)
        indices = indices_or_sections
    except TypeError:
        sections = indices_or_sections
        # Split into equal sections
        section_length = int(arr.shape[axis] // sections)
        indices = section_length * np.arange(1, sections)

    ndim = arr.ndim
    axis = normalize_axis_index(axis, ndim, "axis")

    # Casadi *split requires additional sections compared to numpy:
    # From CasADi docs:
    # To split up an expression horizontally into n smaller expressions,
    # you need to provide, in addition to the expression being split, a
    # vector offset of length n+1. The first element of the offset vector
    # must be 0 and the last element must be the number of columns.
    # Remaining elements must follow in a non-decreasing order.
    indices = [0, *map(int, indices), arr.shape[axis]]

    # Check non-decreasing indices
    ax_sizes = []
    for ix1, ix2 in zip(indices[:-1], indices[1:]):
        if ix1 > ix2:
            raise IndexError("indices to split must be non-decreasing")
        ax_sizes.append(ix2 - ix1)

    arr = _as_casadi_array(arr)
    shapes: list[ShapeLike] = []

    if ndim == 1:
        shapes = [(s,) for s in ax_sizes]
        results = cs.vertsplit(arr, indices)

    else:
        if axis == 0:
            ncol = arr.shape[1]
            shapes = [(s, ncol) for s in ax_sizes]
            results = cs.vertsplit(arr, indices)

        elif axis == 1:
            nrow = arr.shape[0]
            shapes = [(nrow, s) for s in ax_sizes]
            results = cs.horzsplit(arr, indices)

    return tuple(
        SymbolicArray(r, dtype=dtype, shape=s) for (r, s) in zip(results, shapes)
    )


def _vsplit(arr, indices_or_sections):
    return _split(arr, indices_or_sections, 0)


def _hsplit(arr, indices_or_sections):
    return _split(arr, indices_or_sections, 1)


# https://github.com/numpy/numpy/blob/v2.2.0/numpy/lib/_shape_base_impl.py#L803-L879
def _split(ary, indices_or_sections, axis=0):
    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        if ary.shape[axis] % sections != 0:
            raise ValueError(
                "array split does not result in an equal division"
            ) from None
    return np.array_split(ary, indices_or_sections, axis)


def _squeeze(a, axis=None):
    # From NumPy docs:
    # Remove axes of length one from a.
    # https://github.com/numpy/numpy/blob/v2.0.0/numpy/_core/fromnumeric.py#L1564-L1632

    if axis is not None:
        raise NotImplementedError("axis argument is not yet supported")

    shape = tuple(n for n in a.shape if n > 1)
    return np.reshape(a, shape)


def _where(condition, x=None, y=None):
    if x is None or y is None:
        raise ValueError("Calling where with only one argument is not supported")

    condition, x, y = map(array, (condition, x, y))

    shape = shape_inference("broadcast", condition, x, y)
    dtype = _result_type(condition, x, y)

    # CasADi will broadcast scalars, otherwise we need to handle broadcasting
    x = np.broadcast_to(x, shape)
    y = np.broadcast_to(y, shape)
    condition = np.broadcast_to(condition, shape)

    x = x if np.prod(x.shape) <= 1 else np.reshape(x, shape)
    y = y if np.prod(y.shape) <= 1 else np.reshape(y, shape)

    # Convert to NumPy or CasADi arrays only
    condition, x, y = map(_as_casadi_array, (condition, x, y))
    result = cs.if_else(condition, x, y)
    return SymbolicArray(result, dtype=dtype, shape=shape)


def _norm(x, ord=None, axis=None, keepdims=False):
    if axis is not None:
        raise NotImplementedError("axis argument not yet supported for norm")
    if keepdims:
        raise NotImplementedError("keepdims argument not yet supported for norm")

    if ord not in {None, 1, 2, np.inf, "fro"}:
        raise ValueError("Invalid norm order")

    # CasADi will throw an error for 2-norm applied to a matrix,
    # whereas NumPy will assume the Frobenius norm if axis is None.
    # Since we don't support axis, we will just use the Frobenius norm
    if len(x.shape) == 2:
        if ord in {None, 2}:
            ord = "fro"
    else:
        if ord == "fro":
            raise ValueError("Frobenius norm only defined for matrices")
        if ord is None:
            ord = 2

    norm = {
        1: cs.norm_1,
        2: cs.norm_2,
        np.inf: cs.norm_inf,
        "fro": cs.norm_fro,
    }[ord]

    return SymbolicArray(norm(x._sym), dtype=float, shape=())


def _outer(a, b):
    """Compute the outer product of two vectors"""
    a = np.atleast_2d(a.flatten())  # shape: (1, n)
    b = np.atleast_2d(b.flatten())  # shape: (m, 1)
    return a.T @ b


# TODO:
# - use cs.interp1d?
def _interp1d(x, xp, fp, left=None, right=None, period=None, method="linear"):
    # One-dimensional interpolation.  Should be called with
    # np.interp rather than directly.

    # Convert inputs to arrays (either NumPy or SymbolicArray)
    x, xp, fp = map(array, (x, xp, fp))

    if isinstance(x, SymbolicArray) and not isinstance(x._sym, cs.MX):
        raise ValueError("SymbolicArray must use 'kind=\"MX\"' to use interp function")

    # Check for invalid input
    if xp.ndim != 1:
        raise ValueError(f"xp must be 1-dimensional but has shape {xp.shape}")

    if fp.ndim != 1:
        raise ValueError(f"fp must be 1-dimensional but has shape {fp.shape}")

    if left is None:
        left = fp[0]

    if right is None:
        right = fp[-1]

    if period is not None:
        raise NotImplementedError("period argument not yet implemented")

    # CasAdi does not suppoert interpolating symbolic data; only numeric
    if not isinstance(xp, np.ndarray) or not isinstance(fp, np.ndarray):
        raise ValueError(
            f"xp and fp must be NumPy arrays, but have type {type(xp)} and {type(fp)}"
        )

    if not np.isscalar(left) or not np.isscalar(right):
        raise ValueError(
            f"left and right must be scalars, but have type {type(left)} and "
            f"{type(right)}"
        )

    # Output shape is the same as the input shape, dtype is the promotion of
    # the input dtypes
    shape = x.shape
    dtype = type_inference("default", x, xp, fp)

    x_cs = _as_casadi_array(x)  # Map input to either NumPy or CasADi

    # Create the CasADi interpolant object
    lut = cs.interpolant("interpolate", method, [xp], fp)
    f = SymbolicArray(lut(x_cs), shape=shape, dtype=dtype)

    # Limit the ends to the left/right values
    f = np.where(x < xp[0], left, f)  # type: ignore
    f = np.where(x > xp[-1], right, f)  # type: ignore

    return f


def _sum(x, axis=None, dtype=None):
    if dtype is not None:
        raise NotImplementedError("dtype argument not yet supported for sum")

    if np.isscalar(x) or len(x.shape) == 0:
        return x

    if axis is not None and axis >= len(x.shape):
        raise npex.AxisError(
            f"axis {axis} is out of bounds for array of dimension {len(x.shape)}"
        )

    dtype = x.dtype
    shape: tuple[int, ...] = ()
    arg = x._sym if isinstance(x, SymbolicArray) else x

    if axis is None or len(x.shape) == 1:
        shape = ()
        arg = cs.reshape(arg, (-1, 1))
        res = cs.sum1(arg)

    elif axis == 0:
        shape = (x.shape[1],)
        res = cs.sum1(arg)

    elif axis == 1:
        shape = (x.shape[0],)
        res = cs.sum2(arg)

    return SymbolicArray(res, dtype=dtype, shape=shape)


# https://github.com/numpy/numpy/blob/v2.1.0/numpy/_core/numeric.py#L1528-L1747
def _cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3

    axisa = normalize_axis_index(axisa, a.ndim, "axisa")
    axisb = normalize_axis_index(axisb, b.ndim, "axisb")
    axisc = normalize_axis_index(axisc, a.ndim, "axisc")

    if a.shape[axisa] != 3 or b.shape[axisb] != 3:
        raise ValueError("Both arrays must have shape 3 along the axis of cross")

    # CasADi does not support individual axis specification, but only the
    # `dim` argument as in MATLAB
    # (https://www.mathworks.com/help/matlab/ref/cross.html)
    if axisa != axisb:
        raise NotImplementedError(
            "cross product along different axes not yet supported"
        )

    arg1 = a._sym if isinstance(a, SymbolicArray) else a
    arg2 = b._sym if isinstance(b, SymbolicArray) else b

    dtype = type_inference("default", a, b)
    shape = shape_inference("broadcast", a, b)

    res = cs.cross(arg1, arg2, axisa + 1)

    if axisc != axisa:
        res = res.T
        shape = shape[::-1]

    return SymbolicArray(res, dtype=dtype, shape=shape)


def _swapaxes(a, axis1, axis2):
    axis1 = normalize_axis_index(axis1, a.ndim, "axis1")
    axis2 = normalize_axis_index(axis2, a.ndim, "axis2")
    if axis1 == axis2:
        return a

    # Since we're only supporting up to 2D arrays, the only possibilities
    # are that axis1 == axis2 or that we need to transpose the array
    return a.T


def _clip(a, a_min=None, a_max=None):
    if a_min is not None:
        a = np.fmax(a, a_min)
    if a_max is not None:
        a = np.fmin(a, a_max)
    return a


def _roll(a, shift, axis=None):
    a = array(a)
    axis = normalize_axis_index(axis, a.ndim, "axis") if axis is not None else axis

    if a.shape == ():
        return a

    if axis is None:
        a = a.flatten()
        axis = 0

    n = a.shape[axis]
    shift = shift % n  # Handle large or negative shifts
    if shift == 0:
        return a

    if axis == 0:
        top = a[-shift:]
        bottom = a[:-shift]
        return np.concatenate((top, bottom), axis=0)

    elif axis == 1:
        left = a[:, -shift:]
        right = a[:, :-shift]
        return np.concatenate((left, right), axis=1)


# List from numpy.testing.overrides.get_overridable_numpy_array_functions()
SUPPORTED_FUNCTIONS = {
    "array": _dispatch_array,
    "may_share_memory": NotImplemented,
    "is_busday": NotImplemented,
    "identity": NotImplemented,
    "corrcoef": NotImplemented,
    "title": NotImplemented,
    "ptp": NotImplemented,
    "ix_": NotImplemented,
    "allclose": NotImplemented,
    "translate": NotImplemented,
    "busday_offset": NotImplemented,
    "array2string": NotImplemented,
    "upper": NotImplemented,
    "i0": NotImplemented,
    "max": _mmax,
    "isclose": NotImplemented,
    "busday_count": NotImplemented,
    "zfill": NotImplemented,
    "sinc": NotImplemented,
    "rec_join": NotImplemented,
    "multi_dot": NotImplemented,
    "array_equal": NotImplemented,
    "atleast_1d": _atleast_1d,
    "msort": NotImplemented,
    "amax": _mmax,
    "datetime_as_string": NotImplemented,
    "isnumeric": NotImplemented,
    "atleast_2d": _atleast_2d,
    "array_equiv": NotImplemented,
    "median": NotImplemented,
    "isdecimal": NotImplemented,
    "equal": NotImplemented,
    "min": _mmin,
    "choose": NotImplemented,
    "percentile": NotImplemented,
    "not_equal": NotImplemented,
    "fromfunction": NotImplemented,
    "poly": NotImplemented,
    "genfromtxt": NotImplemented,
    "amin": _mmin,
    "quantile": NotImplemented,
    "greater_equal": NotImplemented,
    "roots": NotImplemented,
    "trapz": NotImplemented,
    "prod": NotImplemented,
    "asfarray": NotImplemented,
    "less_equal": NotImplemented,
    "polyint": NotImplemented,
    "cumprod": NotImplemented,
    "polyder": NotImplemented,
    "greater": NotImplemented,
    "meshgrid": NotImplemented,
    "unique": NotImplemented,
    "delete": NotImplemented,
    "intersect1d": NotImplemented,
    "array_repr": NotImplemented,
    "setxor1d": NotImplemented,
    "in1d": NotImplemented,
    "isin": NotImplemented,
    "less": NotImplemented,
    "insert": NotImplemented,
    "polyfit": NotImplemented,
    "ndim": NotImplemented,
    "fix": NotImplemented,
    "polyval": NotImplemented,
    "union1d": NotImplemented,
    "append": _append,
    "repeat": NotImplemented,
    "str_len": NotImplemented,
    "setdiff1d": NotImplemented,
    "size": lambda x: np.prod(x.shape),
    "array_str": NotImplemented,
    "polyadd": NotImplemented,
    "digitize": NotImplemented,
    "isposinf": NotImplemented,
    "add": SUPPORTED_UFUNCS["add"],
    "polysub": NotImplemented,
    "put": NotImplemented,
    "swapaxes": _swapaxes,
    "multiply": binary_op(cs.times),
    "ediff1d": NotImplemented,
    "round": NotImplemented,
    "polymul": NotImplemented,
    "isneginf": NotImplemented,
    "take_along_axis": NotImplemented,
    "mod": SUPPORTED_UFUNCS["remainder"],
    "real": NotImplemented,
    "polydiv": NotImplemented,
    "around": NotImplemented,
    "put_along_axis": NotImplemented,
    "capitalize": NotImplemented,
    "einsum_path": NotImplemented,
    "save": NotImplemented,
    "require": NotImplemented,
    "fft": NotImplemented,
    "transpose": _transpose,
    "imag": NotImplemented,
    "center": NotImplemented,
    "apply_along_axis": NotImplemented,
    "mean": NotImplemented,
    "einsum": NotImplemented,
    "apply_over_axes": NotImplemented,
    "savez": NotImplemented,
    "count": NotImplemented,
    "partition": NotImplemented,
    "iscomplex": NotImplemented,
    "sliding_window_view": NotImplemented,
    "fill_diagonal": NotImplemented,
    "expand_dims": NotImplemented,
    "savez_compressed": NotImplemented,
    "decode": NotImplemented,
    "std": NotImplemented,
    "diag_indices_from": NotImplemented,
    "column_stack": NotImplemented,
    "argpartition": NotImplemented,
    "isreal": NotImplemented,
    "loadtxt": NotImplemented,
    "encode": NotImplemented,
    "broadcast_to": _broadcast_to,
    "dstack": NotImplemented,
    "ifft": NotImplemented,
    "endswith": NotImplemented,
    "var": NotImplemented,
    "array_split": _array_split,
    "sort": NotImplemented,
    "iscomplexobj": NotImplemented,
    "savetxt": NotImplemented,
    "rfft": NotImplemented,
    "broadcast_arrays": NotImplemented,
    "expandtabs": NotImplemented,
    "round_": NotImplemented,
    "argsort": NotImplemented,
    "irfft": NotImplemented,
    "isrealobj": NotImplemented,
    "find": NotImplemented,
    "product": NotImplemented,
    "fliplr": NotImplemented,
    "hsplit": _hsplit,
    "index": NotImplemented,
    "hfft": NotImplemented,
    "vsplit": _vsplit,
    "flipud": NotImplemented,
    "argmax": NotImplemented,
    "cumproduct": NotImplemented,
    "nan_to_num": NotImplemented,
    "inner": NotImplemented,
    "isalnum": NotImplemented,
    "eye": eye,
    "dsplit": NotImplemented,
    "ihfft": NotImplemented,
    "sometrue": NotImplemented,
    "real_if_close": NotImplemented,
    "argmin": NotImplemented,
    "isalpha": NotImplemented,
    "kron": NotImplemented,
    "fftn": NotImplemented,
    "diag": _diag,
    "common_type": NotImplemented,
    "alltrue": NotImplemented,
    "isdigit": NotImplemented,
    "tile": _tile,
    "ifftn": NotImplemented,
    "diagflat": NotImplemented,
    "merge_arrays": NotImplemented,
    "searchsorted": NotImplemented,
    "log": NotImplemented,
    "islower": NotImplemented,
    "fft2": NotImplemented,
    "atleast_3d": NotImplemented,
    "histogram_bin_edges": NotImplemented,
    "tri": NotImplemented,
    "ifft2": NotImplemented,
    "isspace": NotImplemented,
    "log10": NotImplemented,
    "drop_fields": NotImplemented,
    "histogram": NotImplemented,
    "vstack": _vstack,
    "resize": NotImplemented,
    "rfftn": NotImplemented,
    "tril": NotImplemented,
    "istitle": NotImplemented,
    "logn": NotImplemented,
    "tensorsolve": NotImplemented,
    "histogramdd": NotImplemented,
    "where": _where,
    "solve": binary_op(cs.solve, shape_inference="solve"),
    "rec_drop_fields": NotImplemented,
    "rfft2": NotImplemented,
    "hstack": _hstack,
    "tensorinv": NotImplemented,
    "concatenate": _concatenate,
    "squeeze": _squeeze,
    "empty_like": _empty_like,
    "isupper": NotImplemented,
    "log2": NotImplemented,
    "triu": NotImplemented,
    "irfftn": NotImplemented,
    "inv": unary_op(cs.inv),
    "lexsort": NotImplemented,
    "rename_fields": NotImplemented,
    "zeros_like": zeros_like,
    "zeros": zeros,
    "join": NotImplemented,
    "stack": _stack,
    "diagonal": NotImplemented,
    "nanmin": NotImplemented,
    "power": NotImplemented,
    "vander": NotImplemented,
    "rot90": NotImplemented,
    "irfft2": NotImplemented,
    "matrix_power": NotImplemented,
    "take": NotImplemented,
    "ones": ones,
    "recursive_fill_fields": NotImplemented,
    "can_cast": NotImplemented,
    "append_fields": NotImplemented,
    "ljust": NotImplemented,
    "block": NotImplemented,
    "flip": NotImplemented,
    "arccos": NotImplemented,
    "histogram2d": NotImplemented,
    "trace": unary_op(cs.trace, shape_inference="unary_to_scalar"),
    "nanmax": NotImplemented,
    "cholesky": unary_op(cs.chol),
    "min_scalar_type": NotImplemented,
    "rec_append_fields": NotImplemented,
    "lower": NotImplemented,
    "average": NotImplemented,
    "arcsin": NotImplemented,
    "nanargmin": NotImplemented,
    "ones_like": ones_like,
    "qr": NotImplemented,
    "ravel": _ravel,
    "result_type": _result_type,
    "full": NotImplemented,
    "tril_indices_from": NotImplemented,
    "repack_fields": NotImplemented,
    "lstrip": NotImplemented,
    "piecewise": NotImplemented,
    "arctanh": NotImplemented,
    "eigvals": NotImplemented,
    "nanargmax": NotImplemented,
    "select": NotImplemented,
    "dot": _dot,
    "triu_indices_from": NotImplemented,
    "nonzero": NotImplemented,
    "full_like": NotImplemented,
    "structured_to_unstructured": NotImplemented,
    "eigvalsh": NotImplemented,
    "replace": NotImplemented,
    "copy": NotImplemented,
    "nansum": NotImplemented,
    "fftshift": NotImplemented,
    "vdot": NotImplemented,
    "count_nonzero": NotImplemented,
    "rfind": NotImplemented,
    "gradient": NotImplemented,
    "shape": lambda x: x.shape,
    "eig": NotImplemented,
    "unstructured_to_structured": NotImplemented,
    "nanprod": NotImplemented,
    "ifftshift": NotImplemented,
    "bincount": NotImplemented,
    "nancumsum": NotImplemented,
    "eigh": NotImplemented,
    "rindex": NotImplemented,
    "diff": NotImplemented,
    "argwhere": NotImplemented,
    "compress": NotImplemented,
    "apply_along_fields": NotImplemented,
    "interp": _interp1d,
    "ravel_multi_index": NotImplemented,
    "flatnonzero": NotImplemented,
    "svd": NotImplemented,
    "nancumprod": NotImplemented,
    "rjust": NotImplemented,
    "unravel_index": NotImplemented,
    "pad": NotImplemented,
    "cond": NotImplemented,
    "nanmean": NotImplemented,
    "correlate": NotImplemented,
    "rpartition": NotImplemented,
    "linspace": NotImplemented,
    "angle": NotImplemented,
    "clip": _clip,
    "assign_fields_by_name": NotImplemented,
    "copyto": NotImplemented,
    "convolve": NotImplemented,
    "unwrap": NotImplemented,
    "matrix_rank": NotImplemented,
    "rsplit": NotImplemented,
    "nanmedian": NotImplemented,
    "outer": _outer,
    "logspace": NotImplemented,
    "putmask": NotImplemented,
    "require_fields": NotImplemented,
    "astype": _astype,
    "reshape": _reshape,
    "sum": _sum,
    "sort_complex": NotImplemented,
    "pinv": unary_op(cs.pinv, shape_inference="transpose"),
    "rstrip": NotImplemented,
    "nanpercentile": NotImplemented,
    "tensordot": NotImplemented,
    "geomspace": NotImplemented,
    "trim_zeros": NotImplemented,
    "any": NotImplemented,
    "split": _split,
    "packbits": NotImplemented,
    "slogdet": NotImplemented,
    "stack_arrays": NotImplemented,
    "roll": _roll,
    "nanquantile": NotImplemented,
    "det": unary_op(cs.det, shape_inference="unary_to_scalar"),
    "splitlines": NotImplemented,
    "extract": NotImplemented,
    "rollaxis": NotImplemented,
    "unpackbits": NotImplemented,
    "nanvar": NotImplemented,
    "find_duplicates": NotImplemented,
    "all": NotImplemented,
    "place": NotImplemented,
    "moveaxis": NotImplemented,
    "lstsq": NotImplemented,
    "startswith": NotImplemented,
    "sqrt": NotImplemented,
    "shares_memory": NotImplemented,
    "norm": _norm,
    "join_by": NotImplemented,
    "strip": NotImplemented,
    "nanstd": NotImplemented,
    "cross": _cross,
    "cov": NotImplemented,
    "cumsum": NotImplemented,
    "swapcase": NotImplemented,
}
