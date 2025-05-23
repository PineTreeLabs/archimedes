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

from functools import wraps

import casadi as cs
import numpy as np
from numpy import exceptions as npex

from .._array_impl import SymbolicArray, _as_casadi_array, array
from .._type_inference import shape_inference, type_inference


def normalize_axis_index(axis, ndim, msg_prefix="axis"):
    if not isinstance(axis, int):
        raise TypeError(f"integer argument expected, got {type(axis)}")
    if axis >= ndim or axis < -ndim:
        raise npex.AxisError(
            f"{msg_prefix} {axis} is out of bounds for array of dimension {ndim}"
        )
    if axis < 0:
        axis += ndim
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
    shape_inference_rule: str = "none"
    type_inference_rule: str = "default"

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
        cs_inputs = map(_as_casadi_array, inputs)
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
        arr1, arr2 = map(_as_casadi_array, inputs)
        shape1, shape2 = map(np.shape, inputs)  # Original shapes
        return _broadcast_binary_operation(op, arr1, arr2, shape1, shape2, result_shape)


def _repmat(x, reps):
    # There is no NumPy function that does this - for external interfaces
    # `np.tile` should be used instead.  This is only a helper function
    # for broadcasting CasADi arrays.
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

    # Case 6 is a duplicate of case 5 with arguments reversed
    if len(shape1) == 2 and len(shape2) == 1:
        return _broadcast_binary_operation(
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
