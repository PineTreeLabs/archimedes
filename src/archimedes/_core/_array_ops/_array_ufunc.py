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

import casadi as cs
import numpy as np

from .._array_impl import DEFAULT_FLOAT, SymbolicArray, _as_casadi_array
from .._type_inference import shape_inference, type_inference
from ._array_ops import array, binary_op, unary_op


def _dot(x1, x2):
    # If either x1 or x2 are scalars, then use element-wise multiplication.
    x1_is_scalar = np.isscalar(x1) or len(x1.shape) == 0
    x2_is_scalar = np.isscalar(x2) or len(x2.shape) == 0
    if x1_is_scalar or x2_is_scalar:
        return x1 * x2

    # Otherwise, we need to use CasADi's `dot` function for the symbolic
    # component. `dot` should close over the inner dimension of x1 and the outer
    # dimension of x2. The shape inference also checks that the shapes are compatible.
    dtype = type_inference("default", x1, x2)
    x1, x2 = map(array, (x1, x2))
    shape = shape_inference("matmul", x1, x2)

    # Now that type inference is done we can discard the SymbolicArray wrapper and just
    # use the underlying data (SX or NumPy array)
    arg1, arg2 = map(_as_casadi_array, (x1, x2))

    # CasADi's `dot` function requires that the two arrays have the same shape, and
    # acts as a flattened vector dot product. However, for NumPy it is equivalent to
    # matrix multiplication.  Hence, we should only use `cs.dot` if both are column
    # vectors.
    if len(x1.shape) == 1 and len(x2.shape) == 1:
        result = cs.dot(arg1, arg2)

    else:
        # If the shape of the array is (n,) then the CasADi symbolic array will be
        # (n, 1), whereas CasADi will expect a (1, n) array for a column vector.
        # Note that initialization of the SymbolicArray enforces that if x.shape = (n,)
        # then x._sym.shape = (n, 1), so the second argument does not need to be checked
        # here.
        if len(x1.shape) == 1:
            # First argument is a SymbolicArray: transpose the underlying CasADi array
            if isinstance(x1, SymbolicArray) and arg1.shape[1] == 1:
                arg1 = arg1.T
            # First argument is an ndarray: make into a row vector
            elif isinstance(x1, np.ndarray):
                arg1 = arg1.reshape((1, -1))

        result = cs.mtimes(arg1, arg2)

    return SymbolicArray(result, dtype=dtype, shape=shape)


def _matmul(x1, x2):
    # If either x1 or x2 are scalars, then use element-wise multiplication.
    x1_is_scalar = np.isscalar(x1) or len(x1.shape) == 0
    if x1_is_scalar:
        raise ValueError("Matmul input 0 does not have enough dimensions (requires 1)")
    x2_is_scalar = np.isscalar(x2) or len(x2.shape) == 0
    if x2_is_scalar:
        raise ValueError("Matmul input 1 does not have enough dimensions (requires 1)")
    # NumPy matmul is the same as dot for 2D arrays
    return _dot(x1, x2)


@binary_op
def _floor_divide(x1, x2):
    return cs.floor(x1 / x2)


def _divmod(x1, x2, dtype=None):
    y1 = np.floor_divide(x1, x2, dtype=dtype)
    y2 = np.remainder(x1, x2, dtype=dtype)
    return y1, y2


def _hypot(x, y):
    return np.sqrt(x * x + y * y)


def _xor(a, b):
    return cs.logic_or(
        cs.logic_and(a, cs.logic_not(b)), cs.logic_and(cs.logic_not(a), b)
    )


def _radians(x):
    return x * (np.pi / 180.0)


def _degrees(x):
    return x * (180.0 / np.pi)


# List from numpy.testing.overrides.get_overridable_numpy_ufuncs()
SUPPORTED_UFUNCS = {
    "floor": unary_op(cs.floor),
    # Note that np.maximum propagates NaN
    "maximum": binary_op(cs.fmax, result_type=DEFAULT_FLOAT),
    "floor_divide": _floor_divide,
    # Note that np.minimum propagates NaN
    "minimum": binary_op(cs.fmin, result_type=DEFAULT_FLOAT),
    "fmax": binary_op(cs.fmax, result_type=DEFAULT_FLOAT),
    "modf": NotImplemented,
    "fmin": binary_op(cs.fmin, result_type=DEFAULT_FLOAT),
    "multiply": binary_op(cs.times),
    "fmod": binary_op(cs.fmod),
    "negative": unary_op(lambda x: -x),
    "frexp": NotImplemented,
    "nextafter": NotImplemented,
    "gcd": NotImplemented,
    "not_equal": binary_op(cs.ne, result_type=bool),
    "cosh": unary_op(cs.cosh, result_type=DEFAULT_FLOAT),
    "greater": binary_op(cs.gt, result_type=bool),
    "positive": NotImplemented,
    "cos": unary_op(cs.cos, result_type=DEFAULT_FLOAT),
    "greater_equal": binary_op(cs.ge, result_type=bool),
    "power": binary_op(cs.power),
    "copysign": NotImplemented,
    "heaviside": NotImplemented,
    "rad2deg": _degrees,
    "conjugate": NotImplemented,
    "hypot": _hypot,
    "radians": _radians,
    "clip": NotImplemented,
    "invert": unary_op(cs.logic_not, result_type=bool),
    "reciprocal": NotImplemented,
    "ceil": unary_op(cs.ceil),
    "isfinite": NotImplemented,
    "remainder": binary_op(cs.remainder),  # np.remainder and np.mod,
    "cbrt": NotImplemented,
    "isinf": NotImplemented,
    "right_shift": NotImplemented,
    "bitwise_xor": NotImplemented,
    "isnan": NotImplemented,
    "rint": NotImplemented,
    "bitwise_or": NotImplemented,
    "isnat": NotImplemented,
    "sign": unary_op(cs.sign),
    "_ones_like": NotImplemented,
    # Note that np.absolute handles complex
    "absolute": unary_op(cs.fabs),
    "bitwise_and": NotImplemented,
    "lcm": NotImplemented,
    "signbit": NotImplemented,
    "add": binary_op(cs.plus),
    "arctanh": unary_op(cs.atanh, result_type=DEFAULT_FLOAT),
    "ldexp": NotImplemented,
    "sin": unary_op(cs.sin, result_type=DEFAULT_FLOAT),
    "arccos": unary_op(cs.acos, result_type=DEFAULT_FLOAT),
    "arctan2": binary_op(cs.atan2, result_type=DEFAULT_FLOAT),
    "left_shift": NotImplemented,
    "sinh": unary_op(cs.sinh, result_type=DEFAULT_FLOAT),
    "arccosh": unary_op(cs.acosh, result_type=DEFAULT_FLOAT),
    "arctan": unary_op(cs.atan, result_type=DEFAULT_FLOAT),
    "less": binary_op(cs.lt, result_type=bool),
    "spacing": NotImplemented,
    "arcsin": unary_op(cs.asin, result_type=DEFAULT_FLOAT),
    "less_equal": binary_op(cs.le, result_type=bool),
    "sqrt": unary_op(cs.sqrt, result_type=DEFAULT_FLOAT),
    "arcsinh": unary_op(cs.asinh, result_type=DEFAULT_FLOAT),
    "deg2rad": _radians,
    "log": unary_op(cs.log, result_type=DEFAULT_FLOAT),
    "square": NotImplemented,
    "degrees": _degrees,
    "log10": unary_op(cs.log10, result_type=DEFAULT_FLOAT),
    "subtract": binary_op(cs.minus),
    "divide": binary_op(cs.rdivide, result_type=DEFAULT_FLOAT),
    "true_divide": binary_op(cs.rdivide, result_type=DEFAULT_FLOAT),
    "log1p": unary_op(cs.log1p, result_type=DEFAULT_FLOAT),
    "tan": unary_op(cs.tan, result_type=DEFAULT_FLOAT),
    "divmod": _divmod,
    "log2": NotImplemented,
    "tanh": unary_op(cs.tanh, result_type=DEFAULT_FLOAT),
    "equal": binary_op(cs.eq, result_type=bool),
    "logaddexp": NotImplemented,
    "trunc": NotImplemented,
    "exp": unary_op(cs.exp, result_type=DEFAULT_FLOAT),
    "logaddexp2": NotImplemented,
    "exp2": NotImplemented,
    "logical_and": binary_op(cs.logic_and, result_type=bool),
    "expm1": NotImplemented,
    "logical_not": unary_op(cs.logic_not, result_type=bool),
    "fabs": unary_op(cs.fabs),
    "logical_or": binary_op(cs.logic_or, result_type=bool),
    "float_power": NotImplemented,
    "logical_xor": binary_op(_xor, result_type=bool),
    "matmul": _matmul,
}
