from typing import TYPE_CHECKING, Callable, Tuple

import numpy as np

from archimedes.error import ShapeDtypeError

if TYPE_CHECKING:
    from ..typing import DTypeLike

ShapeLike = Tuple[int, ...]

__all__ = ["shape_inference", "type_inference", "ShapeLike", "DTypeLike"]


def _type_inference_default(*inputs):
    # For any scalar inputs, use the smallest scalar type that can represent them
    # For instance, if the operation is x ** 2, we want to get uint8 for 2, not int64
    scalar_types = tuple(np.min_scalar_type(x) for x in inputs if np.isscalar(x))

    # For any vector inputs, use the actual dtype of the array.
    vector_types = tuple(x.dtype for x in inputs if hasattr(x, "dtype"))

    # The result type is the result type of the scalar types and the vector types
    return np.result_type(*scalar_types, *vector_types)


def type_inference(rule, *inputs):
    return {
        "default": _type_inference_default,
    }[rule](*inputs)


def shape_inference(rule, *inputs):
    shapes = list(map(np.shape, inputs))
    inference_fn: Callable[..., ShapeLike] = {  # type: ignore
        "unary": _shape_inference_unary,
        "transpose": _shape_inference_transpose,
        "broadcast": _shape_inference_broadcast,
        "unary_to_scalar": _shape_inference_scalar,
        "matmul": _shape_inference_matmul,
        "solve": _shape_inference_solve,
        # "array": _shape_inference_array,
        "gradient": _shape_inference_gradient,
        "jacobian": _shape_inference_jacobian,
        "hessian": _shape_inference_hessian,
        "jvp": _shape_inference_jvp,
        "vjp": _shape_inference_vjp,
    }[rule]
    return inference_fn(*shapes)


def _shape_inference_scalar(*shapes: Tuple[ShapeLike, ...]) -> ShapeLike:
    return ()


def _shape_inference_unary(*shapes: Tuple[ShapeLike, ...]) -> ShapeLike:
    return shapes[0]  # type: ignore


def _shape_inference_transpose(*shapes: Tuple[ShapeLike, ...]) -> ShapeLike:
    return shapes[0][::-1]  # type: ignore


def _shape_inference_broadcast(*shapes: Tuple[ShapeLike, ...]) -> ShapeLike:
    # https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
    #
    # When operating on two arrays, NumPy compares their shapes element-wise. It starts
    # with the trailing (i.e. rightmost) dimension and works its way left. Two
    # dimensions are compatible when
    #
    #   1. they are equal, or
    #   2. one of them is 1.
    #
    # Input arrays do not need to have the same number of dimensions. The resulting
    # array will have the same number of dimensions as the input array with the
    # greatest number of dimensions, where the size of each dimension is the largest
    # size of the corresponding dimension among the input arrays. Note that missing
    # dimensions are assumed to have size one.
    return np.broadcast_shapes(*shapes)  # type: ignore


def _shape_inference_matmul(shape1: ShapeLike, shape2: ShapeLike) -> ShapeLike:
    # The rule is that the operation is over the trailing axis of x1 and
    # the leading axis of x2.  Hence, the resulting shape is all but the
    # eliminated axes
    if shape1[-1] != shape2[0]:
        raise ShapeDtypeError(
            f"shapes {shape1} and {shape2} not aligned: {shape1[-1]} (dim 0) != "
            f"{shape2[0]} (dim 1)"
        )

    return shape1[:-1] + shape2[1:]


def _shape_inference_solve(shape1: ShapeLike, shape2: ShapeLike) -> ShapeLike:
    # The leading axis of x1 must be the same as the leading axis of x2
    # For now, x2 must also be a vector
    if shape1[0] != shape2[0]:
        raise ShapeDtypeError(
            f"shapes {shape1} and {shape2} not aligned: {shape1[-1]} (dim 0) != "
            f"{shape2[0]} (dim 1)"
        )

    if len(shape2) != 1:
        raise ShapeDtypeError(f"shape {shape2} is not a vector")

    # The result is the trailing axis of x1
    return (shape1[-1],)


def _shape_inference_gradient(expr: ShapeLike, arg: ShapeLike) -> ShapeLike:
    # The expression here must be a scalar.  The gradient will then
    # have the shape of the argument.
    if np.prod(expr) > 1:
        raise ShapeDtypeError(
            f"Gradient expression must be a scalar, but got shape {expr}"
        )
    return arg


def _shape_inference_jacobian(expr: ShapeLike, arg: ShapeLike) -> ShapeLike:
    # Assuming that the shape of the expression is (n,) and the shape of the
    # argument is (m,), the shape of the Jacobian will be (n, m). This should
    # also work for column vectors (n, 1) and (m, 1). The special case is when
    # the shape of the expression is (), in which case the Jacobian should have
    # shape (m,) to be consistent with JAX.

    if len(expr) >= 2 and expr[1] != 1:
        raise ShapeDtypeError("Jacobian expression must be a vector")

    if len(arg) >= 2 and arg[1] != 1:
        raise ShapeDtypeError("Jacobian argument must be a vector")

    if len(arg) == 0:
        if len(expr) == 0:
            return ()  # For scalar expression and argument return a scalar
        else:
            return (expr[0],)

    m = arg[0]

    if len(expr) == 0:
        return (m,)  # For scalar expression return a (1, m) row vector
    else:
        n = expr[0]

    return (n, m)


def _shape_inference_hessian(expr: ShapeLike, arg: ShapeLike) -> ShapeLike:
    if len(expr) != 0:
        raise ShapeDtypeError("Hessian expression must be a scalar")
    if len(arg) != 1:
        raise ShapeDtypeError("Hessian argument must be a vector")
    return (arg[0], arg[0])


def _shape_inference_jvp(f: ShapeLike, x: ShapeLike, _v: ShapeLike) -> ShapeLike:
    """Shape inference for Jacobian-vector product f'(x) * v."""
    # For a Jacobian-vector product, the expression `f` must be "vector-like",
    # i.e. with shape (), (m,), or (m, 1), with a dependence on the vector-like
    # argument `x` with shape (), (n,) or (n, 1).  The Jacobian would have shape
    # (m, n), so with tangent vector `v` with the same shape of `x`, the result
    # of the JVP should have the shape as the expression `f`.

    # Handle error cases first
    if len(x) == 2 and x[1] != 1:
        raise ShapeDtypeError("JVP argument must be a vector or scalar")

    if len(f) == 2 and f[1] != 1:
        raise ShapeDtypeError("JVP expression must be a vector or scalar")

    return f


def _shape_inference_vjp(f: ShapeLike, x: ShapeLike, _w: ShapeLike) -> ShapeLike:
    """Shape inference for transposed-Jacobian-vector product w * f'(x)."""
    # For a vector-Jacobian product, the expression `f` must be "vector-like",
    # i.e. with shape (), (m,), or (m, 1), with a dependence on the vector-like
    # argument `x` with shape (), (n,) or (n, 1).  The Jacobian would have shape
    # (m, n), so with cotangent vector `w` with the same shape of `f`, the result
    # of the VJP should have the shape as the argument `x`.

    # Handle error cases first
    if len(x) == 2 and x[1] != 1:
        raise ShapeDtypeError("VJP argument must be a vector or scalar")

    if len(f) == 2 and f[1] != 1:
        raise ShapeDtypeError("VJP expression must be a vector or scalar")

    return x
