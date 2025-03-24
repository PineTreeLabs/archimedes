"""Autodiff transformations"""

from ._function import SymbolicFunction
from . import SymbolicArray


def grad(
    func,
    argnums=0,
    name=None,
    static_argnums=None,
    static_argnames=None,
):
    """Create a function that evaluates the gradient of `func`.

    This is effectively the same as `jac`, except that it only supports differentiation
    with respect to a scalar return value.  Whereas the Jacobian of such a function
    would be a row vector, the gradient is returned as a column.
    """
    # TODO: expand docstring

    if not isinstance(func, SymbolicFunction):
        func = SymbolicFunction(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    if isinstance(argnums, int):
        argnums = (argnums,)

    if not isinstance(argnums, tuple) or not all(isinstance(i, int) for i in argnums):
        raise ValueError("argnums must be an integer or a tuple of integers")

    if any(i in func.static_argnums for i in argnums):
        raise ValueError("Cannot differentiate with respect to a static argument")

    # Function to evaluate the gradient using the underlying CasADi function,
    # assuming that the arguments are already symbolic arrays. This can then
    # be used to create the gradient SymbolicFunction.
    def _grad(*args):
        # First make sure that the primal function has been compiled for these
        # argument types
        y = func(*args)
        if not isinstance(y, SymbolicArray):
            raise ValueError(
                "The primal function must return a single array. Multiple "
                f"returns are not yet supported.  Return from {func.name} is {y}"
            )
        if y.shape != ():
            raise ValueError(
                "The primal function must return a scalar value with shape (). "
                f"Return from {func.name} is {y} with shape {y.shape}."
            )

        return tuple(y.grad(args[i]) for i in argnums)

    if name is None:
        name = f"grad_{func.name}"

    _grad.__name__ = name

    return SymbolicFunction(
        _grad,
        arg_names=func.arg_names,
        static_argnums=func.static_argnums,
        kind=func._kind,
    )


def jac(
    func,
    argnums=0,
    name=None,
    static_argnums=None,
    static_argnames=None,
):
    """Create a function that evaluates the Jacobian of `func`.

    Similar in spirit to `jax.jacfwd` and `jax.jacrev`.

    Currently only supports functions with a single return value.
    """
    # TODO: expand docstring
    # TODO: Support multiple returns?

    if not isinstance(func, SymbolicFunction):
        func = SymbolicFunction(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    if isinstance(argnums, int):
        argnums = (argnums,)

    if not isinstance(argnums, tuple) or not all(isinstance(i, int) for i in argnums):
        raise ValueError("argnums must be an integer or a tuple of integers")

    if any(i in func.static_argnums for i in argnums):
        raise ValueError("Cannot differentiate with respect to a static argument")

    # From the CasADi docs:
    # f: (x, y) -> (r, s) results in the function
    # df: (x, y, out_r, out_s) -> (jac_r_x, jac_r_y, jac_s_x, jac_s_y)

    # Function to evaluate the Jacobian using the underlying CasADi function,
    # assuming that the arguments are already symbolic arrays. This can then
    # be used to create the Jacobian SymbolicFunction.
    def _jac(*args):
        # First make sure that the primal function has been compiled for these
        # argument types
        y = func(*args)
        if not isinstance(y, SymbolicArray):
            raise ValueError(
                "The primal function must return a single array. Multiple "
                f"returns are not yet supported.  Return from {func.name} is {y}"
            )
        return tuple(y.jac(args[i]) for i in argnums)

    if name is None:
        name = f"jac_{func.name}"

    _jac.__name__ = name

    return SymbolicFunction(
        _jac,
        arg_names=func.arg_names,
        static_argnums=func.static_argnums,
        kind=func._kind,
    )


def hess(
    func,
    argnums=0,
    name=None,
    static_argnums=None,
    static_argnames=None,
):
    """Create a function that evaluates the Hessian of `func`.

    The function must have a single scalar return value (i.e. shape ()).
    """
    # TODO: expand docstring
    # TODO: Support multiple returns?
    
    if not isinstance(func, SymbolicFunction):
        func = SymbolicFunction(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    if isinstance(argnums, int):
        argnums = (argnums,)

    if not isinstance(argnums, tuple) or not all(isinstance(i, int) for i in argnums):
        raise ValueError("argnums must be an integer or a tuple of integers")

    if any(i in func.static_argnums for i in argnums):
        raise ValueError("Cannot differentiate with respect to a static argument")

    def _hess(*args):
        # First make sure that the primal function has been compiled for these
        # argument types
        y = func(*args)
        if not isinstance(y, SymbolicArray):
            raise ValueError(
                "The primal function must return a single array. Multiple "
                f"returns are not yet supported.  Return from {func.name} is {y}"
            )
        if y.shape != ():
            raise ValueError(
                "The primal function must return a scalar value with shape (). "
                f"Return from {func.name} is {y} with shape {y.shape}."
            )
        return tuple(y.hess(args[i]) for i in argnums)

    if name is None:
        name = f"hess_{func.name}"

    _hess.__name__ = name

    return SymbolicFunction(
        _hess,
        arg_names=func.arg_names,
        static_argnums=func.static_argnums,
        kind=func._kind,
    )


def jvp(
    func,
    name=None,
    static_argnums=None,
    static_argnames=None,
):
    """Create a function that evaluates the Jacobian-vector product of `func`.

    This will use forward-mode automatic differentiation to compute the product
    of the Jacobian of `func` with the given vector. For a function `f(x)` the
    returned function will have the signature `jvp_fun(x, v) = f'(x) * v`.

    """
    # TODO: expand docstring

    # Note that the interface here differs from JAX, which has the signature
    # `jvp(func, primals, tangents) -> primals, tangents`. In this case the JVP
    # is computed symbolically up front and can then be evaluated efficiently for
    # every primal/tangent pair.

    if not isinstance(func, SymbolicFunction):
        func = SymbolicFunction(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    # Function to evaluate the JVP using the underlying CasADi function,
    # assuming that the arguments are already symbolic arrays. This can then
    # be used to create the JVP SymbolicFunction.
    def _jvp(x, v):
        print(f"primals: {x}, {type(x)}, {x.shape}")

        # The return values can be a single SymbolicArray or a tuple of these.
        y = func(x)

        if not isinstance(y, SymbolicArray):
            raise ValueError(
                "The primal function must return a single array. Multiple "
                f"returns are not yet supported.  Return from {func.name} is {y}"
            )

        return y.jvp(x, v)

    if name is None:
        name = f"jvp_{func.name}"

    _jvp.__name__ = name

    func_name = func.name
    primal_name = func_name + "_primal"
    tangent_name = func_name + "_tangent"

    return SymbolicFunction(
        _jvp,
        arg_names=[primal_name, tangent_name],
        static_argnums=func.static_argnums,
        kind=func._kind,
    )


def vjp(
    func,
    name=None,
    static_argnums=None,
    static_argnames=None,
):
    """Create a function that evaluates the vector-Jacobian product of `func`.

    This will use reverse-mode automatic differentiation to compute the product
    of the transposed-Jacobian of `func` with the given vector. For a function
    `f(x)` the returned function will have the signature `vjp_fun(x, w) = f'(x)^T * w`.

    """
    # TODO: expand docstring

    # Note that the interface here differs from JAX, which has the signature
    # `vjp(func, *primals) -> primals, vjp_fun`, where `vjp_fun` has the signature
    # `vjp_fun(cotangents) -> cotangents`. In this case the VJP is computed
    # symbolically up front and can then be evaluated efficiently for every
    # primal/cotangent pair.

    if not isinstance(func, SymbolicFunction):
        func = SymbolicFunction(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    # For now, only support functions with a single argument and return value
    if len(func.arg_names) != 1:
        raise NotImplementedError("TODO: Support multiple arguments")

    def _vjp(x, w):
        # First make sure that the primal function has been compiled for these
        # argument types
        y = func(x)
        if not isinstance(y, SymbolicArray):
            raise ValueError(
                "The primal function must return a single array. Multiple "
                f"returns are not yet supported.  Return from {func.name} is {y}"
            )
        return y.vjp(x, w)

    if name is None:
        name = f"vjp_{func.name}"

    _vjp.__name__ = name

    # The new function will be evaluated with "primal" data that has the same shape
    # as the inputs, and "cotangent" data that has the same shape as the output. The
    # names need to match.
    no_ret_names = func.ret_names is None
    if no_ret_names:
        # Since only a single return value is currently supported, we can just assume
        # that the function only outputs one value and give it an arbitrary name. If
        # the function actually returns multiple values, it will throw an error when
        # "compiled" later.
        func.ret_names = ["y0"]
    cotangent_names = [f"d{name}" for name in func.ret_names]

    # Reset to None so that the error about multiple returns is handled by
    # the VJP function and not by a mismatch in the number of return names.
    if no_ret_names:
        func.ret_names = None

    return SymbolicFunction(
        _vjp,
        arg_names=func.arg_names + cotangent_names,
        static_argnums=func.static_argnums,
        kind=func._kind,
    )
