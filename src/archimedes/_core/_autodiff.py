"""Autodiff transformations"""

from typing import Callable, Sequence

from . import SymbolicArray
from ._function import FunctionCache


def grad(
    func: Callable,
    argnums: int | Sequence[int] = 0,
    name: str | None = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
) -> Callable:
    """Create a function that evaluates the gradient of ``func``.

    Transforms a scalar-valued function into a new function that computes
    the gradient (vector of partial derivatives) with respect to one or more
    of its input arguments.

    Parameters
    ----------
    func : callable
        The function to differentiate. Should be a scalar-valued function
        (returns a single value with shape ``()``). If not already a compiled
        function, it will be compiled with specified static arguments.
    argnums : int or tuple of ints, optional
        Specifies which positional argument(s) to differentiate with respect to.
        Default is 0, meaning the first argument.
    name : str, optional
        Name for the created gradient function. If ``None``, a name is automatically
        generated based on the primal function's name.
    static_argnums : tuple of int, optional
        Specifies which positional arguments should be treated as static (not
        differentiated or traced symbolically). Only used if ``func`` is not already
        a compiled function.
    static_argnames : tuple of str, optional
        Specifies which keyword arguments should be treated as static. Only used
        if ``func`` is not already a compiled function.

    Returns
    -------
    callable
        A function that computes the gradient of ``func`` with respect to the
        specified arguments. If multiple arguments are specified in ``argnums``,
        the function returns a tuple of gradients, one for each specified argument.

    Notes
    -----
    When to use this function:
    - When you need to compute derivatives of a scalar-valued cost or objective function
    - For sensitivity analysis of scalar model outputs with respect to parameters

    Internally, CasADi chooses between forward and reverse mode automatic
    differentiation using a heuristic based on the number of required derivative
    calculations in either case.  For gradient calculations with a scalar output,
    typically reverse mode will be used.

    Conceptual model:
    This function uses automatic differentiation (AD) to efficiently compute exact
    derivatives, avoiding the numerical errors and computational cost of finite
    differencing approaches. Unlike numerical differentiation, automatic
    differentiation computes exact derivatives (to machine precision) by applying the
    chain rule to the computational graph generated from your function.

    The ``grad`` function specifically handles scalar-valued functions, returning the
    gradient as a column vector with the same shape as the input. For vector-valued
    functions, use ``jac`` (Jacobian) instead.

    Edge cases:

    - Raises ValueError if ``argnums`` contains a static argument index
    - Raises ValueError if the function does not return a single scalar value
    - Differentiating through non-differentiable operations (like ``abs`` at ``x=0``)
      may return a valid but potentially undefined result

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Basic example with scalar function
    >>> def f(x):
    >>>     return np.sin(x**2)
    >>>
    >>> df = arc.grad(f)
    >>> print(np.allclose(df(1.0), 2.0 * np.cos(1.0)))
    True
    >>>
    >>> # Multi-argument function with gradient w.r.t. specific argument
    >>> def loss(x, A, y):
    >>>     y_pred = A @ x
    >>>     return np.sum((y_pred - y)**2) / len(y)
    >>>
    >>> # Get gradient with respect to weights only
    >>> grad_A = arc.grad(loss, argnums=1)
    >>>
    >>> A = np.random.randn(2, 3)
    >>> x = np.ones(3)
    >>> y = np.array([1, 2])
    >>> print(grad_A(x, A, y))  # df/dA
    >>>
    >>> # Alternatively, differentiate with respect to multiple arguments
    >>> grads = arc.grad(loss, argnums=(0, 1))
    >>> # Returns a tuple (df/dx, df/dA)
    >>> print(grads(x, A, y))

    See Also
    --------
    jac : Compute the Jacobian matrix of a function
    hess : Compute the Hessian matrix of a scalar function
    """

    if not isinstance(func, FunctionCache):
        func = FunctionCache(
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
    # be used to create the gradient FunctionCache.
    def _grad(*args):
        # First make sure that the primal function has been compiled for these
        # argument types
        y = func(*args)
        if not isinstance(y, SymbolicArray):
            raise ValueError(
                "The primal function must return a single array. Multiple "
                "returns are not yet supported.  Return from "
                f"{func.name} is {y}"  # type: ignore[attr-defined]
            )
        if y.shape != ():
            raise ValueError(
                "The primal function must return a scalar value with shape (). "
                f"Return from {func.name} is {y} with shape "  # type: ignore[attr-defined]
                f"{y.shape}."
            )

        return tuple(y.grad(args[i]) for i in argnums)  # type: ignore[union-attr]

    if name is None:
        name = f"grad_{func.name}"

    _grad.__name__ = name

    return FunctionCache(
        _grad,
        arg_names=func.arg_names,
        static_argnums=func.static_argnums,
        kind=func._kind,
    )


def jac(
    func: Callable,
    argnums: int | Sequence[int] = 0,
    name: str | None = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
) -> Callable:
    """Create a function that evaluates the Jacobian of ``func``.

    Transforms a vector-valued function into a new function that computes
    the Jacobian matrix (matrix of all first-order partial derivatives) with
    respect to one or more of its input arguments.

    Parameters
    ----------
    func : callable
        The function to differentiate. Can return a vector or matrix output.
        If not already a compiled function, it will be compiled with the
        specified static arguments.
    argnums : int or tuple of ints, optional
        Specifies which positional argument(s) to differentiate with respect to.
        Default is 0, meaning the first argument.
    name : str, optional
        Name for the created Jacobian function. If ``None``, a name is automatically
        generated based on the primal function's name.
    static_argnums : tuple of int, optional
        Specifies which positional arguments should be treated as static (not
        differentiated or traced symbolically). Only used if ``func`` is not already
        a compiled function.
    static_argnames : tuple of str, optional
        Specifies which keyword arguments should be treated as static. Only used
        if ``func`` is not already a compiled function.

    Returns
    -------
    callable
        A function that computes the Jacobian of ``func`` with respect to the
        specified arguments. If multiple arguments are specified in ``argnums``,
        the function returns a tuple of Jacobians, one for each specified argument.

    Notes
    -----
    When to use this function:

    - When working with derivatives of vector-valued functions
    - For constrained optimization problems where Jacobians are needed
    - For sensitivity analysis of vector outputs with respect to input parameters
    - For linearization of nonlinear models around operating points

    In cases where the full Jacobian is not needed, but only the product of the
    Jacobian with a vector, consider using :py:func:`jvp` (Jacobian-vector product) or
    :py:func:`vjp` (vector-transpose-Jacobian product) for more efficient computation.

    Currently this function only supports creating Jacobians for functions
    with a single return value. If the function has multiple return values,
    the function will raise a ``ValueError``.

    Internally, CasADi chooses between forward and reverse mode automatic
    differentiation using a heuristic based on the number of required derivative
    calculations in either case. For functions with many inputs and few outputs,
    reverse mode is typically more efficient. For functions with few inputs and
    many outputs, forward mode is typically preferred.

    Conceptual model:

    The Jacobian matrix represents the best linear approximation to a function
    near a given point. For a function :math:`f: R^n → R^m`, the Jacobian is an
    :math:`m \\times n` matrix where each element (i,j) represents the partial
    derivative of the i-th output with respect to the j-th input.  Hence, the function
    returned by ``jac`` takes the form :math:`J: R^n → R^{m \\times n}`.

    Edge cases:

    - Raises ``ValueError`` if ``argnums`` contains a static argument index
    - Raises ``ValueError`` if the function does not return a single array
    - Currently only supports functions with a single return value (future versions
      may support multiple returns)

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Example: Jacobian of a simple vector function
    >>> def f(x):
    >>>     return np.array([x[0]**2, x[0]*x[1], np.sin(x[1])], like=x)
    >>>
    >>> J = arc.jac(f)
    >>> x = np.array([2.0, 3.0])
    >>> print(J(x))
    [[ 4.         0.       ]
     [ 3.         2.       ]
     [ 0.        -0.9899925]]
    >>>
    >>> # Multi-argument function with Jacobian w.r.t. specific argument
    >>> def dynamics(t, x, u):
    >>>     # Simple pendulum dynamics with control input.
    >>>     g = 9.81
    >>>     L = 1.0
    >>>     return np.array([
    >>>         x[1], -(g * np.sin(x[0]) + u) / L
    >>>     ], like=x)
    >>>
    >>> # Get Jacobian with respect to state (for linearization)
    >>> Jx = arc.jac(dynamics, argnums=1)
    >>> t = 0.0
    >>> x0 = np.array([0.0, 0.0])  # Equilibrium
    >>> u = 0.0
    >>> print(Jx(t, x0, u))
    [[ 0.    1.  ]
     [-9.81  0.  ]]
    >>>
    >>> # Get Jacobian with respect to control input (for control design)
    >>> Ju = arc.jac(dynamics, argnums=2)
    >>> print(Ju(t, x0, u))
    [0. 1.]

    See Also
    --------
    grad : Compute the gradient of a scalar-valued function
    hess : Compute the Hessian matrix of a scalar function
    jvp : Compute Jacobian-vector products
    vjp : Compute vector-Jacobian products
    """
    # TODO: Support multiple returns using trees?

    if not isinstance(func, FunctionCache):
        func = FunctionCache(
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
    # be used to create the Jacobian FunctionCache.
    def _jac(*args):
        # First make sure that the primal function has been compiled for these
        # argument types
        y = func(*args)
        if not isinstance(y, SymbolicArray):
            raise ValueError(
                "The primal function must return a single array. Multiple "
                "returns are not yet supported.  Return from "
                f"{func.name} is {y}"  # type: ignore[attr-defined]
            )
        return tuple(y.jac(args[i]) for i in argnums)  # type: ignore[union-attr]

    if name is None:
        name = f"jac_{func.name}"

    _jac.__name__ = name

    return FunctionCache(
        _jac,
        arg_names=func.arg_names,
        static_argnums=func.static_argnums,
        kind=func._kind,
    )


def hess(
    func: Callable,
    argnums: int | Sequence[int] = 0,
    name: str | None = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
) -> Callable:
    """Create a function that evaluates the Hessian of ``func``.

    Transforms a scalar-valued function into a new function that computes
    the Hessian matrix (matrix of all second-order partial derivatives) with
    respect to one or more of its input arguments.

    Parameters
    ----------
    func : callable
        The function to differentiate. Must be a scalar-valued function
        (returns a single value with shape ``()``). If not already a compiled
        function, it will be compiled with the specified static arguments.
    argnums : int or tuple of ints, optional
        Specifies which positional argument(s) to differentiate with respect to.
        Default is 0, meaning the first argument.
    name : str, optional
        Name for the created Hessian function. If None, a name is automatically
        generated based on the primal function's name.
    static_argnums : tuple of int, optional
        Specifies which positional arguments should be treated as static (not
        differentiated or traced symbolically). Only used if ``func`` is not already
        a compiled function.
    static_argnames : tuple of str, optional
        Specifies which keyword arguments should be treated as static. Only used
        if `func` is not already a compiled function.

    Returns
    -------
    callable
        A function that computes the Hessian of ``func`` with respect to the
        specified arguments. If multiple arguments are specified in ``argnums``,
        the function returns a tuple of Hessians, one for each specified argument.

    Notes
    -----
    When to use this function:

    - For optimization problems requiring second-derivative information
    - For analyzing the local curvature of a cost function
    - When working with quadratic approximations of nonlinear functions

    Conceptual model:

    The Hessian matrix represents the local curvature of a function. For a function
    :math:`f: R^n → R`, the Hessian is an :math:`n \\times n` symmetric matrix where
    each element (i,j) represents the second partial derivative
    :math:`\\partial^2 f / \\partial x_i \\partial x_j`.

    The Hessian is computed using automatic differentiation by applying the gradient
    operation twice. This ensures high numerical accuracy compared to finite
    differencing methods, especially for functions with complex computational paths.

    Edge cases:

    - Raises ``ValueError`` if ``argnums`` contains a static argument index
    - Currently only supports functions with a single return value

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Example: Hessian of a simple function
    >>> def f(x):
    >>>     return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2  # Rosenbrock function
    >>>
    >>> H = arc.hess(f)
    >>> x = np.array([1.0, 1.0])  # At the minimum
    >>> print(H(x))
    [[ 802. -400.]
    [-400.  200.]]
    >>>
    >>> # Example: Hessian for optimization
    >>> def loss(x, A, y):
    >>>     y_pred = x @ A
    >>>     return np.sum((y_pred - y)**2) / len(y)
    >>>
    >>> # Get Hessian with respect to parameters
    >>> H_params = arc.hess(loss, argnums=1)
    >>>
    >>> # Create some example data
    >>> x = np.random.randn(100, 5)  # 100 samples, 5 features
    >>> y = np.random.randn(100)
    >>> A = np.zeros(5)  # Initial parameters
    >>>
    >>> # Compute Hessian at initial point (useful for Newton optimization)
    >>> H = H_params(x, A, y)
    >>> print(H.shape)  # Should be (5, 5)

    See Also
    --------
    grad : Compute the gradient of a scalar-valued function
    jac : Compute the Jacobian matrix of a function
    minimize : Optimization using automatic differentiation
    """
    # TODO: Support multiple returns using trees?

    if not isinstance(func, FunctionCache):
        func = FunctionCache(
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
                "returns are not yet supported.  Return from "
                f"{func.name} is {y}"  # type: ignore[attr-defined]
            )
        if y.shape != ():
            raise ValueError(
                "The primal function must return a scalar value with shape (). "
                f"Return from {func.name} is {y} with shape "  # type: ignore[attr-defined]
                f"{y.shape}."
            )
        return tuple(y.hess(args[i]) for i in argnums)  # type: ignore[union-attr]

    if name is None:
        name = f"hess_{func.name}"

    _hess.__name__ = name

    return FunctionCache(
        _hess,
        arg_names=func.arg_names,
        static_argnums=func.static_argnums,
        kind=func._kind,
    )


def jvp(
    func: Callable,
    name: str | None = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
) -> Callable:
    """Create a function that evaluates the Jacobian-vector product of ``func``.

    Transforms a function into a new function that efficiently computes the
    product of the Jacobian matrix of ``func`` with a given vector, using
    forward-mode automatic differentiation.

    Parameters
    ----------
    func : callable
        The function to differentiate. If not already a compiled function,
        it will be compiled with the specified static arguments.
    name : str, optional
        Name for the created JVP function. If ``None``, a name is automatically
        generated based on the primal function's name.
    static_argnums : tuple of int, optional
        Specifies which positional arguments should be treated as static (not
        differentiated or traced symbolically). Only used if ``func`` is not already
        a compiled function.
    static_argnames : tuple of str, optional
        Specifies which keyword arguments should be treated as static. Only used
        if ``func`` is not already a compiled function.

    Returns
    -------
    callable
        A function with signature ``jvp_fun(x, v)`` that computes
        :math:`J(x) \\cdot v`, where :math:`J(x)` is the Jacobian of ``func`` evaluated
        at :math:`x`, and :math:`v` is the vector to multiply with. The function
        returns a vector with the same shape as the output of ``func``.

    Notes
    -----
    When to use this function:

    - When you need directional derivatives along a specific vector
    - When computing sensitivities for functions with many outputs and few inputs
    - When the full Jacobian matrix would be too large to compute or store efficiently
    - In iterative algorithms that require repeated Jacobian-vector products

    Conceptual model:

    The Jacobian-vector product (JVP) computes the directional derivative of a
    function in the direction of a given vector, without explicitly forming the
    full Jacobian matrix. For a function :math:`f: R^n \\rightarrow R^m` and a vector
    :math:`v \\in R^n`, the JVP is equivalent to :math:`J(x) \\cdot v`, where
    :math:`J(x)` is the :math:`m \\times n` Jacobian matrix at point :math:`x`.

    Forward-mode automatic differentiation computes JVPs efficiently, with a
    computational cost similar to that of evaluating the original function,
    regardless of the output dimension. This makes JVP particularly effective for
    functions with few inputs but many outputs.

    The JVP also represents how a small change in the input (in the direction of
    :math:`v`) affects the output of the function, making it useful for sensitivity
    analysis.

    Edge cases:

    - Raises ``ValueError`` if the function does not return a single vector-valued\
        array
    - The vector ``v`` must have the same shape as the input ``x``

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Example: JVP of a simple function
    >>> def f(x):
    >>>     return np.array([x[0]**2, x[0]*x[1], np.exp(x[1])], like=x)
    >>>
    >>> # Create the JVP function
    >>> f_jvp = arc.jvp(f)
    >>>
    >>> # Evaluate at a point
    >>> x = np.array([2.0, 1.0])
    >>> v = np.array([1.0, 0.5])  # Direction vector
    >>>
    >>> # Compute JVP: J(x)·v
    >>> auto_jvp = f_jvp(x, v)
    >>> print(auto_jvp)
    [4.         2.         1.35914091]
    >>>
    >>> # Compare with direct computation of Jacobian
    >>> manual_jvp = arc.jac(f)(x) @ v
    >>> print(np.allclose(auto_jvp, manual_jvp))
    True
    >>>
    >>> # Example: Efficient sensitivity analysis for a high-dimensional output
    >>> def high_dim_func(params):
    >>>     # Function with few inputs but many outputs.
    >>>     return np.sin(np.sum(np.outer(params, np.arange(1000)), axis=0))
    >>>
    >>> # JVP is much more efficient than computing the full Jacobian
    >>> sensitivity = arc.jvp(high_dim_func)
    >>> params = np.array([0.1, 0.2])
    >>> direction = np.array([1.0, 0.0])  # Sensitivity in first parameter direction
    >>>
    >>> # Compute how output changes in the direction of the first parameter
    >>> output_sensitivity = sensitivity(params, direction)
    >>> print(output_sensitivity.shape)
    (1000,)

    See Also
    --------
    vjp : Compute vector-Jacobian products (reverse-mode AD)
    jac : Compute the full Jacobian matrix
    grad : Compute the gradient of a scalar-valued function
    """

    # Note that the interface here differs from JAX, which has the signature
    # `jvp(func, primals, tangents) -> primals, tangents`. In this case the JVP
    # is computed symbolically up front and can then be evaluated efficiently for
    # every primal/tangent pair.

    if not isinstance(func, FunctionCache):
        func = FunctionCache(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    # Function to evaluate the JVP using the underlying CasADi function,
    # assuming that the arguments are already symbolic arrays. This can then
    # be used to create the JVP FunctionCache.
    def _jvp(x, v):
        # The return values can be a single SymbolicArray or a tuple of these.
        y = func(x)

        if not isinstance(y, SymbolicArray):
            raise ValueError(
                "The primal function must return a single array. Multiple "
                "returns are not yet supported.  Return from "
                f"{func.name} is {y}"  # type: ignore[attr-defined]
            )

        return y.jvp(x, v)

    if name is None:
        name = f"jvp_{func.name}"

    _jvp.__name__ = name

    func_name = func.name
    primal_name = func_name + "_primal"
    tangent_name = func_name + "_tangent"

    return FunctionCache(
        _jvp,
        arg_names=[primal_name, tangent_name],
        static_argnums=func.static_argnums,
        kind=func._kind,
    )


def vjp(
    func: Callable,
    name: str | None = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
) -> Callable:
    """Create a function that evaluates the vector-Jacobian product of ``func``.

    Transforms a function into a new function that efficiently computes the
    product of a vector with the transposed Jacobian matrix of ``func``, using
    reverse-mode automatic differentiation.

    Parameters
    ----------
    func : callable
        The function to differentiate. If not already a compiled function,
        it will be compiled with the specified static arguments.
    name : str, optional
        Name for the created VJP function. If ``None``, a name is automatically
        generated based on the primal function's name.
    static_argnums : tuple of int, optional
        Specifies which positional arguments should be treated as static (not
        differentiated or traced symbolically). Only used if ``func`` is not already
        a compiled function.
    static_argnames : tuple of str, optional
        Specifies which keyword arguments should be treated as static. Only used
        if ``func`` is not already a compiled function.

    Returns
    -------
    callable
        A function with signature ``vjp_fun(x, w)`` that computes
        :math:`w^T \\cdot J(x)`, where :math:`J(x)` is the Jacobian of ``func``
        evaluated at :math:`x` and :math:`w` is the vector to multiply with.
        The function returns a vector with the same shape as the input ``x``.

    Notes
    -----
    When to use this function:

    - When you need gradients of scalar projections of the output
    - When computing sensitivities for functions with many inputs and few outputs
    - In adjoint-based (e.g PDE-constrained) optimization problems

    Conceptual model:

    The vector-Jacobian product (VJP) computes the gradient of a scalar projection
    of the output without explicitly forming the full Jacobian matrix. For a function
    :math:`f: R^n \\rightarrow R^m` and a vector :math:`w \\in R^m`, the VJP is
    equivalent to :math:`w^T \\cdot J(x)`, where :math:`J(x)` is the
    :math:`m \\times n` Jacobian matrix evaluated at point :math:`x`.

    Reverse-mode automatic differentiation computes VJPs efficiently, with a
    computational cost similar to that of evaluating the original function,
    regardless of the input dimension. This makes VJP particularly effective for
    functions with many inputs but few outputs (which is why it's widely used in
    machine learning for gradient-based optimization). If the function is scalar-valued
    then the VJP with vector ``w=1`` is equivalent to the gradient of the function.

    The VJP represents the sensitivity of a weighted sum of outputs to changes in
    each input dimension.

    Edge cases:

    - Raises ``ValueError`` if the function does not return a single vector-valued array
    - The vector ``w`` must have the same shape as the output of ``func``

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Example: VJP of a simple function
    >>> def f(x):
    >>>     return np.array([x[0]**2, x[0]*x[1], np.exp(x[1])], like=x)
    >>>
    >>> # Create the VJP function
    >>> f_vjp = arc.vjp(f)
    >>>
    >>> # Evaluate at a point
    >>> x = np.array([2.0, 1.0])
    >>> w = np.array([1.0, 0.5, 2.0])  # Weighting vector for outputs
    >>>
    >>> # Compute VJP: w^T·J(x)
    >>> auto_vjp = f_vjp(x, w)
    >>> print(auto_vjp)
    [4.5        6.43656366]
    >>>
    >>> # Compare with direct computation of Jacobian transpose
    >>> J = arc.jac(f)
    >>> full_jacobian = J(x)
    >>> manual_vjp = w @ full_jacobian
    >>> print(np.allclose(input_gradient, manual_vjp))
    True
    >>>
    >>> # Example: Comparing efficiency with JVP
    >>> # For functions with many inputs but few outputs, VJP is more efficient
    >>> def high_dim_input_func(x):
    >>>     # Sum of squares of many inputs, producing a scalar
    >>>     return np.sum(x**2)
    >>>
    >>> # Gradient computation is equivalent to a VJP with w=1
    >>> grad_result = arc.grad(high_dim_input_func)(x)
    >>> x = np.random.randn(1000)  # 1000-dimensional input, scalar output
    >>> w = np.array(1.0)  # Weight for the scalar output
    >>>
    >>> # VJP computes the gradient efficiently without forming full Jacobian
    >>> vjp_result = arc.vjp(high_dim_input_func)(x, w)
    >>> print(np.allclose(vjp_result, grad_result))
    True

    See Also
    --------
    jvp : Compute Jacobian-vector products (forward-mode AD)
    jac : Compute the full Jacobian matrix
    grad : Compute the gradient of a scalar-valued function (special case of VJP)
    """

    # Note that the interface here differs from JAX, which has the signature
    # `vjp(func, *primals) -> primals, vjp_fun`, where `vjp_fun` has the signature
    # `vjp_fun(cotangents) -> cotangents`. In this case the VJP is computed
    # symbolically up front and can then be evaluated efficiently for every
    # primal/cotangent pair.

    if not isinstance(func, FunctionCache):
        func = FunctionCache(
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
                "returns are not yet supported.  Return from "
                f"{func.name} is {y}"  # type: ignore[attr-defined]
            )
        return y.vjp(x, w)

    if name is None:
        name = f"vjp_{func.name}"

    _vjp.__name__ = name

    # The new function will be evaluated with "primal" data that has the same shape
    # as the inputs, and "cotangent" data that has the same shape as the output. The
    # names need to match.
    no_return_names = func.return_names is None
    if no_return_names:
        # Since only a single return value is currently supported, we can just assume
        # that the function only outputs one value and give it an arbitrary name. If
        # the function actually returns multiple values, it will throw an error when
        # "compiled" later.
        func.return_names = ["y0"]
    cotangent_names = [f"d{name}" for name in func.return_names]

    # Reset to None so that the error about multiple returns is handled by
    # the VJP function and not by a mismatch in the number of return names.
    if no_return_names:
        func.return_names = None

    return FunctionCache(
        _vjp,
        arg_names=func.arg_names + cotangent_names,
        static_argnums=func.static_argnums,
        kind=func._kind,
    )
