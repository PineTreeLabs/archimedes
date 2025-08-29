from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Hashable, NamedTuple, Sequence, Tuple

import casadi as cs
import numpy as np

from archimedes import tree
from archimedes._core._array_impl import DEFAULT_SYM_NAME, _as_casadi_array, array

from .. import sym_like

if TYPE_CHECKING:
    from archimedes.tree._flatten_util import HashablePartial

    # Type alias for the key in the compiled dictionary
    # This will be first the shape/dtype of all arguments and then a tuple
    # of the static arguments, which are restricted only by the requirement
    # that they be hashable.
    CompiledKey = Tuple[Tuple[HashablePartial, ...], Tuple[Hashable, ...]]


_lambda_idx = 0  # Global index for naming anonymous functions


class CompiledFunction(NamedTuple):
    """Container for a CasADi function specialized to particular arg types.

    The FunctionCache operates similarly to JAX transformations in that it does
    not need to know the shapes and dtypes of the arguments at creation.  Instead,
    a specialized version of the function is created each time the FunctionCache
    is called with different argument types.  This class stores a single instance
    of each of these specialized functions.

    This class is not intended to be used directly by the user.
    """

    func: cs.Function
    results_unravel: tuple[HashablePartial, ...]

    def __call__(self, *args):
        args = tuple(map(_as_casadi_array, args))
        result = self.func(*args)

        if not isinstance(result, tuple):
            result = (result,)

        # The result will be a CasADi type - convert it to a NumPy array or
        # SymbolicArray
        result = [
            unravel(array(x)) for (x, unravel) in zip(result, self.results_unravel)
        ]

        if len(result) == 1:
            result = result[0]

        return result

    def codegen(self, filename, options):
        # Call CasADi's C codegen function. Typically it will be easier to call
        # `archimedes.codegen(func, filename, args, **options)` instead
        # of trying to access this method directly.
        try:
            self.func.generate(filename, options)
        except RuntimeError as e:
            print(
                f"Error generating C code: {e}.  Note that CasADi does not support "
                "codegen to paths other than the working directory. If the error "
                "indicates that `Function::check_name` failed, likely the filename "
                "includes a path (`/`) or other invalid characters."
            )
            raise e


def _resolve_signature(func, arg_names):
    # Determine the full signature of the Python function, including all static
    # arguments.

    # By default just get the argument names from the function signature
    # Note that this will not work with functions defined with *args, e.g.
    # for function created dynamically by wrapping with `integrator`,
    # `implicit`, etc.
    if arg_names is None:
        signature = inspect.signature(func)

        # So far only POSITIONAL_OR_KEYWORD arguments are allowed until
        # it's more clear how to process other kinds of arguments like
        # varargs and keyword-only arguments.
        valid_kinds = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        for param in signature.parameters.values():
            if param.kind not in valid_kinds:
                raise ValueError(
                    "Currently compiled functions only support explicit arguments "
                    "(i.e. no *args, **kwargs, or keyword-only arguments). Found "
                    f"{func} with parameter {param.name} of kind {param.kind}"
                )

    else:
        # Assume that all the arguments are positional and allowed to
        # be specified by keyword as well
        parameters = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in arg_names
        ]
        signature = inspect.Signature(parameters)

    return signature


class FunctionCache:
    def __init__(
        self,
        func,
        arg_names=None,
        return_names=None,
        static_argnums=None,
        static_argnames=None,
        jit=False,
        kind=DEFAULT_SYM_NAME,
        name=None,
    ):
        self._func = func  # The original Python function

        if name is None:
            if hasattr(func, "__name__"):
                name = func.__name__
                if name == "<lambda>":
                    global _lambda_idx
                    name = f"lambda_{_lambda_idx}"
                    _lambda_idx += 1
            else:
                name = f"function_{id(func)}"

        self.name = name

        # Kind of symbolic object to use
        # TODO: Make this `scalar: bool` instead of `kind: str`
        self._kind = kind

        # Should we JIT compile the function?
        self._jit = jit

        self._compiled: dict[CompiledKey, CompiledFunction] = {}

        # Determine the signature of the original function.  If not
        # specified explicitly, it will be inferred using `inspect.signature`
        self.signature = _resolve_signature(func, arg_names)

        self.default_return_names = return_names is None
        self.return_names = return_names  # Can still be None at this point

        if static_argnums is not None and static_argnames is not None:
            raise ValueError(
                "Only one of static_argnums and static_argnames can be provided"
            )

        self.static_argnums = []
        if static_argnums is not None:
            if not isinstance(static_argnums, (list, tuple)):
                static_argnums = [static_argnums]
            self.static_argnums = static_argnums

        if static_argnames is not None:
            if not isinstance(static_argnames, (list, tuple)):
                static_argnames = [static_argnames]
            for name in static_argnames:
                if name not in self.arg_names:
                    raise ValueError(f"Argument {name} not found in function signature")
                self.static_argnums.append(self.arg_names.index(name))

    @property
    def arg_names(self):
        return list(self.signature.parameters.keys())

    def __repr__(self):
        return f"{self.name}({', '.join(self.arg_names)})"

    def _split_func(self, static_args, sym_args):
        # Wrap the function call by interleaving the static and symbolic arguments
        # in the original function order, then call the function.
        args = []
        sym_idx = 0
        for i in range(len(self.arg_names)):
            if i in self.static_argnums:
                args.append(static_args[self.static_argnums.index(i)])
            else:
                args.append(sym_args[sym_idx])
                sym_idx += 1

        return self._func(*args)

    def _compile(self, specialized_func, args_unravel, *args) -> CompiledFunction:
        # Create a casadi.Function for the particular argument types
        # and store it in a CompiledFunction object along with the
        # type information.

        # The function so far is specialized for the static data, so
        # here it is reduced to only a function of symbolic arguments.
        # So far these are still NumPy arrays, but they will be converted
        # to symbolic objects in the next step.

        # NOTE: Checking for consistency of the argument types is done by
        # `signature.bind` in `specialize`, so at this point we can expect a
        # consistent signature at least

        arg_names = [
            self.arg_names[i]
            for i in range(len(self.arg_names))
            if i not in self.static_argnums
        ]

        # Create symbolic arguments matching the types of the caller arguments
        # At this point all the symbolic arguments will be "flat", meaning that
        # dict- or tuple-structured arguments will be flat arrays.
        sym_args = [
            sym_like(x, name, kind=self._kind) for (x, name) in zip(args, arg_names)
        ]
        cs_args = [x._sym for x in sym_args]

        # Unravel tree-structured arguments before calling the function
        # For instance, if the original function was called with a dict, this will
        # create a dict with symbolic entries matching the original argument structure
        sym_args = [unravel(x) for (x, unravel) in zip(sym_args, args_unravel)]

        # Evaluate the function symbolically.  At this point we will have
        # everything we need to construct the CasADi function.
        sym_ret = specialized_func(sym_args)

        if not isinstance(sym_ret, (tuple, list)) or hasattr(sym_ret, "_fields"):
            sym_ret = (sym_ret,)

        # Ravel all return types to flattened arrays before creating the CasADi function
        # This list will still have the same length as the original returns (so same
        # number of return_names), but each _element_ will be individually flattened.
        sym_ret_flat = []
        results_unravel = []
        for x in sym_ret:
            x_flat, unravel = tree.ravel(x)
            sym_ret_flat.append(x_flat)
            results_unravel.append(unravel)

        cs_ret = [_as_casadi_array(x) for x in sym_ret_flat]

        if self.return_names is None:
            self.return_names = [f"y{i}" for i in range(len(sym_ret_flat))]
        else:
            if len(self.return_names) != len(sym_ret_flat):
                raise ValueError(
                    f"Expected {len(sym_ret_flat)} return values, got "
                    f"{len(self.return_names)} in call to {self.name}"
                )

        options = {
            "jit": self._jit,
        }
        # print(f"Compiling {self.name} for {cs_args} -> {cs_ret}")
        _compiled_func = cs.Function(
            self.name, cs_args, cs_ret, arg_names, self.return_names, options
        )

        return CompiledFunction(_compiled_func, tuple(results_unravel))

    def split_args(self, static_argnums, *args):
        # Given a set of positional arguments, split them into the static
        # and dynamic (i.e. possibly symbolic) arguments.
        static_args = []
        dynamic_args = []
        args_unravel = []
        for i, x in enumerate(args):
            if i in static_argnums:
                if not isinstance(x, Hashable) and not isinstance(x, np.ndarray):
                    raise ValueError(
                        f"Static argument {x} must be hashable or numpy array, but "
                        f"type {type(x)} is unhashable"
                    )
                static_args.append(x)
            else:
                x_flat, unravel = tree.ravel(x)
                dynamic_args.append(x_flat)
                args_unravel.append(unravel)
        return static_args, dynamic_args, args_unravel

    def _specialize(self, *args, **kwargs) -> Tuple[CompiledFunction, Tuple[Any]]:
        """Process the arguments and compile the function if necessary."""
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Since we initially enforced that all arguments can be identified as either
        # positional or keyword, we can now extract the arguments in the order
        # they were defined in the function signature and apply them as strictly
        # positional.
        pos_args = [bound_args.arguments[name] for name in self.arg_names]

        # Map the arguments to their shape and data types
        # For static arguments, add to a separate list for the purpose
        # of constructing the key in the hash table of compiled variants
        static_args, dynamic_args, args_unravel = self.split_args(
            self.static_argnums, *pos_args
        )

        # The key is a tuple of the argument types and the static arguments
        # Only hashable objects and NumPy arrays are allowed as static arguments
        static_arg_keys = tuple(
            arg if isinstance(arg, Hashable) else str(arg) for arg in static_args
        )
        key = tuple(args_unravel), tuple(static_arg_keys)
        # print(f"Calling {self.name} with key {key}")

        if key not in self._compiled:
            # Specialize the function for the static arguments
            func = partial(self._split_func, static_args)
            self._compiled[key] = self._compile(func, args_unravel, *dynamic_args)

        return self._compiled[key], dynamic_args

    def __call__(self, *args, **kwargs):
        func, args = self._specialize(*args, **kwargs)
        return func(*args)


# Decorator for transforming functions into FunctionCache
def compile(
    func: Callable | None = None,
    *,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
    return_names: str | Sequence[str] | None = None,
    jit: bool = False,
    kind: str = DEFAULT_SYM_NAME,
    name: str | None = None,
) -> Callable:
    """Create a "compiled" function from a Python function.

    Transforms a standard NumPy-based Python function into an efficient computational
    graph that can be executed in C++ and supports automatic differentiation, code
    generation, and other advanced capabilities. This decorator is the primary entry
    point for converting Python functions into Archimedes' symbolic computation system.

    Parameters
    ----------
    func : callable, optional
        A Python function to be evaluated symbolically.
    static_argnums : int or sequence of int, optional
        Indices of arguments to treat as static (constant) in the function.
        Static arguments aren't converted to symbolic variables but are passed directly
        to the function during tracing.
    static_argnames : str or sequence of str, optional
        Names of arguments to treat as static (constant) in the function.
        Alternative to static_argnums when using keyword arguments.
    return_names : str or sequence of str, optional
        Names of the return values of the function. If not specified, the return
        values will be named ``y0``, ``y1``, etc. This does not need to be specified,
        but is recommended for debugging and C code generation.
    jit : bool, default=False
        Whether to compile the function with JIT for additional performance.
        Not fully implemented in the current version.
    kind : str, default="MX"
        The type of the symbolic variables. Options:

        - SX: Trace the function with scalar-valued symbolic type

        - MX: Trace the function with array-valued symbolic type

    name : str, optional
        The name of the function. If ``None``, taken from the function name.
        Required if the function is a lambda function.

    Returns
    -------
    FunctionCache
        A compiled function that can be called with either symbolic or numeric
        arguments, while maintaining the same function signature.

    Notes
    -----
    When to use this function:

    - To accelerate numerical code by converting it to C++
    - To enable automatic differentiation of your functions
    - To generate C code from your functions
    - When embedding functions in optimization problems or ODE solvers

    Choosing a symbolic type:

    - This option determines the type of symbolic variables used to construct the
        computational graph.  The choice determines the efficiency and type of
        supported operations.
    - ``"SX"`` produces scalar symbolic arrays, meaning that every entry in the array
        has its own scalar symbol. This can produce highly efficient code, but is
        limited to a subset of operations. For example, ``"SX"`` symbolics don't
        support interpolation with lookup tables.
    - ``"MX"`` symbolics are array-valued, meaning that the entire array is represented
        by a single symbol. This allows for embedding more general operations like
        interpolation, ODE solves, and optimization solves into the computational
        graph, but may not be as fast as ``"SX"`` for functions that are dominated by
        scalar operations.
    - The current default is ``"MX"`` and the current recommendation is to use ``"MX"``
        symbolics unless you want to do targeted performance optimizations and feel
        comfortable with the symbolic array concepts.

    When a compiled function is called, Archimedes:

    1. Replaces arguments with symbolic variables of the same shape and dtype
    2. Traces the execution of your function with these symbolic arguments
    3. Creates a computational graph representing all operations
    4. Caches this graph based on input shapes/dtypes and static arguments
    5. Evaluates the graph with the provided numeric inputs

    The function is only traced once for each unique combination of argument shapes,
    dtypes, and static argument values. Subsequent calls with the same shapes reuse
    the cached graph, improving performance.

    Static arguments:

    Static arguments aren't converted to symbolic variables. This is useful for:

    - Configuration flags that affect control flow
    - Constants that shouldn't be differentiated through
    - Values that would be inefficient to recalculate online

    Examples
    --------
    Basic usage as a decorator:

    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> @arc.compile
    ... def rotate(x, theta):
    ...     R = np.array([
    ...         [np.cos(theta), -np.sin(theta)],
    ...         [np.sin(theta), np.cos(theta)],
    ...     ], like=x)
    ...     return R @ x
    >>>
    >>> x = np.array([1.0, 0.0])
    >>> rotate(x, 0.5)

    Using static arguments that modify the function behavior:

    >>> @arc.compile(static_argnames=("use_boundary_conditions",))
    ... def solve_system(A, b, use_boundary_conditions=True):
    ...     if use_boundary_conditions:
    ...         b[[0, -1]] = 0.0  # Apply boundary conditions
    ...     return np.linalg.solve(A, b)

    Different symbolic types:

    >>> # Simple mathematical function - use SX for efficiency
    >>> @arc.compile(kind="SX")
    ... def norm(x):
    ...     return np.sqrt(np.sum(x**2))
    >>>
    >>> # Function with interpolation - requires MX
    >>> @arc.compile(kind="MX", static_argnames=("xp", "fp"))
    ... def interpolate(x, xp, fp):
    ...     return np.interp(x, xp, fp)

    See Also
    --------
    grad : Compute gradients of compiled functions
    jac : Compute Jacobians of compiled functions
    codegen : Generate C code from compiled functions
    """

    kwargs = {
        "static_argnums": static_argnums,
        "static_argnames": static_argnames,
        "return_names": return_names,
        "jit": jit,
        "kind": kind,
        "name": name,
    }

    # If used as @compile(...)
    if func is None:

        def decorator(f):
            return FunctionCache(f, **kwargs)

        return decorator

    # If used as @compile
    if isinstance(func, FunctionCache):
        return func

    return FunctionCache(func, **kwargs)
