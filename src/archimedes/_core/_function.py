from __future__ import annotations
from functools import partial
import inspect
from typing import NamedTuple, Hashable, Tuple, Any

import numpy as np
import casadi as cs

from archimedes import tree
from archimedes.tree._flatten_util import HashablePartial

from . import sym_like
from ._array_impl import array, _as_casadi_array


# Type alias for the key in the compiled dictionary
# This will be first the shape/dtype of all arguments and then a tuple
# of the static arguments, which are restricted only by the requirement
# that they be hashable.
CompiledKey = Tuple[Tuple[HashablePartial, ...], Tuple[Hashable, ...]]


class CompiledFunction(NamedTuple):
    """Container for a CasADi function specialized to particular arg types.
    
    The SymbolicFunction operates similarly to JAX transformations in that it does
    not need to know the shapes and dtypes of the arguments at creation.  Instead,
    a specialized version of the function is created each time the SymbolicFunction
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
                    "Currently symbolic functions only support explicit arguments "
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


class SymbolicFunction:
    def __init__(
        self,
        func,
        arg_names=None,
        ret_names=None,
        static_argnums=None,
        static_argnames=None,
        jit=False,
        kind="SX",
        name=None,
    ):
        self._func = func  # The original Python function
        self.name = name if name is not None else func.__name__

        # Kind of symbolic object to use
        # TODO: Make this `scalar: bool` instead of `kind: str`
        self._kind = kind

        # Should we JIT compile the function?
        self._jit = jit

        self._compiled: dict[CompiledKey, CompiledFunction] = {}
    
        # Determine the signature of the original function.  If not
        # specified explicitly, it will be inferred using `inspect.signature`
        self.signature = _resolve_signature(func, arg_names)

        self.ret_names = ret_names  # Can still be None at this point

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

        if not isinstance(sym_ret, tuple):
            sym_ret = (sym_ret,)

        # print(f"Compiling {self.name} for {sym_args} -> {sym_ret}")

        # Ravel all return types to flattened arrays before creating the CasADi function
        sym_ret_flat = []
        results_unravel = []
        for x in sym_ret:
            x_flat, unravel = tree.ravel(x)
            sym_ret_flat.append(x_flat)
            results_unravel.append(unravel)

        cs_ret = [_as_casadi_array(x) for x in sym_ret_flat]

        if self.ret_names is None:
            self.ret_names = [f"y{i}" for i in range(len(sym_ret_flat))]
        else:
            if len(self.ret_names) != len(sym_ret_flat):
                raise ValueError(
                    f"Expected {len(sym_ret_flat)} return values, got "
                    f"{len(self.ret_names)} in call to {self.name}"
                )

        options = {
            "jit": self._jit,
        }
        # print(f"Compiling {self.name} for {cs_args} -> {cs_ret}")
        _compiled_func = cs.Function(
            self.name, cs_args, cs_ret, arg_names, self.ret_names, options
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
        static_args, dynamic_args, args_unravel = self.split_args(self.static_argnums, *pos_args)

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


# Decorator for transforming functions into SymbolicFunction
def compile(
    func=None, *, static_argnums=None, static_argnames=None, jit=False, kind="SX", name=None
):
    """Create a "symbolic function" from a Python function.
    
    Parameters
    ----------
    func : callable
        A Python function to be evaluated symbolically.
    static_argnums : int or sequence of int
        Indices of arguments to treat as static (constant) in the function.
        Defaults to None.
    static_argnames : str or sequence of str
        Names of arguments to treat as static (constant) in the function.
        Defaults to None.
    jit : bool
        Whether to compile the function with JIT. Defaults to False.
    kind : str
        The type of the symbolic variables. Defaults to "SX". For "plain" math
        functions, the scalar symbolic type "SX" is generally recommended for
        efficiency.  When the function includes more general operations like embedded
        ODE solves, root-finding, interpolation, or optimization problems, the matrix
        symbolic type "MX" is required.  See the CasADi documentation for more details.
    name : str
        The name of the function. Defaults to None (taken from the function name).
        Required if the function is a lambda function.

    Returns
    -------
    SymbolicFunction
        A symbolic function that can be called with either symbolic or numeric
        arguments.
    """
    # TODO: Link to documentation

    # If used as @compile(...)
    if func is None:
        def decorator(f):
            return SymbolicFunction(
                f,
                static_argnums=static_argnums,
                static_argnames=static_argnames,
                jit=jit,
                kind=kind,
                name=name,
            )
        return decorator

    # If used as @compile
    return SymbolicFunction(
        func,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        jit=jit,
        kind=kind,
        name=name,
    )


def scan(
    func,
    init_carry,
    xs=None,
    length=None,
    unroll=10,
):
    """Loop evaluation of a function while carrying the previous result.

    Roughly similar to `jax.lax.scan`, or the following pure Python code:

    ```python

    def scan(func, init_carry, xs, length):
        if xs is None:
            xs = range(length)
        carry = init_carry
        ys = []
        for x in xs:
            carry, y = func(carry, x)
            ys.append(y)
        return carry, np.stack(ys)
    ```

    Either `xs` or `length` must be provided, and if both are provided, `xs.shape[0]`
    must be equal to `length`.  `xs` can be a symbolic or numeric array, but `length`
    must be a known integer value.

    While the pure Python code above is valid, it generates computational graphs
    that grow linearly with the length of the loop, increasing compile time and memory
    usage.  On the other hand, `scan` will "roll up" the loop, resulting in much
    smaller computational graphs.  The number of iterations that are combined into a
    single node in the graph is controlled by the `unroll` argument, which effectively
    trades off faster compile times and less memory usage (larger `unroll`) against
    somewhat slower execution times (smaller `unroll`).  A value of `-1` will fully
    unroll the loop.

    Parameters
    ----------
    func :
        A function f(carry, x) -> (carry, y) applied at each loop iteration.
    init_carry :
        The initial value of the "carry".
    xs :
        The values of the loop variable (optional if `length` is provided).
    length :
        The length of the loop (optional if `xs` is provided).
    unroll :
        The number of iterations to combine into a single node in the graph.

    Returns
    -------
    carry :
        A tuple of the final carry and the values of the loop variable.
    """

    if not isinstance(func, SymbolicFunction):
        func = SymbolicFunction(func)

    # Check the input signature of the function
    if len(func.arg_names) != 2:
        raise ValueError(
            f"The scanned function ({func.name}) must accept exactly two "
            f"arguments.  The provided function call signature is {func.signature}."
        )

    if xs is None:
        if length is None:
            raise ValueError("Either xs or length must be provided")
        xs = np.arange(length)
    else:
        if length is not None:
            if xs.shape[0] != length:
                raise ValueError(
                    f"xs.shape[0] ({xs.shape[0]}) must be equal to length ({length})"
                )
        else:
            length = xs.shape[0]

    # Compile the function for the provided arguments at the first loop iteration
    # We've checked the arguments already, so we don't need those here
    specialized_func, _args = func._specialize(init_carry, xs[0])
    results_unravel = specialized_func.results_unravel

    # Check that the specialized function returns exactly two outputs
    if len(results_unravel) != 2:
        raise ValueError(
            f"The scanned function ({func.name}) must return exactly two outputs.  "
            f"The provided function returned {results_unravel}."
        )

    carry_out, x_out = specialized_func(init_carry, xs[0])
    carry_in_treedef = tree.structure(init_carry)
    carry_out_treedef = tree.structure(carry_out)
    if carry_in_treedef != carry_out_treedef:
        raise ValueError(
            f"The scanned function ({func.name}) must return the same type for the "
            f"carry as the initial value ({carry_in_treedef}) but returned "
            f"{carry_out_treedef}."
        )
    if len(x_out.shape) > 1:
        raise ValueError(
            "The second return of a scanned function can only be 0- or 1-D."
            f"The return shape of {func.name} is {x_out.shape}."
        )

    # Create the CasADi function that will perform the scan
    # scan_func = specialized_func.func.mapaccum(length, {"base": unroll})
    scan_func = specialized_func.func.fold(length)

    # Convert arguments to either CasADi expressions or NumPy arrays
    # Note that CasADi will map over the _second_ axis of `xs`, so we need to
    # transpose the array before passing it.
    cs_args = tuple(_as_casadi_array(arg) for arg in (init_carry, xs.T))
    cs_carry, cs_ys = scan_func(*cs_args)

    # Ensure that the return has shape and dtype consistent with the inputs
    carry = array(cs_carry, carry_out.dtype).reshape(carry_out.shape)
    # Reshape so that the shape is (length, ...) (note transposing the CasADi result)
    ys = array(cs_ys.T, x_out.dtype).reshape((length,) + x_out.shape)

    # Return the CasADi outputs as NumPy or SymbolicArray types
    return carry, ys