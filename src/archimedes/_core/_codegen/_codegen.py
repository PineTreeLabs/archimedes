"""C code generation"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Sequence

import numpy as np

from .._function import FunctionCache
from ._renderer import _render_template

dtype_to_c = {
    float: "float",
    int: "long int",
    bool: "bool",
    np.float64: "double",
    np.int64: "long long int",
    np.uint: "unsigned int",
    np.bool_: "bool",
    np.float32: "float",
    np.int32: "int",
    np.dtype("float32"): "float",
    np.dtype("int32"): "int",
    np.dtype("float64"): "double",
    np.dtype("int64"): "long long int",
    np.dtype("bool_"): "bool",
    np.dtype("uint"): "unsigned int",
}


DEFAULT_OPTIONS = {
    "verbose": False,
    "cpp": False,
    "main": False,
    "with_mem": False,
    "indent": 4,
}


def codegen(
    func: Callable | FunctionCache,
    args: Sequence[Any],
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
    return_names: Sequence[str] | None = None,
    kwargs: dict[str, Any] | None = None,
    float_type: type = float,
    int_type: type = int,
    input_descriptions: dict[str, str] | None = None,
    output_descriptions: dict[str, str] | None = None,
    output_dir: str | None = None,
    options: dict[str, Any] | None = None,
) -> None:
    """Generate C/C++ code from a compiled function.

    Creates standalone C or C++ code that implements the computational graph
    defined by the function. This allows Archimedes models to be deployed on
    embedded systems, integrated into C/C++ codebases, or compiled to native
    executables for maximum performance.

    Parameters
    ----------
    func : Callable | FunctionCache
        The compiled function to generate code for. If not already a FunctionCache,
        it will be compiled automatically.
    args : tuple
        Arguments to the function that specify shapes and dtypes. These can be:
        - SymbolicArray objects
        - NumPy arrays with the same shape and dtype as expected inputs
        - The actual values for static arguments
        Note: For dynamic arguments, the numeric values are ignored.
    static_argnums : tuple, optional
        The indices of the static arguments to the function. Will be ignored if
        `func` is already a FunctionCache.
    static_argnames : tuple, optional
        The names of the static arguments to the function. Will be ignored if
        `func` is already a FunctionCache.
    return_names : tuple, optional
        The names of the return values of the function. Ignored if `func` is
        already a FunctionCache. For the sake of readability, this argument is
        required to be provided, either directly or when separately compiling
        the function to a FunctionCache.
    kwargs : dict, optional
        Keyword arguments to pass to the function during specialization.
    float_type : type, default=float
        The C type to use for floating point numbers.
    int_type : type, default=int
        The C type to use for integers.
    input_descriptions : dict[str, str], optional
        Descriptions for the input arguments. Used for generating comments in the code.
    output_descriptions : dict[str, str], optional
        Descriptions for the output values. Used for generating comments in the code.
    output_dir : str, optional
        Path where the generated code will be written.
    options : dict, optional
        Additional options for code generation. This can include:

        - verbose: If True, include additional comments in the generated code.
        - with_mem: If True, generate a simplified C API with memory helpers.
        - indent: The number of spaces to use for indentation in the generated code.

    Returns
    -------
    None
        The function writes the generated code to the specified file(s).

    Notes
    -----
    When to use this function:
    - For deploying models on embedded systems or hardware without Python
    - For integrating Archimedes algorithms into C/C++ applications
    - For maximum runtime performance by removing Python interpretation overhead
    - For creating standalone, portable implementations of your algorithm

    This function specializes your computational graph to specific input shapes
    and types, then uses CasADi's code generation capabilities to produce C code
    that implements the same computation. The generated code has no dependencies
    on Archimedes, CasADi, or Python.

    Currently, this function uses CasADi's code generation directly, so the
    generated code will contain CASADI_* prefixes and follow CasADi's conventions.
    The function will also generate an "interface" API layer with struct definitions
    for inputs and outputs, along with convenience functions for initialization and
    function calls.

    To store numerical constants in the generated code, either:
    1. "Close over" the values in your function definition, or
    2. Pass them as hashable static arguments

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Define a simple function
    >>> @arc.compile
    ... def rotate(x, theta):
    ...     R = np.array([
    ...         [np.cos(theta), -np.sin(theta)],
    ...         [np.sin(theta), np.cos(theta)],
    ...     ], like=x)
    ...     return R @ x
    >>>
    >>> # Create templates with appropriate shapes and dtypes
    >>> x_type = np.zeros((2,), dtype=float)
    >>> theta_type = np.array(0.0, dtype=float)
    >>>
    >>> # Generate C code
    >>> arc.codegen(rotate, (x_type, theta_type))

    The above code will generate files including 'rotate.c' and 'rotate.h'
    that implement the rotation function in C.

    To use numerical constants, declaring arguments as static will fix the
    value in the generated code:

    >>> @arc.compile(static_argnames=("scale",))
    ... def scaled_rotation(x, theta, scale=2.0):
    ...     R = np.array([
    ...         [np.cos(theta), -np.sin(theta)],
    ...         [np.sin(theta), np.cos(theta)],
    ...     ], like=x)
    ...     return scale * (R @ x)
    >>>
    >>> arc.codegen(scaled_rotation, (x_type, theta_type, 5.0))

    See Also
    --------
    compile : Create a compiled function for use with codegen
    """
    # TODO: Automatic type inference if not specified

    # Check that all arguments are arrays
    for arg in args:
        if not np.all(np.isreal(arg)):
            raise TypeError(f"Argument {arg} is not numeric or a NumPy array.")

    if not isinstance(func, FunctionCache):
        func = FunctionCache(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            return_names=return_names,
        )

    elif func.default_return_names is not None:
        return_names = func.return_names

    # Design choice: enforce return_names to be provided by user.
    # Otherwise there's no way to know meaningful names and we'd
    # have to autogenerate names like y0, y1, etc.
    # This results in hard-to-read code, so we require names.
    if return_names is None and func.default_return_names:
        raise ValueError(
            "Return names must be provided, either as an argument to `codegen` "
            "or `compile`."
        )

    if options is None:
        options = {}

    options = {
        **DEFAULT_OPTIONS,
        "casadi_real": dtype_to_c[float_type],
        "casadi_int": dtype_to_c[int_type],
        **options,
        "with_header": True,  # Always generate a header file
    }

    # Next we have to compile the function to get the signature
    if kwargs is None:
        kwargs = {}

    # Compile the function for this set of arguments.
    specialized_func, sym_args = func._specialize(*args, **kwargs)

    # Evaluate for the template arguments to get the correct return types
    results = specialized_func(*sym_args)

    # Now we can generate the "kernel" function code with CasADi.
    # This will also generate the header file with the function signature.
    file_base = func.name
    specialized_func.codegen(f"{file_base}_kernel.c", options)

    # Generate the "runtime" code that calls the kernel functions
    if output_descriptions is None:
        output_descriptions = {}

    if input_descriptions is None:
        input_descriptions = {}

    context = {
        "filename": file_base,
        "function_name": func.name,
        "float_type": dtype_to_c[float_type],
        "int_type": dtype_to_c[int_type],
        "inputs": [],
        "outputs": [],
    }

    input_helper = ContextHelper(float_type, int_type, input_descriptions)
    for name, arg in zip(func.arg_names, args):
        input_context = input_helper(arg, name)
        context["inputs"].append(input_context)

    for name, val in kwargs.items():
        input_context = input_helper(val, name)
        context["inputs"].append(input_context)

    output_helper = ContextHelper(float_type, int_type, output_descriptions)
    for name, ret in zip(return_names, results):
        output_context = output_helper(ret, name)
        context["outputs"].append(output_context)

    _render_template("api", context, output_path=f"{file_base}.c")
    _render_template("api_header", context, output_path=f"{file_base}.h")

    # Move files to the specified output path if provided
    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for ext in ["c", "h"]:
            for suffix in ["_kernel", ""]:
                src_file = f"{file_base}{suffix}.{ext}"
                dst_file = os.path.join(output_dir, os.path.basename(src_file))
                print(f"Moving {src_file} to {dst_file}")
                os.rename(src_file, dst_file)


@dataclasses.dataclass
class ContextHelper:
    float_type: str
    int_type: str
    descriptions: dict[str, str]

    def __call__(self, arg, name):
        arg = np.asarray(arg)
        if np.issubdtype(arg.dtype, np.floating):
            dtype = self.float_type
        else:
            dtype = self.int_type
        if np.isscalar(arg) or arg.shape == ():
            initial_value = str(arg)
            dims = None
            is_addr = True
        else:
            initial_value = "{" + ", ".join(map(str, arg.flatten())) + "}"
            # dims = str(arg.size)
            dims = arg.size
            is_addr = False

        # At this point we have the actual dtype information.  However,
        # CasADi treats everything as a float, so here we discard this
        # and just use the float_type for all arguments and returns.
        dtype = self.float_type

        return {
            "type": dtype_to_c[dtype],
            "name": name,
            "dims": dims,
            "initial_value": initial_value,
            "initial_data": arg,
            "description": self.descriptions.get(name, None),
            "is_addr": is_addr,
        }
