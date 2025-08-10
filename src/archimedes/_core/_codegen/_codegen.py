"""C code generation"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from .._function import FunctionCache
from ._renderer import _render_template

if TYPE_CHECKING:
    from ._renderer import RendererBase

dtype_to_c = {
    float: "double",
    int: "long long int",
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
    "with_header": True,
    "with_mem": False,
    "indent": 4,
}


def codegen(
    func: Callable | FunctionCache,
    filename: str,
    args: Sequence[Any],
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
    return_names: Sequence[str] | None = None,
    kwargs: dict[str, Any] | None = None,
    float_type: type = float,
    int_type: type = int,
    options: dict[str, Any] | None = None,
    application: str | RendererBase | None = None,
    app_config: dict[str, str] | None = None,
    app_context: dict[str, Any] | None = None,
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
    filename : str
        The file to write the code to. Must be specified as string is not supported.
        A header file will also be generated if `header=True`.
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
        already a FunctionCache. If not provided, the function will use default
        names like ``y0``, ``y1``, etc.
    kwargs : dict, optional
        Keyword arguments to pass to the function during specialization.
    float_type : type, default=float
        The C type to use for floating point numbers.
    int_type : type, default=int
        The C type to use for integers.
    options : dict, optional
        Additional options for code generation. This can include:

        - verbose: If True, include additional comments in the generated code.
        - cpp: If True, generate C++ code instead of C code.
        - main: If True, generate a main function entry point.
        - with_header: If True, also generate a header file with the extension `.h`.
        - with_mem: If True, generate a simplified C API with memory helpers.
        - indent: The number of spaces to use for indentation in the generated code.

    application : str | RendererBase, optional
        Name of the template to use for generating application code, or a custom
        implementation of ``RendererBase``. If None, no application code will be
        generated.
    app_config : dict[str, str], optional
        Additional options for rendering the application template.  This might include
        the following keys:

        - template_path: Path to a custom template file, if not using the default.
        - output_path: Path where the generated code will be written.
        - sample_rate: Sample rate for the loop function in seconds (not used by all
          templates).
        - input_descriptions: Dictionary mapping input names to descriptions.
          Used for generating comments in the code.
        - output_descriptions: Dictionary mapping output names to descriptions.
          Used for generating comments in the code.

    app_context : dict, optional
        Additional context for the application template. This can include additional
        variables that are used by custom templates. This context is passed directly
        to the Jinja2 template renderer.

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

    To store numerical constants in the generated code, either:
    1. "Close over" the values in your function definition, or
    2. Pass them as hashable static arguments

    Application generation:
    Optionally, this function can also be used to generate templated "application" code
    for different use cases.  For example, this can be used to create a basic C program
    that allocates memory and calls the generated function, or to create code for
    deployment to an embedded system.

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
    >>> arc.codegen(rotate, "rotate_function.c", (x_type, theta_type))

    The above code will generate 'rotate_function.c' and 'rotate_function.h'
    files that implement the rotation function in C.

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
    >>> arc.codegen(scaled_rotation, "scaled_rotation.c",
    ...            (x_type, theta_type, 5.0))

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

    if filename is None:
        raise ValueError(
            "Must provide filename. Returning code as a string is not yet supported"
        )

    if options is None:
        options = {}

    options = {
        **DEFAULT_OPTIONS,
        "casadi_real": dtype_to_c[float_type],
        "casadi_int": dtype_to_c[int_type],
        **options,
    }

    # Next we have to compile the function to get the signature
    if kwargs is None:
        kwargs = {}

    # Compile the function for this set of arguments.
    specialized_func, sym_args = func._specialize(*args, **kwargs)

    # Evaluate for the template arguments to get the correct return types
    results = specialized_func(*sym_args)

    # Now we can generate the "kernel" function code with CasADi
    file_base = filename.split(".")[0]
    specialized_func.codegen(f"{file_base}_kernel.c", options)

    # Generate the "runtime" code that calls the kernel functions

    # Generate a template/application file if requested
    if application is None:
        return

    if app_config is None:
        app_config = {}

    if app_context is None:
        app_context = {}

    # Build the context for the renderer
    context = {
        **app_context,
        "filename": filename.split(".")[0],
        "function_name": func.name,
        "float_type": dtype_to_c[float_type],
        "int_type": dtype_to_c[int_type],
        "inputs": [],
        "outputs": [],
    }

    ts = app_config.pop("sample_rate", None)
    if ts is not None:
        context["sample_rate"] = {
            "hz": int(1 / ts),  # Approximate, used for comments
            "s": ts,  # Sample rate in seconds
            "ms": int(ts * 1e3),  # Sample rate in microseconds
            "us": int(ts * 1e6),  # Sample rate in microseconds
        }

    input_helper = ContextHelper(
        float_type, int_type, app_config.pop("input_descriptions", {})
    )
    for name, arg in zip(func.arg_names, args):
        input_context = input_helper(arg, name)
        context["inputs"].append(input_context)

    for name, val in kwargs.items():
        input_context = input_helper(val, name)
        context["inputs"].append(input_context)

    output_helper = ContextHelper(
        float_type, int_type, app_config.pop("output_descriptions", {})
    )
    for name, ret in zip(func.return_names, results):
        output_context = output_helper(ret, name)
        context["outputs"].append(output_context)

    _render_template(application, context, **app_config)


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
            dims = str(arg.size)
            is_addr = False
        return {
            "type": dtype_to_c[dtype],
            "name": name,
            "dims": dims,
            "initial_value": initial_value,
            "description": self.descriptions.get(name, None),
            "is_addr": is_addr,
        }
