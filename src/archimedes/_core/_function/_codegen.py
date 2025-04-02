"""C code generation"""

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from ._compile import FunctionCache

if TYPE_CHECKING:
    pass

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
}


def codegen(
    func: Callable | FunctionCache,
    filename: str,
    args: Sequence[Any],
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
    kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
    cpp: bool = False,
    main: bool = False,
    float_type: type = float,
    int_type: type = int,
    header: bool = True,
    with_mem: bool = False,
    indent: int = 4,
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
    kwargs : dict, optional
        Keyword arguments to pass to the function during specialization.
    verbose : bool, default=False
        If True, include additional comments in the generated code.
    cpp : bool, default=False
        If True, generate C++ code instead of C code.
    main : bool, default=False
        If True, generate a main function entry point.
    float_type : type, default=float
        The C type to use for floating point numbers.
    int_type : type, default=int
        The C type to use for integers.
    header : bool, default=True
        If True, also generate a header file with the extension `.h`.
    with_mem : bool, default=False
        If True, generate a simplified C API with memory management helpers.
    indent : int, default=4
        The number of spaces to use for indentation in the generated code.

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

    Conceptual model:
    This function specializes your computational graph to specific input shapes
    and types, then uses CasADi's code generation capabilities to produce C code
    that implements the same computation. The generated code has no dependencies
    on Archimedes, CasADi, or Python.

    Currently, this function uses CasADi's code generation directly, so the
    generated code will contain CASADI_* prefixes and follow CasADi's conventions.

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
    >>> arc.codegen(rotate, "rotate_function.c", (x_type, theta_type),
    ...            header=True, verbose=True)

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

    if not isinstance(func, FunctionCache):
        func = FunctionCache(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    if filename is None:
        raise ValueError(
            "Must provide filename. Returning code as a string is not yet supported"
        )

    options = {
        "verbose": verbose,
        "cpp": cpp,
        "main": main,
        "casadi_real": dtype_to_c[float_type],
        "casadi_int": dtype_to_c[int_type],
        "with_header": header,
        "with_mem": with_mem,
        "indent": indent,
    }

    # Next we have to compile the function to get the signature
    if kwargs is None:
        kwargs = {}

    # Compile the function for this set of arguments.  We don't need to
    # actually evaluate the function here, just make sure that a CasADi
    # function is generated.
    specialized_func, _sym_args = func._specialize(*args, **kwargs)

    # Now we can generate the code
    specialized_func.codegen(filename, options)
