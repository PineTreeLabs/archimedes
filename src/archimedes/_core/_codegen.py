"""C code generation"""
import numpy as np

from ._function import SymbolicFunction

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
    func,
    filename,
    args,
    static_argnums=None,
    static_argnames=None,
    kwargs=None,
    verbose=False,
    cpp=False,
    main=False,
    float_type=float,
    int_type=int,
    header=True,
    with_mem=False,
    indent=4,
):
    """Generate C code from a symbolic function.

    Currently this just calls CasADi's code generation directly, so the generated
    code will have CASADI_* prefixes.

    The function must be specialized to a particular set of static arguments as well
    as a set of shapes and dtypes for the "dynamic" arguments.  Dynamic arguments
    can be given as SymbolicArray types, or as NumPy arrays with the same shape and
    dtype as the expected inputs to the function.  In the latter case, the numeric
    values of the array will be ignored.  In order to store the numeric values in
    the generated code, the function should either "close over" the numeric values
    or accept them as hashable static arguments.

    Parameters
    ----------
    func : Callable | SymbolicFunction
        The symbolic function to generate code for.
    filename : str
        The file to write the code to.  If None, the code is returned as a string.
    args : tuple
        Arguments to the function.  If not static arguments, these should either be
        SymbolicArray objects or NumPy arrays with the same shape and dtype as the
        expected inputs to the function.
    static_argnums : tuple
        The indices of the static arguments to the function. Will be ignored if the
        function is already a SymbolicFunction.
    static_argnames : tuple
        The names of the static arguments to the function. Will be ignored if the
        function is already a SymbolicFunction.
    verbose : bool
        If True, include comments in the generated code.
    cpp : bool
        If True, generate C++ code instead of C code.
    main : bool
        If True, generate a main entrypoint.
    float_type : type
        The type to use for floating point numbers.
    int_type : type
        The type to use for integers.
    header : bool
        If True, also generate a header file with the extension `.h`.
    with_mem : bool
        If True, generate a simplified C API.
    indent : int
        The number of spaces to use for indentation.

    Returns
    -------
    None
    """
    # TODO: Automatic type inference if not specified

    if not isinstance(func, SymbolicFunction):
        func = SymbolicFunction(
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