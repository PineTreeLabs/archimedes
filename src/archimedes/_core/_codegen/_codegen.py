"""C code generation"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import (
    Any,
    Callable,
    NamedTuple,
    OrderedDict,
    Protocol,
    Sequence,
    cast,
)

import numpy as np

from archimedes import tree

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


class CodegenError(ValueError):
    pass


def _to_snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


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
    debug: bool = False,
) -> None:
    """Generate C/C++ code from a compiled function.

    Creates standalone C or C++ code that implements the computational graph
    defined by the function. This allows Archimedes models to be deployed on
    embedded systems, integrated into C/C++ codebases, or compiled to native
    executables for maximum performance.

    For a detailed description of the codegen functionality, see the
    :doc:`guide to code
    generation and hardware deployment <../../tutorials/codegen/codegen00>`.

    Parameters
    ----------
    func : Callable | FunctionCache
        The compiled function to generate code for. If not already a FunctionCache,
        it will be compiled automatically.
    args : tuple
        Arguments to the function that specify shapes and dtypes. These can be:

        - SymbolicArray objects
        - NumPy arrays with the same shape and dtype as expected inputs
        - [tree-structured data types](../../trees.md) matching expected inputs
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
    debug: bool, default=False
        If True, print debugging information about the codegen process. Useful for
        development purposes only and subject to removal in future versions.

    Returns
    -------
    None
        The function writes the generated code to the specified file(s).

    Notes
    -----

    **Use this for:**

    - Deploying models on embedded systems or hardware without Python
    - Integrating Archimedes algorithms into C/C++ applications
    - Maximum runtime performance by removing Python interpreter overhead
    - Creating standalone, portable implementations of your algorithm

    This function specializes the computational graph of ``func`` to specific input
    shapes and types, then uses CasADi's code generation capabilities to produce C code
    that implements the same computation. The generated code has no dependencies
    on Archimedes, CasADi, or Python.

    Currently, this function uses CasADi's code generation directly, so the
    generated code will contain ``CASADI_*`` prefixes and follow CasADi's conventions.
    The function will also generate an "interface" API layer with struct definitions
    for inputs and outputs, along with convenience functions for initialization and
    function calls.

    **Generated API**

    The :doc:`codegen guide <../../tutorials/codegen/codegen03>` has a detailed
    description of how to use the generated API.  In short, a Python function named
    ``func`` will generate three top-level C structs:

    1. ``func_arg_t``: Arguments to ``func``
    2. ``func_res_t``: Return values of ``func``
    3. ``func_work_t``: Pre-allocated temporary workspace variables

    There will also be two generated functions:

    1. ``func_init``: Initializes the argument and workspace structs
    2. ``func_step``: Calls the main computation function

    The basic pattern for using this API is:

    .. code-block:: c

        // Allocate all structs on the stack
        func_arg_t arg;
        func_res_t res;
        func_work_t work;

        func_init(&arg, &res, &work);  // Initialize using the template values
        func_step(&arg, &res, &work);  // Numerically evaluate the function

    If the function is "stateful", meaning some outputs are recursively looped back
    to the inputs, manual copying of the state will be required. See below for details.

    **Numerical constants**

    To store numerical constants in the generated code, either:

    1. "Close over" the values in your function definition
    2. Pass them as hashable static arguments (same effect as a closure)
    3. Pass as "dynamic" arguments that could be edited in the generated code

    **Support for structured data types**

    The code generation system supports structured data types, either as homogeneous
    arrays (lists or tuples with all elements of the same type) or as heterogeneous
    containers (e.g. dictionaries, named tuples, or :py:func:`struct` classes.
    The former will be represented as C arrays, while the latter will be represented
    as C structs.

    For example, a struct argument defined with

    .. code-block:: python3

        @struct
        class Point:
            x: float
            y: float

    will autogenerate a C struct:

    .. code-block:: c

        typedef struct {
            float x;
            float y;
        } point_t;

    **Stateful functions**

    A common pattern for dynamics models or control algorithms is to have functions
    that implement a generic discrete-time state-space model of the form

    .. math::

       x_{k+1} = f(x_k, u_k)

       y_k = g(x_k, u_k)

    This can be combined into a single function with the signature
    ``func(x[k], u[k]) -> (x[k+1], y[k])``.  When working with functions of this form
    (or any similar stateful functions), the generated code will still be "functionally
    pure", meaning that the result data will store ``x[k+1]`` and the argument data
    will store ``x[k]``, but the updated state will *not* automatically be copied back
    to the input.

    If the state is implemented as a `@struct`, it will correspond to a C struct and the
    copy operation can be implemented simply using direct copy semantics.  For example,
    a function with the signature ``func(state, inputs) -> (state_new, outputs)`` for
    which the state is a `@struct` (or dict, or named tuple) will generate a C struct
    named ``state_t``.  The ``arg`` structure will include a field ``state``, and the
    ``res`` structure will include a field ``state_new``, both of which have type
    ``state_t``.  With direct assignment copying, the updated state can be copied back
    to the input with

    .. code-block:: c

        arg.state = res.state_new;

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

    if not isinstance(func, FunctionCache):
        func = FunctionCache(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            return_names=return_names,
        )

    # Design choice: enforce return_names to be provided by user.
    # Otherwise there's no way to know meaningful names and we'd
    # have to autogenerate names like y0, y1, etc.
    # This results in hard-to-read code, so we require names.
    if return_names is None and func.default_return_names:
        raise CodegenError(
            "Return names must be provided, either as an argument to `codegen` "
            "or `compile`."
        )

    # If return_names is not provided but known by the function,
    # use the function's names.  If both are known, use the names
    # provided by the user to this function - overriding the FunctionCache's names
    elif return_names is None and not func.default_return_names:
        return_names = func.return_names

    # Now we're sure `return_names` is not None
    return_names = cast(Sequence[str], return_names)

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
        "input_size": tree.ravel(sym_args)[0].size,
        "outputs": [],
        "output_size": tree.ravel(results)[0].size,
    }

    input_helper = ContextHelper(float_type, int_type, input_descriptions, debug=debug)
    for name, arg in zip(func.arg_names, args):
        input_context = input_helper(arg, name)
        context["inputs"].append(input_context)

    for name, val in kwargs.items():
        input_context = input_helper(val, name)
        context["inputs"].append(input_context)

    output_helper = ContextHelper(
        float_type, int_type, output_descriptions, debug=debug
    )
    if not isinstance(results, (tuple, list)) and not hasattr(results, "_fields"):
        results = (results,)

    for name, ret in zip(return_names, results):
        output_context = output_helper(ret, name)
        context["outputs"].append(output_context)

    context["unique_types"] = _unique_types(context["inputs"])
    context["unique_types"].update(_unique_types(context["outputs"]))
    context["assignments"] = _generate_assignments(context["inputs"], prefix="arg")

    if debug:
        print("\ninputs")
        for ctx in context["inputs"]:
            print("\t", ctx)

        print("\nflat inputs")
        flat_ctx = tree.leaves(
            context["inputs"], is_leaf=lambda x: isinstance(x, LeafContext)
        )
        for ctx in flat_ctx:
            print("\t", ctx)

        print("\nassignments")
        for assn in context["assignments"]:
            print("\t", assn)
        print("types", context["unique_types"])

        print("\noutputs")
        for ctx in context["outputs"]:
            print("\t", ctx)

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
                os.rename(src_file, dst_file)


class Context(Protocol):
    type_: str
    name: str
    description: str | None
    ctx_type: str


@dataclasses.dataclass
class NoneContext:
    type_: str = "none"
    name: str = "none"
    description: str | None = None
    ctx_type: str = dataclasses.field(default="none")


@dataclasses.dataclass
class LeafContext:
    type_: str
    name: str
    is_addr: bool
    dims: int | None
    initial_data: np.ndarray
    description: str | None
    ctx_type: str = "leaf"


@dataclasses.dataclass
class StructContext:
    type_: str
    name: str
    children: list[Context]
    description: str | None = None
    ctx_type: str = "node"


@dataclasses.dataclass
class ListContext(Context):
    type_: str = "list"
    name: str = "list"
    length: int = 0
    elements: list[Context] = dataclasses.field(default_factory=list)
    description: str | None = None
    ctx_type: str = "list"


@dataclasses.dataclass
class ContextHelper:
    float_type: type
    int_type: type
    descriptions: dict[str, str]
    debug: bool

    def _process_none(self, name: str) -> NoneContext:
        if self.debug:
            print(f"\tProcessing None '{name}' for codegen...")
        return NoneContext(name=name, description=self.descriptions.get(name, None))

    def _process_leaf(self, arg: Any, name: str) -> LeafContext | NoneContext:
        """Process a 'leaf' value (scalar or array)."""
        if self.debug:
            print(f"\tProcessing leaf '{name}' ({arg}) for codegen...")

        arr = np.asarray(arg)
        if arr.size == 0:
            return self._process_none(name)

        if np.issubdtype(arr.dtype, np.floating):
            dtype = self.float_type
        else:
            dtype = self.int_type
        if np.isscalar(arr) or arr.shape == ():
            dims = None
            is_addr = True
        else:
            dims = arr.size
            is_addr = False

        # At this point we have the actual dtype information.  However,
        # CasADi treats everything as a float, so here we discard this
        # and just use the float_type for all arguments and returns.
        dtype = self.float_type

        return LeafContext(
            type_=dtype_to_c[dtype],
            name=name,
            is_addr=is_addr,
            dims=dims,
            initial_data=arr,
            description=self.descriptions.get(name, None),
        )

    def _process_list(self, arg: list | tuple, name: str) -> ListContext | NoneContext:
        if self.debug:
            print(f"\tProcessing list '{name}' ({arg}) for codegen...")
        # Empty lists are treated as None
        if not arg:
            return self._process_none(name)

        # Items can be any valid tree, but they all must have an identical structure
        treedef0 = tree.structure(arg[0])
        elements: list[Context] = []
        for i, item in enumerate(arg):
            treedef = tree.structure(item)
            if treedef != treedef0:
                raise CodegenError(
                    f"All items in list '{name}' must have the same structure. "
                    f"Found {treedef0.tree_str} and {treedef.tree_str}.  To return "
                    "heterogeneous data, use a structured data type (e.g. dict, "
                    "NamedTuple, or @struct)."
                )
            elements.append(self._process_arg(item, f"{name}[{i}]"))

        return ListContext(
            type_=elements[0].type_,
            name=name,
            elements=elements,
            description=self.descriptions.get(name, None),
            length=len(elements),
        )

    def _process_struct(self, arg: Any, name: str) -> StructContext | NoneContext:
        if self.debug:
            print(f"\tProcessing struct '{name}' ({arg}) for codegen...")
        # Note that all "static" data will be embedded in the computational
        # graph, so here we just need to process the child nodes and leaves
        children = []
        for field in tree.fields(arg):
            if field.metadata.get("static", False):
                continue
            child_name = field.name
            child_context = self._process_arg(getattr(arg, child_name), child_name)
            children.append(child_context)

        if len(children) == 0 or all(isinstance(c, NoneContext) for c in children):
            return self._process_none(name)

        return StructContext(
            type_=f"{_to_snake_case(arg.__class__.__name__)}_t",
            name=name,
            children=children,
            description=self.descriptions.get(name, None),
        )

    def _process_dict(
        self, arg: dict[str, Any], name: str, type_: str | None = None
    ) -> StructContext | NoneContext:
        if self.debug:
            print(f"\tProcessing dict '{name}' ({arg}) for codegen...")

        # A dict is treated as a struct since it has named fields
        children = []
        for key, value in arg.items():
            child_context = self._process_arg(value, key)
            children.append(child_context)

        if len(children) == 0 or all(isinstance(c, NoneContext) for c in children):
            return self._process_none(name)

        if type_ is None:
            type_ = name
        return StructContext(
            type_=f"{_to_snake_case(type_)}_t",
            name=name,
            children=children,
            description=self.descriptions.get(name, None),
        )

    # mypy: ignore[return] because the if/else statements catch all the valid cases
    # (see note at end of function).  If we added dead code to make the type checker
    # happy, codecov would fail...
    def _process_arg(self, arg: Any, name: str) -> Context:  # type: ignore[return]
        """Process a generic arg, which may be a leaf or a node"""
        # Check that the name doesn't end with `_t`, which could be
        # confused with the typedef
        if name.endswith("_t"):
            raise CodegenError(
                f"Argument name '{name}' cannot end with '_t', since this suffix is "
                "reserved for struct typedefs."
            )

        if self.debug:
            print(f"Processing arg {name} ({arg}) with type {type(arg)}")

        if arg is None:
            return self._process_none(name)
        if tree.is_leaf(arg):
            return self._process_leaf(arg, name)
        elif tree.is_struct(arg):
            return self._process_struct(arg, name)
        elif isinstance(arg, tuple) and hasattr(arg, "_fields"):
            # Special case for named tuples: convert to dict
            arg = cast(NamedTuple, arg)
            type_ = _to_snake_case(arg.__class__.__name__)
            return self._process_dict(arg._asdict(), name, type_)
        elif isinstance(arg, (list, tuple)):
            # If it's a homogeneous container, it can be turned into a C array
            return self._process_list(arg, name)
        elif isinstance(arg, dict):
            # Dicts have named fields, so can be used to generate a C struct
            return self._process_dict(arg, name)

        # Unsupported arguments will raise a TypeError during specialization
        # raise CodegenError(f"Unsupported type for \"{name}\": {type(arg)}")

    def __call__(self, arg: Any, name: str) -> Context:
        return self._process_arg(arg, name)


def _unique_types(
    contexts: list[StructContext | LeafContext],
) -> dict[str, StructContext]:
    """Collect all unique StructContext types, keyed by type_ field."""
    unique_types = OrderedDict()

    def _traverse(ctx):
        if isinstance(ctx, StructContext) and ctx.type_ not in unique_types:
            unique_types[ctx.type_] = ctx
            # Recursively traverse children to find nested types
            for child in ctx.children:
                _traverse(child)

        elif isinstance(ctx, ListContext):
            if ctx.type_ not in unique_types:
                # All elements share a type, so we only need to recursively
                # traverse the first element's context for subtypes
                _traverse(ctx.elements[0])

    for ctx in contexts:
        _traverse(ctx)

    # Since the children are added last but need to be defined first,
    # reverse the order of the unique_types dictionary.
    return OrderedDict(reversed(list(unique_types.items())))


@dataclasses.dataclass
class Assignment:
    path: str  # "arg->clusters[0].points[1].x"
    value: str | None = None  # "2.5f"


def _generate_assignments(
    contexts: list[Context], prefix: str = "arg"
) -> list[Assignment]:
    """Generate flat list of all non-zero assignments."""
    assignments = []

    def _format_value(value: Any, type_: str) -> str:
        if type_ == "float":
            return f"{value:f}f"
        return str(value)

    def _traverse(ctx: Context, current_path: str):
        if isinstance(ctx, LeafContext):
            if not np.all(ctx.initial_data == 0):
                if ctx.dims:  # Array
                    # Handle array initialization from initial data
                    # Note that CasADi uses column-major (Fortran-style, b/c of MATLAB)
                    # ordering (unusual for C), so we have to flatten with "F" style.
                    # This will then generate assignments in the correct order.
                    for i, val in enumerate(ctx.initial_data.flatten("F")):
                        if val != 0:
                            assignments.append(
                                Assignment(
                                    path=f"{current_path}[{i}]",
                                    value=_format_value(val, ctx.type_),
                                )
                            )
                else:  # Scalar
                    assignments.append(
                        Assignment(
                            path=current_path,
                            value=_format_value(ctx.initial_data, ctx.type_),
                        )
                    )

        elif isinstance(ctx, ListContext):
            for i, child in enumerate(ctx.elements):
                child_path = f"{current_path}[{i}]"
                _traverse(child, child_path)

        elif isinstance(ctx, StructContext):
            for child in ctx.children:
                child_path = f"{current_path}.{child.name}"
                _traverse(child, child_path)

    for ctx in contexts:
        base_path = f"{prefix}->{ctx.name}" if ctx.name else prefix
        _traverse(ctx, base_path)

    return assignments
