"""C code generation"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import (
    Any,
    Callable,
    Sequence,
    OrderedDict,
    NamedTuple,
    cast,
    Protocol,
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


def _to_snake_case(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

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

    # # Check that all arguments are arrays
    # for arg in args:
    #     if not np.all(np.isreal(arg)):
    #         raise TypeError(f"Argument {arg} is not numeric or a NumPy array.")

    if not isinstance(func, FunctionCache):
        func = FunctionCache(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            return_names=return_names,
        )

    # If return_names is not provided but known by the function,
    # use the function's names.  If both are known, use the names
    # provided by the user to this function - overriding the FunctionCache's names
    elif return_names is None and not func.default_return_names:
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

    context["unique_types"] = _unique_types(context["inputs"])

    print(context["inputs"])
    flat_ctx = tree.leaves(context["inputs"], is_leaf=lambda x: isinstance(x, LeafContext))
    for ctx in flat_ctx:
        print("\t", ctx)
    print()

    context["assignments"] = _generate_assignments(context["inputs"], prefix="arg")
    # print("assignments")
    # for assn in context["assignments"]:
    #     print("\t", assn)
    # print("types", context["unique_types"])

    output_helper = ContextHelper(float_type, int_type, output_descriptions)
    if not isinstance(results, tuple) and not hasattr(results, "_fields"):
        results = (results,)

    for name, ret in zip(return_names, results):
        output_context = output_helper(ret, name)
        context["outputs"].append(output_context)

    context["unique_types"].update(_unique_types(context["outputs"]))

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


@tree.struct.pytree_node
class LeafContext(Context):
    type_: str
    name: str
    is_addr: bool
    dims: int
    initial_data: np.ndarray
    description: str | None
    ctx_type: str = tree.struct.field(default="leaf")


@tree.struct.pytree_node
class NodeContext(Context):
    type_: str = tree.struct.field(static=True)
    name: str = tree.struct.field(static=True)
    children: list[NodeContext | LeafContext]
    description: str | None = tree.struct.field(static=True, default=None)
    ctx_type: str = tree.struct.field(default="node", static=True)


@tree.struct.pytree_node
class ListContext(Context):
    type_: str = tree.struct.field(static=True)  # Type name of the list elements
    name: str = tree.struct.field(static=True)
    length: int = tree.struct.field(static=True)
    elements: list[NodeContext | LeafContext]
    description: str | None = tree.struct.field(static=True, default=None)
    ctx_type: str = tree.struct.field(static=True, default="list")


@tree.struct.pytree_node
class ContextHelper:
    float_type: str
    int_type: str
    descriptions: dict[str, str]

    def _process_leaf(self, arg: Any, name: str) -> LeafContext:
        """Process a 'leaf' value (scalar or array)."""
        print(f"Processing leaf '{name}' ({arg}) for codegen...")
        
        arg = np.asarray(arg)
        if np.issubdtype(arg.dtype, np.floating):
            dtype = self.float_type
        else:
            dtype = self.int_type
        if np.isscalar(arg) or arg.shape == ():
            dims = None
            is_addr = True
        else:
            dims = arg.size
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
            initial_data=arg,
            description=self.descriptions.get(name, None),
        )

    def _process_list(self, arg: list | tuple, name: str) -> NodeContext:
        print(f"Processing list '{name}' ({arg}) for codegen...")
        # Items can be any valid PyTree, but they all must have an identical structure
        treedef0 = tree.structure(arg[0])
        elements: list[Context] = []
        for i, item in enumerate(arg):
            treedef = tree.structure(item)
            if treedef != treedef0:
                raise ValueError(
                    f"All items in list '{name}' must have the same structure. "
                    f"Found {treedef0.tree_str} and {treedef.tree_str}.  To return "
                    "heterogeneous data, use a structured data type (e.g. dict, "
                    "NamedTuple, or pytree_node)."
                )
            elements.append(self._process_arg(item, f"{name}[{i}]"))

        return ListContext(
            type_=elements[0].type_,
            name=name,
            elements=elements,
            description=self.descriptions.get(name, None),
            length=len(elements)
        )

    def _process_struct(self, arg: Any, name: str) -> NodeContext:
        print(f"Processing struct '{name}' ({arg}) for codegen...")
        # Note that all "static" data will be embedded in the computational
        # graph, so here we just need to process the child nodes and leaves
        children = []
        for field in tree.struct.fields(arg):
            if field.metadata.get("static", False):
                continue
            child_name = field.name
            child_context = self._process_arg(getattr(arg, child_name), child_name)
            children.append(child_context)
        return NodeContext(
            type_=_to_snake_case(arg.__class__.__name__),
            name=name,
            children=children,
            description=self.descriptions.get(name, None),
        )

    def _process_dict(self, arg: dict[str, Any], name: str) -> NodeContext:
        print(f"Processing dict '{name}' ({arg}) for codegen...")
        # A dict is treated as a struct since it has named fields
        children = []
        for key, value in arg.items():
            child_context = self._process_arg(value, key)
            children.append(child_context)
        return NodeContext(
            type_=name,
            name=name,
            children=children,
            description=self.descriptions.get(name, None),
        )

    def _process_arg(self, arg: Any, name: str) -> Context:
        """Process a generic arg, which may be a leaf or a node"""
        # Check that the name doesn't end with `_t`, which could be
        # confused with the typedef
        if name.endswith("_t"):
            raise ValueError(
                f"Argument name '{name}' cannot end with '_t', since this suffix is "
                "reserved for struct typedefs."
            )

        if tree.is_leaf(arg):
            return self._process_leaf(arg, name)
        elif tree.struct.is_pytree_node(arg):
            return self._process_struct(arg, name)
        elif isinstance(arg, tuple) and hasattr(arg, "_fields"):
            # Special case for named tuples: convert to dict
            arg = cast(NamedTuple, arg)
            return self._process_dict(arg._asdict(), name)
        elif isinstance(arg, (list, tuple)):
            # If it's a homogeneous container, it can be turned into a C array
            return self._process_list(arg, name)
        elif isinstance(arg, dict):
            # Dicts have named fields, so can be used to generate a C struct
            return self._process_dict(arg, name)
        raise ValueError(f"Unsupported type for \"{name}\": {type(arg)}")

    def __call__(self, arg: Any, name: str) -> Context:
        return self._process_arg(arg, name)


def _unique_types(contexts: list[NodeContext | LeafContext]) -> dict[str, NodeContext]:
    """Collect all unique NodeContext types, keyed by type_ field."""
    unique_types = OrderedDict()

    def _traverse(ctx):
        if isinstance(ctx, NodeContext) and ctx.type_ not in unique_types:
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


@tree.struct.pytree_node
class Assignment:
    path: str          # "arg->clusters[0].points[1].x"
    value: str         # "2.5f"


def _generate_assignments(
    contexts: list[NodeContext | LeafContext | ListContext], 
    prefix: str = "arg"
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
                    for i, val in enumerate(ctx.initial_data):
                        if val != 0:
                            assignments.append(Assignment(
                                path=f"{current_path}[{i}]",
                                value=_format_value(val, ctx.type_),
                            ))
                else:  # Scalar
                    assignments.append(Assignment(
                        path=current_path,
                        value=_format_value(ctx.initial_data, ctx.type_),
                    ))

        elif isinstance(ctx, ListContext):
            for i, child in enumerate(ctx.elements):
                child_path = f"{current_path}[{i}]"
                _traverse(child, child_path)

        elif isinstance(ctx, NodeContext):
            for child in ctx.children:
                child_path = f"{current_path}.{child.name}"
                _traverse(child, child_path)

    for ctx in contexts:
        base_path = f"{prefix}->{ctx.name}" if ctx.name else prefix
        _traverse(ctx, base_path)
    
    return assignments