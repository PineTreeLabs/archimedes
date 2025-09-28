# This code modifies code from JAX:
# - vmap: https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L831-L1033

# Copyright (c) 2021 The JAX Authors
# Licensed under Apache License 2.0
# https://github.com/jax-ml/jax

# Modifications and additions to the original code:
# Copyright (c) 2025 Pine Tree Labs, LLC
# Licensed under the GNU General Public License v3.0

# As a combined work, use of this code requires compliance with the GNU GPL v3.0.

# SPDX-FileCopyrightText: 2021 The JAX Authors
# SPDX-FileCopyrightText: 2025 Pine Tree Labs, LLC
# SPDX-License-Identifier: GPL-3.0-or-later

# The original license terms are included below for attribution:

# === Apache License 2.0 ===
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, TypeVar

import casadi as cs
import numpy as np

from archimedes import tree
from archimedes._core._array_impl import (
    DEFAULT_SYM_NAME,
    _as_casadi_array,
    array,
)

from .._array_ops._array_ops import normalize_axis_index
from ._compile import FunctionCache, compile

if TYPE_CHECKING:
    from ...typing import ArrayLike, Tree

    T = TypeVar("T", bound=Tree)

__all__ = [
    "scan",
    "switch",
    "vmap",
]


def scan(
    func: Callable,
    init_carry: T,
    xs: ArrayLike | None = None,
    length: int | None = None,
) -> tuple[T, ArrayLike]:
    """
    Apply a function repeatedly while carrying state between iterations.

    Efficiently implements a loop that accumulates state and collects outputs at each
    iteration. Similar to functional fold/reduce operations but also accumulates the
    intermediate outputs. This provides a structured way to express iterative
    algorithms in a functional style that can be efficiently compiled and
    differentiated.

    Parameters
    ----------
    func : callable
        A function with signature ``func(carry, x) -> (new_carry, y)`` to be applied at
        each loop iteration. The function must:

        - Accept exactly two arguments: the current carry value and loop variable

        - Return exactly two values: the updated carry value and an output for this step

        - Return a carry with the same structure as the input carry

    init_carry : array_like or Tree
        The initial value of the carry state. Can be a scalar, array, or structured
        data type. The structure of this value defines what ``func`` must return as
        its first output.
    xs : array_like, optional
        The values to loop over, with shape ``(length, ...)``. Each value is passed as
        the second argument to ``func``. Required unless length is provided.
    length : int, optional
        The number of iterations to run. Required if ``xs`` is None. If both are
        provided, ``xs.shape[0]`` must equal ``length``.

    Returns
    -------
    final_carry : same type as ``init_carry``
        The final carry value after all iterations.
    ys : array
        The stacked outputs from each iteration, with shape ``(length, ...)``.

    Notes
    -----
    When to use this function:

    - To keep computational graph size manageable for large loops
    - For implementing recurrent computations (filters, RNNs, etc.)
    - For iterative numerical methods (e.g., fixed-point iterations)

    Conceptual model:
    Each iteration applies ``func`` to the current carry value and the current loop
    value: ``(carry, y) = func(carry, x)``

    The ``carry`` is threaded through all iterations, while each ``y`` output is
    collected. This pattern is common in many iterative algorithms and can be more
    efficient than explicit Python loops because it creates a single node in the
    computational graph regardless of the number of iterations.

    The standard Python equivalent would be:

    .. highlight:: python
    .. code-block:: python

        def scan_equivalent(func, init_carry, xs=None, length=None):
            if xs is None:
                xs = range(length)
            carry = init_carry
            ys = []
            for x in xs:
                carry, y = func(carry, x)
                ys.append(y)
            return carry, np.stack(ys)


    However, the compiled ``scan`` is more efficient for long loops because it creates a
    fixed-size computational graph regardless of loop length.

    Examples
    --------
    Basic summation:

    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> @arc.compile
    ... def sum_func(carry, x):
    ...     new_carry = carry + x
    ...     return new_carry, new_carry
    >>>
    >>> xs = np.array([1, 2, 3, 4, 5])
    >>> final_sum, intermediates = arc.scan(sum_func, 0, xs)
    >>> print(final_sum)  # 15
    >>> print(intermediates)  # [1, 3, 6, 10, 15]

    Implementing a discrete-time IIR filter:

    >>> @arc.compile
    ... def iir_step(state, x):
    ...     # Simple first-order IIR filter: y[n] = 0.9*y[n-1] + 0.1*x[n]
    ...     new_state = 0.9 * state + 0.1 * x
    ...     return new_state, new_state
    >>>
    >>> # Apply to a step input
    >>> input_signal = np.ones(50)
    >>> initial_state = 0.0
    >>> final_state, filtered = arc.scan(iir_step, initial_state, input_signal)

    Implementing Euler's method for ODE integration:

    >>> @arc.compile
    ... def euler_step(state, t):
    ...     # Simple harmonic oscillator: d²x/dt² = -x
    ...     dt = 0.001
    ...     x, v = state
    ...     new_x = x + dt * v
    ...     new_v = v - dt * x
    ...     return (new_x, new_v), new_x
    >>>
    >>> ts = np.linspace(0, 1.0, 1001)
    >>> initial_state = (1.0, 0.0)  # x=1, v=0
    >>> final_state, trajectory = arc.scan(euler_step, initial_state, ts)

    See Also
    --------
    jax.lax.scan : JAX equivalent function
    arc.tree : Module for working with structured data in scan loops
    """

    if not isinstance(func, FunctionCache):
        func = FunctionCache(func)

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

    carry_out, x_out = func(init_carry, xs[0])
    init_carry_flat, unravel = tree.ravel(init_carry)

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
    cs_args = tuple(_as_casadi_array(arg) for arg in (init_carry_flat, xs.T))
    cs_carry, cs_ys = scan_func(*cs_args)

    # Ensure that the return has shape and dtype consistent with the inputs
    carry = unravel(array(cs_carry))

    # Reshape so that the shape is (length, ...) (note transposing the CasADi result)
    ys = array(cs_ys.T, x_out.dtype).reshape((length,) + x_out.shape)

    # Return the outputs as NumPy or SymbolicArray types
    return carry, ys


def switch(
    index: int,
    branches: tuple[Callable, ...],
    *args: Tree,
    name: str | None = None,
    kind: str = DEFAULT_SYM_NAME,
) -> Tree:
    """
    Selectively apply one of several functions based on an index.

    This function provides a conditional branching mechanism that selects and applies
    one of the provided branch functions based on the index value. The function is
    similar to a switch/case statement but can be embedded within computational graphs.

    Semantically, this function is equivalent to the following Python code:

    .. highlight:: python
    .. code-block:: python

        def switch(index, branches, *args):
            index = min(max(index, 0), len(branches) - 1)
            return branches[index](*args)

    Parameters
    ----------
    index : int
        The branch selector. Must be an integer. If the index is out of bounds,
        it will be clamped to the valid range ``[0, len(branches)-1]``.
    branches : tuple of callables
        A tuple of functions to choose from. Each function must accept the same
        arguments and return compatible structures.
    *args : Tree
        Arguments to pass to the selected branch function. All branches must
        accept these arguments.
    name : str, optional
        Name for the resulting function. Used for debugging and visualization.
        Default is "switch_{index}".
    kind : str, optional
        The kind of symbolics to use when constructing the function. Default is "MX".

    Returns
    -------
    Tree
        The result of applying the selected branch function to the provided arguments.
        All branches must return the same structure.

    Notes
    -----
    This function converts conditional branching into a computational graph construct.
    Unlike Python's if/else, which doesn't work with symbolic values, ``switch`` is
    compatible with symbolic/numeric execution.

    The function evaluates each branch at compilation time to ensure they have
    compatible output structures. At runtime, only the selected branch is executed.

    Behavior notes:

    - If `index` is out of bounds, it will be clamped to the valid range.
    - All branches must return the same tree structure, or a ValueError will be
      raised.
    - At least two branches must be provided, or a ValueError will be raised.
    - Functions are traced at compilation time, meaning any side effects will occur
      for all branches during tracing, even though only one branch executes at runtime.
      It is strongly recommended to avoid side effects.
    - This function supports automatic differentiation

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Define functions for each branch
    >>> def branch0(x):
    ...     return x**2
    ...
    >>> def branch1(x):
    ...     return np.sin(x)
    ...
    >>> def branch2(x):
    ...     return -x
    ...
    >>> # Create a switch function
    >>> @arc.compile
    ... def apply_operation(x, op_index):
    ...     return arc.switch(op_index, (branch0, branch1, branch2), x)
    ...
    >>> # Apply different branches based on the index
    >>> x = np.array([0.5, 1.0, 1.5])
    >>> apply_operation(x, 0)  # Returns x**2
    >>> apply_operation(x, 1)  # Returns sin(x)
    >>> apply_operation(x, 2)  # Returns -x

    # Example with tree-structured data
    >>> def multiply(data, factor):
    ...     return {k: v * factor for k, v in data.items()}
    ...
    >>> def add_offset(data, offset):
    ...     return {k: v + offset for k, v in data.items()}
    ...
    >>> @arc.compile
    ... def process_data(data, op_index, param):
    ...     return arc.switch(op_index, (multiply, add_offset), data, param)
    ...
    >>> data = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    >>> process_data(data, 0, 2.0)  # Multiplies all values by 2.0
    >>> process_data(data, 1, 1.0)  # Adds 1.0 to all values

    See Also
    --------
    np.where : Element-wise conditional selection between two arrays
    scan : Functional for-loop construct for repeated operations
    """

    if len(branches) < 2:
        raise ValueError("switch requires at least two branches")

    if name is None:
        name = f"switch_{index}"

    # Wrap this within a compile decorator to ensure everything is symbolic
    @compile(name=name, static_argnums=(1,))
    def _switch(index, branches, args):
        # Ravel all args to a single flat argument for constructing
        # an equivalent flattened CasADi function
        args_flat, args_unravel = tree.ravel(args)

        # Check that each branch evaluates to the same output structure
        # Note that this evaluates each branch at compile time, but at
        # runtime the only evaluation is the CasADi conditional below, which
        # is short-circuiting.
        results_treedef = None
        results_size = None
        cs_branches = []  # Flat inputs -> flat outputs
        for branch in branches:
            if not isinstance(branch, FunctionCache):
                branch = FunctionCache(branch)

            # Evaluate the branch for type checking of results
            results = branch(*args)

            # Flatten the results to a single array for each branch
            # Note that all branches must have the same treedef, so it
            # doesn't matter which unravel function we use
            results_flat, results_unravel = tree.ravel(results)

            # Check for consistency of the output structure
            if results_treedef is None:
                results_treedef = tree.structure(results)
                results_size = results_flat.size
            else:
                results_treedef_i = tree.structure(results)
                if results_size != results_flat.size:
                    raise ValueError(
                        "All branches of a switch must return the same number of "
                        f"elements, but got {results_size} for branch 0 and "
                        f"{results_flat.size} for branch {branch.name}."
                    )
                if results_treedef != results_treedef_i:
                    raise ValueError(
                        "All branches of a switch must return the same tree "
                        f"structure, but got {results_treedef} for branch 0 and "
                        f"{results_treedef_i} for branch {branch.name}."
                    )

            # Save a flattened version of the branch function
            cs_branches.append(
                cs.Function(
                    branch.name,
                    [_as_casadi_array(args_flat)],
                    [_as_casadi_array(results_flat)],
                )
            )

        # Create the CasADi function that will perform the switch
        cs_switch = cs.Function.conditional(name, cs_branches[:-1], cs_branches[-1])

        # Evaluate the CasADi function symbolically
        index = np.fmax(np.fmin(index, len(branches) - 1), 0)  # Clamp to valid range
        cs_results = cs_switch(_as_casadi_array(index), _as_casadi_array(args_flat))

        # Convert back to a flat SymbolicArray
        results_flat = array(cs_results, dtype=results_flat.dtype)

        # Unravel the flat result to the original structure
        return results_unravel(results_flat)

    # Call the switch function with the provided arguments
    return _switch(index, branches, args)


def normalize_vmap_index(axis, data, insert=False):
    if isinstance(axis, int):
        axis = tree.map(lambda x: axis, data)

    # Create a tree of normalized axis indices
    def _normalize_leaf_index(axis, leaf):
        ndim = leaf.ndim
        if insert:
            ndim += 1
        return normalize_axis_index(axis, ndim)

    return tree.map(_normalize_leaf_index, axis, data)


def vmap(
    func: Callable,
    in_axes: int | None | tuple[int | None, ...] = 0,
    out_axes: int = 0,
    name: str | None = None,
) -> Callable:
    """
    Vectorize a function along specified argument axes.

    The `vmap` transformation takes a function that operates on individual elements
    and transforms it into one that operates on batches of elements in a vectorized
    manner. This enables efficient computation without writing explicit loops or
    broadcasting logic.

    Parameters
    ----------
    func : callable
        Function to be vectorized. The function can accept ordinary NumPy arrays
        or tree-structured data.
    in_axes : int, None, or tuple of ints/None, optional
        Specifies which axis of each input argument should be mapped over.

        - int: Use the same axis for all arguments (e.g., 0 for the first dimension)

        - None: Don't map this argument (broadcast it to all mapped elements)

        - tuple: Specify a different axis for each argument

        Default is 0 (map over the first axis of each argument).
    out_axes : int, optional
        Specifies where the mapped axis should appear in the output.
        Default is 0 (mapped axis is the first dimension of the output).
    name : str, optional
        Name for the transformed function. If None, derives a name from the
        original function.

    Returns
    -------
    vectorized_func : callable
        A function with the same signature as `func` that operates on batches of inputs.

    Notes
    -----
    When to use this function:

    - When you need to apply the same operation to many inputs efficiently
    - To convert a single-example function into one that handles batches
    - To selectively vectorize over some arguments while broadcasting others
    - To "unflatten" tree-structured data by mapping the unravel function

    Conceptual model:

    `vmap` transforms functions to operate along array axes. For example, a function
    f(x) that takes a vector and returns a scalar can be transformed into one that
    takes a batch of vectors (an array) and returns a batch of scalars (a vector),
    without explicitly writing loops.

    Each argument can be mapped differently:
    - Mapped arguments (in_axes is an int): Batched processing along the specified axis
    - Broadcasted arguments (in_axes is None): Same value used for all batch elements

    The vectorized function ensures that all mapped arguments have the same size
    along their mapped dimensions.

    Examples
    --------
    Basic vectorization of a dot product:

    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> def dot(a, b):
    ...     return np.dot(a, b)
    >>>
    >>> # Vectorize to compute multiple dot products at once
    >>> batched_dot = arc.vmap(dot)
    >>>
    >>> # Input: batch of vectors (3 vectors of length 2)
    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([[7, 8], [9, 10], [11, 12]])
    >>>
    >>> # Output: batch of scalars (3 dot products)
    >>> batched_dot(x, y)
    array([ 23,  67, 127])

    Working with tree-structured data:

    >>> from archimedes import struct
    >>>
    >>> @struct
    >>> class Particle:
    ...     x: np.ndarray
    ...     v: np.ndarray
    >>>
    >>> def update(p, dt):
    ...     return p.replace(x=p.x + dt * p.v)
    >>>
    >>> # Vectorize to update multiple particles at once
    >>> map_update = arc.vmap(update, in_axes=(0, None))
    >>>
    >>> # Batch of 10 particles
    >>> x = np.random.randn(10, 3)  # 10 particles in 3D space
    >>> v = np.random.randn(10, 3)
    >>> particles = Particle(x=x, v=v)
    >>>
    >>> # Update all 10 particles at once
    >>> new_particles = map_update(particles)

    See Also
    --------
    scan : Transform that applies a function sequentially to array elements
    """

    if not isinstance(func, FunctionCache):
        func = FunctionCache(func)

    num_args = len(func.arg_names)

    if isinstance(in_axes, list):
        in_axes = tuple(in_axes)

    if not (in_axes is None or type(in_axes) in {int, tuple}):
        raise TypeError(
            "vmap in_axes must be an int, None, or a tuple of entries corresponding "
            f"to the positional arguments passed to the function, but got {in_axes}."
        )

    if in_axes is None or isinstance(in_axes, int):
        in_axes = (in_axes,) * num_args

    if not isinstance(out_axes, int):
        raise TypeError(f"vmap out_axes must be an int, but got {out_axes}.")

    def _vmap_func(*args):
        if isinstance(in_axes, tuple) and len(in_axes) != len(args):
            raise ValueError(
                "vmap in_axes must be an int, None, or a tuple of entries "
                "corresponding to the positional arguments passed to the function, "
                f"but got {len(in_axes)=}, {len(args)=}"
            )

        # Split the arguments into fixed (in_axes = None) and mapped (in_axes != None)
        # This way we don't have to scan over the fixed data, only the mapped args
        fixed_kwargs = OrderedDict()
        mapped_args = OrderedDict()
        for ax, key, arg in zip(
            in_axes,  # type: ignore
            func.signature.parameters.keys(),  # type: ignore
            args,
        ):
            if ax is None:
                fixed_kwargs[key] = arg
            else:
                mapped_args[key] = arg

        # The logic here is complicated, because we need to do two levels of
        # "flattening":
        # 1. flatten/unflatten: this can handle data with an additional axis, as long
        #    as it has the right number of leaves.  Basically, this "puts the leaves
        #    back in the container".  However, this isn't compatible with `scan`.
        # 2. ravel/unravel: this requires that the data has exactly the same number of
        #    elements, so we can't flexibly pack/unpack the tree containers. However,
        #    when the arrays are "raveled" we can scan over inputs/outputs efficiently
        #
        # The "outer" step therefore flattens the tree to leaves (a list of arrays),
        # and the "inner" steps stack the arguments and then splits the outputs.

        # Outer step: tree -> leaves
        args_flat, in_tree = tree.flatten(mapped_args)

        # Wrap to (0, ndim) and construct trees of the mapped indices
        # This will be a tuple of one tree for each argument, with each
        # leaf corresponding to the normalized index for that leaf
        in_axes_normalized = tuple(
            normalize_vmap_index(a, arg)
            for a, arg in zip(in_axes, args)  # type: ignore
            if a is not None  # type: ignore
        )

        # Swap axes so that the mapped axis is the leading axis
        args_flat = [
            np.swapaxes(arg, 0, a)  # type: ignore
            for a, arg in zip(tree.leaves(in_axes_normalized), args_flat)
            if a is not None
        ]

        # Check that the leading axis of all mapped args is the same
        leading_axes = [arg.shape[0] for arg in args_flat]
        if len(set(leading_axes)) != 1:
            raise ValueError(
                "vmap requires that all mapped arguments have the same mapped axis "
                f"length, but got {leading_axes}."
            )

        init_args = tuple(arg[0] for arg in args_flat)

        # Inner step: leaves -> vectors (for scanning)
        # Also note that the "fixed vec" has to be passed to the scan function
        # or there is an error about a dangling symbolic variable (because we're
        # doing nested traced functions, which is a little dicey).
        init_args_vec, unravel_mapped_args = tree.ravel(init_args)
        fixed_args_vec, unravel_fixed_args = tree.ravel(fixed_kwargs)

        def flat_func(args_vec, fixed_vec):
            args_leaves = unravel_mapped_args(args_vec)  # vec -> leaves
            fixed_args = unravel_fixed_args(fixed_vec)  # vec -> leaves
            _kwargs = {**fixed_args, **tree.unflatten(in_tree, args_leaves)}
            return tree.flatten(func(**_kwargs))

        # Call once to determine output structure
        out_template_leaves, out_tree = flat_func(init_args_vec, fixed_args_vec)
        out_template = out_tree.unflatten(out_template_leaves)

        # Wrap out axes to (0, ndim)
        out_axes_normalized = normalize_vmap_index(out_axes, out_template, True)

        # Another inner step: manually "ravel"/"unravel" the arguments to/from a
        # 2D array for passing to/from `scan`.
        #
        # Ensure that all arrays are 2D, with the mapped axis in front
        # Since we have ensured that the mapped axis is consistent, the
        # only possibilities are 2D (do nothing) and 1D (expand and transpose)
        stacked_args = np.concatenate(
            [arg if arg.ndim == 2 else arg[:, None] for arg in args_flat], axis=1
        )

        # Determine the indices of the output arrays to split to get the leaves
        # after calling `scan`.
        split_idx = np.cumsum([out.size for out in out_template_leaves])

        # Here is the main work: `scan` will loop over the 2D stacked inputs and
        # return 2D stacked outputs
        def scan_func(carry, x):
            result, _ = flat_func(x, carry)
            return carry, tree.ravel(result)[0]

        _, out_array = scan(scan_func, fixed_args_vec, stacked_args)

        # Inner step: split the array to have the right number of leaves
        out_leaves = np.split(out_array, split_idx, axis=1)[:-1]  # type: ignore

        # Make sure that the leaves are the same as the input plus
        # the additional axis
        map_length = leading_axes[0]
        out_leaves = [
            np.reshape(leaf, (map_length,) + template.shape)
            for leaf, template in zip(out_leaves, out_template_leaves)
        ]

        # Outer step: recreate the tree structure with the output leaves
        out = out_tree.unflatten(out_leaves)

        # Swap axes again according to out_axes
        return tree.map(
            lambda a, arg: np.swapaxes(arg, 0, a),
            out_axes_normalized,
            out,
        )

    if name is None:
        name = f"vmap_{func.name}"

    _vmap_func.__name__ = name

    return FunctionCache(
        _vmap_func,
        arg_names=func.arg_names,
        kind=func._kind,
    )
