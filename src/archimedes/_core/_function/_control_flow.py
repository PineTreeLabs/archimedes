from __future__ import annotations

import numpy as np

from archimedes import tree
from archimedes._core._array_impl import array, _as_casadi_array


__all__ = [
    "scan",
]

def scan(
    func,
    init_carry,
    xs=None,
    length=None,
):
    """Apply a function repeatedly while carrying state between iterations.
    
    Efficiently implements a loop that accumulates state and collects outputs at each 
    iteration. Similar to functional fold/reduce operations but also accumulates the 
    intermediate outputs. This provides a structured way to express iterative algorithms 
    in a functional style that can be efficiently compiled and differentiated.
    
    Parameters
    ----------
    func : callable
        A function with signature f(carry, x) -> (new_carry, y) to be applied at each 
        loop iteration. The function must:
        - Accept exactly two arguments: the current carry value and loop variable
        - Return exactly two values: the updated carry value and an output for this step
        - Return a carry with the same structure as the input carry
    init_carry : array_like or PyTree
        The initial value of the carry state. Can be a scalar, array, or nested PyTree.
        The structure of this value defines what func must return as its first output.
    xs : array_like, optional
        The values to loop over, with shape (length, ...). Each value is passed as the 
        second argument to func. Required unless length is provided.
    length : int, optional
        The number of iterations to run. Required if xs is None. If both are provided, 
        xs.shape[0] must equal length.
    
    Returns
    -------
    final_carry : same type as init_carry
        The final carry value after all iterations.
    ys : array
        The stacked outputs from each iteration, with shape (length, ...).
        
    Notes
    -----
    When to use this function:
    - To keep computational graph size manageable for large loops
    - For implementing recurrent computations (filters, RNNs, etc.)
    - For iterative numerical methods (e.g., fixed-point iterations)
    
    Conceptual model:
    Each iteration applies func to the current carry value and the current loop value:
    (carry, y) = func(carry, x)
    
    The carry is threaded through all iterations, while each y output is collected.
    This pattern is common in many iterative algorithms and can be more efficient 
    than explicit Python loops because it creates a single node in the computational 
    graph regardless of the number of iterations.
    
    The standard Python equivalent would be:
    ```python
    def scan_equivalent(func, init_carry, xs=None, length=None):
        if xs is None:
            xs = range(length)
        carry = init_carry
        ys = []
        for x in xs:
            carry, y = func(carry, x)
            ys.append(y)
        return carry, np.stack(ys)
    ```
    
    However, the compiled `scan` is more efficient for long loops because it creates a 
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

    carry_out, x_out = specialized_func(init_carry, xs[0])
    _, unravel = tree.ravel(carry_out)

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
    carry = unravel(array(cs_carry))

    # Reshape so that the shape is (length, ...) (note transposing the CasADi result)
    ys = array(cs_ys.T, x_out.dtype).reshape((length,) + x_out.shape)

    # Return the CasADi outputs as NumPy or SymbolicArray types
    return carry, ys