"""Interface to casadi.interpolant for N-dimensional interpolation."""

import casadi as cs
import numpy as np

from ._array_impl import SymbolicArray, type_inference, casadi_array
from ._array_ops import array
from ._function import SymbolicFunction


# Wrap as a SymbolicFunction with an input for each grid dimension
def _eval_interpolant(x, cs_interp, grid, data, name):
    x = map(array, x)  # Convert any lists, tuples, etc to arrays
    x = np.atleast_1d(*x)

    # All arguments must either be symbolic or numeric, and must be 0- or 1-dimensional
    if not isinstance(x, tuple):
        x = (x,)

    if not all(x_i.ndim < 2 for x_i in x):
        raise ValueError(
            f"All arguments to {name} must be 0- or 1-dimensional but input shapes are {tuple(x_i.shape for x_i in x)}"
        )

    # The lengths of the arguments must be consistent with each other
    lengths = {len(x_i) for x_i in x}
    if len(lengths) != 1:
        raise ValueError(
            f"All arguments to {name} must have the same length but input shapes "
            f"are {tuple(x_i.shape for x_i in x)}"
        )

    # Stack arguments into a single 2D array
    x = np.stack(x, axis=0)

    # The output shape is the input shape with leading dimension removed
    shape = () if x.shape[1] == 1 else x.shape[1:]
    
    # The output dtype is the promotion of the data and input dtypes
    dtype = type_inference("default", data, x)

    x_cs = casadi_array(x)  # Either CasADi symbol or np.ndarray
    return SymbolicArray(cs_interp(x_cs), shape=shape, dtype=dtype)


# TODO:
# - extrapolation handling?
def interpolant(
    grid: list[np.ndarray],
    data: np.ndarray,
    method: str = "linear",
    arg_names: list[str] = None,
    ret_name: str = None,
    name: str = "interpolant",
):
    """Create a callable N-dimensional interpolant
    
    Args:
        grid: List of 1D arrays defining the grid
        data: Array of data values
        method: Interpolation method (one of "linear", "bspline")
        arg_names: Names of the arguments to the interpolant
        ret_name: Name of the return value of the interpolant
        name: Name of the interpolant

    Returns:
        Interpolant

    The `data` array should be a 1D array with length equal to the product of the
    lengths of the grid arrays.  For instance, in 2D with grid arrays `xgrid` and
    `ygrid`, if the grid arrays are extended to 2D with

    ```python
    X, Y = np.meshgrid(xgrid, ygrid, indexing="ij")
    ```

    and the data array is a function `F = f(X, Y)`, then the data array is

    ```python
    data = F.ravel(order='F')
    ```

    The interpolant can then be created with `interpolant([X, Y], data)`.
    """
    # Convert inputs to NumPy arrays
    grid = [np.asarray(grid_i) for grid_i in grid]
    data = np.asarray(data)

    # Check for invalid input
    for i, grid_i in enumerate(grid):
        if grid_i.ndim != 1:
            raise ValueError(f"grid[{i}] must be 1-dimensional but has shape {grid_i.shape}")

    if data.ndim != 1:
        raise ValueError(f"data must be 1-dimensional but has shape {data.shape}")

    N = np.prod([len(grid_i) for grid_i in grid])
    if data.size != N:
        raise ValueError(
            f"data must have length {N} but has length {data.size}"
        )

    if method not in ("linear", "bspline"):
        raise ValueError(f"method must be one of 'linear', 'bspline' but is {method}")
    
    if arg_names is None:
        arg_names = [f"x_{i}" for i in range(len(grid))]

    else:
        if len(arg_names) != len(grid):
            raise ValueError(
                f"arg_names must have length {len(grid)} but has length {len(arg_names)}"
            )
        if not all([isinstance(arg_name, str) for arg_name in arg_names]):
            raise ValueError(
                f"arg_names must be a list of strings but has type {type(arg_names)}"
            )

    if ret_name is None:
        ret_name = "f"

    elif not isinstance(ret_name, str):
            raise ValueError(
                f"ret_name must be a string but has type {type(ret_name)}"
            )

    # Create CasADi interpolant
    cs_interp = cs.interpolant(name, method, grid, data)
    args = (cs_interp, grid, data, name)

    # Wrap the interpolant in an evaluation function with the right number of args
    if len(grid) == 1:
        def _interp(x):
            return _eval_interpolant((x,), *args)
        
    else:
        def _interp(*x):
            return _eval_interpolant(x, *args)

    _interp.__name__ = name

    return SymbolicFunction(
        _interp,
        arg_names=arg_names,
        ret_names=[ret_name],
        kind="MX",
    )
