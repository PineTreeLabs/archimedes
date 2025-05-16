"""Interface to casadi.interpolant for N-dimensional interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import casadi as cs
import numpy as np

from ._array_impl import SymbolicArray, _as_casadi_array, array, type_inference
from ._function import FunctionCache

if TYPE_CHECKING:
    from ..typing import ArrayLike


# Wrap as a FunctionCache with an input for each grid dimension
def _eval_interpolant(x, cs_interp, grid, data, name) -> ArrayLike:
    x = map(array, x)  # Convert any lists, tuples, etc to arrays
    x = np.atleast_1d(*x)

    # All arguments must either be symbolic or numeric, and must be 0- or 1-dimensional
    if not isinstance(x, tuple):
        x = (x,)

    if not all(x_i.ndim < 2 for x_i in x):
        raise ValueError(
            f"All arguments to {name} must be 0- or 1-dimensional but input shapes are "
            f"{tuple(x_i.shape for x_i in x)}"
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

    x_cs = _as_casadi_array(x)  # Either CasADi symbol or np.ndarray
    return SymbolicArray(cs_interp(x_cs), shape=shape, dtype=dtype)


# TODO:
# - extrapolation handling?
def interpolant(
    grid: list[np.ndarray],
    data: np.ndarray,
    method: str = "linear",
    arg_names: str | list[str] | None = None,
    ret_name: str | None = None,
    name: str = "interpolant",
) -> Callable:
    """Create a callable N-dimensional interpolant function.

    Constructs an efficient interpolation function from grid data that can be
    evaluated at arbitrary points and embedded in Archimedes computational graphs.

    Parameters
    ----------
    grid : list of array_like
        List of N 1D arrays defining the grid points in each dimension.
        Each array must be strictly monotonic (increasing or decreasing).
    data : array_like
        N-D array containing the function values at all grid points.
    method : str, optional
        Interpolation method to use. Options are:
        - "linear": Piecewise linear interpolation (default)
        - "bspline": Cubic B-spline interpolation (smoother)
    arg_names : list of str, optional
        Names for the input arguments to the interpolant function.
        Default is ``["x_0", "x_1", ...]``.
    ret_name : str, optional
        Name for the output of the interpolant function.
        Default is ``"f"``.
    name : str, optional
        Name for the interpolant function itself.
        Default is ``"interpolant"``.

    Returns
    -------
    callable
        A callable function that interpolates values at specified points.
        The function takes N arguments corresponding to each dimension
        and returns the interpolated value.

    Notes
    -----
    When to use this function:

    - To create smooth approximations of complex functions from tabular data
    - To incorporate lookup tables into optimization or simulation workflows
    - To evaluate experimental or simulation data at arbitrary points
    - When embedding interpolation within other Archimedes functions

    Data organization:
    For a 2D example with grid arrays ``xgrid`` and ``ygrid``, organize the data into
    a 2D data array ``Z`` where ``Z[i,j] = f(xgrid[i], ygrid[j])``.  Corresponding 2D
    grid arrays may be created using ``np.meshgrid(xgrid, ygrid, indexing='ij')``.

    This function creates a CasADi interpolant internally, which can be used with
    both numeric and symbolic inputs. It supports gradient computation through
    automatic differentiation for both interpolation methods, but only with respect
    to the input arguments (not the data).

    Limitations:

    - Evaluating outside the grid boundaries will return the nearest grid value
    - For multi-dimensional grids with N > 1, the interpolant expects N separate args

    Examples
    --------
    >>> import numpy as np
    >>> import archimedes as arc
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # 1D interpolation example
    >>> x = np.linspace(0, 10, 11)
    >>> y = np.sin(x)
    >>>
    >>> # Create an interpolant
    >>> sin_interp = arc.interpolant([x], y, method="bspline")
    >>>
    >>> # Evaluate at new points
    >>> x_fine = np.linspace(0, 10, 101)
    >>> y_interp = np.array([sin_interp(xi) for xi in x_fine])
    >>> plt.figure()
    >>> plt.plot(x_fine, y_interp)
    >>> plt.plot(x, y, 'k.')
    >>> plt.plot(x_fine, np.sin(x_fine), 'k--')
    >>> plt.show()
    >>>
    >>> # 2D interpolation example
    >>> xgrid = np.linspace(-5, 5, 11)
    >>> ygrid = np.linspace(-4, 4, 9)
    >>> X, Y = np.meshgrid(xgrid, ygrid, indexing='ij')
    >>>
    >>> # Create a 2D function to interpolate
    >>> R = np.sqrt(X**2 + Y**2) + 1
    >>> Z = np.sin(R) / R
    >>>
    >>> # Create the interpolant
    >>> f = arc.interpolant(
    >>>     [xgrid, ygrid],
    >>>     Z,
    >>>     method="bspline",
    >>>     arg_names=["x", "y"],
    >>>     ret_name="z"
    >>> )
    >>>
    >>> # Use in optimization or with automatic differentiation
    >>> df_dx = arc.grad(f, argnums=0)  # Gradient with respect to x
    >>> print(df_dx(0.5, 1.0))
    -0.19490596565158205
    >>>
    >>> # Combining with other Archimedes functions
    >>> @arc.compile
    >>> def combined_func(x, y):
    >>>     interp_value = f(x, y)
    >>>     return interp_value**2 + np.sin(x * y)

    See Also
    --------
    compile : Compile a function for use with symbolic arrays
    grad : Compute the gradient of a function
    numpy.interp : 1D interpolation in NumPy
    scipy.interpolate.RegularGridInterpolator : SciPy's equivalent for ``"linear"``
    scipy.interpolate.RectBivariateSpline : SciPy's equivalent for ``"bspline"``
    """
    # Convert inputs to NumPy arrays
    grid = [np.asarray(grid_i) for grid_i in grid]
    data = np.asarray(data)

    # Check for invalid input
    for i, grid_i in enumerate(grid):
        if grid_i.ndim != 1:
            raise ValueError(
                f"grid[{i}] must be 1-dimensional but has shape {grid_i.shape}"
            )

    ndim = len(grid)
    if data.ndim != ndim:
        raise ValueError(f"data must be {ndim}-dimensional but has shape {data.shape}")

    data = data.flatten(order="F")  # Fortran order (column-major), expected by CasADi

    n = np.prod([len(grid_i) for grid_i in grid])
    if data.size != n:
        raise ValueError(f"data must have length {n} but has length {data.size}")

    if method not in ("linear", "bspline"):
        raise ValueError(f"method must be one of 'linear', 'bspline' but is {method}")

    if arg_names is None:
        arg_names = [f"x_{i}" for i in range(len(grid))]

    else:
        if len(arg_names) != len(grid):
            raise ValueError(
                f"arg_names must have length {len(grid)} but has length "
                f"{len(arg_names)}"
            )
        if not all([isinstance(arg_name, str) for arg_name in arg_names]):
            raise ValueError(
                f"arg_names must be a list of strings but has type {type(arg_names)}"
            )

    if ret_name is None:
        ret_name = "f"

    elif not isinstance(ret_name, str):
        raise ValueError(f"ret_name must be a string but has type {type(ret_name)}")

    # Create CasADi interpolant
    cs_interp = cs.interpolant(name, method, grid, data)
    args = (cs_interp, grid, data, name)

    def _interp(*x: ArrayLike) -> ArrayLike:
        return _eval_interpolant(x, *args)

    _interp.__name__ = name

    return FunctionCache(
        _interp,
        arg_names=arg_names,
        return_names=[ret_name],
        kind="MX",
    )
