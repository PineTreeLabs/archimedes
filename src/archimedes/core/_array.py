from __future__ import annotations
import numpy as np
import casadi as cs


from ._array_ops import (
    SymbolicArray, array, zeros, ones, zeros_like, ones_like, eye
)


__all__ = [
    "array",
    "sym",
    "sym_like",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "eye",
]


def sym(name, shape=None, dtype=np.float64, kind="SX") -> SymbolicArray:
    """Create a symbolic array.
    
    Parameters
    ----------
    name : str
        Name of the symbolic variable.
    shape : tuple of int, optional
        Shape of the array. Default is ().
    dtype : numpy.dtype, optional
        Data type of the array. Default is np.float64.
    kind : str, optional
        Kind of the symbolic variable (`"SX"` or `"MX"`). Default is `"SX"`.
        See CasADi documentation for details on the differences between the two.

    Returns
    -------
    SymbolicArray
        Symbolic array with the given name, shape, dtype, and kind.
    """
    # TODO: Use `scalar: bool` instead of `kind: str`
    if shape is None:
        shape = ()
    if isinstance(shape, int):
        shape = (shape,)
    assert (
        isinstance(shape, tuple) and len(shape) >= 0
    ), "shape must be a tuple of length >= 0"
    assert all(isinstance(s, int) for s in shape), "shape must be a tuple of ints"
    assert len(shape) <= 2, "Only scalars, vectors, and matrices are supported for now"
    # Note that CasADi creates variables in column-major order, so to be consistent
    # with NumPy we have to reverse the shape and transpose the result.  This only
    # applies to SX, since for MX it's just a single symbol anyway.
    if kind == "SX":
        _sym = cs.SX.sym(name, *shape[::-1]).T
    elif kind == "MX":
        _sym = cs.MX.sym(name, *shape)
    else:
        raise ValueError(f"Unknown symbolic kind {kind}")
    return SymbolicArray(_sym, dtype=dtype, shape=shape)


def sym_like(x, name, dtype=None, kind="SX") -> SymbolicArray:
    """Create a symbolic array similar to an existing array
    
    Parameters
    ----------
    x : array_like
        Array to copy shape and dtype from.
    name : str
        Name of the symbolic variable.
    dtype : numpy.dtype, optional
        Data type of the array. Default is the dtype of `x`.
    kind : str, optional
        Kind of the symbolic variable (`"SX"` or `"MX"`). Default is `"SX"`.
        See CasADi documentation for details on the differences between the two.

    Returns
    -------
    SymbolicArray
        Symbolic array with the given name and kind, and with the shape of `x`.
    """
    if not isinstance(x, (np.ndarray, SymbolicArray)):
        x = np.asarray(x)
    return sym(name, x.shape, dtype=dtype or x.dtype, kind=kind)

