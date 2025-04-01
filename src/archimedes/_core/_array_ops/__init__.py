"""Implementation of Numpy array dispatch for SymbolicArray objects.

No part of this module should need to be imported directly or accessed by the user.
However, when `SymbolicArray` is used internally it should be imported from here,
since this is where the `__array_ufunc__` and `__array_function__` methods are defined.
"""

from .._array_impl import SymbolicArray
from ._array_function import SUPPORTED_FUNCTIONS
from ._array_ufunc import SUPPORTED_UFUNCS


def __array_ufunc__(self: SymbolicArray, ufunc, method, *inputs, **kwargs):  # noqa: N807
    if method == "__call__":
        # Try to dispatch to the equivalent wrapped compiled function
        if (
            ufunc.__name__ not in SUPPORTED_UFUNCS
            or SUPPORTED_UFUNCS[ufunc.__name__] is NotImplemented
        ):
            return NotImplemented
        return SUPPORTED_UFUNCS[ufunc.__name__](*inputs, **kwargs)
    else:
        raise NotImplementedError(
            f"__array__ufunc__ method {method} not implemented for {ufunc}"
        )


def __array_function__(self: SymbolicArray, func, types, args, kwargs):  # noqa: N807
    if (
        func.__name__ not in SUPPORTED_FUNCTIONS
        or SUPPORTED_FUNCTIONS[func.__name__] is NotImplemented
    ):
        return NotImplemented
    # # Note: this allows subclasses that don't override
    # # __array_function__ to handle SymbolicArray objects.
    # if not all(issubclass(t, self.__class__) for t in types):
    #     print(f"__array_function__ returning NotImplemented for {func.__name__}")
    #     return NotImplemented
    return SUPPORTED_FUNCTIONS[func.__name__](*args, **kwargs)


# The ufuncs and array functions are defined here, so we have to manually go in
# and inject the SymbolicArray function.  This isn't really necessary if everything
# was in one file, just helps to logically separate all the function overloading
# from the core class definition.
SymbolicArray.__array_ufunc__ = __array_ufunc__  # type: ignore[method-assign]
SymbolicArray.__array_function__ = __array_function__  # type: ignore[method-assign]


__all__ = ["SymbolicArray"]
