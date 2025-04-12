from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Hashable, NamedTuple, Sequence, Tuple

import casadi as cs
from casadi import Callback, Sparsity
import numpy as np

from .._array_impl import _as_casadi_array, array
from ._compile import FunctionCache


_callback_refs: list[Callback] = []


def _exec_callback(cb, arg_flat):
    print("Exec callback")
    arg_flat = _as_casadi_array(arg_flat)  # Convert any lists, tuples, etc to arrays
    print(arg_flat, type(arg_flat))
    ret_cs = cb(arg_flat)
    print()
    ret = array(ret_cs)
    return ret


def callback(func, *args):
    from archimedes import tree  # HACK: avoid circular imports

    # Create a FunctionCache for the function - we don't actually
    # want to "compile" this, but the FunctionCache is still helpful for
    # signature handling, etc.
    cache = FunctionCache(func)

    arg_flat, arg_unravel = tree.ravel(args)
    arg_shape = (len(arg_flat), 1)
    print(f"arg_flat: {arg_flat}")

    # Need to evaluate once to know the expected return size
    ret = func(*args)
    ret_flat, ret_unravel = tree.ravel(ret)
    ret_shape = (len(ret_flat), 1)
    print(f"ret_flat: {ret_flat}")

    class _Callback(Callback):
        def __init__(self, name, opts={}):
            Callback.__init__(self)
            self.construct(name, opts)

        # Number of inputs and outputs
        def get_n_in(self):
            return 1

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return Sparsity.dense(*arg_shape)

        def get_sparsity_out(self, i):
            return Sparsity.dense(*ret_shape)

        # Evaluate numerically
        def eval(self, dm_arg):
            # Here cb_args is a list with a single flattened DM array
            # -> convert to NumPy and unravel back to tree
            print(f"dm_arg = {dm_arg}")
            dm_arg = np.asarray(dm_arg[0])
            print(f"dm_arg = {dm_arg}")
            cb_args = arg_unravel(dm_arg)
            print(f"_Callback.eval for {func} with args {cb_args}")
            ret = func(*cb_args)
            print(f"_Callback.eval returns {ret}")
            # Callback expects DM returns, so flatten this to an array
            ret = tree.map(np.asarray, ret)
            print(f"Mapped -> {ret}")
            ret, _ = tree.ravel(ret)
            return [ret]

    if hasattr(func, "__name__"):
        name = f"cb_{func.__name__}"
    else:
        name = "cb"

    cb = _Callback(name)
    print(cb, cb.size_in(0), cb.size_out(0))
    def _call(*args):
        print(f"args= {[arg.shape for arg in args]}")
        arg_flat, _ = tree.ravel(args)
        print(arg_flat.shape)
        return _exec_callback(cb, arg_flat)

    _call.__name__ = name
    _call = FunctionCache(
        _call,
        arg_names=cache.arg_names,
    )

    # Store this or the memory reference will get cleaned up
    # and raise a null error when the callback gets executed
    # with data.
    _callback_refs.append(cb)

    print(f"args = {args}")
    return _call(*args)