from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Hashable, NamedTuple, Sequence, Tuple

import casadi as cs
from casadi import Callback
import numpy as np

from .._array_impl import _as_casadi_array, array
from ._compile import FunctionCache


_callback_refs: dict[int, Callback] = {}


def _exec_callback(cb, arg_flat):
    print("Exec callback")
    arg_flat = _as_casadi_array(arg_flat)  # Convert any lists, tuples, etc to arrays
    print(arg_flat, type(arg_flat))
    print(cb)
    ret_cs = cb(arg_flat)
    print()
    ret = array(ret_cs)
    return ret


class TreeStore:
    def __init__(self):
        self.arg_unravel = None
        self.ret_unravel = None


def callback(func, *args):
    from archimedes import tree  # HACK: avoid circular imports

    # Create a FunctionCache for the function - we don't actually
    # want to "compile" this, but the FunctionCache is still helpful for
    # signature handling, etc.
    cache = FunctionCache(func)

    store = TreeStore()  # For unraveling types
    arg_flat, store.arg_unravel = tree.ravel(args)
    print(f"arg_flat: {arg_flat}")

    class _Callback(Callback):
        def __init__(self, name, opts={}):
            Callback.__init__(self)
            self.construct(name, opts)

        # Number of inputs and outputs
        def get_n_in(self):
            return 1

        def get_n_out(self):
            return 1

        # Evaluate numerically
        def eval(self, dm_arg):
            # Here cb_args is a list with a single flattened DM array
            # -> convert to NumPy and unravel back to tree
            print(f"dm_arg = {dm_arg}")
            dm_arg = np.asarray(dm_arg[0])
            print(f"dm_arg = {dm_arg}")
            cb_args = store.arg_unravel(dm_arg)
            print(f"_Callback.eval for {func} with args {cb_args}")
            ret = func(*cb_args)
            print(f"_Callback.eval returns {ret}")
            # Callback expects DM returns, so flatten this to an array
            ret = tree.map(np.asarray, ret)
            print(f"Mapped -> {ret}")
            ret, store.unravel = tree.ravel(ret)
            return [ret]

    if hasattr(func, "__name__"):
        name = f"cb_{func.__name__}"
    else:
        name = "cb"

    cb = _Callback(name)
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
    _callback_refs[hash(cb)] = cb

    print(f"args = {args}")
    return _call(*args)