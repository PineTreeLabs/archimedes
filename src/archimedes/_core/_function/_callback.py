from __future__ import annotations

from typing import Any, Callable

import numpy as np
from casadi import Callback, Sparsity

from .._array_impl import _as_casadi_array, array
from ._compile import FunctionCache

_callback_refs: list[Callback] = []


def _exec_callback(cb, arg_flat):
    arg_flat = _as_casadi_array(arg_flat)  # Convert any lists, tuples, etc to arrays
    ret_cs = cb(arg_flat)
    ret = array(ret_cs)
    return ret


def callback(func: Callable, result_shape_dtypes, *args) -> Any:
    """
    Execute an arbitrary Python function within an symbolic computational graph.

    This function allows arbitrary Python functions to be incorporated into
    computational graphs.  This makes it possible to use functions that cannot be
    traced symbolically within functions created with :py:func:`compile`.

    Parameters
    ----------
    func : callable
        The Python function to wrap. This function should accept the same number of
        arguments as provided in ``*args`` and should return values that can be
        converted to NumPy arrays.
    result_shape_dtypes : Tree
        A tree structure that defines the expected shape and data types of the
        function's output. This is used to determine the output shape of the
        callback wrapper without calling the function itself.
    *args : Any
        Arguments to pass to ``func``. These are used to determine the input and output
        shapes for the callback wrapper.

    Returns
    -------
    Any
        The result of calling ``func(*args)``, structured as a tree if applicable.

    Notes
    -----
    When to use this function:

    - When you need to incorporate external functions that cannot be directly
      evaluated symbolically into Archimedes computational graphs
    - When interfacing with legacy code or external libraries that need to be
      called during symbolic execution
    - When implementing custom numerical algorithms that don't map cleanly to
      Archimedes' symbolic operations
    - For testing and debugging purposes to inspect the numerical values at
      some point in an otherwise symbolically compiled function

    The callback is executed numerically in interpreted Python at each evaluation,
    which means:

    1. It won't benefit from symbolic optimization
    2. It cannot be differentiated through automatically
    3. It may be slower than native symbolic operations

    Note that while it is _possible_ to use this function to circumvent the
    requirement that Archimedes code be functionally pure, this is strongly
    recommended against, primarily because the number of evaluation times is
    not guaranteed, so side effects may be unpredictable.

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Define an external function
    >>> def unsupported_code(x):
    ...     print("Evaluating unsupported_code")
    ...     # The "math" library is not supported symbolically
    ...     return math.tanh(x[0]) * math.exp(-0.1 * x[1]**2)
    >>>
    >>>
    >>> # Use in a compiled function
    >>> @arc.compile
    ... def model(x):
    ...     result_shape_dtypes = 0.0  # Output is a scalar
    ...     y = arc.callback(unsupported_code, result_shape_dtypes, x)
    ...     return y * 2
    >>>
    >>> model(np.array([0.5, 1.5]))
    Evaluating unsupported_code
    array(0.73801609)

    See Also
    --------
    compile : Function for symbolically compiling Python functions
    integrator : Specialized solver transformation for ODEs
    implicit : Specialized solver transformation for implicit functions
    """
    from archimedes import tree  # HACK: avoid circular imports

    # Create a FunctionCache for the function - we don't actually
    # want to "compile" this, but the FunctionCache is still helpful for
    # signature handling, etc.
    cache = func if isinstance(func, FunctionCache) else FunctionCache(func)

    arg_flat, arg_unravel = tree.ravel(args)
    arg_shape = (len(arg_flat), 1)

    # Need to evaluate once to know the expected return size
    ret_flat, ret_unravel = tree.ravel(result_shape_dtypes)
    ret_shape = (len(ret_flat), 1)

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
            dm_arg = np.asarray(dm_arg[0])
            cb_args = arg_unravel(dm_arg)

            ret = func(*cb_args)

            # Callback expects DM returns, so flatten this to an array
            ret = tree.map(np.asarray, ret)
            ret, _ = tree.ravel(ret)
            return [ret]

    name = f"cb_{cache.name}"
    cb = _Callback(name)

    def _call(*args):
        arg_flat, _ = tree.ravel(args)
        ret_flat = _exec_callback(cb, arg_flat)
        return ret_unravel(ret_flat)

    _call.__name__ = name
    _call = FunctionCache(
        _call,
        arg_names=cache.arg_names,
    )

    # Store this or the memory reference will get cleaned up
    # and raise a null error when the callback gets executed
    # with data.
    _callback_refs.append(cb)

    return _call(*args)
