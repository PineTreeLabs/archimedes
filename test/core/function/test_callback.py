import numpy as np
import pytest

import archimedes as arc


def test_callback():
    def f(x, y):
        print(f"f called: {x=}, {y=}")
        return x * (y + 3)

    @arc.compile
    def call_f(x, y):
        return arc.callback(f, x, y)

    z = call_f(np.array([1.0, 2.0]), 3.0)
    print(z, z.shape)

    assert False


# def test_casadi_callback():
#     import casadi as cs
#     from casadi import Callback

    
#     class _Callback(Callback):
#         def __init__(self, name, opts={}):
#             Callback.__init__(self)
#             self.construct(name, opts)

#         # Number of inputs and outputs
#         def get_n_in(self):
#             return 1

#         def get_n_out(self):
#             return 1

#         # Evaluate numerically
#         def eval(self, cb_args):
#             x = cb_args[0]
#             print(f"_Callback.eval")
#             print(f"f called: {x=}")
#             ret = x ** 2
#             print(f"_Callback.eval returns {ret}")
#             return [ret]

#     cb = _Callback("f")
#     x = cs.MX.sym("x")
#     z = cb(x)
#     print(z)

#     z = cb(2.0)
#     print(z)

#     assert False