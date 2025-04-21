"""Utilities for discretizing a continuous ODE function"""

import numpy as np

from archimedes import scan
from archimedes._core import FunctionCache
from archimedes._optimization import implicit

# # NOTE: This implementation fails because alpha[i] is not allowed
# # when i is symbolic.  In theory this would be a better way to do it
# # because it has fewer "call sites" of the RHS function
# def _discretize_rk4_scan(f, h):
#     # RK4 Butcher tableau
#     alpha = np.array([0, 1 / 2, 1 / 2, 1])
#     beta = np.array([
#         [0, 0, 0, 0],
#         [1 / 2, 0, 0, 0],
#         [0, 1 / 2, 0, 0],
#         [0, 0, 1, 0],
#     ])
#     c_sol = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])

#     # Take a single RK4 step
#     def step(t0, x0, p):
#         def body_fun(k, i):
#             ti = t0 + h * alpha[i]
#             yi = x0 + h * np.dot(beta[i, :], k)
#             k[i] = f(ti, yi, p)
#             return k, np.array([])

#         k = np.zeros((4, x0.size), dtype=x0.dtype)
#         k, _ = scan(body_fun, k, length=4)
#         return x0 + h * np.dot(c_sol, k)

#     return step


def _discretize_rk4(f, dt, n_steps=1):
    # Take a single RK4 step
    # TODO: Use scan here?
    h = dt / n_steps

    # def step(t0, x0, p):
    #     k1 = f(t0, x0, p)
    #     k2 = f(t0 + h / 2, x0 + h * k1 / 2, p)
    #     k3 = f(t0 + h / 2, x0 + h * k2 / 2, p)
    #     k4 = f(t0 + h, x0 + h * k3, p)
    #     return x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def scan_fun(carry, i):
        t0, x0, p = carry

        k1 = f(t0, x0, p)
        k2 = f(t0 + h / 2, x0 + h * k1 / 2, p)
        k3 = f(t0 + h / 2, x0 + h * k2 / 2, p)
        k4 = f(t0 + h, x0 + h * k3, p)
        x1 = x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        new_carry = (t0 + h, x1, p)
        return new_carry, np.array([])

    def step(t0, x0, p):
        carry = (t0, x0, p)

        if n_steps == 1:
            # Slightly faster compilation if scan is not used
            carry, _ = scan_fun(carry, 0)
        else:
            carry, _ = scan(scan_fun, carry, length=n_steps)

        _, xf, _ = carry
        return xf

    return FunctionCache(step, name=f"rk4_{f.__name__}", arg_names=["t", "x", "p"])


def _discretize_radau5(rhs, h, newton_solver="fast_newton"):
    c = np.array([(4 - np.sqrt(6)) / 10, (4 + np.sqrt(6)) / 10, 1])
    b = np.array([(16 - np.sqrt(6)) / 36, (16 + np.sqrt(6)) / 36, 1 / 9])
    a = np.array(
        [
            [
                (88 - 7 * np.sqrt(6)) / 360,
                (296 - 169 * np.sqrt(6)) / 1800,
                (-2 + 3 * np.sqrt(6)) / 225,
            ],
            [
                (296 + 169 * np.sqrt(6)) / 1800,
                (88 + 7 * np.sqrt(6)) / 360,
                (-2 - 3 * np.sqrt(6)) / 225,
            ],
            [(16 - np.sqrt(6)) / 36, (16 + np.sqrt(6)) / 36, 1 / 9],
        ]
    )

    if not isinstance(rhs, FunctionCache):
        rhs = FunctionCache(rhs)

    sym_kind = rhs._kind  # TODO: Use a better way to get the kind

    # Define the residual function used in the Newton solver
    def F(k, t, y, p):
        n = y.size
        k = np.reshape(k, (3, n))
        f = np.zeros_like(k)

        ts = t + h * c
        ys = y + h * a @ k

        # TODO: Use scan here?
        for i in range(3):
            f[i] = rhs(ts[i], ys[i], p)

        f, k = np.reshape(f, (3 * n,)), np.reshape(k, (3 * n,))
        return f - k

    F = FunctionCache(
        F, kind=sym_kind, arg_names=["k", "t", "y", "p"], return_names=["r"]
    )
    solve = implicit(F, solver=newton_solver)

    def step(t, y, p):
        n = y.size
        # Solve the nonlinear system using Newton's method
        k0 = np.hstack([y, y, y])
        k = solve(k0, t, y, p)
        k = np.reshape(k, (3, n))
        return y + h * np.dot(b, k)

    return step
