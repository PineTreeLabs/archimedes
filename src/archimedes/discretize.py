"""Utilities for discretizing a continuous ODE function"""
# ruff: noqa: N802, N806

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Protocol, cast

import numpy as np

from archimedes import scan
from archimedes._core import FunctionCache
from archimedes.optimize import implicit

if TYPE_CHECKING:
    from archimedes.typing import Tree

    DynArgs = tuple[float, np.ndarray, np.ndarray, Tree]
    DynFunc = Callable[[*DynArgs], np.ndarray]


__all__ = ["discretize"]

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


class RKSolver(Protocol):
    func: DynFunc
    h: float

    def __call__(self, carry: DynArgs, i: int) -> tuple[DynArgs, np.ndarray]:
        """Implement a single integration step in the scan loop."""


@dataclasses.dataclass
class EulerSolver:
    func: DynFunc
    h: float

    def __call__(self, carry: DynArgs, i: int) -> tuple[DynArgs, np.ndarray]:
        t0, x0, u, p = carry
        x1 = x0 + self.h * self.func(t0, x0, u, p)
        new_carry = (t0 + self.h, x1, u, p)
        return new_carry, np.array([])  # Dummy output for scan


@dataclasses.dataclass
class RK4Solver:
    func: DynFunc
    h: float

    def __call__(self, carry: DynArgs, i: int) -> tuple[DynArgs, np.ndarray]:
        t0, x0, u, p = carry

        k1 = self.func(t0, x0, u, p)
        k2 = self.func(t0 + self.h / 2, x0 + self.h * k1 / 2, u, p)
        k3 = self.func(t0 + self.h / 2, x0 + self.h * k2 / 2, u, p)
        k4 = self.func(t0 + self.h, x0 + self.h * k3, u, p)
        x1 = x0 + self.h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        new_carry = (t0 + self.h, x1, u, p)
        return new_carry, np.array([])  # Dummy output for scan


@dataclasses.dataclass
class Radau5Solver:
    func: DynFunc
    h: float
    newton_solver: dataclasses.InitVar[str] = "fast_newton"
    solve: Callable[[np.ndarray, *DynArgs], np.ndarray] = dataclasses.field(init=False)

    @property
    def a(self) -> np.ndarray:
        return np.array(
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

    @property
    def b(self) -> np.ndarray:
        return np.array([(16 - np.sqrt(6)) / 36, (16 + np.sqrt(6)) / 36, 1 / 9])

    @property
    def c(self) -> np.ndarray:
        return np.array([(4 - np.sqrt(6)) / 10, (4 + np.sqrt(6)) / 10, 1])

    def __post_init__(self, newton_solver: str):
        func = cast(FunctionCache, self.func)
        sym_kind: str = func._kind

        # Define the residual function used in the Newton solver
        def F(k, t, y, u, p):
            n = y.size
            k = np.reshape(k, (3, n))
            f = np.zeros_like(k)

            ts = t + self.h * self.c
            ys = y + self.h * self.a @ k

            # TODO: Use scan here?
            for i in range(3):
                f[i] = self.func(ts[i], ys[i], u, p)

            f, k = np.reshape(f, (3 * n,)), np.reshape(k, (3 * n,))
            return f - k

        F = FunctionCache(
            F, kind=sym_kind, arg_names=["k", "t", "y", "u", "p"], return_names=["r"]
        )
        self.solve = implicit(F, solver=newton_solver)

    def __call__(self, carry: DynArgs, i: int) -> tuple[DynArgs, np.ndarray]:
        t, x0, u, p = carry
        n = x0.size
        # Solve the nonlinear system using Newton's method
        k0 = np.hstack([x0, x0, x0])
        k = self.solve(k0, t, x0, u, p)
        k = np.reshape(k, (3, n))
        t1 = t + self.h
        x1 = x0 + self.h * np.dot(self.b, k)
        new_carry = (t1, x1, u, p)
        return new_carry, np.array([])


def discretize(func=None, dt=None, method="rk4", n_steps=1, name=None, **options):
    """Convert continuous-time dynamics to discrete-time using numerical integration.

    Transforms a continuous-time ordinary differential equation (ODE) function
    into a discrete-time difference equation.

    The function implements numerical integration schemes while maintaining automatic
    differentiation compatibility for gradient-based optimization, extended Kalman
    filtering, etc. The resulting discrete-time function preserves the same calling
    convention as the original continuous-time dynamics.

    Given a continuous-time system:

    .. code-block:: text

        dx/dt = ode(t, x, u, params)

    this function returns a discrete-time approximation:

    .. code-block:: text

        x[k+1] = f(t[k], x[k], u[k], params)

    where the discrete-time function ``f`` integrates the continuous dynamics over
    the time interval ``dt`` using the specified numerical method.

    Can also be used as a decorator to convert a continuous-time function
    into a discrete-time function:

    .. code-block:: python

        @discretize(dt=0.1, method="rk4")
        def dyn(t, x, u, params):
            # ... continuous-time dynamics implementation

    Parameters
    ----------
    func : callable or FunctionCache
        Continuous-time dynamics function with signature ``func(t, x, u, params)``:

        - ``t`` : Current time (scalar)
        - ``x`` : State vector of shape ``(nx,)``
        - ``u`` : Input vector of shape ``(nu,)``
        - ``params`` : Parameters (any tree-compatible structure)

        Must return the time derivative ``dx/dt`` as an array of shape ``(nx,)``.
        The function can be a regular Python function or a pre-compiled
        :class:`FunctionCache` object.
    dt : float
        Sampling time step for discretization. Represents the time interval
        between discrete samples ``t[k+1] - t[k]``. Smaller values provide
        better accuracy but increase computational cost.
    method : str, optional
        Numerical integration method. Default is ``"rk4"``. Available methods:

        - ``"rk4"``: Fourth-order Runge-Kutta method. Explicit, O(h⁴) accuracy.
          Excellent balance of accuracy and computational efficiency for most
          systems. Recommended for well-behaved, non-stiff dynamics.

        - ``"radau5"``: Fifth-order Radau IIA implicit method. Implicit,
          A-stable with excellent stability properties. Suitable for stiff
          systems and when high accuracy is required. Involves solving
          nonlinear equations at each step using Newton's method.

    n_steps : int, optional
        Number of integration sub-steps within each sampling interval ``dt``.
        Default is 1. When ``n_steps > 1``, the integration step size becomes
        ``h = dt / n_steps``, improving accuracy for rapid dynamics or large
        sampling intervals. Useful for maintaining accuracy when ``dt`` is
        constrained by measurement availability rather than dynamics.
    name : str, optional
        Name for the resulting discrete-time function. If None, automatically
        generated as ``"{method}_{func.name}"``. Used for debugging and
        performance profiling of compiled functions.
    **options
        Additional method-specific options:

        For ``method="radau5"``:
            - ``newton_solver`` : str, default ``"fast_newton"``
              Newton solver for implicit equations. Other options inclue ``"kinsol"``
              and ``"newton"``.

    Returns
    -------
    discrete_func : FunctionCache
        Discrete-time dynamics function with signature
        ``discrete_func(t, x, u, params)`` that returns the next state
        ``x[k+1]`` given current state ``x[k]``. The function is automatically
        compiled for efficient evaluation and supports automatic differentiation
        for gradient computation.

    Notes
    -----
    **Method Selection Guide:**

    **RK4 Method** (``method="rk4"``):
        - **Best for**: Non-stiff systems, general-purpose discretization
        - **Advantages**: Fast, explicit, well-tested, excellent accuracy/cost ratio
        - **Limitations**: Can become unstable for stiff systems or large time steps
        - **Typical use**: Mechanical systems, mild nonlinearities, most applications

    **Radau5 Method** (``method="radau5"``):
        - **Best for**: Stiff systems, high accuracy requirements, DAE systems
        - **Advantages**: A-stable, handles stiff dynamics, high-order accuracy
        - **Limitations**: Computational overhead from implicit solve
        - **Typical use**: Chemical kinetics, electrical circuits, multiscale dynamics

    **Accuracy Considerations:**

    The discretization error depends on both the method order and the time step:

    - **RK4**: Error ∼ O(dt⁴), typically accurate for ``dt < T/10`` where ``T``
      is the fastest system time constant
    - **Radau5**: Error ∼ O(dt⁵), maintains accuracy even for moderate ``dt``
    - **Sub-stepping**: Use ``n_steps > 1`` when measurement rate limits ``dt``
      but dynamics require smaller integration steps

    **Automatic Differentiation:**

    The discretized function preserves automatic differentiation through the
    integration process, enabling efficient gradient computation for:

    - Parameter estimation via :func:`pem`
    - Sensitivity analysis and uncertainty quantification
    - Gradient-based optimal control

    **Performance Optimization:**

    The returned function is automatically compiled using CasADi's just-in-time
    compilation for efficient repeated evaluation. For system identification
    applications involving hundreds of function evaluations, this provides
    significant performance benefits over pure Python implementations.

    Examples
    --------
    **Basic Discretization:**

    >>> import numpy as np
    >>> import archimedes as arc
    >>>
    >>> # Define continuous-time harmonic oscillator
    >>> def oscillator(t, x, u, params):
    ...     omega = params["omega"]  # Natural frequency
    ...     return np.hstack([
    ...         x[1],                    # dx1/dt = x2 (velocity)
    ...         -omega**2 * x[0] + u[0]  # dx2/dt = -ω²x1 + u (acceleration)
    ...     ])
    >>>
    >>> # Discretize with 50ms sampling
    >>> dt = 0.05
    >>> oscillator_discrete = arc.discretize(oscillator, dt, method="rk4")
    >>>
    >>> # Simulate one step
    >>> t0 = 0.0
    >>> x0 = np.array([1.0, 0.0])  # Initial position and velocity
    >>> u0 = np.array([0.0])       # No input
    >>> params = {"omega": 2.0}    # 2 rad/s natural frequency
    >>>
    >>> x1 = oscillator_discrete(t0, x0, u0, params)
    >>> print(f"Next state: {x1}")
    Next state: [0.995004 -0.199334]

    **Stiff System with Radau5:**

    >>> # Stiff Van der Pol oscillator
    >>> def van_der_pol(t, x, u, params):
    ...     mu = params["mu"]  # Large damping parameter
    ...     return np.hstack([
    ...         x[1],
    ...         mu * (1 - x[0]**2) * x[1] - x[0]
    ...     ])
    >>>
    >>> # Use implicit method for stiff dynamics
    >>> dt = 0.1  # Larger time step acceptable with Radau5
    >>> vdp_discrete = arc.discretize(
    ...     van_der_pol, dt,
    ...     method="radau5",
    ...     newton_solver="fast_newton"
    ... )
    >>>
    >>> params_stiff = {"mu": 10.0}  # Stiff parameter
    >>> x0 = np.array([2.0, 0.0])
    >>> x1 = vdp_discrete(0.0, x0, np.array([0.0]), params_stiff)

    See Also
    --------
    archimedes.sysid.pem : Parameter estimation using discrete-time models
    odeint : Continuous-time integration for comparison and validation

    References
    ----------
    .. [1] Hairer, E., Nørsett, S.P., and Wanner, G. "Solving Ordinary Differential
           Equations I: Nonstiff Problems." 2nd edition, Springer, 1993.
    .. [2] Hairer, E. and Wanner, G. "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems." 2nd edition, Springer, 1996.
    """

    def _discretize_impl(f, dt_val):
        """Implementation of the discretization logic."""
        h = dt_val / n_steps

        if not isinstance(f, FunctionCache):
            f = FunctionCache(f)

        if name is None:
            func_name = f"{method}_{f.name}"
        else:
            func_name = name

        h = dt_val / n_steps
        scan_fun: RKSolver = {
            "euler": EulerSolver,
            "rk4": RK4Solver,
            "radau5": Radau5Solver,
        }[method](f, h, **options)

        def step(t0, x0, u, p):
            carry = (t0, x0, u, p)

            if n_steps == 1:
                # Slightly faster compilation if scan is not used
                carry, _ = scan_fun(carry, 0)
            else:
                carry, _ = scan(scan_fun, carry, length=n_steps)

            _, xf, _, _ = carry
            return xf

        return FunctionCache(step, name=func_name, arg_names=["t", "x", "u", "p"])

    # Check if we're in decorator mode (func is None) or direct mode (func provided)
    if func is None:
        # Decorator mode: @discretize(dt=0.1, method="rk4")
        if dt is None:
            raise ValueError("dt must be specified")

        def decorator(f):
            # Decorator function that applies discretization to the wrapped function
            return _discretize_impl(f, dt)

        return decorator
    else:
        # Direct mode: discretize(func, dt=0.1, method="rk4", ...)
        if dt is None:
            raise ValueError("dt must be specified")

        return _discretize_impl(func, dt)

    # h = dt / n_steps

    # if not isinstance(func, FunctionCache):
    #     func = FunctionCache(func)

    # if name is None:
    #     name = f"{method}_{func.name}"

    # h = dt / n_steps
    # scan_fun = {
    #     "rk4": _rk4,
    #     "radau5": _radau5,
    # }[method](func, h, **options)

    # def step(t0, x0, u, p):
    #     carry = (t0, x0, u, p)

    #     if n_steps == 1:
    #         # Slightly faster compilation if scan is not used
    #         carry, _ = scan_fun(carry, 0)
    #     else:
    #         carry, _ = scan(scan_fun, carry, length=n_steps)

    #     _, xf, _, _ = carry
    #     return xf

    # return FunctionCache(step, name=name, arg_names=["t", "x", "u", "p"])
