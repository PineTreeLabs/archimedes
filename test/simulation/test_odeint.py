# ruff: noqa: N806, N803

import numpy as np
import pytest

from archimedes._core import SymbolicArray, jac, sym
from archimedes.simulate import integrator, odeint

RTOL = 1e-10
ATOL = 1e-12


class TestIntegrator:
    def test_integrator(self):
        def f(t, x):
            return -x

        solver = integrator(f, rtol=RTOL, atol=ATOL)
        x0 = 1.0
        tf = 2.0
        tspan = (0.0, tf)
        xf = solver(x0, tspan)
        assert np.allclose(xf, x0 * np.exp(-tf))

    def test_integrator_with_param(self):
        def f(t, x, a=1.0):
            return -a * x

        solver = integrator(f, rtol=RTOL, atol=ATOL)
        x0 = 1.0
        tf = 2.0
        tspan = (0.0, tf)
        a = 2.0
        xf = solver(x0, tspan, a)
        assert np.allclose(xf, x0 * np.exp(-a * tf))

    def test_autodiff(self):
        def f(t, x, a=1.0):
            return -a * x

        # xf = x0 * exp(-a * tf)
        # dxf/dx0 = exp(-a * tf)
        # dxf/da = -x0 * tf * exp(-a * tf)

        solver = integrator(f, rtol=RTOL, atol=ATOL)
        tf = 2.0
        tspan = (0.0, tf)
        dxf = jac(solver, argnums=(0, 2))

        # Test symbolic evaluation
        x0 = sym("x", kind="MX")
        a = sym("a", kind="MX")
        dxf(x0, tspan, a)

        # Test numeric evaluation
        x0 = np.random.randn()
        a = np.random.randn()
        xf = solver(x0, tspan, a)
        dxf, da = dxf(x0, tspan, a)

        assert np.allclose(xf, x0 * np.exp(-a * tf))
        assert np.allclose(dxf, np.exp(-a * tf))
        assert np.allclose(da, -x0 * tf * np.exp(-a * tf))

    def test_integrator_with_static_arg(self):
        def f(t, x, decay=False):
            if decay:
                return -x
            return 0 * x

        solver = integrator(f, rtol=RTOL, atol=ATOL, static_argnames=("decay",))
        x0 = 1.0
        tf = 2.0
        tspan = (0.0, tf)

        xf = solver(x0, tspan, False)
        assert np.allclose(xf, x0)

        xf = solver(x0, tspan, True)
        assert np.allclose(xf, x0 * np.exp(-tf))

        # Should have recompiled with the new static arg
        assert len(solver._compiled) == 2


class TestOdeint:
    def test_odeint(self):
        # Simple scalar integration
        def f(t, x):
            return -x

        x0 = 1.0
        tf = 2.0
        t_span = (0.0, tf)
        xf = odeint(f, t_span, x0, rtol=RTOL, atol=ATOL)
        assert np.allclose(xf, x0 * np.exp(-tf))

    def test_odeint_args(self):
        # Simple scalar integration
        def f(t, x, a, b):
            return -a * x + b

        x0 = 1.0
        tf = 2.0
        t_span = (0.0, tf)
        a = 2.0
        b = 0.0
        xf = odeint(f, t_span, x0, args=(a, b), rtol=RTOL, atol=ATOL)
        assert np.allclose(xf, (b / a) + x0 * np.exp(-a * tf))

        # Symbolic evaluation
        x0 = sym("x", kind="MX")
        a = sym("a", kind="MX")
        b = sym("b", kind="MX")

        xf = odeint(f, t_span, x0, args=(a, b), rtol=RTOL, atol=ATOL)
        assert isinstance(xf, SymbolicArray)

    def test_odeint_t_eval(self):
        # Simple scalar integration
        def f(t, x):
            return -x

        x0 = 1.0
        tf = 2.0
        t_span = (0.0, tf)
        n = 100
        t_eval = np.linspace(0, tf, n)
        xf = odeint(f, t_span, x0, t_eval=t_eval, rtol=RTOL, atol=ATOL)
        assert xf.shape == (n,)
        assert np.allclose(xf, x0 * np.exp(-t_eval))

    def test_odeint_t_eval_args(self):
        # Simple scalar integration
        def f(t, x, a=1.0):
            return -a * x

        x0 = 1.0
        tf = 2.0
        t_span = (0.0, tf)
        n = 100
        ts = np.linspace(0, tf, n)
        a = 2.0
        xs = odeint(f, t_span, x0, t_eval=ts, args=(a,), rtol=RTOL, atol=ATOL)
        assert xs.shape == (n,)
        assert np.allclose(xs, x0 * np.exp(-a * ts))

    def test_odeint_vector(self):
        # Linear system
        A = np.array([[-2, 0], [0, -1]])

        def f(t, x):
            return A @ x

        # Check for shapes (n,) and (n, 1)
        for x0 in (
            np.array([1.0, 1.0]),
            np.array([[1.0], [1.0]]),
        ):
            tf = 2.0
            t_span = (0.0, tf)
            n = 100
            t_eval = np.linspace(0, tf, n)
            xf = odeint(f, t_span, x0, t_eval=t_eval, rtol=RTOL, atol=ATOL)
            assert xf.shape == (len(x0), n)
            xf_ex = x0[:, None] * np.exp(np.outer(np.diag(A), t_eval))
            assert np.allclose(xf, xf_ex)

        with pytest.raises(ValueError):
            # Invalid shape
            x0 = np.zeros((2, 2))
            odeint(f, x0=x0, t_span=(0, 1))
