import numpy as np
import pytest

from archimedes._core import SymbolicArray, array, compile, sym
from archimedes.optimize import minimize

METHODS = ("ipopt", "sqpmethod", "BFGS")
BOUNDED_METHODS = ("ipopt", "sqpmethod", "L-BFGS-B")


class TestMinimize:
    @pytest.mark.parametrize("method", METHODS)
    def test_minimize(self, method):
        # Basic functionality test
        def f(x):
            return x**2

        result = minimize(f, x0=1.0, method=method)
        assert np.allclose(result.x, 0.0)

    def test_minimize_with_param(self):
        # Test with a parameter
        def f(x, a=1.0):
            return a * x**2

        result = minimize(f, x0=1.0, args=(2.0,))
        assert np.allclose(result.x, 0.0)

    @pytest.mark.parametrize("method", BOUNDED_METHODS)
    def test_minimize_with_bounds(self, method):
        def f(x):
            return x**2

        result = minimize(f, x0=1.0, bounds=[-2.0, 2.0], method=method)
        assert np.allclose(result.x, 0.0)

    @pytest.mark.parametrize("method", ("ipopt", "sqpmethod"))
    def test_minimize_constrained(self, method):
        # Test with function from the CasADi docs, using additional parameters
        def f(x, a, b):
            return x[0] ** 2 + a * x[2] ** 2

        def g(x, a, b):
            return x[2] + b * (1 - x[0]) ** 2 - x[1]

        x0 = np.random.randn(3)
        args = (100.0, 1.0)
        result = minimize(f, constr=g, x0=x0, args=args, method=method)
        assert np.allclose(result.x, [0.0, 1.0, 0.0])

        # Test with bounds
        result_bounded = minimize(
            f,
            constr=g,
            x0=x0,
            args=args,
            bounds=(np.full(3, -10), np.full(3, 10)),
            method=method,
        )
        assert np.allclose(result.x, result_bounded.x)

        # Test symbolic evaluation
        x0 = sym("x", shape=(3,), kind="MX")
        result = minimize(f, x0=x0, args=args, constr=g)
        assert isinstance(result.x, SymbolicArray)

    def test_minimize_rosenbrock(self):
        # Test the Rosenbrock function
        def f(x, a):
            return a * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        result = minimize(f, x0=[-1.0, 1.0], static_argnames=("a",), args=(100.0,))
        assert np.allclose(result.x, [1.0, 1.0])

    def test_minimize_tree(self):
        def f(params):
            x, y = params["x"], params["y"]
            return 100 * (y - x**2) ** 2 + (1 - x) ** 2

        def g(params):
            x, y = params["x"], params["y"]
            return x + y - 1.5  # x + y >= 1.5

        # Tree-structured initial guess
        x0 = {"x": 2.0, "y": 1.0}

        # Solve with inequality constraint
        result = minimize(f, x0, constr=g, constr_bounds=(0.0, np.inf))

        assert np.allclose(result.x["x"], 1.0, atol=1e-3)
        assert np.allclose(result.x["y"], 1.0, atol=1e-3)

    def test_minimize_rosenbrock_constrained(self):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        #
        # This has a local minimum at (0, 0) and a global minimum at (1, 1)

        def f(x):
            return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        def g(x):
            g1 = (x[0] - 1) ** 3 - x[1] + 1
            g2 = x[0] + x[1] - 2
            return array([g1, g2])

        result = minimize(f, constr=g, x0=[2.0, 0.0], constr_bounds=(-np.inf, 0))
        assert np.allclose(result.x, [1.0, 1.0], atol=1e-3)

    def test_error_handling(self):
        # Inconsistent arguments
        def f(x):
            return x**2

        def g(x, y):
            return x + y

        with pytest.raises(ValueError, match=r".*must have the same number"):
            minimize(f, constr=g, x0=1.0)

        # Unsupported method
        with pytest.raises(ValueError):
            minimize(f, x0=1.0, method="unsupported_method")

        # Inconsistent static arguments
        def f(a, x):
            return a * x**2

        def g(a, x):
            return a * x

        f = compile(f, static_argnames=["a"])
        g = compile(g)

        with pytest.raises(ValueError, match=r".*must have the same number"):
            minimize(f, constr=g, x0=1.0)
