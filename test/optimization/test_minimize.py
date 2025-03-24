import pytest
import numpy as np

from archimedes._core import array, sym, sym_function, SymbolicArray
from archimedes._optimization import minimize, root, implicit
from archimedes.error import ShapeDtypeError


class TestMinimize:
    def test_minimize(self):
        # Basic functionality test
        def f(x):
            return x**2

        x_opt = minimize(f, x0=1.0)
        assert np.allclose(x_opt, 0.0)

    def test_minimize_with_param(self):
        # Test with a parameter
        def f(x, a=1.0):
            return a * x**2

        x_opt = minimize(f, x0=1.0, args=(2.0,))
        assert np.allclose(x_opt, 0.0)

    def test_minimize_with_bounds(self):
        def f(x):
            return x**2

        x_opt = minimize(f, x0=1.0, bounds=[-2.0, 2.0])
        assert np.allclose(x_opt, 0.0)

    def test_minimize_constrained(self):
        # Test with function from the CasADi docs, using additional parameters
        def f(x, a, b):
            return x[0] ** 2 + a * x[2] ** 2

        def g(x, a, b):
            return x[2] + b * (1 - x[0]) ** 2 - x[1]

        x0 = np.random.randn(3)
        args = (100.0, 1.0)
        x_opt = minimize(f, constr=g, x0=x0, args=args)
        assert np.allclose(x_opt, [0.0, 1.0, 0.0])

        # Test with bounds
        x_opt_bounded = minimize(
            f, constr=g, x0=x0, args=args, bounds=(np.full(3, -10), np.full(3, 10))
        )
        assert np.allclose(x_opt, x_opt_bounded)

        # Test symbolic evaluation
        x0 = sym("x", shape=(3,), kind="MX")
        x_opt = minimize(f, x0=x0, args=args, constr=g)
        assert isinstance(x_opt, SymbolicArray)

    def test_minimize_rosenbrock(self):
        # Test the Rosenbrock function
        def f(x, a):
            return a * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        x_opt = minimize(f, x0=[-1.0, 1.0], static_argnames=("a",), args=(100.0,))
        assert np.allclose(x_opt, [1.0, 1.0])

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

        x_opt = minimize(f, constr=g, x0=[2.0, 0.0], constr_bounds=(-np.inf, 0))
        assert np.allclose(x_opt, [1.0, 1.0], atol=1e-3)

    def test_error_handling(self):
        # Inconsistent arguments
        def f(x):
            return x ** 2
        
        def g(x, y):
            return x + y
        
        with pytest.raises(ValueError):
            minimize(f, constr=g, x0=1.0)

        # Inconsistent static arguments
        def f(a, x):
            return a * x ** 2
        
        def g(a, x):
            return a * x
        
        f = sym_function(f, static_argnames=["a"])
        g = sym_function(g)

        with pytest.raises(ValueError):
            minimize(f, constr=g, x0=1.0)

        # Matrix-valued decision variables
        def f(x):
            return np.dot(x, x)
        
        with pytest.raises(ValueError):
            minimize(f, x0=np.ones((2, 2)))

