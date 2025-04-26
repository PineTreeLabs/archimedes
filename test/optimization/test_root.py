# ruff: noqa: N802
# ruff: noqa: N803
# ruff: noqa: N806

import numpy as np

from archimedes.error import ShapeDtypeError
from archimedes.optimize import implicit, root


class TestImplicitFunction:
    def test_implicit(self):
        # An implicit function with no other arguments
        def F(x):
            return x**2 - 1

        # The implicit function will return the root of F given
        # an initial guess x0
        g = implicit(F)
        x = g(x0=2.0)

        assert np.allclose(x, 1.0)

    def test_implicit_args(self):
        # An implicit function with additional arguments
        def F(x, z):
            return x**2 - z

        g = implicit(F)
        z = 4.0
        x = g(z=z, x0=1.0)

        assert np.allclose(x, 2.0)

    def test_implicit_static_args(self):
        # An implicit function with additional arguments
        def F(x, z, a=1.0):
            return a * x**2 - z

        g = implicit(F, static_argnames=["a"])
        z = 4.0
        x = g(z=z, a=2.0, x0=1.0)
        print(x)

        assert np.allclose(x, np.sqrt(2))

    def test_implicit_incorrect_shape(self):
        # Test that the implicit function raises an error if the shape
        # of the output is incorrect
        def F(x):
            return np.dot(x, x)

        g = implicit(F)

        with np.testing.assert_raises(ShapeDtypeError):
            g(x0=np.ones(2))


class TestRoot:
    def test_root(self):
        # A simple root-finding problem
        def f(x):
            return x**2 - 1

        x = root(f, x0=2.0, tol=1e-6)
        assert np.allclose(x, 1.0)

    def test_root_args(self):
        # A root-finding problem with additional arguments
        def f(x, a):
            return x**2 - a

        x = root(f, x0=1.0, args=(4.0,))
        assert np.allclose(x, 2.0)

    def test_root_static_args(self):
        def f(x, shift=True):
            res = x**2
            if shift:
                res -= 1
            return res

        for shift in [True, False]:
            x = root(
                f,
                x0=2.0,
                args=(shift,),  # This should not shift the result
                static_argnames=["shift"],
            )

            x_ex = 1.0 if shift else 0.0
            assert np.allclose(x, x_ex, atol=1e-6)

    def test_root_error_handling(self):
        def f(x):
            return np.dot(x, x)

        with np.testing.assert_raises(ShapeDtypeError):
            root(f, x0=np.ones(2))

        with np.testing.assert_raises(ValueError):
            root(f, x0=np.ones((2, 2)))
