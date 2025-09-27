import numpy as np
import pytest

from archimedes._core import FunctionCache, SymbolicArray, compile, sym


def f(x):
    return x
    return x**2 + 1


class TestCompile:
    def test_construction(self):
        f_sym = compile(f)
        assert isinstance(f_sym, FunctionCache)
        assert f_sym.name == "f"
        assert f_sym.arg_names == ["x"]
        assert f_sym._compiled == {}

        # Test recompilation
        f_sym2 = compile(f_sym)
        assert f_sym2 is f_sym

    def test_naming(self):
        # Lambda function
        f_sym = compile(lambda x: x)
        assert f_sym.name.startswith("lambda")

        # Callable class
        class _Functor:
            def __call__(self, x):
                return 2 * x

        func = _Functor()
        assert not hasattr(func, "__name__")
        f_sym = compile(func)
        assert f_sym.name.startswith("function")

    def test_eval_numeric(self):
        f_sym = compile(f)

        # Scalar
        x = 2
        assert f_sym(x) == f(x)

        # NumPy Arrays of different shapes (will have to "recompile")
        for x in (
            np.array([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
        ):
            y = f_sym(x)
            print(y)
            assert isinstance(y, np.ndarray)
            assert np.all(y == f(x))
            assert y.shape == x.shape

    def test_eval_symbolic(self):
        f_sym = compile(f)

        x = sym("x", shape=(), dtype=np.int32)
        y = f_sym(x)
        assert isinstance(y, SymbolicArray)
        assert y.shape == ()
        assert y.dtype == np.int32

    def test_multiple_returns(self):
        # Test a function that returns multiple values
        def f2(x):
            return x**2, x + 1

        f_sym = compile(f2)

        x = np.array(2.0, dtype=np.float64)
        y1, y2 = f_sym(x)
        y1_expected, y2_expected = f2(x)
        assert y1 == y1_expected
        assert y2 == y2_expected

        assert isinstance(y1, np.ndarray)
        assert isinstance(y2, np.ndarray)
        assert y1.dtype == x.dtype
        assert y2.dtype == x.dtype

    def test_tree_returns(self):
        @compile
        def g(x, y):
            return (x, y), 2 * x

        out = g(0.0, 1.0)
        print(f"{out=}")

    def test_static(self):
        # Test compiling function with static argument specified by number

        def f2(a, x):
            return a * x

        # Create functions specifying static data both by number and by name
        for f_sym in (
            compile(f2, static_argnums=0),
            compile(f2, static_argnames="a"),
        ):
            # Nothing compiled yet
            assert len(f_sym._compiled) == 0

            a = 2.0
            x = np.array(3.0, dtype=np.float64)
            y = f_sym(a, x)
            assert y == a * x
            # Only one function compiled so far
            assert len(f_sym._compiled) == 1

            # Call again with different static data
            a = 3.0
            y = f_sym(a, x)
            assert y == a * x
            # Should have compiled a new function
            assert len(f_sym._compiled) == 2

    def test_kwargs(self):
        a0 = 2.0

        def f(x, a=a0):
            return a * x

        f_sym = compile(f)
        x = 3.0

        # Call without specifying `a`
        y = f_sym(x)
        assert y == a0 * x

        # Specify `a` by keyword
        a = 3.0
        y = f_sym(x, a=a)
        assert y == a * x

        # Specify `a` by position
        y = f_sym(x, a)
        assert y == a * x

        # Specify both args by keyword
        y = f_sym(x=x, a=a0)
        assert y == a0 * x

    def test_numeric_returns(self):
        # Should be able to return both numeric and symbolic arrays
        @compile
        def f(x):
            return x, np.array([3.0, 4.0]), True, 3

        x = np.array(2.0, dtype=np.float64)
        y1, y2, y3, y4 = f(x)

        assert all(isinstance(y, np.ndarray) for y in (y1, y2, y3, y4))

        assert y1 == x
        assert np.all(y2 == np.array([3.0, 4.0]))
        assert y3.dtype == np.bool_
        assert y3
        assert y4.dtype == int
        assert y4 == 3

    def test_error_handling(self):
        # Test function with varargs
        def f(x, *args):
            return x + sum(args)

        with pytest.raises(ValueError):
            compile(f)

        # Specify both static_argnums and static_argnames
        def f(a, x):
            return a * x

        with pytest.raises(ValueError):
            compile(f, static_argnums=(0,), static_argnames=("a",))

        # Static arg not in signature
        with pytest.raises(ValueError):
            compile(f, static_argnames=("b",))

        # Call with incorrect number of arguments (raised by `inspect`)
        f = compile(f, static_argnames=("a",))
        with pytest.raises(TypeError):
            f(2.0)

        # Return something besides an array
        def f(x):
            return "abc"

        with pytest.raises(TypeError):
            compile(f)(0.0)

        # Return variable number of values
        def f(x, flag):
            if flag:
                return 2 * x
            else:
                return 2 * x, 3 * x

        f = compile(f, static_argnames=("flag",))
        f(2.0, True)  # OK the first time
        with pytest.raises(ValueError):
            f(2.0, False)  # Fail the second time

        # Unhashable static arg (list)
        def f(a, x):
            return x

        f = compile(f, static_argnames=("a",))
        with pytest.raises(ValueError):
            f([1.0], 2.0)

    def test_modified_array(self):
        # Check that in-place modification of an array doesn't change
        # the original value (because dummy args are created during compilation)
        x = np.zeros((2, 3))

        @compile
        def f(x):
            x[0] = x[0] + 1.0
            return x

        y = f(x)
        assert np.allclose(x, 0.0)
        assert np.allclose(y[0], 1.0)
        assert np.allclose(y[1], 0.0)
