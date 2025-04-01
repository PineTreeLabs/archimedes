# ruff: noqa: N806
# ruff: noqa: N803

import casadi as cs
import numpy as np
import pytest

from archimedes import (
    compile,
    grad,
    hess,
    jac,
    jvp,
    vjp,
)
from archimedes import (
    sym as _sym,
)

# NOTE: Most tests here use SX instead of the default MX, since the is_equal
# tests struggle with the array-valued MX type.  This doesn't indicate an error
# in the MX representation, just a difficulty of checking for equality between
# array-valued symbolic expressions


# Override the default symbolic kind to use SX
def sym(*args, kind="SX", **kwargs):
    return _sym(*args, kind=kind, **kwargs)


class TestGrad:
    def test_grad(self):
        # Test evaluating the gradient symbolically
        def f(x):
            return 0.5 * x.T @ x

        df = grad(f)

        # Test with different input shapes (will "recompile" the underlying functions)
        for x in (
            sym("x", shape=(3,), dtype=np.float64),
            sym("x", shape=(2,), dtype=np.float64),
        ):
            J = df(x)
            assert J.shape == x.shape

            assert cs.is_equal(
                cs.simplify(J._sym),
                cs.simplify(x._sym),
                1,
            )

        # Will get an error if the output is not a scalar
        x = sym("x", shape=(3, 1), dtype=np.float64)
        with pytest.raises(ValueError):
            J = df(x)

    def test_grad_multiple_args(self):
        # Test evaluating the gradient with respect to multiple inputs.
        # This will also evaluate the gradient function numerically (though it
        # can also be evaluated symbolically, as in test_grad).
        @compile
        def f(x, y):
            return x.T @ y

        df = grad(f, argnums=(0, 1))

        # Test with different input shapes (will "recompile" the underlying functions)
        for n in (2, 3):
            x = np.random.randn(n)
            y = np.random.randn(n)
            Jx, Jy = df(x, y)
            assert Jx.shape == y.shape
            assert Jy.shape == x.shape

            assert np.allclose(Jx, y)
            assert np.allclose(Jy, x)

    def test_grad_static_args(self):
        def f(a, x):
            return np.sin(a * x[0]) * np.cos(x[1])

        f_sym = compile(f, static_argnames=("a",))
        grad_f = grad(f_sym, argnums=1)

        a = 2.0
        x = np.random.randn(2)
        df = grad_f(a, x)
        assert df.shape == x.shape

        df_ex = np.array(
            [a * np.cos(a * x[0]) * np.cos(x[1]), -np.sin(a * x[0]) * np.sin(x[1])]
        )
        assert np.allclose(df, df_ex)

        # Cannot differentiate with respect to a static arg
        with pytest.raises(ValueError):
            grad_f = grad(f_sym, argnums=0)

    def test_error_handling(self):
        def f(x):
            return 0.5 * x.T @ x

        # Test with invalid argnums
        with pytest.raises(ValueError):
            grad(f, argnums=(0.5,))

        # Test with invalid return type (multiple returns not supported)
        def f(x):
            return x, 3 * x

        x = sym("x", shape=(3,))
        df = grad(f)
        with pytest.raises(ValueError):
            df(x)


class TestJac:
    def test_jac(self):
        def f(x):
            return 0.5 * x.T @ x

        df = jac(f)

        # Test with different input shapes (will "recompile" the underlying functions)
        for x in (
            sym("x", shape=(3,), dtype=np.float64),
            sym("x", shape=(3, 1), dtype=np.float64),
            sym("x", shape=(2,), dtype=np.float64),
        ):
            J = df(x)
            assert J.shape == x.T.shape

            assert cs.is_equal(
                cs.simplify(J._sym),
                cs.simplify(x.T._sym),
                1,
            )

    def test_jac_static_args(self):
        def _test(df, n=3):
            for a in (2.0, 3.0):
                x = np.random.randn(3)
                J = df(a, x)
                assert J.shape == (n, n)
                assert np.allclose(J, a * np.eye(n))

        # Test evaluating the Jacobian with respect to a single input, with a
        # static argument.
        def f(a, x):
            return a * x

        # Test with no static args, just with specifying the second argument
        f_sym = compile(f)
        df = jac(f_sym, argnums=1)
        _test(df)

        # Test with a static argument specified by name
        f_sym = compile(f, static_argnames=("a",))
        df = jac(f_sym, argnums=1)
        _test(df)

        # Cannot differentiate with respect to a static arg
        with pytest.raises(ValueError):
            df = jac(f_sym, argnums=0)

    def test_error_handling(self):
        def f(x):
            return 0.5 * x.T @ x

        # Test with invalid argnums
        with pytest.raises(ValueError):
            jac(f, argnums=(0.5,))

        # Test with invalid return type (multiple returns not supported)
        def f(x):
            return x, 3 * x

        x = sym("x", shape=(3,))
        df = jac(f)
        with pytest.raises(ValueError):
            df(x)


class TestHess:
    def test_hess(self):
        def f(Q, x):
            return 0.5 * x.T @ Q @ x

        hess_fun = hess(f, argnums=1)

        for n in (2, 3):
            Q = np.diag(np.arange(n))
            x = np.random.randn(n)
            H = hess_fun(Q, x)
            assert H.shape == (n, n)
            assert np.allclose(H, Q)

    def test_error_handling(self):
        def f(x):
            return 0.5 * x.T @ x

        # Test with invalid argnums
        with pytest.raises(ValueError):
            hess(f, argnums=(0.5,))

        # Differentiation with respect to static arg
        def f(a, x):
            return a * x.T @ x

        f = compile(f, static_argnames=("a",))
        with pytest.raises(ValueError):
            hess(f, argnums=0)

        # Test with invalid return type (multiple returns not supported)
        def f(Q, x):
            return 0.5 * x.T @ x, x.T @ Q @ x

        Q = np.random.randn(3, 3)
        x = sym("x", shape=(3,))
        hess_fun = hess(f, argnums=1)
        with pytest.raises(ValueError):
            hess_fun(Q, x)

        # Test with invalid return type (vector-valued)
        def f(x):
            return x

        x = sym("x", shape=(3,))
        hess_fun = hess(f)
        with pytest.raises(ValueError):
            hess_fun(x)


class TestJVP:
    def test_jvp(self):
        # Test with a simple scalar-valued function
        def f(x):
            return 0.5 * x.T @ x

        jvp_fun = jvp(f)

        # Test with different input shapes (will "recompile" the underlying functions)
        for shape in ((3,), (2,), (3, 1)):
            x = np.random.randn(*shape)
            v = np.random.randn(*shape)

            # Expect the result to be equivalent to dot(x, v)
            df = jvp_fun(x, v)
            df_ex = x.T @ v

            assert df.shape == df_ex.shape
            assert np.allclose(df, df_ex)

    def test_jvp_vec(self):
        # Test with a simple vector-valued function
        m, n = 3, 2
        A = np.random.randn(m, n)

        @compile
        def f(x):
            return A @ x

        jvp_fun = jvp(f)

        x = np.random.randn(n)
        v = np.random.randn(n)
        df = jvp_fun(x, v)
        df_ex = A @ v

        assert df.shape == df_ex.shape
        assert np.allclose(df, df_ex)

    def test_error_handling(self):
        # Invalid return types
        def f(x):
            return x, 3 * x

        x = sym("x", shape=(3,))
        jvp_fun = jvp(f)
        with pytest.raises(ValueError):
            jvp_fun(x, x)


class TestVJP:
    def test_vjp(self):
        def f(x):
            return 0.5 * x.T @ x

        vjp_fun = vjp(f)

        # Test with different input shapes (will "recompile" the underlying functions)
        for shape in ((3,), (2,), (3, 1)):
            x = np.random.randn(*shape)
            w = np.random.randn()  # Scalar output

            # Expect the result to be equivalent to dot(w, x)
            df = vjp_fun(x, w)
            df_ex = w * x

            assert df.shape == df_ex.shape
            assert np.allclose(df, df_ex)

    def test_vjp_vec(self):
        # Test with a simple vector-valued function
        m, n = 3, 2
        A = np.random.randn(m, n)

        @compile
        def f(x):
            return A @ x

        vjp_fun = vjp(f)

        x = np.random.randn(n)
        w = np.random.randn(m)

        df = vjp_fun(x, w)
        df_ex = A.T @ w

        assert df.shape == df_ex.shape
        assert np.allclose(df, df_ex)

    def test_error_handling(self):
        # Multiple arguments currently unsupported
        def f(x, y):
            return x.T @ y

        with pytest.raises(NotImplementedError):
            vjp(f)

        # Invalid return types
        def f(x):
            return x, 3 * x

        x = sym("x", shape=(3,))
        vjp_fun = vjp(f)
        with pytest.raises(ValueError):
            vjp_fun(x, x)
