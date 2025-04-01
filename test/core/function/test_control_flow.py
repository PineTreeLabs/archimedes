# ruff: noqa: N802
# ruff: noqa: N803
# ruff: noqa: N806

import casadi as cs
import numpy as np
import pytest

import archimedes as arc
from archimedes import struct
from archimedes._core import SymbolicArray
from archimedes._core import sym as _sym

# NOTE: Most tests here use SX instead of the default MX, since the is_equal
# tests struggle with the array-valued MX type.  This doesn't indicate an error
# in the MX representation, just a difficulty of checking for equality between
# array-valued symbolic expressions


# Override the default symbolic kind to use SX
def sym(*args, kind="SX", **kwargs):
    return _sym(*args, kind=kind, **kwargs)


def f(carry, x):
    carry = carry + x
    return carry, 2 * x


class TestScan:
    def test_numeric(self):
        # Test with 1D array input
        xs = np.arange(10)
        carry, ys = arc.scan(f, 0.0, xs)
        assert ys.shape == xs.shape
        assert carry == sum(xs)
        assert np.allclose(ys, 2 * xs)

        # Test with 2D array
        xs = np.stack([xs, ys], axis=1)
        init_carry = np.array([0, 0])
        carry, ys = arc.scan(f, init_carry, xs)
        assert ys.shape == xs.shape
        assert ys.dtype == xs.dtype
        assert carry.dtype == xs.dtype
        assert np.allclose(ys, 2 * xs)
        assert np.allclose(carry, sum(xs))

        # Test with length argument
        xs = np.arange(10)
        carry, ys = arc.scan(f, 0.0, length=len(xs))
        assert ys.shape == xs.shape
        assert carry == sum(xs)
        assert np.allclose(ys, 2 * xs)

        # Test with both xs and length arguments
        carry2, ys2 = arc.scan(f, 0.0, xs=xs, length=len(xs))
        assert np.allclose(ys, ys2)
        assert np.allclose(carry2, carry)

        with pytest.raises(ValueError, match=r".*must be equal to length.*"):
            arc.scan(f, 0.0, xs=xs, length=42)

    def test_symbolic(self):
        # Test with 1D array
        xs = sym("x", shape=(3,))
        carry, ys = arc.scan(f, 0.0, xs)
        assert isinstance(ys, SymbolicArray)
        assert ys.shape == xs.shape
        assert ys.dtype == xs.dtype
        assert isinstance(carry, SymbolicArray)
        assert carry.dtype == xs.dtype
        assert cs.is_equal(ys._sym, 2 * xs._sym, 1)
        assert cs.is_equal(carry._sym, cs.sum1(xs._sym), 2)

        # Test with 2D array
        xs = sym("x", shape=(3, 2), dtype=int)
        carry, ys = arc.scan(f, np.array([0, 0]), xs)
        assert ys.shape == xs.shape
        assert ys.dtype == xs.dtype
        assert carry.dtype == xs.dtype
        assert cs.is_equal(ys._sym, 2 * xs._sym, 1)
        assert cs.is_equal(carry._sym, cs.sum1(xs._sym).T, 2)

    def test_dummy_return(self):
        def f(carry, x):
            carry = carry + x
            return carry, np.array([])

        xs = np.arange(10)
        carry, ys = arc.scan(f, 0.0, xs)
        assert np.allclose(carry, sum(xs))
        assert ys.size == 0

    def test_error_handling(self):
        # Invalid function signature
        def g(x):
            return x**2

        with pytest.raises(ValueError, match=r".*exactly two arguments.*"):
            arc.scan(g, 0.0, length=3)

        # No length or xs provided
        with pytest.raises(ValueError, match=r".*xs or length.*"):
            arc.scan(f, 0.0)

        # Invalid number of returns
        def g(carry, x):
            return 2 * x

        with pytest.raises(ValueError, match=r".*exactly two outputs.*"):
            arc.scan(g, 0.0, length=3)

        # Inconsistent output shape for carry
        def g(carry, x):
            return (carry, carry), 2 * x

        with pytest.raises(ValueError, match=r".*same type for the carry.*"):
            arc.scan(g, 0.0, length=3)

        # Too many output dimensions
        def g(carry, x):
            return carry, np.atleast_2d(x)

        with pytest.raises(ValueError, match=r".*can only be 0- or 1-D.*"):
            arc.scan(g, 0.0, length=3)


class TestVmap:
    def test_product(self):
        # Example from JAX docs: matrix-matrix product
        def vv(a, b):
            return np.dot(a, b)

        mv = arc.vmap(
            vv, (0, None), 0
        )  #  ([b,a], [a]) -> [b]      (b is the mapped axis)

        A = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([1, 2])
        c = mv(A, b)
        assert c.shape == (3,)
        assert np.allclose(c, A @ b)

        mm = arc.vmap(
            vv, (None, 1), 1
        )  #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)

        A = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
        B = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # (2, 4)
        C = mm(A, B)
        assert C.shape == (3, 4)
        assert np.allclose(C, A @ B)

    def test_vmap_dot(self):
        def dot(a, b):
            return np.dot(a, b)

        batched_dot = arc.vmap(dot)
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([[7, 8], [9, 10], [11, 12]])

        z = batched_dot(x, y)
        assert z.shape == (3,)
        assert np.allclose(z, np.array([np.dot(x[i], y[i]) for i in range(3)]))

    def test_vmap_unravel(self):
        @struct.pytree_node
        class PointMass:
            pos: np.ndarray
            vel: np.ndarray

        p = PointMass(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        p_flat, unravel = arc.tree.ravel(p)
        unravel = arc.compile(unravel, kind="SX")

        ps_flat = np.random.randn(10, 4)
        ps = arc.vmap(unravel)(ps_flat)
        assert isinstance(ps, PointMass)

        assert ps.pos.shape == (10, 2)
        assert np.allclose(ps.pos, ps_flat[:, :2])
        assert ps.vel.shape == (10, 2)
        assert np.allclose(ps.vel, ps_flat[:, 2:])

    def test_vmap_with_arg(self):
        @struct.pytree_node
        class PointMass:
            pos: np.ndarray
            vel: np.ndarray

        def update(p, dt):
            return p.replace(pos=p.pos + dt * p.vel)

        map_update = arc.vmap(update, in_axes=(0, None))

        x = np.random.randn(10, 3)
        v = np.random.randn(10, 3)
        particles = PointMass(pos=x, vel=v)

        dt = 0.1
        new_particles = map_update(particles, dt)
        assert isinstance(new_particles, PointMass)
        assert new_particles.pos.shape == (10, 3)
        assert new_particles.vel.shape == (10, 3)

        assert np.allclose(new_particles.pos, x + dt * v)
