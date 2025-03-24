import pytest

import numpy as np
import casadi as cs

import archimedes as arc
from archimedes.core import sym, SymbolicArray


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

        with pytest.raises(ValueError, match=r'.*must be equal to length.*'):
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
            return x ** 2
    
        with pytest.raises(ValueError, match=r'.*exactly two arguments.*'):
            arc.scan(g, 0.0, length=3)

        # No length or xs provided
        with pytest.raises(ValueError, match=r'.*xs or length.*'):
            arc.scan(f, 0.0)

        # Invalid number of returns
        def g(carry, x):
            return 2 * x
        
        with pytest.raises(ValueError, match=r'.*exactly two outputs.*'):
            arc.scan(g, 0.0, length=3)

        # Inconsistent output shape for carry
        def g(carry, x):
            return (carry, carry), 2 * x

        with pytest.raises(ValueError, match=r'.*same type for the carry.*'):
            arc.scan(g, 0.0, length=3)

        # Too many output dimensions
        def g(carry, x):
            return carry, np.atleast_2d(x)

        with pytest.raises(ValueError, match=r'.*can only be 0- or 1-D.*'):
            arc.scan(g, 0.0, length=3)