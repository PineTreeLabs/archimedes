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


class TestSwitch:
    # Basic functionality tests with different indices
    def test_basic_indices(self):
        branches = (
            lambda x: x,
            lambda x: x**2,
            lambda x: x**3,
        )

        @arc.compile
        def apply_operation(x, op_index):
            return arc.switch(op_index, branches, x)

        x = np.array([2.0, 3.0])

        # Test each valid index
        assert np.allclose(apply_operation(x, 0), x)
        assert np.allclose(apply_operation(x, 1), x**2)
        assert np.allclose(apply_operation(x, 2), x**3)

    # Edge cases - empty or single branches
    def test_empty_branches(self):
        @arc.compile
        def apply_operation(x, op_index):
            return arc.switch(op_index, (), x)

        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            apply_operation(x, 0)

        @arc.compile
        def apply_operation(x, op_index):
            return arc.switch(op_index, (lambda x: x**2,), x)

        with pytest.raises(ValueError):
            apply_operation(x, 0)

    # Index bounds handling
    def test_index_out_of_bounds(self):
        branches = (
            lambda x: x,
            lambda x: 2 * x,
            lambda x: 3 * x,
        )

        @arc.compile
        def apply_operation(x, op_index):
            return arc.switch(op_index, branches, x)

        x = np.array([5.0])

        # Test negative indices (should clamp to first branch)
        assert np.allclose(apply_operation(x, -1), x)
        assert np.allclose(apply_operation(x, -100), x)

        # Test too large indices (should clamp to last branch)
        assert np.allclose(apply_operation(x, 3), 3 * x)
        assert np.allclose(apply_operation(x, 999), 3 * x)

    # Error handling - branch return structure mismatch
    def test_branch_structure_mismatch(self):
        @arc.compile
        def apply_operation(x, op_index):
            return arc.switch(
                op_index,
                (
                    lambda x: x,  # Returns array
                    lambda x: x[0],  # Returns scalar
                ),
                x,
            )

        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match=r".*must return the same number.*"):
            apply_operation(x, 0)

        # Tree-structured returns
        @arc.compile
        def apply_operation(x, op_index):
            return arc.switch(
                op_index,
                (
                    lambda x: x,  # Returns array
                    lambda x: {"a": x},  # Returns dict
                ),
                x,
            )

        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match=r".*must return the same tree.*"):
            apply_operation(x, 0)

    # Multiple arguments to branches
    def test_multiple_arguments(self):
        @arc.compile
        def apply_operation(x, y, op_index):
            return arc.switch(
                op_index,
                (
                    lambda x, y: x + y,
                    lambda x, y: x * y,
                ),
                x,
                y,
            )

        assert np.isclose(apply_operation(3.0, 4.0, 0), 7.0)
        assert np.isclose(apply_operation(3.0, 4.0, 1), 12.0)

    # Tree handling
    def test_tree_arguments(self):
        def process1(data):
            return {k: v * 2 for k, v in data.items()}

        def process2(data):
            return {k: v + 10 for k, v in data.items()}

        @arc.compile
        def process_data(data, op_index):
            return arc.switch(op_index, (process1, process2), data)

        data = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}

        # Test both branches
        result1 = process_data(data, 0)
        assert np.allclose(result1["a"], np.array([2.0, 4.0]))
        assert np.allclose(result1["b"], np.array([6.0, 8.0]))

        result2 = process_data(data, 1)
        assert np.allclose(result2["a"], np.array([11.0, 12.0]))
        assert np.allclose(result2["b"], np.array([13.0, 14.0]))

    # Compare numeric vs symbolic evaluation
    def test_numeric_vs_symbolic(self):
        def branch0(x):
            return x**2

        def branch1(x):
            return np.sin(x)

        # Non-compiled version (numeric)
        def numeric_switch(x, op_index):
            branches = (branch0, branch1)
            op_index = max(0, min(op_index, len(branches) - 1))
            return branches[op_index](x)

        # Compiled version (symbolic)
        @arc.compile
        def symbolic_switch(x, op_index):
            return arc.switch(op_index, (branch0, branch1), x)

        x = np.array([0.5, 1.0, 1.5])

        # Test both branches
        for i in [0, 1]:
            numeric_result = numeric_switch(x, i)
            symbolic_result = symbolic_switch(x, i)
            assert np.allclose(numeric_result, symbolic_result)

    # Test gradient computation through switch
    def test_gradient(self):
        @arc.compile
        def apply_operation(x, op_index):
            return arc.switch(
                op_index,
                (
                    lambda x: x**2,
                    lambda x: np.sin(x),
                ),
                x,
            )

        # Get gradient with respect to x
        grad_op = arc.grad(apply_operation, argnums=0)

        x = 0.5

        # Gradient of x^2 should be 2x
        grad_val = grad_op(x, 0)
        assert np.isclose(grad_val, 2 * x)

        # Gradient of sin(x) should be cos(x)
        grad_val = grad_op(x, 1)
        assert np.isclose(grad_val, np.cos(x))

    # Named switch
    def test_named_switch(self):
        @arc.compile
        def apply_operation(x, op_index):
            return arc.switch(
                op_index,
                (
                    lambda x: x**2,
                    lambda x: np.sin(x),
                ),
                x,
                name="custom_switch_name",
            )

        x = np.array([2.0])
        assert np.allclose(apply_operation(x, 0), x**2)

    # Test with compiled branch functions
    def test_compiled_branch_functions(self):
        @arc.compile
        def branch0(x):
            return x**2

        @arc.compile
        def branch1(x):
            return np.sin(x)

        @arc.compile
        def apply_operation(x, op_index):
            return arc.switch(op_index, (branch0, branch1), x)

        x = np.array([0.5, 1.0, 1.5])

        assert np.allclose(apply_operation(x, 0), x**2)
        assert np.allclose(apply_operation(x, 1), np.sin(x))

    # Nested switch operations
    def test_nested_switch(self):
        @arc.compile
        def nested_operation(x, outer_idx, inner_idx):
            branches = (
                lambda x: x**2,
                lambda x: x**3,
            )

            return arc.switch(
                outer_idx,
                (
                    # Outer branch 0: apply inner switch
                    lambda x, inner_idx: arc.switch(inner_idx, branches, x),
                    # Outer branch 1: negate the inner switch result
                    lambda x, inner_idx: -arc.switch(inner_idx, branches, x),
                ),
                x,
                inner_idx,
            )

        x = 2.0

        assert np.isclose(nested_operation(x, 0, 0), 4.0)  # x^2
        assert np.isclose(nested_operation(x, 0, 1), 8.0)  # x^3
        assert np.isclose(nested_operation(x, 1, 0), -4.0)  # -x^2
        assert np.isclose(nested_operation(x, 1, 1), -8.0)  # -x^3

    # Test branch side effects to verify all branches are traced
    def test_branch_tracing(self):
        side_effects = []

        def branch0(x):
            side_effects.append("branch0")
            return x**2

        def branch1(x):
            side_effects.append("branch1")
            return np.sin(x)

        # Clear side effects before compilation
        side_effects.clear()

        @arc.compile
        def apply_operation(x, op_index):
            return arc.switch(op_index, (branch0, branch1), x)

        # First call should trace both branches during compilation
        x = np.array([1.0])
        apply_operation(x, 0)

        # Both branches should have been traced
        assert set(side_effects) == {"branch0", "branch1"}

        # Clear and call again - no retracing should occur
        side_effects.clear()
        apply_operation(x, 1)
        assert len(side_effects) == 0


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
            vv, in_axes=[None, 1], out_axes=1
        )  #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)

        A = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
        B = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # (2, 4)
        C = mm(A, B)
        assert C.shape == (3, 4)
        assert np.allclose(C, A @ B)

    def test_vmap_errors(self):
        def f(x, y):
            return x**2 + np.sin(y)

        # Test in_axes errors
        with pytest.raises(TypeError, match=r".*in_axes must be an.*"):
            arc.vmap(f, in_axes="invalid")

        with pytest.raises(TypeError, match=r".*out_axes must be an.*"):
            arc.vmap(f, out_axes="invalid")

        # Map over inconsistent arrays
        x = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
        y = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # (2, 4)

        with pytest.raises(
            ValueError, match=r".*all mapped arguments have the same mapped axis.*"
        ):
            arc.vmap(f)(x, y)

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
        @struct
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
        @struct
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
