import numpy as np
import pytest

import archimedes as arc
from archimedes.experimental.approximation import (
    Function,
    FunctionSpace,
    OrthogonalPolynomialBasis,
)
from archimedes.measure import LegendreMeasure
from archimedes.quadrature import gauss_legendre


@pytest.fixture
def space():
    return FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5), domain=(-1.0, 1.0)
    )


@pytest.fixture
def quad_rule():
    return gauss_legendre(10)


@pytest.fixture
def quadratic(space, quad_rule):
    return space.project(lambda x: x**2, quad_rule)


def test_call_matches_target_function(quadratic):
    x = np.linspace(-1, 1, 13)
    np.testing.assert_allclose(quadratic(x), x**2, atol=1e-10)


def test_add_same_space(quadratic):
    doubled = quadratic + quadratic
    np.testing.assert_allclose(doubled.coefficients, 2 * quadratic.coefficients)
    x = np.linspace(-1, 1, 9)
    np.testing.assert_allclose(doubled(x), 2 * x**2, atol=1e-10)


def test_add_mismatched_space_raises(quadratic, quad_rule):
    other_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5), domain=(0.0, 2.0)
    )
    other = other_space.project(lambda x: x**2, quad_rule)
    with pytest.raises(ValueError):
        quadratic + other


def test_scalar_multiplication(quadratic):
    scaled = 3.0 * quadratic
    np.testing.assert_allclose(scaled.coefficients, 3.0 * quadratic.coefficients)
    rscaled = quadratic * 3.0
    np.testing.assert_allclose(rscaled.coefficients, scaled.coefficients)


def test_is_struct_pytree(quadratic):
    leaves, treedef = arc.tree.flatten(quadratic)
    assert len(leaves) == 1  # only `coefficients` is a leaf; `space` is static
    np.testing.assert_array_equal(leaves[0], quadratic.coefficients)

    rebuilt = arc.tree.unflatten(treedef, [quadratic.coefficients * 2])
    np.testing.assert_allclose(rebuilt.coefficients, quadratic.coefficients * 2)
    assert rebuilt.space == quadratic.space


# -- symbolic tracing / autodiff --


def test_call_compiles_and_matches_static_evaluation(space, quadratic):
    x = np.linspace(-1, 1, 9)
    static_vals = np.array([quadratic(xi) for xi in x])

    @arc.compile
    def traced(x, c):
        return Function(c, space)(x)

    dynamic_vals = np.array([float(traced(xi, quadratic.coefficients)) for xi in x])
    np.testing.assert_allclose(static_vals, dynamic_vals, atol=1e-12)


def test_grad_wrt_x(space, quadratic):
    @arc.compile
    def traced(x, c):
        return Function(c, space)(x)

    dfdx = arc.grad(traced, argnums=0)
    x = np.linspace(-0.8, 0.8, 5)
    computed = np.array([float(dfdx(xi, quadratic.coefficients)) for xi in x])
    np.testing.assert_allclose(computed, 2 * x, atol=1e-10)


def test_grad_wrt_coefficients_matches_basis_values(space, quadratic):
    @arc.compile
    def traced(x, c):
        return Function(c, space)(x)

    dfdc = arc.grad(traced, argnums=1)
    x0 = 0.37
    grad = np.asarray(dfdc(x0, quadratic.coefficients)).ravel()
    phi0 = space.basis.evaluate(np.array([x0]), a=-1.0, b=1.0)[0, :]
    np.testing.assert_allclose(grad, phi0, atol=1e-10)
