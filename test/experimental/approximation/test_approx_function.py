import numpy as np
import pytest

import archimedes as arc
from archimedes.experimental.approximation import (
    Function,
    FunctionSpace,
    OrthogonalPolynomialBasis,
)
from archimedes.measure import LegendreMeasure, UnitInterval
from archimedes.quadrature import gauss_legendre


@pytest.fixture
def quad_rule():
    return gauss_legendre(10)


@pytest.fixture
def space(quad_rule):
    return FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5),
        domain=UnitInterval.Parameters(a=-1.0, b=1.0),
        quad_rule=quad_rule,
    )


@pytest.fixture
def quadratic(space):
    return space.project(lambda x: x**2)


def test_call_matches_target_function(quadratic):
    x = np.linspace(-1, 1, 13)
    np.testing.assert_allclose(quadratic(x), x**2, atol=1e-10)


def test_add_same_space(quadratic):
    doubled = quadratic + quadratic
    np.testing.assert_allclose(doubled.coefficients, 2 * quadratic.coefficients)
    x = np.linspace(-1, 1, 9)
    np.testing.assert_allclose(doubled(x), 2 * x**2, atol=1e-10)


def test_add_structurally_mismatched_space_raises(quadratic, quad_rule):
    # Different n_basis -> different basis -> genuinely incompatible.
    other_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=4),
        domain=UnitInterval.Parameters(a=-1.0, b=1.0),
        quad_rule=quad_rule,
    )
    other = other_space.project(lambda x: x**2)
    with pytest.raises(ValueError):
        quadratic + other


def test_add_numerically_mismatched_domain_is_not_caught(quadratic, quad_rule):
    # Documented limitation: `_is_compatible_with` compares the domain only
    # structurally, since values are undecidable once traced. Two spaces
    # differing *only* in domain values are therefore accepted -- the caller
    # is responsible for ensuring they agree numerically.
    other_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5),
        domain=UnitInterval.Parameters(a=0.0, b=2.0),
        quad_rule=quad_rule,
    )
    other = other_space.project(lambda x: x**2)
    result = quadratic + other  # does not raise
    np.testing.assert_allclose(
        result.coefficients, quadratic.coefficients + other.coefficients
    )


def test_scalar_multiplication(quadratic):
    scaled = 3.0 * quadratic
    np.testing.assert_allclose(scaled.coefficients, 3.0 * quadratic.coefficients)
    rscaled = quadratic * 3.0
    np.testing.assert_allclose(rscaled.coefficients, scaled.coefficients)


def test_is_struct_pytree(quadratic):
    # coefficients + the domain's (a, b): `basis`/`quad_rule` are static, but
    # the domain parameters are leaves so they can be traced/optimized.
    leaves, treedef = arc.tree.flatten(quadratic)
    assert len(leaves) == 3
    np.testing.assert_array_equal(leaves[0], quadratic.coefficients)
    assert leaves[1:] == [-1.0, 1.0]

    rebuilt = arc.tree.unflatten(treedef, [quadratic.coefficients * 2, -1.0, 1.0])
    np.testing.assert_allclose(rebuilt.coefficients, quadratic.coefficients * 2)
    assert rebuilt.space._is_compatible_with(quadratic.space)


def test_domain_parameters_are_traceable(space, quadratic):
    # The point of making FunctionSpace a struct: the domain endpoints are
    # pytree leaves, so they can be traced and differentiated through.
    x0 = 0.4

    @arc.compile
    def evaluate(c, a, b):
        moving = FunctionSpace(
            space.basis,
            domain=UnitInterval.Parameters(a=a, b=b),
            quad_rule=space.quad_rule,
        )
        return Function(c, moving)(np.array([x0]))[0]

    # Matches the equivalent static-domain evaluation
    val = evaluate(quadratic.coefficients, -1.0, 1.0)
    np.testing.assert_allclose(float(val), float(quadratic(np.array([x0]))[0]))

    # d/da and d/db agree with finite differences of the same function
    h = 1e-6
    for argnum, (da, db) in enumerate([(h, 0.0), (0.0, h)], start=1):
        analytic = float(
            arc.grad(evaluate, argnums=argnum)(quadratic.coefficients, -1.0, 1.0)
        )
        fd = (
            float(evaluate(quadratic.coefficients, -1.0 + da, 1.0 + db))
            - float(evaluate(quadratic.coefficients, -1.0 - da, 1.0 - db))
        ) / (2 * h)
        np.testing.assert_allclose(analytic, fd, atol=1e-6)


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


# -- dot / norm --


def test_dot_matches_space_inner_product(space, quadratic):
    cubic = space.project(lambda x: x**3)
    expected = space.inner_product(quadratic.coefficients, cubic.coefficients)
    np.testing.assert_allclose(quadratic.dot(cubic), expected)


def test_dot_of_odd_and_even_function_on_symmetric_domain_vanishes(space):
    # x^2 (even) and x^3 (odd) are L2-orthogonal on [-1, 1].
    even_fn = space.project(lambda x: x**2)
    odd_fn = space.project(lambda x: x**3)
    np.testing.assert_allclose(even_fn.dot(odd_fn), 0.0, atol=1e-10)


def test_norm_matches_sqrt_self_dot(quadratic):
    np.testing.assert_allclose(quadratic.norm() ** 2, quadratic.dot(quadratic))


def test_dot_structurally_mismatched_space_raises(quadratic, quad_rule):
    other_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=4),
        domain=UnitInterval.Parameters(a=-1.0, b=1.0),
        quad_rule=quad_rule,
    )
    other = other_space.project(lambda x: x**2)
    with pytest.raises(ValueError):
        quadratic.dot(other)
