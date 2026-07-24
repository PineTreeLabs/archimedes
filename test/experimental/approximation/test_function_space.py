import numpy as np
import pytest

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
    # 10 points: exact to degree 19, plenty for degree <= 8 mass/stiffness
    # integrands (2 * (n_basis - 1) for n_basis=5) and low-degree projections.
    return gauss_legendre(10)


def test_n_basis_forwarded(space):
    assert space.n_basis == 5


def test_evaluate_matches_direct_basis_contraction(space):
    coefficients = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    x = np.linspace(-1, 1, 9)
    phi = space.basis.evaluate(x, a=-1.0, b=1.0)  # (npts, n_basis)
    expected = phi @ coefficients
    np.testing.assert_allclose(space.evaluate(coefficients, x), expected)


def test_mass_matrix_is_identity_on_reference_domain(space, quad_rule):
    # OrthogonalPolynomialBasis is orthonormal by construction, so the mass
    # matrix should be the identity (not merely diagonal).
    M = space.mass_matrix(quad_rule)
    np.testing.assert_allclose(M, np.eye(space.n_basis), atol=1e-10)


def test_stiffness_matrix_is_symmetric(space, quad_rule):
    K = space.stiffness_matrix(quad_rule)
    np.testing.assert_allclose(K, K.T, atol=1e-10)


def test_mass_matrix_is_identity_on_non_reference_domain(quad_rule):
    # The basis renormalizes to stay orthonormal w.r.t. the *target* domain's
    # measure (not just the reference one), so this holds for any domain --
    # unlike a fixed classical normalization, it isn't Jacobian-scaled.
    wide_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=4), domain=(0.0, 4.0)
    )
    M = wide_space.mass_matrix(quad_rule)
    np.testing.assert_allclose(M, np.eye(4), atol=1e-10)


def test_project_recovers_exact_polynomial(space, quad_rule):
    # x^2 is even and degree 2, so -- regardless of the basis normalization
    # -- its expansion in a Legendre-derived basis has nonzero coefficients
    # only at (even) degrees 0 and 2.
    fn = space.project(lambda x: x**2, quad_rule)
    np.testing.assert_allclose(fn.coefficients[[1, 3, 4]], 0.0, atol=1e-10)
    assert abs(fn.coefficients[0]) > 1e-6
    assert abs(fn.coefficients[2]) > 1e-6

    x = np.linspace(-1, 1, 13)
    np.testing.assert_allclose(fn(x), x**2, atol=1e-10)
    assert isinstance(fn, Function)


def test_project_on_non_reference_domain(quad_rule):
    space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5), domain=(0.0, 4.0)
    )

    def f(x):
        return x**2 - 3 * x + 1

    fn = space.project(f, quad_rule)
    x = np.linspace(0, 4, 17)
    np.testing.assert_allclose(fn(x), f(x), atol=1e-8)


def test_project_of_function_outside_basis_degree_is_approximate(quad_rule):
    # n_basis=2 (degree <= 1) can't exactly represent x^2; the L2 projection
    # should be a genuine (inexact) least-squares fit, not a crash.
    space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=2), domain=(-1.0, 1.0)
    )
    fn = space.project(lambda x: x**2, quad_rule)
    x = np.linspace(-1, 1, 9)
    residual = fn(x) - x**2
    assert np.max(np.abs(residual)) > 1e-3  # not exact
    assert np.max(np.abs(residual)) < 1.0  # but still a reasonable fit
