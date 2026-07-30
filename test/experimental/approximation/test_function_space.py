# ruff: noqa: N806  (M, K are the conventional names for these matrices)
import numpy as np
import pytest

from archimedes.experimental.approximation import (
    Function,
    FunctionSpace,
    OrthogonalPolynomialBasis,
)
from archimedes.measure import (
    HermiteNormMeasure,
    LegendreMeasure,
    RealLine,
    UnitInterval,
)
from archimedes.quadrature import gauss_legendre


@pytest.fixture
def quad_rule():
    # 10 points: exact to degree 19, plenty for degree <= 8 mass/stiffness
    # integrands (2 * (n_basis - 1) for n_basis=5) and low-degree projections.
    return gauss_legendre(10)


@pytest.fixture
def space(quad_rule):
    return FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5),
        domain=UnitInterval.Parameters(a=-1.0, b=1.0),
        quad_rule=quad_rule,
    )


def test_n_basis_forwarded(space):
    assert space.n_basis == 5


def test_domain_must_match_basis_parameters_type(quad_rule):
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5)
    with pytest.raises(TypeError):
        FunctionSpace(basis, domain=(-1.0, 1.0), quad_rule=quad_rule)
    with pytest.raises(TypeError):
        # Wrong domain type -- RealLine's Parameters takes mean/std, not
        # a/b, so it isn't a UnitInterval.Parameters.
        FunctionSpace(basis, domain=RealLine.Parameters(), quad_rule=quad_rule)


def test_evaluate_matches_direct_basis_contraction(space):
    coefficients = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    x = np.linspace(-1, 1, 9)
    phi = space.basis.evaluate(x, a=-1.0, b=1.0)  # (npts, n_basis)
    expected = phi @ coefficients
    np.testing.assert_allclose(space.evaluate(coefficients, x), expected)


def test_mass_matrix_is_identity_on_reference_domain(space):
    # OrthogonalPolynomialBasis is orthonormal by construction, so the mass
    # matrix should be the identity (not merely diagonal).
    M = space.mass_matrix()
    np.testing.assert_allclose(M, np.eye(space.n_basis), atol=1e-10)


def test_stiffness_matrix_is_symmetric(space):
    K = space.stiffness_matrix()
    np.testing.assert_allclose(K, K.T, atol=1e-10)


def test_mass_matrix_is_identity_on_non_reference_domain(quad_rule):
    # The basis renormalizes to stay orthonormal w.r.t. the *target* domain's
    # measure (not just the reference one), so this holds for any domain --
    # unlike a fixed classical normalization, it isn't Jacobian-scaled.
    wide_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=4),
        domain=UnitInterval.Parameters(a=0.0, b=4.0),
        quad_rule=quad_rule,
    )
    M = wide_space.mass_matrix()
    np.testing.assert_allclose(M, np.eye(4), atol=1e-10)


def test_project_recovers_exact_polynomial(space):
    # x^2 is even and degree 2, so -- regardless of the basis normalization
    # -- its expansion in a Legendre-derived basis has nonzero coefficients
    # only at (even) degrees 0 and 2.
    fn = space.project(lambda x: x**2)
    np.testing.assert_allclose(fn.coefficients[[1, 3, 4]], 0.0, atol=1e-10)
    assert abs(fn.coefficients[0]) > 1e-6
    assert abs(fn.coefficients[2]) > 1e-6

    x = np.linspace(-1, 1, 13)
    np.testing.assert_allclose(fn(x), x**2, atol=1e-10)
    assert isinstance(fn, Function)


def test_project_uses_space_quad_rule_by_default(space, quad_rule):
    default = space.project(lambda x: x**2)
    explicit = space.project(lambda x: x**2, quad_rule=quad_rule)
    np.testing.assert_allclose(default.coefficients, explicit.coefficients)


def test_project_accepts_quad_rule_override(space):
    # A coarser rule than space.quad_rule (but still enough points to keep
    # the mass matrix nonsingular and resolve the degree-8 mass-matrix
    # integrand for n_basis=5) still recovers x^2 exactly.
    coarse_rule = gauss_legendre(6)
    fn = space.project(lambda x: x**2, quad_rule=coarse_rule)
    x = np.linspace(-1, 1, 9)
    np.testing.assert_allclose(fn(x), x**2, atol=1e-10)


def test_project_rejects_quad_rule_with_too_few_points(space):
    # Fewer quadrature points than n_basis makes the mass matrix exactly
    # singular (phi is (npts, n_basis), rank <= npts) -- should raise
    # cleanly rather than hand back garbage from np.linalg.solve.
    too_coarse = gauss_legendre(3)
    with pytest.raises(ValueError):
        space.project(lambda x: x**2, quad_rule=too_coarse)


def test_project_on_non_reference_domain(quad_rule):
    space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5),
        domain=UnitInterval.Parameters(a=0.0, b=4.0),
        quad_rule=quad_rule,
    )

    def f(x):
        return x**2 - 3 * x + 1

    fn = space.project(f)
    x = np.linspace(0, 4, 17)
    np.testing.assert_allclose(fn(x), f(x), atol=1e-8)


def test_project_of_function_outside_basis_degree_is_approximate(quad_rule):
    # n_basis=2 (degree <= 1) can't exactly represent x^2; the L2 projection
    # should be a genuine (inexact) least-squares fit, not a crash.
    space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=2),
        domain=UnitInterval.Parameters(a=-1.0, b=1.0),
        quad_rule=quad_rule,
    )
    fn = space.project(lambda x: x**2)
    x = np.linspace(-1, 1, 9)
    residual = fn(x) - x**2
    assert np.max(np.abs(residual)) > 1e-3  # not exact
    assert np.max(np.abs(residual)) < 1.0  # but still a reasonable fit


# -- inner_product --


def test_inner_product_matches_mass_matrix_quadratic_form(space):
    c1 = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    c2 = np.array([0.2, 1.0, -1.0, 4.0, 0.1])
    M = space.mass_matrix()
    np.testing.assert_allclose(space.inner_product(c1, c2), c1 @ M @ c2, atol=1e-10)


def test_inner_product_of_orthonormal_basis_vectors_is_kronecker_delta(space):
    for i in range(space.n_basis):
        for j in range(space.n_basis):
            ci = np.eye(space.n_basis)[i]
            cj = np.eye(space.n_basis)[j]
            expected = 1.0 if i == j else 0.0
            np.testing.assert_allclose(
                space.inner_product(ci, cj), expected, atol=1e-10
            )


def test_inner_product_accepts_quad_rule_override(space):
    c1 = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    c2 = np.array([0.2, 1.0, -1.0, 4.0, 0.1])
    coarse = gauss_legendre(6)
    np.testing.assert_allclose(
        space.inner_product(c1, c2, quad_rule=coarse),
        space.inner_product(c1, c2),
        atol=1e-10,
    )


# -- quad_points / quad_weights / galerkin --


def test_quad_points_matches_scaled_points(space, quad_rule):
    expected = quad_rule.scaled_points(a=-1.0, b=1.0)
    np.testing.assert_allclose(space.quad_points(), expected)


def test_quad_points_accepts_quad_rule_override(space):
    coarse = gauss_legendre(6)
    np.testing.assert_allclose(
        space.quad_points(quad_rule=coarse), coarse.scaled_points(a=-1.0, b=1.0)
    )


def test_quad_weights_matches_scaled_weights(space, quad_rule):
    expected = quad_rule.scaled_weights(a=-1.0, b=1.0, density=space.basis.density)
    np.testing.assert_allclose(space.quad_weights(), expected)


def test_quad_weights_accepts_quad_rule_override(space):
    coarse = gauss_legendre(6)
    np.testing.assert_allclose(
        space.quad_weights(quad_rule=coarse),
        coarse.scaled_weights(a=-1.0, b=1.0, density=space.basis.density),
    )


def test_quad_points_and_weights_reproduce_galerkin(space):
    # quad_points/quad_weights are exactly what `galerkin` uses internally,
    # so assembling the same contraction by hand must match it exactly.
    def f(x):
        return x**2 - x

    x, w = space.quad_points(), space.quad_weights()
    phi = space.basis.evaluate(x, a=-1.0, b=1.0)
    by_hand = phi.T @ (w * f(x))
    np.testing.assert_allclose(by_hand, space.galerkin(f), atol=1e-12)


def test_galerkin_of_basis_itself_reproduces_mass_matrix(space):
    # mass_matrix is `galerkin` with residual_fn fixed to the (undifferentiated)
    # basis design matrix itself: R_j = int phi_i phi_j w dx = M_ij.
    def all_basis_functions(x):
        return space.basis.evaluate(x, a=-1.0, b=1.0)

    R = space.galerkin(all_basis_functions)
    np.testing.assert_allclose(R, space.mass_matrix(), atol=1e-10)


def test_galerkin_of_derivative_matches_stiffness_matrix(space):
    def all_basis_derivatives(x):
        return space.basis.evaluate(x, deriv=1, a=-1.0, b=1.0)

    R = space.galerkin(all_basis_derivatives, test_deriv=1)
    np.testing.assert_allclose(R, space.stiffness_matrix(), atol=1e-10)


def test_galerkin_of_plain_function_matches_project_rhs(space):
    # project's right-hand side is `galerkin(f)` solved against the mass
    # matrix; for an orthonormal basis M = I, so project(f).coefficients ==
    # galerkin(f).
    def f(x):
        return x**2

    fn = space.project(f)
    np.testing.assert_allclose(space.galerkin(f), fn.coefficients, atol=1e-10)


def test_galerkin_accepts_vector_valued_residual(space):
    def f(x):
        return np.stack([x, x**2], axis=-1)  # (npts, 2)

    R = space.galerkin(f)
    assert R.shape == (space.n_basis, 2)
    np.testing.assert_allclose(R[:, 0], space.galerkin(lambda x: x), atol=1e-12)
    np.testing.assert_allclose(R[:, 1], space.galerkin(lambda x: x**2), atol=1e-12)


def test_galerkin_accepts_quad_rule_override(space):
    coarse = gauss_legendre(6)
    np.testing.assert_allclose(
        space.galerkin(lambda x: x**2, quad_rule=coarse),
        space.galerkin(lambda x: x**2),
        atol=1e-10,
    )


# -- density=True: a Hermite space projects directly to PCE moments --


@pytest.fixture
def hermite_space():
    basis = OrthogonalPolynomialBasis(HermiteNormMeasure(), n_basis=4, density=True)
    return FunctionSpace(basis, domain=basis.Parameters(mean=0.0, std=2.0))


def test_density_mass_matrix_is_identity(hermite_space):
    # Same identity result as the (density=False) Legendre case above, but
    # now against the probability measure rather than the raw weight.
    np.testing.assert_allclose(
        hermite_space.mass_matrix(), np.eye(hermite_space.n_basis), atol=1e-8
    )


def test_density_project_gives_mean_and_variance_directly(hermite_space):
    # For X ~ N(0, std^2): E[X^2] = std^2, Var(X^2) = 2 * std^4. With
    # density=True, project's c_0 and sum(c[k>=1]^2) recover these directly
    # -- no rescaling by the measure's mass, unlike a density=False basis.
    std = hermite_space.domain.std
    fn = hermite_space.project(lambda x: x**2)
    np.testing.assert_allclose(fn.coefficients[0], std**2, atol=1e-6)
    np.testing.assert_allclose(np.sum(fn.coefficients[1:] ** 2), 2 * std**4, atol=1e-6)


def test_density_false_project_does_not_give_moments_directly(hermite_space):
    # Contrast case: the default (density=False) convention is orthonormal
    # against the *raw* weight, so c_0 is off from the true mean by a factor
    # of sqrt(mass) -- confirming the two conventions really do differ.
    raw_basis = OrthogonalPolynomialBasis(HermiteNormMeasure(), n_basis=4)
    raw_space = FunctionSpace(raw_basis, domain=hermite_space.domain)
    std = hermite_space.domain.std
    fn = raw_space.project(lambda x: x**2)
    assert abs(fn.coefficients[0] - std**2) > 1e-3
