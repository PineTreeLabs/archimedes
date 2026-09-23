# ruff: noqa: N806  (M, K are the conventional names for these matrices)
import numpy as np
import pytest
from _helpers import mass_matrix, stiffness_matrix

from archimedes.approximation import (
    Function,
    FunctionSpace,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
)
from archimedes.measure import (
    LegendreMeasure,
    ProbabilistsHermiteMeasure,
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
        reference_quad_rule=quad_rule,
    )


def test_n_basis_forwarded(space):
    assert space.n_basis == 5


def test_domain_type_validation(quad_rule):
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5)
    with pytest.raises(TypeError):
        FunctionSpace(basis, domain=(-1.0, 1.0), reference_quad_rule=quad_rule)
    with pytest.raises(TypeError):
        # Wrong domain type -- RealLine's Parameters takes loc/scale, not
        # a/b, so it isn't a UnitInterval.Parameters.
        FunctionSpace(
            basis, domain=RealLine.Parameters(), reference_quad_rule=quad_rule
        )


def test_evaluate(space):
    coefficients = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    x = np.linspace(-1, 1, 9)
    phi = space.basis.evaluate(x, a=-1.0, b=1.0)  # (npts, n_basis)
    expected = phi @ coefficients
    np.testing.assert_allclose(space._evaluate(coefficients, x), expected)


def test_mass_matrix(space, quad_rule):
    # OrthogonalPolynomialBasis is orthonormal by construction, so the mass
    # matrix should be the identity (not merely diagonal).
    M = mass_matrix(space)
    np.testing.assert_allclose(M, np.eye(space.n_basis), atol=1e-10)

    # The basis renormalizes to stay orthonormal w.r.t. the *target* domain's
    # measure (not just the reference one), so this holds for any domain --
    # unlike a fixed classical normalization, it isn't Jacobian-scaled.
    wide_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=4),
        domain=UnitInterval.Parameters(a=0.0, b=4.0),
        reference_quad_rule=quad_rule,
    )
    M = mass_matrix(wide_space)
    np.testing.assert_allclose(M, np.eye(4), atol=1e-10)


def test_stiffness_matrix(space):
    K = stiffness_matrix(space)
    np.testing.assert_allclose(K, K.T, atol=1e-10)


# -- project() --


def test_project_exact_polynomial(space):
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


def test_project_quad_rule(space, quad_rule):
    default = space.project(lambda x: x**2)
    explicit = space.project(lambda x: x**2, quad_rule=quad_rule)
    np.testing.assert_allclose(default.coefficients, explicit.coefficients)

    # A coarser rule than space.quad_rule (but still enough points to keep
    # the mass matrix nonsingular and resolve the degree-8 mass-matrix
    # integrand for n_basis=5) still recovers x^2 exactly.
    coarse_rule = gauss_legendre(6)
    fn = space.project(lambda x: x**2, quad_rule=coarse_rule)
    x = np.linspace(-1, 1, 9)
    np.testing.assert_allclose(fn(x), x**2, atol=1e-10)


# -- function() --


def test_function(space):
    coefficients = np.arange(space.n_basis, dtype=float)
    fn = space.function(coefficients)
    assert isinstance(fn, Function)
    assert fn.space is space
    np.testing.assert_array_equal(fn.coefficients, coefficients)

    # Defaults to zero coefficients.
    fn = space.function()
    assert isinstance(fn, Function)
    assert fn.space is space
    np.testing.assert_array_equal(fn.coefficients, np.zeros(space.n_basis))


def test_project_quad_rule_too_few_points(space):
    # Fewer quadrature points than n_basis makes the mass matrix exactly
    # singular (phi is (npts, n_basis), rank <= npts) -- should raise
    # cleanly rather than hand back garbage from np.linalg.solve.
    too_coarse = gauss_legendre(3)
    with pytest.raises(ValueError):
        space.project(lambda x: x**2, quad_rule=too_coarse)


def test_project_non_reference_domain(quad_rule):
    space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5),
        domain=UnitInterval.Parameters(a=0.0, b=4.0),
        reference_quad_rule=quad_rule,
    )

    def f(x):
        return x**2 - 3 * x + 1

    fn = space.project(f)
    x = np.linspace(0, 4, 17)
    np.testing.assert_allclose(fn(x), f(x), atol=1e-8)


def test_diff_matrix_uses_mapped_rule_not_reference(quad_rule):
    # `_diff_matrix`/`_integral_matrix` must evaluate both bases at the
    # mapped (physical) quadrature nodes given by the `quad_rule` property,
    # not the reference-domain nodes. Check this on a domain that differs
    # visibly from the reference one.
    wide_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5),
        domain=UnitInterval.Parameters(a=0.0, b=4.0),
        reference_quad_rule=quad_rule,
    )
    fn = wide_space.project(lambda x: x**2)
    deriv = fn.derivative()
    x = np.linspace(0, 4, 13)
    np.testing.assert_allclose(deriv(x), 2 * x, atol=1e-8)


def test_quad_rule_property_is_mapped_not_reference(quad_rule):
    # `reference_quad_rule` stays on the reference domain; the public
    # `quad_rule` property is what a caller should read directly.
    wide_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5),
        domain=UnitInterval.Parameters(a=0.0, b=4.0),
        reference_quad_rule=quad_rule,
    )
    assert not np.allclose(
        wide_space.reference_quad_rule.nodes, wide_space.quad_rule.nodes
    )
    np.testing.assert_allclose(
        wide_space.quad_rule.nodes, quad_rule.map_to(a=0.0, b=4.0).nodes
    )
    assert np.all(wide_space.quad_rule.nodes >= 0.0)
    assert np.all(wide_space.quad_rule.nodes <= 4.0)


def test_quad_rule_shared_across_spaces_lands_on_physical_nodes(quad_rule):
    # A coupled (mixed-field) assembly evaluates two different
    # FunctionSpaces at one shared set of physical nodes, by passing one
    # space's `quad_rule` as an explicit override to another's
    # `basis_matrix()`. Check both fields are assembled at the shared
    # physical nodes, not the reference-domain ones, when domain != [-1, 1].
    domain = UnitInterval.Parameters(a=2.0, b=6.0)
    space_a = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5),
        domain=domain,
        reference_quad_rule=quad_rule,
    )
    space_b = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=3),
        domain=domain,
        reference_quad_rule=quad_rule,
    )

    shared_quad = space_a.quad_rule
    Phi_a = space_a.basis_matrix(quad_rule=shared_quad)
    Phi_b = space_b.basis_matrix(quad_rule=shared_quad)

    np.testing.assert_allclose(Phi_a.nodes, Phi_b.nodes)
    assert np.all(Phi_a.nodes >= 2.0)
    assert np.all(Phi_a.nodes <= 6.0)


def test_project_test_space_validation(space, quad_rule):
    test_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=3),
        domain=UnitInterval.Parameters(a=-1.0, b=1.0),
        reference_quad_rule=quad_rule,
    )
    with pytest.raises(ValueError, match="n_basis"):
        space.project(lambda x: x**2, test_space=test_space)

    test_space = FunctionSpace(
        OrthogonalPolynomialBasis(ProbabilistsHermiteMeasure(), n_basis=5),
        domain=RealLine.Parameters(),
    )
    with pytest.raises(ValueError, match="domain"):
        space.project(lambda x: x**2, test_space=test_space)


def test_project_petrov_galerkin(space):
    # A genuinely different test space -- 5 discontinuous piecewise-constant
    # "bumps", not another basis for the same degree-4 polynomial span --
    # still recovers x^2 exactly: the true expansion's residual is
    # identically zero, so it satisfies *any* set of orthogonality
    # conditions, as long as the resulting (square) system is nonsingular.
    test_space = FunctionSpace(
        PiecewiseBasis(
            OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=1),
            breakpoints=np.linspace(-1, 1, 6),
            continuity=-1,
        ),
        domain=UnitInterval.Parameters(a=-1.0, b=1.0),
    )
    assert test_space.n_basis == space.n_basis

    fn = space.project(lambda x: x**2, test_space=test_space)
    assert fn.space is space

    x = np.linspace(-1, 1, 13)
    np.testing.assert_allclose(fn(x), x**2, atol=1e-10)


def test_project_outside_basis_degree(quad_rule):
    # n_basis=2 (degree <= 1) can't exactly represent x^2; the L2 projection
    # should be a genuine (inexact) least-squares fit, not a crash.
    space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=2),
        domain=UnitInterval.Parameters(a=-1.0, b=1.0),
        reference_quad_rule=quad_rule,
    )
    fn = space.project(lambda x: x**2)
    x = np.linspace(-1, 1, 9)
    residual = fn(x) - x**2
    assert np.max(np.abs(residual)) > 1e-3  # not exact
    assert np.max(np.abs(residual)) < 1.0  # but still a reasonable fit


# -- inner_product --


def test_inner_product(space):
    c1 = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    c2 = np.array([0.2, 1.0, -1.0, 4.0, 0.1])
    M = mass_matrix(space)
    np.testing.assert_allclose(space._inner_product(c1, c2), c1 @ M @ c2, atol=1e-10)

    coarse = gauss_legendre(6)
    np.testing.assert_allclose(
        space._inner_product(c1, c2, quad_rule=coarse),
        space._inner_product(c1, c2),
        atol=1e-10,
    )

    # Orthonormal basis vectors give the Kronecker delta.
    for i in range(space.n_basis):
        for j in range(space.n_basis):
            ci = np.eye(space.n_basis)[i]
            cj = np.eye(space.n_basis)[j]
            expected = 1.0 if i == j else 0.0
            np.testing.assert_allclose(
                space._inner_product(ci, cj), expected, atol=1e-10
            )


# -- quadrature / basis_matrix / BasisMatrix.T --


def test_quadrature(space, quad_rule):
    x, w = space.quadrature()
    mapped = quad_rule.map_to(a=-1.0, b=1.0)
    np.testing.assert_allclose(x, mapped.nodes)
    expected_w = mapped.weights
    if space.basis.density:
        expected_w = expected_w / np.sum(expected_w)
    np.testing.assert_allclose(w, expected_w)

    # space's domain happens to be [-1, 1], the reference domain itself. So
    # an unmapped override coincides numerically with one mapped onto it,
    # which is not true in general on a non-reference domain.
    coarse = gauss_legendre(6)
    x, w = space.quadrature(quad_rule=coarse)
    np.testing.assert_allclose(x, coarse.nodes)
    expected_w = coarse.weights
    if space.basis.density:
        expected_w = expected_w / np.sum(expected_w)
    np.testing.assert_allclose(w, expected_w)


def test_quadrature_non_reference_domain(quad_rule):
    # An explicit override is used exactly as given, with no further domain
    # mapping applied -- so on a genuinely non-reference domain, an
    # unmapped override rule gives *reference*-domain nodes back, not nodes
    # mapped onto the space's domain.
    wide_space = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=4),
        domain=UnitInterval.Parameters(a=0.0, b=4.0),
        reference_quad_rule=quad_rule,
    )
    coarse = gauss_legendre(6)
    x, w = wide_space.quadrature(quad_rule=coarse)
    np.testing.assert_allclose(x, coarse.nodes)
    np.testing.assert_allclose(w, coarse.weights)

    # The correct way to use an override on a non-identity domain: map it
    # yourself first.
    pre_mapped = quad_rule.map_to(a=0.0, b=4.0)
    x, w = wide_space.quadrature(quad_rule=pre_mapped)
    x_default, w_default = wide_space.quadrature()
    np.testing.assert_allclose(x, x_default)
    np.testing.assert_allclose(w, w_default)


def test_basis_matrix(space):
    x, w = space.quadrature()
    expected = space.basis.evaluate(x, a=-1.0, b=1.0)
    phi = space.basis_matrix()
    np.testing.assert_allclose(phi.matrix, expected)
    np.testing.assert_allclose(phi.weights, w)
    assert phi.shape == expected.shape

    coarse = gauss_legendre(6)
    x, _ = space.quadrature(quad_rule=coarse)
    expected = space.basis.evaluate(x, a=-1.0, b=1.0)
    np.testing.assert_allclose(space.basis_matrix(quad_rule=coarse).matrix, expected)


def test_basis_matrix_of_derivative(space):
    x, _ = space.quadrature()
    expected = space.basis.evaluate(x, deriv=1, a=-1.0, b=1.0)
    np.testing.assert_allclose(space.basis_matrix(deriv=1).matrix, expected)


def test_basis_matrix_matmul(space):
    c = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    phi = space.basis_matrix()
    np.testing.assert_allclose(phi @ c, phi.matrix @ c)


def test_basis_matrix_adjoint(space):
    # project's right-hand side is `phi.T @ f(x)` solved against the mass
    # matrix; for an orthonormal basis M = I, so project(f).coefficients ==
    # phi.T @ f(x).
    def f(x):
        return x**2

    fn = space.project(f)
    x, _ = space.quadrature()
    phi = space.basis_matrix()
    np.testing.assert_allclose(phi.T @ f(x), fn.coefficients, atol=1e-10)

    # Also accepts a vector-valued integrand.
    def fv(x):
        return np.stack([x, x**2], axis=-1)  # (npts, 2)

    R = phi.T @ fv(x)
    assert R.shape == (space.n_basis, 2)
    np.testing.assert_allclose(R[:, 0], phi.T @ x, atol=1e-12)
    np.testing.assert_allclose(R[:, 1], phi.T @ x**2, atol=1e-12)


def test_basis_matrix_adjoint_round_trip():
    space_local = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5),
        domain=UnitInterval.Parameters(a=-1.0, b=1.0),
    )
    phi = space_local.basis_matrix()
    assert phi.T.T is phi


def test_basis_matrix_petrov_galerkin(space, quad_rule):
    # The adjoint doesn't require the "trial side" to have anything to do
    # with phi's own column count -- a differently-sized test basis on the
    # same quadrature nodes (a stand-in for a genuinely different test
    # space) contracts fine.
    other = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=8),
        domain=UnitInterval.Parameters(a=-1.0, b=1.0),
        reference_quad_rule=quad_rule,
    )
    test_phi = other.basis_matrix()
    x, _ = space.quadrature()
    R = test_phi.T @ (x**2)
    assert R.shape == (8,)


# -- density=True: a Hermite space projects directly to PCE moments --


@pytest.fixture
def hermite_space():
    basis = OrthogonalPolynomialBasis(
        ProbabilistsHermiteMeasure(), n_basis=4, density=True
    )
    return FunctionSpace(basis, domain=basis.Parameters(loc=0.0, scale=2.0))


def test_density_mass_matrix(hermite_space):
    # Same identity result as the (density=False) Legendre case above, but
    # now against the probability measure rather than the raw weight.
    np.testing.assert_allclose(
        mass_matrix(hermite_space), np.eye(hermite_space.n_basis), atol=1e-8
    )


def test_density_project(hermite_space):
    # For X ~ N(0, scale^2): E[X^2] = scale^2, Var(X^2) = 2 * scale^4. With
    # density=True, project's c_0 and sum(c[k>=1]^2) recover these directly
    # -- no rescaling by the measure's mass, unlike a density=False basis.
    scale = hermite_space.domain.scale
    fn = hermite_space.project(lambda x: x**2)
    np.testing.assert_allclose(fn.coefficients[0], scale**2, atol=1e-6)
    np.testing.assert_allclose(
        np.sum(fn.coefficients[1:] ** 2), 2 * scale**4, atol=1e-6
    )


def test_density_false_project(hermite_space):
    # Contrast case: the default (density=False) convention is orthonormal
    # against the *raw* weight, so c_0 is off from the true mean by a factor
    # of sqrt(mass) -- confirming the two conventions really do differ.
    raw_basis = OrthogonalPolynomialBasis(ProbabilistsHermiteMeasure(), n_basis=4)
    raw_space = FunctionSpace(raw_basis, domain=hermite_space.domain)
    scale = hermite_space.domain.scale
    fn = raw_space.project(lambda x: x**2)
    assert abs(fn.coefficients[0] - scale**2) > 1e-3
