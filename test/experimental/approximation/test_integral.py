"""Antiderivatives and definite integrals of Functions.

Integrating raises the degree, so the exact result lives in a *larger*
space -- the mirror of a derivative, which needs a smaller one (and the same
direction as a product). ``boundary`` pins the extra degree(s) of freedom
that introduces by requiring the antiderivative (and its lower derivatives,
for ``order > 1``) to vanish at one end of the domain, rather than leaving it
ambiguous the way a bare "constant of integration" would.

``integrate()`` is the scalar/definite counterpart -- the
``scipy.interpolate.PPoly`` ``antiderivative()``/``integrate(a, b)`` split --
built directly on ``integral()`` wherever that is defined, with a
whole-domain-only fallback where it isn't.
"""

import numpy as np
import pytest

import archimedes as arc
from archimedes.experimental.approximation import (
    Basis,
    CubicHermiteBasis,
    FourierBasis,
    Function,
    FunctionSpace,
    LagrangeBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
)
from archimedes.measure import (
    JacobiMeasure,
    LegendreMeasure,
    PhysicistsHermiteMeasure,
    UnitInterval,
)
from archimedes.quadrature import gauss_lobatto, quadint

A, B = 0.0, 2.0
DOMAIN = UnitInterval.Parameters(a=A, b=B)
BREAKS = np.linspace(-1.0, 1.0, 4)
X = np.linspace(0.07, 1.93, 15)


def _lobatto(n):
    return LagrangeBasis(reference_nodes=gauss_lobatto(n).nodes)


SPACE_BUILDERS = {
    "modal": lambda: FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), 6), domain=DOMAIN
    ),
    "jacobi": lambda: FunctionSpace(
        OrthogonalPolynomialBasis(JacobiMeasure(1.5, 0.5), 6), domain=DOMAIN
    ),
    "nodal": lambda: FunctionSpace(_lobatto(6), domain=DOMAIN),
    # Unlike PiecewiseBasis (see the dedicated "-- piecewise --" section
    # below), a B-spline's knot vector is one global object with no running
    # constant needed across elements, so `.integral()` is fully defined --
    # this is the concrete end-to-end check that BSplineBasis.Parameters'
    # redundant (a, b) echo (see BSplineBasis's docstring) correctly anchors
    # FunctionSpace._integral_matrix's boundary condition.
    "bspline": lambda: FunctionSpace.bspline(3, np.linspace(A, B, 4)),
}


@pytest.fixture(params=sorted(SPACE_BUILDERS))
def space(request):
    return SPACE_BUILDERS[request.param]()


def f_(x):
    return 2 * x**3 - x**2 + 5 * x - 1


def f_antideriv(x):
    # An antiderivative of f_, with no particular constant fixed.
    return x**4 / 2 - x**3 / 3 + 2.5 * x**2 - x


def f_left(x):
    return f_antideriv(x) - f_antideriv(A)


def f_right(x):
    return f_antideriv(x) - f_antideriv(B)


def g_(x):
    # A separate, simpler integrand: with A = 0 the order=2 closed form
    # (see test_order_two_matches_the_cauchy_repeated_integral) stays simple.
    return x**2


# -- exactness --


def test_integral_is_exact(space):
    antideriv = space.project(f_).integral()
    np.testing.assert_allclose(antideriv(X), f_left(X), atol=1e-10)


def test_integral_undoes_derivative_up_to_the_dropped_constant(space):
    # The FTC round trip: integrating the derivative back up recovers f_,
    # shifted so it vanishes at the left endpoint (which the derivative
    # itself has no memory of).
    u = space.project(f_)
    np.testing.assert_allclose(u.derivative().integral()(X), f_(X) - f_(A), atol=1e-9)


def test_derivative_undoes_integral():
    # The other direction of the round trip: differentiating the
    # antiderivative recovers the original function exactly.
    space = SPACE_BUILDERS["modal"]()
    u = space.project(f_)
    np.testing.assert_allclose(u.integral().derivative()(X), f_(X), atol=1e-9)


def test_repeated_integration_matches_a_single_call_at_higher_order():
    space = SPACE_BUILDERS["modal"]()
    u = space.project(g_)
    np.testing.assert_allclose(
        u.integral().integral()(X), u.integral(order=2)(X), atol=1e-9
    )


def test_order_two_matches_the_cauchy_repeated_integral():
    # With A = 0, the standard closed form for the twice-iterated integral
    # of x**2 vanishing (with its first derivative) at the origin is x**4/12.
    space = SPACE_BUILDERS["modal"]()
    antideriv2 = space.project(g_).integral(order=2)
    np.testing.assert_allclose(antideriv2(X), X**4 / 12, atol=1e-9)
    np.testing.assert_allclose(antideriv2(np.array([A]))[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(antideriv2(np.array([A]), deriv=1)[0], 0.0, atol=1e-10)


def test_order_zero_is_the_identity(space):
    u = space.project(f_)
    same = u.integral(order=0)
    assert same.space.n_basis == space.n_basis
    np.testing.assert_allclose(same(X), f_(X), atol=1e-10)


# -- boundary semantics --


def test_left_boundary_vanishes_at_a(space):
    antideriv = space.project(f_).integral(boundary="left")
    np.testing.assert_allclose(antideriv(np.array([A]))[0], 0.0, atol=1e-10)


def test_right_boundary_vanishes_at_b(space):
    antideriv = space.project(f_).integral(boundary="right")
    np.testing.assert_allclose(antideriv(np.array([B]))[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(antideriv(X), f_right(X), atol=1e-10)


def test_left_and_right_differ_by_the_whole_domain_integral(space):
    u = space.project(f_)
    total = u.integrate()
    left = u.integral(boundary="left")
    right = u.integral(boundary="right")
    np.testing.assert_allclose(right(X), left(X) - total, atol=1e-9)


def test_invalid_boundary_rejected(space):
    with pytest.raises(ValueError, match="boundary must be"):
        space.project(f_).integral(boundary="middle")


# -- the space is the maximal one (dual of derivative's minimal) --


def test_integral_space_is_larger(space):
    assert space._integral_space().n_basis == space.n_basis + 1
    assert space._integral_space(3).n_basis == space.n_basis + 3


def test_integral_matrix_shape(space):
    assert space._integral_matrix().shape == (space.n_basis + 1, space.n_basis)
    assert space._integral_matrix(order=2).shape == (space.n_basis + 2, space.n_basis)


def test_integral_matrix_order_zero_is_identity(space):
    np.testing.assert_allclose(space._integral_matrix(order=0), np.eye(space.n_basis))


def test_polynomial_families_keep_their_measure_and_normalization():
    for measure in (LegendreMeasure(), JacobiMeasure(1.5, 0.5)):
        basis = OrthogonalPolynomialBasis(measure, 5, density=True)
        derived = basis._integral_basis()
        assert derived.measure == measure
        assert derived.density is True
        assert derived.n_basis == 6


def test_integral_basis_construction_does_not_require_a_finite_domain():
    # Basis construction is purely about span/degree and doesn't know about
    # a target domain; the finite-endpoint requirement only bites once
    # `FunctionSpace._integral_matrix` needs somewhere to evaluate at.
    basis = OrthogonalPolynomialBasis(PhysicistsHermiteMeasure(), 5)
    derived = basis._integral_basis()
    assert derived.n_basis == 6


# -- explicit result space --


def test_explicit_result_space_is_honored(space):
    u = space.project(f_)
    target = space._integral_space()
    antideriv = u.integral(space=target)
    assert antideriv.space is target
    np.testing.assert_allclose(antideriv(X), f_left(X), atol=1e-10)


def test_wrong_size_explicit_space_is_rejected(space):
    u = space.project(f_)
    with pytest.raises(ValueError, match="expected"):
        u.integral(space=space)  # same size as self, not self.n_basis + 1


def test_wrong_size_explicit_space_is_rejected_at_order_zero(space):
    u = space.project(f_)
    with pytest.raises(ValueError, match="expected"):
        u.integral(order=0, space=space._integral_space())


# -- vector-valued --


def test_vector_valued_integral(space):
    def fv(x):
        return np.stack([x**2, 3 * x], axis=-1)

    antideriv = space.project(fv).integral()
    expected = np.stack([X**3 / 3 - A**3 / 3, 1.5 * X**2 - 1.5 * A**2], axis=-1)
    np.testing.assert_allclose(antideriv(X), expected, atol=1e-9)


def test_vector_valued_integrate(space):
    def fv(x):
        return np.stack([x**2, 3 * x], axis=-1)

    total = space.project(fv).integrate()
    expected = np.array([(B**3 - A**3) / 3, 1.5 * (B**2 - A**2)])
    np.testing.assert_allclose(total, expected, atol=1e-9)


# -- symbolic --


def test_integral_traces(space):
    u = space.project(f_)
    expected = u.integral()(X)

    @arc.compile
    def traced(c):
        return Function(c, space).integral()(X)

    np.testing.assert_allclose(
        np.asarray(traced(u.coefficients)).ravel(), expected, atol=1e-9
    )


# -- errors --


@pytest.mark.parametrize(
    "basis",
    [
        OrthogonalPolynomialBasis(LegendreMeasure(), 4),
        LagrangeBasis(reference_nodes=gauss_lobatto(4).nodes),
        FourierBasis(3, kind="sine"),
    ],
    ids=["modal", "nodal", "fourier"],
)
def test_negative_integral_order_rejected(basis):
    with pytest.raises(ValueError, match="order must be >= 0"):
        basis._integral_basis(-1)


def test_integral_matrix_rejects_negative_order(space):
    with pytest.raises(ValueError, match="order must be >= 0"):
        space._integral_matrix(order=-1)


def test_zeroth_integral_returns_the_same_basis():
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), 4)
    assert basis._integral_basis(0) is basis
    lagrange = _lobatto(4)
    assert lagrange._integral_basis(0) is lagrange
    fourier = FourierBasis(3, kind="sine")
    assert fourier._integral_basis(0) is fourier


def test_basis_without_integral_support_raises():
    class Constant(Basis):
        n_basis = 1

        @property
        def Parameters(self):  # noqa: N802
            return UnitInterval.Parameters

        def default_quadrature(self):
            return gauss_lobatto(2)

        def evaluate(self, x, deriv=0, a=None, b=None):
            column = np.ones_like(x) if deriv == 0 else np.zeros_like(x)
            return column[:, None]

    with pytest.raises(NotImplementedError, match="does not define an integral basis"):
        Constant()._integral_basis()


def test_hermite_has_no_integral_basis():
    # CubicHermiteBasis doesn't override `_integral_basis`, so this is the
    # base class's NotImplementedError, propagated unchanged through
    # PiecewiseBasis's own (differently-worded) override.
    hermite = CubicHermiteBasis()
    with pytest.raises(NotImplementedError, match="does not define an integral basis"):
        hermite._integral_basis()


def test_piecewise_has_no_integral_basis():
    basis = PiecewiseBasis(_lobatto(4), BREAKS, continuity=0)
    with pytest.raises(
        NotImplementedError, match="running constant carried across elements"
    ):
        basis._integral_basis()


def test_piecewise_function_integral_raises():
    space = FunctionSpace(
        PiecewiseBasis(_lobatto(4), BREAKS, continuity=0), domain=DOMAIN
    )
    with pytest.raises(
        NotImplementedError, match="running constant carried across elements"
    ):
        space.project(f_).integral()


# -- FourierBasis: only "sine" has an integral basis, and only at order=1 --
# Fourier's own basis functions aren't polynomials, so this doesn't fit the
# generic `space`/`f_`/`g_` fixture above -- a dedicated section, same as
# the Hermite/piecewise-specific tests above it.

FOURIER_DOMAIN = UnitInterval.Parameters(a=-1.0, b=1.0)
XF = np.linspace(-0.93, 0.93, 15)


def test_fourier_full_and_cosine_integral_basis_raises():
    # Both contain the constant/DC basis function, whose antiderivative is a
    # non-periodic linear ramp -- a different reason than Hermite/piecewise
    # raise for theirs.
    for basis in (FourierBasis(5, kind="full"), FourierBasis(4, kind="cosine")):
        with pytest.raises(NotImplementedError, match="linear ramp"):
            basis._integral_basis()


def test_fourier_sine_integral_basis_grows_by_one():
    basis = FourierBasis(3, kind="sine")
    grown = basis._integral_basis(1)
    assert grown.kind == "cosine"
    assert grown.n_basis == 4


def test_fourier_sine_integral_order_two_raises():
    # The order-1 result is "cosine", which can't itself be integrated.
    basis = FourierBasis(3, kind="sine")
    with pytest.raises(NotImplementedError, match="only order=1 is supported"):
        basis._integral_basis(2)


def test_fourier_sine_integral_dc_coefficient_is_generically_nonzero():
    # Documents *why* the integral basis must grow: pinning the antiderivative
    # to vanish at a boundary forces a nonzero constant term back in, since
    # cos(k*theta(boundary)) = (-1)**k != 0 for every mode.
    space = FunctionSpace(FourierBasis(3, kind="sine"), domain=FOURIER_DOMAIN)
    u = space.project(lambda x: np.sin(np.pi * x))
    antideriv = u.integral()
    assert abs(antideriv.coefficients[0]) > 1e-6


@pytest.mark.parametrize("boundary", ["left", "right"])
def test_fourier_sine_integral_matches_analytic_antiderivative(boundary):
    # F(x) = -cos(pi*x)/pi + C. Since cos(theta(a)) == cos(theta(b)) always
    # (cos is even and theta(a) = -pi, theta(b) = pi), both boundary choices
    # pin the same C here -- both are checked to confirm the kwarg is honored
    # at its own endpoint, not because the results are expected to differ.
    space = FunctionSpace(FourierBasis(3, kind="sine"), domain=FOURIER_DOMAIN)
    u = space.project(lambda x: np.sin(np.pi * x))
    antideriv = u.integral(boundary=boundary)
    assert antideriv.space.basis.kind == "cosine"

    def expected(x):
        return -np.cos(np.pi * x) / np.pi - 1.0 / np.pi

    np.testing.assert_allclose(antideriv(XF), expected(XF), atol=1e-9)
    endpoint = np.array([-1.0 if boundary == "left" else 1.0])
    np.testing.assert_allclose(antideriv(endpoint)[0], 0.0, atol=1e-10)


def test_fourier_sine_integral_traces():
    space = FunctionSpace(FourierBasis(3, kind="sine"), domain=FOURIER_DOMAIN)
    u = space.project(lambda x: np.sin(np.pi * x))
    expected = u.integral()(XF)

    @arc.compile
    def traced(c):
        return Function(c, space).integral()(XF)

    np.testing.assert_allclose(
        np.asarray(traced(u.coefficients)).ravel(), expected, atol=1e-9
    )


def test_unbounded_domain_rejected_for_integral():
    space = FunctionSpace.hermite(6)
    u = space.project(lambda x: np.exp(-(x**2)))
    with pytest.raises(ValueError, match="finite endpoints"):
        u.integral()


def test_unbounded_domain_rejected_for_laguerre_too():
    space = FunctionSpace.laguerre(6)
    u = space.project(lambda x: np.exp(-x))
    with pytest.raises(ValueError, match="finite endpoints"):
        u.integral()


def test_order_zero_is_exempt_from_the_finite_domain_requirement():
    # order=0 needs no boundary condition at all, so it should work even on
    # a space with no finite endpoint to anchor at.
    space = FunctionSpace.hermite(6)
    u = space.project(lambda x: np.exp(-(x**2)))
    np.testing.assert_allclose(u.integral(order=0)(X[:1]), u(X[:1]), atol=1e-10)


# -- integrate() --


def test_integrate_matches_the_whole_domain_analytic_value(space):
    u = space.project(f_)
    np.testing.assert_allclose(
        u.integrate(), f_antideriv(B) - f_antideriv(A), atol=1e-9
    )


def test_integrate_matches_integral_at_the_right_endpoint(space):
    # Since the default boundary is "left", antideriv(A) == 0 exactly, so the
    # whole-domain integral is just antideriv(B).
    u = space.project(f_)
    antideriv = u.integral()
    np.testing.assert_allclose(u.integrate(), antideriv(np.array([B]))[0], atol=1e-10)


def test_integrate_supports_an_arbitrary_sub_interval(space):
    u = space.project(f_)
    lo, hi = 0.3, 1.7
    np.testing.assert_allclose(
        u.integrate(lo, hi), f_antideriv(hi) - f_antideriv(lo), atol=1e-9
    )
    np.testing.assert_allclose(u.integrate(lo, hi), quadint(u, lo, hi), atol=1e-8)


def test_integrate_is_unweighted_even_for_a_measure_weighted_family():
    # Jacobi/Chebyshev's own quadrature is calibrated to their orthogonality
    # weight, not the plain Lebesgue measure -- `integrate()` must not leak
    # that in, since "the integral of this function" should mean the
    # ordinary calculus integral regardless of which family represents it.
    space = FunctionSpace.chebyshev(8, a=-1.0, b=1.0)
    u = space.project(lambda x: x**2)
    np.testing.assert_allclose(u.integrate(), 2.0 / 3.0, atol=1e-9)


def test_piecewise_whole_domain_integrate_works():
    space = FunctionSpace(
        PiecewiseBasis(_lobatto(4), BREAKS, continuity=0), domain=DOMAIN
    )
    u = space.project(f_)
    np.testing.assert_allclose(
        u.integrate(), f_antideriv(B) - f_antideriv(A), atol=1e-9
    )


def test_piecewise_sub_interval_integrate_raises():
    space = FunctionSpace(
        PiecewiseBasis(_lobatto(4), BREAKS, continuity=0), domain=DOMAIN
    )
    u = space.project(f_)
    with pytest.raises(
        NotImplementedError, match="running constant carried across elements"
    ):
        u.integrate(0.3, 1.7)


def test_unbounded_domain_rejected_for_integrate_even_whole_domain():
    # `.integral()` raises ValueError here, not NotImplementedError, so
    # `integrate()`'s fallback (which only catches NotImplementedError)
    # does not swallow it -- there is no well-defined fallback anyway, since
    # an unweighted integral over an unbounded domain need not converge.
    space = FunctionSpace.hermite(6)
    u = space.project(lambda x: np.exp(-(x**2)))
    with pytest.raises(ValueError, match="finite endpoints"):
        u.integrate()
