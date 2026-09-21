"""Tests for ``BSplineBasis``: general (non-uniform, arbitrary-multiplicity,
clamped or open) B-spline evaluation via de Boor's BSPLVB recurrence.
"""

import numpy as np
import pytest
from scipy.interpolate import BSpline as ScipyBSpline

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.approximation import (
    BSplineBasis,
    Function,
    FunctionSpace,
    ProductParameters,
    TensorBasis,
)
from archimedes.measure import UnitInterval
from archimedes.quadrature import composite_quad, gauss_legendre

CLAMPED_UNIFORM = (3, np.array([0.0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]))
CLAMPED_NONUNIFORM = (3, np.array([0.0, 0, 0, 0, 0.5, 2.5, 3.0, 4, 4, 4, 4]))
OPEN_QUADRATIC = (2, np.array([0.0, 0, 1, 2, 3, 4, 5, 6]))
DEGREE_ZERO = (0, np.array([0.0, 1, 2, 3]))
INTERIOR_DOUBLE_KNOT = (3, np.array([0.0, 0, 0, 0, 1, 1, 3, 4, 4, 4, 4]))
INTERIOR_MAX_MULT = (3, np.array([0.0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4]))

CASES = [
    CLAMPED_UNIFORM,
    CLAMPED_NONUNIFORM,
    OPEN_QUADRATIC,
    DEGREE_ZERO,
    INTERIOR_DOUBLE_KNOT,
    INTERIOR_MAX_MULT,
]
CASE_IDS = [
    "clamped_uniform",
    "clamped_nonuniform",
    "open_quadratic",
    "degree_zero",
    "interior_double_knot",
    "interior_max_mult",
]


@pytest.fixture(params=CASES, ids=CASE_IDS)
def case(request):
    degree, knots = request.param
    return degree, knots


@pytest.fixture
def basis(case):
    degree, knots = case
    return BSplineBasis(degree, knots)


# A two-case witness pair for tests whose branch coverage doesn't depend on
# knot layout: a clamped/nonuniform case and a repeated-interior-knot case.
# Together they exercise the two edge-sensitive code paths in
# `_local_values`: clamped-boundary indexing, and the multiplicity-stepping
# loop for a repeated interior knot.
WITNESS_CASES = [CLAMPED_NONUNIFORM, INTERIOR_MAX_MULT]
WITNESS_CASE_IDS = ["clamped_nonuniform", "interior_max_mult"]


@pytest.fixture(params=WITNESS_CASES, ids=WITNESS_CASE_IDS)
def witness_case(request):
    degree, knots = request.param
    return degree, knots


@pytest.fixture
def witness_basis(witness_case):
    degree, knots = witness_case
    return BSplineBasis(degree, knots)


def _sample_points(basis, extrapolate=True):
    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    x = np.linspace(a, b, 37)
    if extrapolate:
        span = b - a
        x = np.concatenate([[a - 0.3 * span], x, [b + 0.3 * span]])
    return x


# -- construction validation --


def test_construction_validation():
    with pytest.raises(ValueError, match="degree must be"):
        BSplineBasis(-1, [0.0, 0, 1, 1])

    with pytest.raises(ValueError, match="1-D"):
        BSplineBasis(1, [[0.0, 0], [1, 1]])

    with pytest.raises(ValueError, match="at least 2\\*degree\\+2"):
        BSplineBasis(3, [0.0, 0, 0, 0, 1])

    with pytest.raises(ValueError, match="nondecreasing"):
        BSplineBasis(1, [0.0, 0, 1, 0.5, 2, 2])

    with pytest.raises(ValueError, match="repeat at most"):
        BSplineBasis(1, [0.0, 0, 0, 1, 2])

    with pytest.raises(ValueError, match="degenerate"):
        BSplineBasis(1, [-5.0, 0, 0, 5])


def test_n_basis():
    degree, knots = CLAMPED_UNIFORM
    basis = BSplineBasis(degree, knots)
    assert basis.n_basis == len(knots) - degree - 1 == 7


def test_equality_and_hash():
    a = BSplineBasis(3, [0.0, 0, 0, 0, 1, 1, 1, 1])
    b = BSplineBasis(3, [0.0, 0, 0, 0, 1, 1, 1, 1])
    c = BSplineBasis(2, [0.0, 0, 0, 1, 1, 1])
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != "not a basis"


def test_negative_deriv_rejected(witness_basis):
    # Raises before touching knots/degree, so this doesn't depend on knot
    # layout -- the witness pair is enough.
    basis = witness_basis
    x = np.array([basis.knots[basis.degree]])
    with pytest.raises(ValueError, match="deriv must be >= 0"):
        basis.evaluate(x, deriv=-1)
    with pytest.raises(ValueError, match="deriv must be >= 0"):
        basis.evaluate_expansion(np.zeros(basis.n_basis), x, deriv=-1)


def test_invalid_side_rejected(witness_basis):
    # Validated before any knot-dependent logic runs, so this doesn't
    # depend on knot layout -- the witness pair is enough.
    basis = witness_basis
    x = np.array([basis.knots[basis.degree]])
    with pytest.raises(ValueError, match="side must be"):
        basis.evaluate(x, side="up")
    with pytest.raises(ValueError, match="side must be"):
        basis.evaluate_expansion(np.zeros(basis.n_basis), x, side="up")


# -- values against the scipy oracle --


def test_values_match_scipy(basis):
    x = _sample_points(basis)
    phi = basis.evaluate(x)
    assert phi.shape == (len(x), basis.n_basis)
    for i in range(basis.n_basis):
        c = np.zeros(basis.n_basis)
        c[i] = 1.0
        expected = ScipyBSpline(basis.knots, c, basis.degree, extrapolate=True)(x)
        np.testing.assert_allclose(phi[:, i], expected, atol=1e-9)


def test_first_derivative_matches_scipy(basis):
    # Full 6-case breadth: this is the core oracle check, and the
    # knot-difference derivative formula is genuinely case-sensitive.
    deriv = 1
    x = _sample_points(basis)
    phi = basis.evaluate(x, deriv=deriv)
    for i in range(basis.n_basis):
        c = np.zeros(basis.n_basis)
        c[i] = 1.0
        sp = ScipyBSpline(basis.knots, c, basis.degree, extrapolate=True)
        np.testing.assert_allclose(phi[:, i], sp(x, nu=deriv), atol=1e-6)


@pytest.mark.parametrize("deriv", [2, 3])
def test_higher_derivatives_match_scipy(witness_basis, deriv):
    # The witness pair (both degree 3) is enough for deriv=2,3; the
    # deriv=1 case gets the full 6-case breadth check separately.
    basis = witness_basis
    x = _sample_points(basis)
    phi = basis.evaluate(x, deriv=deriv)
    for i in range(basis.n_basis):
        c = np.zeros(basis.n_basis)
        c[i] = 1.0
        sp = ScipyBSpline(basis.knots, c, basis.degree, extrapolate=True)
        np.testing.assert_allclose(phi[:, i], sp(x, nu=deriv), atol=1e-6)


def test_deriv_past_degree(witness_basis):
    # Case-invariant early-return branch -- only degree matters, and the
    # witness pair covers degree 3.
    basis = witness_basis
    x = _sample_points(basis)
    phi = basis.evaluate(x, deriv=basis.degree + 1)
    np.testing.assert_allclose(phi, 0.0, atol=1e-12)
    phi2 = basis.evaluate(x, deriv=basis.degree + 4)
    np.testing.assert_allclose(phi2, 0.0, atol=1e-12)


def test_derivative_matches_finite_difference(witness_basis):
    # An independent finite-difference oracle for the first derivative.
    # This doesn't depend on knot layout, so the witness pair (both
    # degree 3, where a first derivative always exists) is enough.
    basis = witness_basis
    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    x = np.linspace(a + 0.05 * (b - a), b - 0.05 * (b - a), 25)
    h = 1e-6
    dphi = basis.evaluate(x, deriv=1)
    lo = basis.evaluate(x - h)
    hi = basis.evaluate(x + h)
    np.testing.assert_allclose(dphi, (hi - lo) / (2 * h), atol=1e-4)


# -- structural properties --


def test_partition_of_unity(witness_basis):
    # A generic recurrence property, independent of knot layout.
    basis = witness_basis
    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    x = np.linspace(a + 1e-9, b - 1e-9, 101)
    phi = basis.evaluate(x)
    np.testing.assert_allclose(phi.sum(axis=1), 1.0, atol=1e-8)


def test_local_support(witness_basis):
    # A generic support-window property, independent of knot layout.
    basis = witness_basis
    knots = basis.knots
    p = basis.degree
    a, b = knots[p], knots[-1 - p]
    x = np.linspace(a, b, 201)
    phi = basis.evaluate(x)
    for i in range(basis.n_basis):
        lo, hi = knots[i], knots[i + p + 1]
        outside = (x < lo) | (x > hi)
        np.testing.assert_allclose(phi[outside, i], 0.0, atol=1e-10)


def test_side_away_from_knots(witness_basis):
    # Away from knots, the multiplicity-stepping loop never fires
    # meaningfully -- the genuinely side-sensitive cases have their own
    # dedicated tests below.
    basis = witness_basis
    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    x = np.linspace(a + 0.01, b - 0.01, 23)
    # Avoid landing exactly on an interior knot by construction of the grid.
    np.testing.assert_allclose(
        basis.evaluate(x, side="left"), basis.evaluate(x, side="right"), atol=1e-10
    )


def test_side_at_discontinuous_knot():
    degree, knots = INTERIOR_MAX_MULT
    basis = BSplineBasis(degree, knots)
    xk = np.array([1.0])  # the multiplicity-(degree+1) interior knot
    left = basis.evaluate(xk, side="left")
    right = basis.evaluate(xk, side="right")
    assert not np.allclose(left, right)

    eps = 1e-8
    num_left = np.stack(
        [
            ScipyBSpline(knots, np.eye(basis.n_basis)[i], degree, extrapolate=True)(
                1.0 - eps
            )
            for i in range(basis.n_basis)
        ]
    )
    num_right = np.stack(
        [
            ScipyBSpline(knots, np.eye(basis.n_basis)[i], degree, extrapolate=True)(
                1.0 + eps
            )
            for i in range(basis.n_basis)
        ]
    )
    np.testing.assert_allclose(left.ravel(), num_left, atol=1e-4)
    np.testing.assert_allclose(right.ravel(), num_right, atol=1e-4)


def test_side_at_c1_knot():
    # Multiplicity degree - 1 (< degree + 1) still leaves the basis itself
    # (deriv=0) continuous, unlike the maximal-multiplicity case above.
    degree, knots = INTERIOR_DOUBLE_KNOT
    basis = BSplineBasis(degree, knots)
    xk = np.array([1.0])
    left = basis.evaluate(xk, side="left")
    right = basis.evaluate(xk, side="right")
    np.testing.assert_allclose(left, right, atol=1e-8)


# -- evaluate_expansion (fused local-support path) --


def test_evaluate_expansion_matches_dense(witness_basis):
    # Algebraic identity, independent of knot values beyond the _gather
    # boundary indexing exercised by the witness pair.
    basis = witness_basis
    rng = np.random.default_rng(0)
    x = _sample_points(basis)
    c = rng.normal(size=basis.n_basis)
    phi = basis.evaluate(x)
    np.testing.assert_allclose(basis.evaluate_expansion(c, x), phi @ c, atol=1e-9)


@pytest.mark.parametrize("deriv", [1, 2, 3])
def test_evaluate_expansion_matches_dense_derivative(witness_basis, deriv):
    # Algebraic identity, independent of knot layout.
    basis = witness_basis
    rng = np.random.default_rng(1)
    x = _sample_points(basis)
    c = rng.normal(size=basis.n_basis)
    phi = basis.evaluate(x, deriv=deriv)
    np.testing.assert_allclose(
        basis.evaluate_expansion(c, x, deriv=deriv), phi @ c, atol=1e-7
    )


def test_evaluate_expansion_vector_valued(witness_basis):
    # Exercises only the vector-valued branch, which doesn't depend on knot
    # layout.
    basis = witness_basis
    rng = np.random.default_rng(2)
    x = _sample_points(basis)
    c = rng.normal(size=(basis.n_basis, 3))
    phi = basis.evaluate(x)
    np.testing.assert_allclose(basis.evaluate_expansion(c, x), phi @ c, atol=1e-9)


def test_evaluate_expansion_deriv_past_degree(witness_basis):
    # Exercises the early-return branch through evaluate_expansion, when
    # the derivative order exceeds the basis degree.
    basis = witness_basis
    x = _sample_points(basis)
    c = np.ones(basis.n_basis)
    result = basis.evaluate_expansion(c, x, deriv=basis.degree + 1)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)

    cvec = np.ones((basis.n_basis, 2))
    result_vec = basis.evaluate_expansion(cvec, x, deriv=basis.degree + 1)
    assert result_vec.shape == (len(x), 2)
    np.testing.assert_allclose(result_vec, 0.0, atol=1e-12)


# -- boundary_dofs --


def test_boundary_dofs_clamped():
    degree, knots = CLAMPED_UNIFORM
    basis = BSplineBasis(degree, knots)
    assert basis.boundary_dofs() == (0, basis.n_basis - 1)


def test_boundary_dofs_open():
    degree, knots = OPEN_QUADRATIC
    basis = BSplineBasis(degree, knots)
    assert basis.boundary_dofs() == (None, None)


def test_boundary_dofs_nonzero_order(witness_basis):
    # `order != 0` returns immediately without reading `self`, so this
    # doesn't depend on knot layout.
    assert witness_basis.boundary_dofs(order=1) == (None, None)


# -- _derivative_basis / _integral_basis --


def test_derivative_basis():
    degree, knots = CLAMPED_NONUNIFORM
    basis = BSplineBasis(degree, knots)
    assert basis._derivative_basis(0) is basis
    d1 = basis._derivative_basis(1)
    assert d1.degree == degree - 1
    assert len(d1.knots) == len(knots) - 2
    assert d1.n_basis == basis.n_basis - 1

    d_full = basis._derivative_basis(degree)
    assert d_full.degree == 0

    with pytest.raises(ValueError, match="deriv must be >= 0"):
        basis._derivative_basis(-1)
    with pytest.raises(ValueError, match="at or past the degree"):
        basis._derivative_basis(degree + 1)


def test_integral_basis():
    degree, knots = CLAMPED_NONUNIFORM
    basis = BSplineBasis(degree, knots)
    assert basis._integral_basis(0) is basis
    i1 = basis._integral_basis(1)
    assert i1.degree == degree + 1
    assert len(i1.knots) == len(knots) + 2
    assert i1.n_basis == basis.n_basis + 1

    i2 = basis._integral_basis(2)
    assert i2.degree == degree + 2
    assert i2.n_basis == basis.n_basis + 2
    # Interior knots are untouched; only the ends grow.
    np.testing.assert_allclose(
        i2.knots[degree + 2 : -(degree + 2)], knots[degree:-degree]
    )

    with pytest.raises(ValueError, match="order must be >= 0"):
        basis._integral_basis(-1)


# -- FunctionSpace round trip --


def _space(basis):
    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    return FunctionSpace(basis, UnitInterval.Parameters(a=a, b=b))


def test_default_quadrature_exactness(basis):
    space = _space(basis)
    coeffs = np.linspace(0.0, 1.0, basis.degree + 1)

    def f(x):
        return sum(c * x**k for k, c in enumerate(coeffs))

    nodes, weights = space.quadrature()
    got = np.dot(weights, f(nodes))

    def antideriv(x):
        return sum(c * x ** (k + 1) / (k + 1) for k, c in enumerate(coeffs))

    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    expected = antideriv(b) - antideriv(a)
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_project_exactness(basis):
    space = _space(basis)
    coeffs = np.linspace(0.5, 1.5, basis.degree + 1)

    def f(x):
        return sum(c * x**k for k, c in enumerate(coeffs))

    u = space.project(f)
    x = _sample_points(basis, extrapolate=False)
    np.testing.assert_allclose(u(x), f(x), atol=1e-7)


def test_derivative_round_trip():
    degree, knots = CLAMPED_NONUNIFORM
    basis = BSplineBasis(degree, knots)
    space = _space(basis)

    def f(x):
        return 2 * x**3 - x**2 + 5 * x - 1

    def df(x):
        return 6 * x**2 - 2 * x + 5

    u = space.project(f)
    x = _sample_points(basis, extrapolate=False)
    du = u.derivative()
    np.testing.assert_allclose(du(x), df(x), atol=1e-6)
    np.testing.assert_allclose(du(x), u(x, deriv=1), atol=1e-6)


def test_integral_round_trip():
    degree, knots = CLAMPED_NONUNIFORM
    basis = BSplineBasis(degree, knots)
    space = _space(basis)

    def f(x):
        return 2 * x**3 - x**2 + 5 * x - 1

    def f_antideriv(x):
        return x**4 / 2 - x**3 / 3 + 2.5 * x**2 - x

    u = space.project(f)
    a = knots[degree]
    x = _sample_points(basis, extrapolate=False)
    antideriv = u.antiderivative()
    np.testing.assert_allclose(antideriv(x), f_antideriv(x) - f_antideriv(a), atol=1e-6)


# -- FunctionSpace.bspline() / .clamped_bspline() constructors --


def test_bspline_constructor():
    # Matches a manual FunctionSpace(BSplineBasis(...), domain) construction.
    knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.5, 4.0, 4.0, 4.0, 4.0])
    degree = 3
    manual = FunctionSpace(
        BSplineBasis(degree, knots), UnitInterval.Parameters(a=0.0, b=4.0)
    )
    space = FunctionSpace.bspline(degree, knots)
    assert space.n_basis == manual.n_basis
    assert space.domain == manual.domain
    phi_m, phi_s = manual.basis_matrix(), space.basis_matrix()
    np.testing.assert_allclose(phi_m.matrix, phi_s.matrix)
    np.testing.assert_allclose(phi_m.weights, phi_s.weights)

    # Domain is derived from the knots, and the knots pass through unchanged.
    knots = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    degree = 2
    space = FunctionSpace.bspline(degree, knots)
    assert space.domain == UnitInterval.Parameters(
        a=knots[degree], b=knots[-1 - degree]
    )
    np.testing.assert_allclose(space.basis.knots, knots)

    # quad_rule is forwarded when it's compatible with the basis.
    knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    degree = 2
    basis = BSplineBasis(degree, knots)
    rule = composite_quad(gauss_legendre(5), basis.required_breakpoints)
    space = FunctionSpace.bspline(degree, knots, quad_rule=rule)
    assert space.reference_quad_rule is rule

    # Projection is exact for a polynomial within the basis's own degree.
    knots = np.array([0.0, 0.0, 0.0, 0.0, 1.5, 3.0, 3.0, 3.0, 3.0])
    degree = 3
    space = FunctionSpace.bspline(degree, knots)

    def f(x):
        return x**3 - 2 * x + 1

    x = np.linspace(0.0, 3.0, 15)
    u = space.project(f)
    np.testing.assert_allclose(u(x), f(x), atol=1e-8)


def test_clamped_bspline_constructor():
    # Matches a manual FunctionSpace(BSplineBasis(...), domain) construction
    # built from the equivalent hand-assembled clamped knot vector.
    breakpoints = np.array([0.0, 1.0, 2.5, 4.0])
    degree = 3
    knots = np.concatenate(
        [np.full(degree, breakpoints[0]), breakpoints, np.full(degree, breakpoints[-1])]
    )
    manual = FunctionSpace(
        BSplineBasis(degree, knots),
        UnitInterval.Parameters(a=breakpoints[0], b=breakpoints[-1]),
    )
    space = FunctionSpace.clamped_bspline(degree, breakpoints)
    assert space.n_basis == manual.n_basis
    assert space.domain == manual.domain
    phi_m, phi_s = manual.basis_matrix(), space.basis_matrix()
    np.testing.assert_allclose(phi_m.matrix, phi_s.matrix)
    np.testing.assert_allclose(phi_m.weights, phi_s.weights)

    # The knot vector is clamped (simple interior knots, full multiplicity
    # at both ends), and the domain is derived from the breakpoints.
    space = FunctionSpace.clamped_bspline(2, [0.0, 1.0, 2.0])
    knots = space.basis.knots
    np.testing.assert_allclose(knots[:3], 0.0)
    np.testing.assert_allclose(knots[-3:], 2.0)
    np.testing.assert_allclose(knots[3:-3], [1.0])
    assert space.basis.boundary_dofs() == (0, space.n_basis - 1)

    space = FunctionSpace.clamped_bspline(3, [2.0, 3.0, 5.0, 7.0])
    assert space.domain == UnitInterval.Parameters(a=2.0, b=7.0)

    # Default quadrature is one Gauss-Legendre rule per element.
    space = FunctionSpace.clamped_bspline(3, [0.0, 1.0, 2.0, 3.0, 4.0])
    rule = space.basis.default_quadrature()
    assert len(rule) == 4 * len(gauss_legendre(4))

    # Projection is exact for a polynomial within the basis's own degree.
    breakpoints = np.array([0.0, 1.5, 3.0])
    degree = 3
    space = FunctionSpace.clamped_bspline(degree, breakpoints)

    def f(x):
        return x**3 - 2 * x + 1

    x = np.linspace(0.0, 3.0, 15)
    u = space.project(f)
    np.testing.assert_allclose(u(x), f(x), atol=1e-8)

    # Breakpoints must be a 1-D array with at least two strictly increasing
    # entries.
    with pytest.raises(ValueError, match="1-D with at least 2 entries"):
        FunctionSpace.clamped_bspline(2, [1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        FunctionSpace.clamped_bspline(2, [0.0, 1.0, 1.0, 2.0])


@pytest.mark.parametrize(
    "ctor",
    [FunctionSpace.bspline, FunctionSpace.clamped_bspline],
    ids=["bspline", "clamped_bspline"],
)
def test_degree_zero(ctor):
    # degree=0 needs no repeated end knots to be "clamped", so both
    # constructors agree on the same knot vector for this special case.
    space = ctor(0, [0.0, 1.0, 2.0, 3.0])
    assert space.basis.degree == 0
    assert space.n_basis == 3
    np.testing.assert_allclose(space.basis.knots, [0.0, 1.0, 2.0, 3.0])


# -- symbolic tracing --


def test_static_and_dynamic_evaluation_agree(basis):
    x = _sample_points(basis, extrapolate=False)
    static_phi = basis.evaluate(x)

    @arc.compile
    def traced(x):
        assert isinstance(x, SymbolicArray)
        return basis.evaluate(np.atleast_1d(x))

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-9)


def test_static_and_dynamic_side_left():
    # At a repeated interior knot, `_locate`'s left-side span needs more
    # than a single-step backup. Check the traced (symbolic) evaluation
    # matches the eager one at this case.
    degree, knots = INTERIOR_MAX_MULT
    basis = BSplineBasis(degree, knots)
    x = np.array([1.0])
    static_phi = basis.evaluate(x, side="left")

    @arc.compile
    def traced(x):
        return basis.evaluate(np.atleast_1d(x), side="left")

    dynamic_phi = np.asarray(traced(x[0]))
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-9)


def test_gradient_through_traced_coefficients():
    degree, knots = CLAMPED_NONUNIFORM
    basis = BSplineBasis(degree, knots)
    space = _space(basis)
    x = _sample_points(basis, extrapolate=False)

    @arc.compile
    def energy(c):
        return Function(c, space)(x) @ Function(c, space)(x)

    c0 = np.linspace(-1.0, 1.0, basis.n_basis)
    step = 1e-6
    direction = np.ones(basis.n_basis)
    fd = (energy(c0 + step * direction) - energy(c0 - step * direction)) / (2 * step)
    grad = arc.grad(energy)(c0) @ direction
    assert float(grad) == pytest.approx(float(fd), rel=1e-4)


# -- TensorBasis composition --
#
# Unlike every other family, a BSplineBasis's knots are physical, and its
# `evaluate` ignores the `a`/`b` domain kwargs `TensorBasis` forwards per
# dimension (see the class docstring). Each dimension here therefore needs
# its own BSplineBasis instance already built on its own physical range,
# rather than one basis reused and remapped across dimensions.


def test_tensor_composition_exactness():
    basis_x = FunctionSpace.clamped_bspline(3, np.linspace(0.0, 2.0, 4)).basis
    basis_y = FunctionSpace.clamped_bspline(2, np.linspace(-1.0, 1.0, 3)).basis
    tensor = TensorBasis((basis_x, basis_y))
    domain = ProductParameters(
        dims=(
            UnitInterval.Parameters(a=0.0, b=2.0),
            UnitInterval.Parameters(a=-1.0, b=1.0),
        )
    )
    space = FunctionSpace(tensor, domain=domain)

    def f(xy):
        # Non-separable, but within each factor's exactly-representable
        # degree (cubic in x, quadratic in y).
        return xy[:, 0] ** 3 * xy[:, 1] ** 2 + xy[:, 0] - xy[:, 1]

    u = space.project(f)
    grid_x = np.linspace(0.1, 1.9, 6)
    grid_y = np.linspace(-0.9, 0.9, 5)
    xy = np.stack(np.meshgrid(grid_x, grid_y, indexing="ij"), axis=-1).reshape(-1, 2)
    np.testing.assert_allclose(u(xy), f(xy), atol=1e-6)
