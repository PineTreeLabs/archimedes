"""Tests for ``BSplineBasis``: general (non-uniform, arbitrary-multiplicity,
clamped or open) B-spline evaluation via de Boor's BSPLVB recurrence.

Values and derivatives are checked against ``scipy.interpolate.BSpline`` as
an independent oracle, across clamped/open, uniform/non-uniform, and
repeated-interior-knot (reduced continuity, up to a genuine discontinuity at
maximal multiplicity) knot vectors.
"""

import numpy as np
import pytest
from scipy.interpolate import BSpline as ScipyBSpline

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import (
    BSplineBasis,
    Function,
    FunctionSpace,
    ProductParameters,
    TensorBasis,
)
from archimedes.measure import UnitInterval

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


def _sample_points(basis, extrapolate=True):
    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    x = np.linspace(a, b, 37)
    if extrapolate:
        span = b - a
        x = np.concatenate([[a - 0.3 * span], x, [b + 0.3 * span]])
    return x


# -- construction validation --


def test_degree_must_be_nonnegative():
    with pytest.raises(ValueError, match="degree must be"):
        BSplineBasis(-1, [0.0, 0, 1, 1])


def test_knots_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        BSplineBasis(1, [[0.0, 0], [1, 1]])


def test_knots_must_have_minimum_length():
    with pytest.raises(ValueError, match="at least 2\\*degree\\+2"):
        BSplineBasis(3, [0.0, 0, 0, 0, 1])


def test_knots_must_be_nondecreasing():
    with pytest.raises(ValueError, match="nondecreasing"):
        BSplineBasis(1, [0.0, 0, 1, 0.5, 2, 2])


def test_knot_multiplicity_is_bounded():
    with pytest.raises(ValueError, match="repeat at most"):
        BSplineBasis(1, [0.0, 0, 0, 1, 2])


def test_degenerate_basic_interval_is_rejected():
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


def test_negative_deriv_rejected(basis):
    x = np.array([basis.knots[basis.degree]])
    with pytest.raises(ValueError, match="deriv must be >= 0"):
        basis.evaluate(x, deriv=-1)
    with pytest.raises(ValueError, match="deriv must be >= 0"):
        basis.evaluate_expansion(np.zeros(basis.n_basis), x, deriv=-1)


def test_invalid_side_rejected(basis):
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


@pytest.mark.parametrize("deriv", [1, 2, 3])
def test_derivatives_match_scipy(basis, deriv):
    if deriv > basis.degree:
        pytest.skip("covered by test_deriv_past_degree_is_zero")
    x = _sample_points(basis)
    phi = basis.evaluate(x, deriv=deriv)
    for i in range(basis.n_basis):
        c = np.zeros(basis.n_basis)
        c[i] = 1.0
        sp = ScipyBSpline(basis.knots, c, basis.degree, extrapolate=True)
        np.testing.assert_allclose(phi[:, i], sp(x, nu=deriv), atol=1e-6)


def test_deriv_past_degree_is_zero(basis):
    x = _sample_points(basis)
    phi = basis.evaluate(x, deriv=basis.degree + 1)
    np.testing.assert_allclose(phi, 0.0, atol=1e-12)
    phi2 = basis.evaluate(x, deriv=basis.degree + 4)
    np.testing.assert_allclose(phi2, 0.0, atol=1e-12)


def test_derivative_matches_finite_difference(basis):
    if basis.degree < 1:
        pytest.skip("no first derivative to check")
    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    x = np.linspace(a + 0.05 * (b - a), b - 0.05 * (b - a), 25)
    h = 1e-6
    dphi = basis.evaluate(x, deriv=1)
    lo = basis.evaluate(x - h)
    hi = basis.evaluate(x + h)
    np.testing.assert_allclose(dphi, (hi - lo) / (2 * h), atol=1e-4)


# -- structural properties --


def test_partition_of_unity(basis):
    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    x = np.linspace(a + 1e-9, b - 1e-9, 101)
    phi = basis.evaluate(x)
    np.testing.assert_allclose(phi.sum(axis=1), 1.0, atol=1e-8)


def test_local_support(basis):
    knots = basis.knots
    p = basis.degree
    a, b = knots[p], knots[-1 - p]
    x = np.linspace(a, b, 201)
    phi = basis.evaluate(x)
    for i in range(basis.n_basis):
        lo, hi = knots[i], knots[i + p + 1]
        outside = (x < lo) | (x > hi)
        np.testing.assert_allclose(phi[outside, i], 0.0, atol=1e-10)


def test_side_left_and_right_agree_away_from_knots(basis):
    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    x = np.linspace(a + 0.01, b - 0.01, 23)
    # Avoid landing exactly on an interior knot by construction of the grid.
    np.testing.assert_allclose(
        basis.evaluate(x, side="left"), basis.evaluate(x, side="right"), atol=1e-10
    )


def test_side_disagrees_at_a_discontinuous_interior_knot():
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


def test_side_agrees_at_a_c1_interior_knot():
    # Multiplicity degree - 1 (< degree + 1) still leaves the basis itself
    # (deriv=0) continuous, unlike the maximal-multiplicity case above.
    degree, knots = INTERIOR_DOUBLE_KNOT
    basis = BSplineBasis(degree, knots)
    xk = np.array([1.0])
    left = basis.evaluate(xk, side="left")
    right = basis.evaluate(xk, side="right")
    np.testing.assert_allclose(left, right, atol=1e-8)


# -- evaluate_expansion (fused local-support path) --


def test_evaluate_expansion_matches_dense(basis):
    rng = np.random.default_rng(0)
    x = _sample_points(basis)
    c = rng.normal(size=basis.n_basis)
    phi = basis.evaluate(x)
    np.testing.assert_allclose(basis.evaluate_expansion(c, x), phi @ c, atol=1e-9)


@pytest.mark.parametrize("deriv", [1, 2, 3])
def test_evaluate_expansion_matches_dense_derivatives(basis, deriv):
    if deriv > basis.degree:
        pytest.skip("covered by test_evaluate_expansion_deriv_past_degree")
    rng = np.random.default_rng(1)
    x = _sample_points(basis)
    c = rng.normal(size=basis.n_basis)
    phi = basis.evaluate(x, deriv=deriv)
    np.testing.assert_allclose(
        basis.evaluate_expansion(c, x, deriv=deriv), phi @ c, atol=1e-7
    )


def test_evaluate_expansion_vector_valued(basis):
    rng = np.random.default_rng(2)
    x = _sample_points(basis)
    c = rng.normal(size=(basis.n_basis, 3))
    phi = basis.evaluate(x)
    np.testing.assert_allclose(basis.evaluate_expansion(c, x), phi @ c, atol=1e-9)


def test_evaluate_expansion_deriv_past_degree(basis):
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


def test_boundary_dofs_nonzero_order_is_always_none(basis):
    assert basis.boundary_dofs(order=1) == (None, None)


# -- _derivative_basis / _integral_basis --


def test_derivative_basis_size_and_degree():
    degree, knots = CLAMPED_NONUNIFORM
    basis = BSplineBasis(degree, knots)
    assert basis._derivative_basis(0) is basis
    d1 = basis._derivative_basis(1)
    assert d1.degree == degree - 1
    assert len(d1.knots) == len(knots) - 2
    assert d1.n_basis == basis.n_basis - 1

    d_full = basis._derivative_basis(degree)
    assert d_full.degree == 0


def test_derivative_basis_rejects_invalid_order():
    degree, knots = CLAMPED_NONUNIFORM
    basis = BSplineBasis(degree, knots)
    with pytest.raises(ValueError, match="deriv must be >= 0"):
        basis._derivative_basis(-1)
    with pytest.raises(ValueError, match="at or past the degree"):
        basis._derivative_basis(degree + 1)


def test_integral_basis_size_and_degree():
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


def test_integral_basis_rejects_negative_order():
    degree, knots = CLAMPED_NONUNIFORM
    basis = BSplineBasis(degree, knots)
    with pytest.raises(ValueError, match="order must be >= 0"):
        basis._integral_basis(-1)


# -- FunctionSpace round trip --


def _space(basis):
    a, b = basis.knots[basis.degree], basis.knots[-1 - basis.degree]
    return FunctionSpace(basis, UnitInterval.Parameters(a=a, b=b))


def test_default_quadrature_is_exact_for_a_polynomial_of_the_basis_degree(basis):
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


def test_project_is_exact_for_a_polynomial_of_the_basis_degree(basis):
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


def test_static_and_dynamic_evaluation_agree_side_left():
    # Regression check for the multiplicity-aware left-side span location:
    # `_locate`'s single-step backup is not enough at a repeated knot, and
    # the fix must trace the same as it runs eagerly.
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
# Unlike every other family, a BSplineBasis's knots are physical and its
# `evaluate` ignores the `a`/`b` domain kwargs `TensorBasis` forwards per
# dimension (see the class docstring) -- so, unlike the generic
# `FACTORIES`/`space` fixture in test_tensor_basis.py (which builds *one*
# basis and reuses it, remapped, across dimensions with different target
# domains), each dimension here needs its *own* BSplineBasis instance
# already built on its own physical range. Forcing this family into that
# shared harness would silently test the wrong thing (both dimensions
# evaluating the same physical knots despite claiming different domains),
# so this is a dedicated, correctly-constructed check instead.


def test_tensor_composition_is_exact_on_a_nonseparable_polynomial():
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
