"""Which element owns a point on a breakpoint.

A piecewise basis is two-valued at its interior breakpoints -- always in the
derivative, and in the value too when discontinuous -- so evaluating exactly
on one requires deciding which element it belongs to. There are two
mechanisms, and the point of these tests is that they are *different*:

- Coordinate-only evaluation has nothing but the point, so it follows the
  ``side`` argument.
- Quadrature uses the rule's record of which element each node was generated
  for, which is strictly more information: a composite Lobatto rule places a
  node on each interior breakpoint from *both* sides, and no coordinate
  convention can tell those two copies apart.
"""

import numpy as np
import pytest

import archimedes as arc
from archimedes.experimental.approximation import (
    FunctionSpace,
    LagrangeBasis,
    PiecewiseBasis,
    ProductParameters,
    TensorBasis,
)
from archimedes.measure import UnitInterval
from archimedes.quadrature import (
    composite,
    gauss_legendre,
    gauss_lobatto,
    gauss_radau,
    tensor,
)

A, B = 0.0, 1.0
DOMAIN = UnitInterval.Parameters(a=A, b=B)
BREAKS = np.linspace(-1.0, 1.0, 3)
KNOT = 0.5  # the interior breakpoint, mapped onto [0, 1]

# Exact to degree 12 with strictly interior nodes: no ownership question can
# arise, so this is the reference every other rule is compared against.
REFERENCE = composite(gauss_legendre(12), BREAKS)


def _element(n=3):
    return LagrangeBasis(reference_nodes=gauss_lobatto(n).nodes)


def _basis(continuity, n=3, breaks=BREAKS):
    return PiecewiseBasis(_element(n), breaks, continuity=continuity)


# Every rule below is exact to at least degree 4, which the element mass
# integrand needs, so any remaining error is ownership and nothing else.
BOUNDARY_NODE_RULES = {
    "lobatto": composite(gauss_lobatto(4), BREAKS),
    "radau_left": composite(gauss_radau(3, endpoint="left"), BREAKS),
    "radau_right": composite(gauss_radau(3, endpoint="right"), BREAKS),
    "gauss": composite(gauss_legendre(3), BREAKS),
}


# -- quadrature uses recorded ownership --


@pytest.mark.parametrize("continuity", [-1, 0])
@pytest.mark.parametrize("rule_name", sorted(BOUNDARY_NODE_RULES))
def test_mass_matrix_is_exact_whatever_the_rules_nodes(continuity, rule_name):
    # Before ownership was recorded, `lobatto` was 15.6% wrong and
    # `radau_right` 20.8% wrong for continuity=-1, because the half-open
    # coordinate rule assigned each element's right-endpoint node to its
    # neighbour. `radau_left` passed only by accident of that convention.
    basis = _basis(continuity)
    expected = FunctionSpace(basis, DOMAIN, quad_rule=REFERENCE).mass_matrix()
    got = FunctionSpace(
        basis, DOMAIN, quad_rule=BOUNDARY_NODE_RULES[rule_name]
    ).mass_matrix()
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("continuity", [-1, 0])
def test_stiffness_matrix_is_exact_with_boundary_nodes(continuity):
    basis = _basis(continuity, n=4)
    expected = FunctionSpace(basis, DOMAIN, quad_rule=REFERENCE).stiffness_matrix()
    rule = composite(gauss_lobatto(5), BREAKS)
    got = FunctionSpace(basis, DOMAIN, quad_rule=rule).stiffness_matrix()
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_projection_is_exact_with_boundary_nodes():
    basis = _basis(-1, n=4)
    space = FunctionSpace(basis, DOMAIN, quad_rule=composite(gauss_lobatto(6), BREAKS))
    x = np.linspace(A, B, 41)
    np.testing.assert_allclose(
        space.project(lambda x: x**3 - 2 * x)(x), x**3 - 2 * x, atol=1e-11
    )


def test_refined_rule_whose_breakpoints_strictly_contain_the_basis():
    # Ownership maps rule elements to the basis element containing them, so a
    # rule refined beyond the basis is still handled exactly.
    basis = _basis(-1)
    expected = FunctionSpace(basis, DOMAIN, quad_rule=REFERENCE).mass_matrix()
    refined = composite(gauss_lobatto(4), np.linspace(-1.0, 1.0, 5))
    got = FunctionSpace(basis, DOMAIN, quad_rule=refined).mass_matrix()
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_derivative_of_a_c0_space_integrates_exactly():
    # `derivative()` always lands in a discontinuous space, so this is the
    # path that made the issue reachable from ordinary use.
    space = FunctionSpace(_basis(0, n=4), DOMAIN)
    u = space.project(lambda x: x**3)
    du = u.derivative()
    exact = FunctionSpace(du.space.basis, DOMAIN, quad_rule=REFERENCE)
    assert du.dot(du) == pytest.approx(
        exact.inner_product(du.coefficients, du.coefficients), rel=1e-10
    )


def test_falls_back_to_coordinates_without_recorded_ownership():
    # A rule with no element structure has no provenance to use; the result
    # must still match plain coordinate evaluation.
    basis = _basis(-1)
    rule = gauss_legendre(8)
    assert rule.elements is None
    np.testing.assert_allclose(
        basis._evaluate_at_nodes(rule, a=A, b=B),
        basis.evaluate(rule.scaled_points(a=A, b=B), a=A, b=B),
    )


def test_tensor_of_piecewise_factors_with_boundary_nodes():
    basis = TensorBasis((_basis(-1), _basis(-1)))
    domain = ProductParameters(dims=(DOMAIN, DOMAIN))
    lobatto = composite(gauss_lobatto(4), BREAKS)
    got = FunctionSpace(basis, domain, quad_rule=tensor(lobatto, lobatto)).mass_matrix()
    expected = FunctionSpace(
        basis, domain, quad_rule=tensor(REFERENCE, REFERENCE)
    ).mass_matrix()
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_tensor_mixes_piecewise_and_smooth_factors():
    from archimedes.experimental.approximation import OrthogonalPolynomialBasis
    from archimedes.measure import LegendreMeasure

    basis = TensorBasis((_basis(-1), OrthogonalPolynomialBasis(LegendreMeasure(), 3)))
    domain = ProductParameters(dims=(DOMAIN, DOMAIN))
    got = FunctionSpace(
        basis,
        domain,
        quad_rule=tensor(composite(gauss_lobatto(4), BREAKS), gauss_legendre(4)),
    ).mass_matrix()
    expected = FunctionSpace(
        basis, domain, quad_rule=tensor(REFERENCE, gauss_legendre(4))
    ).mass_matrix()
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


# -- coordinate evaluation follows `side` --


@pytest.mark.parametrize("continuity,jump", [(-1, True), (0, False)])
def test_side_gives_the_one_sided_limits(continuity, jump):
    space = FunctionSpace(_basis(continuity), DOMAIN)
    u = space.project(lambda x: np.where(x < KNOT, x, 2 * x - 0.25))
    eps = 1e-9
    at = np.array([KNOT])
    left = u(at, side="left")[0]
    right = u(at, side="right")[0]
    np.testing.assert_allclose(left, u(np.array([KNOT - eps]))[0], atol=1e-7)
    np.testing.assert_allclose(right, u(np.array([KNOT + eps]))[0], atol=1e-7)
    # A C0 space is single-valued; a discontinuous one is not.
    assert bool(abs(right - left) > 1e-3) is jump


@pytest.mark.parametrize("continuity", [-1, 0])
def test_derivative_jumps_even_when_the_value_does_not(continuity):
    # The gradient-jump error indicator for a C0 space needs exactly this.
    space = FunctionSpace(_basis(continuity), DOMAIN)
    u = space.project(lambda x: np.where(x < KNOT, x, 2 * x - 0.25))
    at = np.array([KNOT])
    assert u(at, deriv=1, side="right")[0] - u(at, deriv=1, side="left")[0] == (
        pytest.approx(1.0, rel=1e-6)
    )


def test_right_is_the_default():
    space = FunctionSpace(_basis(-1), DOMAIN)
    u = space.project(lambda x: np.where(x < KNOT, 1.0, 2.0))
    at = np.array([KNOT])
    assert u(at)[0] == u(at, side="right")[0]


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("continuity", [-1, 0])
@pytest.mark.parametrize("deriv", [0, 1])
def test_dense_and_fused_paths_agree_on_both_sides(side, continuity, deriv):
    # `evaluate` and `evaluate_expansion` resolve breakpoints independently
    # (masking vs. locate-and-gather), so they must be checked to agree.
    basis = _basis(continuity)
    coefficients = np.arange(basis.n_basis, dtype=float)
    x = np.concatenate([np.linspace(A, B, 11), [KNOT]])
    dense = basis.evaluate(x, deriv=deriv, a=A, b=B, side=side) @ coefficients
    fused = basis.evaluate_expansion(coefficients, x, deriv=deriv, a=A, b=B, side=side)
    np.testing.assert_allclose(dense, fused, atol=1e-10)


@pytest.mark.parametrize("side", ["left", "right"])
def test_side_traces_symbolically(side):
    basis = _basis(-1)
    coefficients = np.arange(basis.n_basis, dtype=float)
    x = np.array([KNOT])
    expected = basis.evaluate_expansion(coefficients, x, a=A, b=B, side=side)

    @arc.compile
    def traced(xx):
        return basis.evaluate_expansion(coefficients, xx, a=A, b=B, side=side)

    np.testing.assert_allclose(np.asarray(traced(x)).ravel(), expected, atol=1e-12)


def test_side_at_the_outer_endpoints():
    # The domain's own endpoints are owned by the end elements under either
    # convention -- there is no element beyond them to hand the point to.
    basis = _basis(-1)
    ends = np.array([A, B])
    for side in ("left", "right"):
        phi = basis.evaluate(ends, a=A, b=B, side=side)
        np.testing.assert_allclose(phi.sum(axis=1), 1.0, atol=1e-10)


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.evaluate(np.array([KNOT]), a=A, b=B, side="up"),
        lambda b: b.evaluate_expansion(
            np.zeros(b.n_basis), np.array([KNOT]), a=A, b=B, side="up"
        ),
    ],
    ids=["evaluate", "evaluate_expansion"],
)
def test_invalid_side_rejected(call):
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        call(_basis(-1))
