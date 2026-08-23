"""Tests for the two ``BSplineBasis``-backed ``FunctionSpace`` classmethod
constructors:

- ``FunctionSpace.bspline(degree, knots)`` -- general, thin sugar over
  ``FunctionSpace(BSplineBasis(degree, knots), domain)``, with ``domain``
  derived automatically from ``knots``.
- ``FunctionSpace.clamped_bspline(degree, breakpoints)`` -- sugar over
  ``bspline()`` for the common case, building a clamped knot vector (simple
  interior knots, multiplicity ``degree + 1`` at both ends) from physical
  breakpoints.

Each constructor has exactly one required argument that's always meaningful
-- unlike an earlier combined design, neither ever silently ignores part of
its input.
"""

import numpy as np
import pytest

from archimedes.experimental.approximation import BSplineBasis, FunctionSpace
from archimedes.measure import UnitInterval
from archimedes.quadrature import composite_quad, gauss_legendre

# -- bspline() (general, explicit knots) --


def test_bspline_matches_manual_construction():
    knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.5, 4.0, 4.0, 4.0, 4.0])
    degree = 3
    manual = FunctionSpace(
        BSplineBasis(degree, knots), UnitInterval.Parameters(a=0.0, b=4.0)
    )
    sugar = FunctionSpace.bspline(degree, knots)

    assert sugar.n_basis == manual.n_basis
    assert sugar.domain == manual.domain
    phi_m, phi_s = manual.basis_matrix(), sugar.basis_matrix()
    np.testing.assert_allclose(phi_m.matrix, phi_s.matrix)
    np.testing.assert_allclose(phi_m.weights, phi_s.weights)


def test_bspline_domain_is_derived_from_knots():
    knots = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    degree = 2
    space = FunctionSpace.bspline(degree, knots)
    assert space.domain == UnitInterval.Parameters(
        a=knots[degree], b=knots[-1 - degree]
    )


def test_bspline_open_knot_vector():
    knots = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    degree = 2
    space = FunctionSpace.bspline(degree, knots)
    np.testing.assert_allclose(space.basis.knots, knots)
    assert space.basis.boundary_dofs() == (None, None)


def test_bspline_degree_zero():
    space = FunctionSpace.bspline(0, [0.0, 1.0, 2.0, 3.0])
    assert space.basis.degree == 0
    assert space.n_basis == 3


def test_bspline_project_is_exact_for_its_own_degree():
    knots = np.array([0.0, 0.0, 0.0, 0.0, 1.5, 3.0, 3.0, 3.0, 3.0])
    degree = 3
    space = FunctionSpace.bspline(degree, knots)

    def f(x):
        return x**3 - 2 * x + 1

    x = np.linspace(0.0, 3.0, 15)
    u = space.project(f)
    np.testing.assert_allclose(u(x), f(x), atol=1e-8)


def test_bspline_quad_rule_passthrough_accepts_compatible_rule():
    degree = 2
    knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    basis = BSplineBasis(degree, knots)
    rule = composite_quad(gauss_legendre(5), basis.required_breakpoints)
    sugar = FunctionSpace.bspline(degree, knots, quad_rule=rule)
    assert sugar.quad_rule is rule


def test_bspline_quad_rule_passthrough_rejects_incompatible_rule():
    # A plain (non-composite) rule can't integrate a piecewise-smooth basis
    # exactly -- FunctionSpace's own validation catches this, same as it
    # always has for PiecewiseBasis.
    knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="only piecewise smooth"):
        FunctionSpace.bspline(2, knots, quad_rule=gauss_legendre(5))


# -- clamped_bspline() (clamped, from physical breakpoints) --


def test_clamped_bspline_matches_manual_construction():
    breakpoints = np.array([0.0, 1.0, 2.5, 4.0])
    degree = 3
    knots = np.concatenate(
        [np.full(degree, breakpoints[0]), breakpoints, np.full(degree, breakpoints[-1])]
    )
    manual = FunctionSpace(
        BSplineBasis(degree, knots),
        UnitInterval.Parameters(a=breakpoints[0], b=breakpoints[-1]),
    )
    sugar = FunctionSpace.clamped_bspline(degree, breakpoints)

    assert sugar.n_basis == manual.n_basis
    assert sugar.domain == manual.domain
    phi_m, phi_s = manual.basis_matrix(), sugar.basis_matrix()
    np.testing.assert_allclose(phi_m.matrix, phi_s.matrix)
    np.testing.assert_allclose(phi_m.weights, phi_s.weights)


def test_clamped_bspline_knot_vector_is_clamped_with_simple_interior_knots():
    space = FunctionSpace.clamped_bspline(2, [0.0, 1.0, 2.0])
    knots = space.basis.knots
    np.testing.assert_allclose(knots[:3], 0.0)
    np.testing.assert_allclose(knots[-3:], 2.0)
    np.testing.assert_allclose(knots[3:-3], [1.0])
    assert space.basis.boundary_dofs() == (0, space.n_basis - 1)


def test_clamped_bspline_domain_is_derived_from_breakpoints():
    space = FunctionSpace.clamped_bspline(3, [2.0, 3.0, 5.0, 7.0])
    assert space.domain == UnitInterval.Parameters(a=2.0, b=7.0)


def test_clamped_bspline_degree_zero():
    space = FunctionSpace.clamped_bspline(0, [0.0, 1.0, 2.0, 3.0])
    assert space.basis.degree == 0
    assert space.n_basis == 3
    np.testing.assert_allclose(space.basis.knots, [0.0, 1.0, 2.0, 3.0])


def test_clamped_bspline_breakpoints_must_be_1d_with_at_least_two_entries():
    with pytest.raises(ValueError, match="1-D with at least 2 entries"):
        FunctionSpace.clamped_bspline(2, [1.0])


def test_clamped_bspline_breakpoints_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        FunctionSpace.clamped_bspline(2, [0.0, 1.0, 1.0, 2.0])


def test_clamped_bspline_default_quadrature_is_composite_gauss_legendre():
    space = FunctionSpace.clamped_bspline(3, [0.0, 1.0, 2.0, 3.0, 4.0])
    rule = space.basis.default_quadrature()
    assert len(rule) == 4 * len(gauss_legendre(4))


def test_clamped_bspline_quad_rule_forwarded():
    degree = 2
    breakpoints = np.array([0.0, 1.0, 2.0])
    knots = np.concatenate(
        [np.full(degree, breakpoints[0]), breakpoints, np.full(degree, breakpoints[-1])]
    )
    basis = BSplineBasis(degree, knots)
    rule = composite_quad(gauss_legendre(5), basis.required_breakpoints)
    sugar = FunctionSpace.clamped_bspline(degree, breakpoints, quad_rule=rule)
    assert sugar.quad_rule is rule


def test_clamped_bspline_project_matches_manual_construction():
    breakpoints = np.array([0.0, 1.5, 3.0])
    degree = 3
    sugar = FunctionSpace.clamped_bspline(degree, breakpoints)

    def f(x):
        return x**3 - 2 * x + 1

    x = np.linspace(0.0, 3.0, 15)
    u = sugar.project(f)
    np.testing.assert_allclose(u(x), f(x), atol=1e-8)
