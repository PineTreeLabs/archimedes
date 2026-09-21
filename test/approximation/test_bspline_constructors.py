import numpy as np
import pytest

from archimedes.approximation import BSplineBasis, FunctionSpace
from archimedes.measure import UnitInterval
from archimedes.quadrature import composite_quad, gauss_legendre


def test_bspline():
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


def test_clamped_bspline():
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
