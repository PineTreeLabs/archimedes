"""``OrthogonalPolynomialBasis``: values, derivatives, and orthonormality
against the measure each polynomial family is built on.
"""

# ruff: noqa: N806  (M is the conventional name for a Gram matrix)
import numpy as np
import pytest
from scipy.special import eval_legendre

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.approximation import FunctionSpace, OrthogonalPolynomialBasis
from archimedes.measure import (
    HalfLine,
    JacobiMeasure,
    LaguerreMeasure,
    LegendreMeasure,
    PhysicistsHermiteMeasure,
    ProbabilistsHermiteMeasure,
    RealLine,
    UnitInterval,
)
from archimedes.quadrature import gauss_hermite, gauss_laguerre, gauss_legendre


def test_construction_validation():
    with pytest.raises(ValueError):
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=0)

    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=3)
    with pytest.raises(ValueError):
        basis.evaluate(np.array([0.0]), deriv=-1)


@pytest.mark.parametrize("n_basis", [1, 2, 6])
def test_legendre_values_match_scipy_up_to_orthonormal_scale(n_basis):
    # Orthonormal p_k = P_k / ||P_k||, with ||P_k||^2 = 2 / (2k + 1) on [-1, 1]
    # (the classical Legendre normalization constant).
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=n_basis)
    x = np.linspace(-1, 1, 11)
    phi = basis.evaluate(x)
    assert phi.shape == (len(x), n_basis)
    for k in range(n_basis):
        classical = eval_legendre(k, x)
        norm = np.sqrt(2.0 / (2 * k + 1))
        np.testing.assert_allclose(phi[:, k], classical / norm, atol=1e-10)


@pytest.mark.parametrize("deriv,h,atol", [(1, 1e-6, 1e-5), (2, 1e-5, 1e-3)])
def test_derivative_matches_finite_difference(deriv, h, atol):
    # Not hand-derived per order -- a second derivative falls out of the
    # same generic recurrence as the first.
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=6)
    x = np.linspace(-0.9, 0.9, 13)
    dphi = basis.evaluate(x, deriv=deriv)
    lo = basis.evaluate(x - h, deriv=deriv - 1)
    hi = basis.evaluate(x + h, deriv=deriv - 1)
    np.testing.assert_allclose(dphi, (hi - lo) / (2 * h), atol=atol)


def test_orthonormal_on_reference_domain():
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=6)
    rule = gauss_legendre(15)
    phi = basis.evaluate(rule.nodes)
    M = phi.T @ (rule.weights[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(6), atol=1e-10)


def test_orthonormal_on_mapped_domain():
    a, b = 2.0, 7.0
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=6)
    rule = gauss_legendre(15).map_to(a, b)
    x = rule.nodes
    w = rule.weights
    phi = basis.evaluate(x, a=a, b=b)
    M = phi.T @ (w[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(6), atol=1e-10)


# -- `density`: orthonormal against the probability measure, not the raw weight --


def test_density():
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=3)
    assert basis.density is False

    # Only norm[0] (beta_0) differs between the two conventions -- 1 instead
    # of measure.mass(...) -- so phi_density = phi_raw * sqrt(mass) exactly,
    # for every degree.
    loc, scale = 0.0, 3.0
    raw = OrthogonalPolynomialBasis(ProbabilistsHermiteMeasure(), n_basis=4)
    density = OrthogonalPolynomialBasis(
        ProbabilistsHermiteMeasure(), n_basis=4, density=True
    )
    x = np.linspace(-5, 5, 11)
    phi_raw = raw.evaluate(x, loc=loc, scale=scale)
    phi_density = density.evaluate(x, loc=loc, scale=scale)
    mass = ProbabilistsHermiteMeasure().mass(loc=loc, scale=scale)
    np.testing.assert_allclose(phi_density, phi_raw * np.sqrt(mass), atol=1e-10)

    # With density=True, integrating phi_i * phi_j against density-normalized
    # (mass-1) quadrature weights gives the identity, same as the raw-weight
    # case does for density=False.
    loc, scale = 1.5, 2.0
    basis = OrthogonalPolynomialBasis(
        ProbabilistsHermiteMeasure(), n_basis=5, density=True
    )
    rule = gauss_hermite(15, kind="prob").map_to(loc, scale)
    x = rule.nodes
    w = rule.weights / np.sum(rule.weights)
    phi = basis.evaluate(x, loc=loc, scale=scale)
    M = phi.T @ (w[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(5), atol=1e-8)


# -- genericity: same class, no family-specific code, for other measures --


@pytest.mark.parametrize(
    "measure,rule",
    [
        (PhysicistsHermiteMeasure(), gauss_hermite(15, kind="phys")),
        (LaguerreMeasure(), gauss_laguerre(15)),
    ],
)
def test_generic_orthonormality(measure, rule):
    basis = OrthogonalPolynomialBasis(measure, n_basis=5)
    phi = basis.evaluate(rule.nodes)
    M = phi.T @ (rule.weights[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(5), atol=1e-8)


# -- static (NumPy) vs. dynamic (symbolic, via arc.compile) equivalence --


@pytest.mark.parametrize("deriv", [0, 1])
def test_static_and_dynamic_evaluation_agree(deriv):
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5)
    x = np.linspace(-1, 1, 9)
    static_phi = basis.evaluate(x, deriv=deriv)

    @arc.compile
    def traced(x):
        assert isinstance(x, SymbolicArray)
        return basis.evaluate(x, deriv=deriv)

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-12)


# -- FunctionSpace classmethod constructors --


def test_legendre_constructor():
    manual = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), 8),
        UnitInterval.Parameters(a=-2.0, b=3.0),
    )
    space = FunctionSpace.legendre(8, a=-2.0, b=3.0)
    assert space.n_basis == manual.n_basis
    assert space.domain == manual.domain
    np.testing.assert_allclose(
        space.basis_matrix().matrix, manual.basis_matrix().matrix
    )


@pytest.mark.parametrize("second_kind, expected_exponent", [(False, -0.5), (True, 0.5)])
def test_chebyshev_constructor(second_kind, expected_exponent):
    space = FunctionSpace.chebyshev(6, second_kind=second_kind)
    assert space.basis.measure == JacobiMeasure(expected_exponent, expected_exponent)


def test_jacobi_constructor():
    space = FunctionSpace.jacobi(0.5, 1.5, 5, a=0.0, b=2.0)
    assert space.basis.measure == JacobiMeasure(0.5, 1.5)
    assert space.domain == UnitInterval.Parameters(a=0.0, b=2.0)
    assert space.n_basis == 5


def test_hermite_constructor():
    space = FunctionSpace.hermite(4, loc=1.0, scale=2.0)
    assert isinstance(space.basis.measure, ProbabilistsHermiteMeasure)
    assert space.domain == RealLine.Parameters(loc=1.0, scale=2.0)

    space = FunctionSpace.hermite(4, kind="phys")
    assert isinstance(space.basis.measure, PhysicistsHermiteMeasure)

    with pytest.raises(ValueError, match="Hermite kind must be"):
        FunctionSpace.hermite(4, kind="bogus")

    # density=True normalizes to a probability measure, for either kind.
    for kind in ("phys", "prob"):
        assert FunctionSpace.hermite(4, kind=kind, density=True).basis.density is True
        assert FunctionSpace.hermite(4, kind=kind).basis.density is False


def test_laguerre_constructor():
    space = FunctionSpace.laguerre(4, rate=2.0, start=1.0)
    assert isinstance(space.basis.measure, LaguerreMeasure)
    assert space.domain == HalfLine.Parameters(rate=2.0, start=1.0)
