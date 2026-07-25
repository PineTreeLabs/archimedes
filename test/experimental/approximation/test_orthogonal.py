# ruff: noqa: N806  (M, K are the conventional names for these matrices)
import numpy as np
import pytest
from scipy.special import eval_legendre

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import OrthogonalPolynomialBasis
from archimedes.measure import HermiteMeasure, LaguerreMeasure, LegendreMeasure
from archimedes.quadrature import gauss_hermite, gauss_laguerre, gauss_legendre


def test_n_basis_validation():
    with pytest.raises(ValueError):
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=0)


def test_negative_deriv_rejected():
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


def test_derivative_matches_finite_difference():
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=6)
    x = np.linspace(-0.9, 0.9, 13)
    h = 1e-6
    dphi = basis.evaluate(x, deriv=1)
    dphi_fd = (basis.evaluate(x + h) - basis.evaluate(x - h)) / (2 * h)
    np.testing.assert_allclose(dphi, dphi_fd, atol=1e-5)


def test_second_derivative_matches_finite_difference_of_first():
    # Not hand-derived per family -- differentiating the recurrence twice
    # falls out of the same generic implementation.
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=6)
    x = np.linspace(-0.9, 0.9, 13)
    h = 1e-5
    d2phi = basis.evaluate(x, deriv=2)
    d2phi_fd = (basis.evaluate(x + h, deriv=1) - basis.evaluate(x - h, deriv=1)) / (
        2 * h
    )
    np.testing.assert_allclose(d2phi, d2phi_fd, atol=1e-3)


def test_orthonormal_on_reference_domain():
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=6)
    rule = gauss_legendre(15)
    phi = basis.evaluate(rule.nodes)
    M = phi.T @ (rule.weights[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(6), atol=1e-10)


def test_orthonormal_on_mapped_domain():
    a, b = 2.0, 7.0
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=6)
    rule = gauss_legendre(15)
    x = rule.scaled_points(a, b)
    w = rule.scaled_weights(a, b)
    phi = basis.evaluate(x, a=a, b=b)
    M = phi.T @ (w[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(6), atol=1e-10)


# -- genericity: same class, no family-specific code, for other measures --


def test_generic_orthonormality_hermite():
    basis = OrthogonalPolynomialBasis(HermiteMeasure(), n_basis=5)
    rule = gauss_hermite(15, kind="phys")
    phi = basis.evaluate(rule.nodes)
    M = phi.T @ (rule.weights[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(5), atol=1e-8)


def test_generic_orthonormality_laguerre():
    basis = OrthogonalPolynomialBasis(LaguerreMeasure(), n_basis=5)
    rule = gauss_laguerre(15)
    phi = basis.evaluate(rule.nodes)
    M = phi.T @ (rule.weights[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(5), atol=1e-8)


# -- static (NumPy) vs. dynamic (symbolic, via arc.compile) equivalence --


def test_static_and_dynamic_evaluation_agree():
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5)
    x = np.linspace(-1, 1, 9)
    static_phi = basis.evaluate(x)

    @arc.compile
    def traced(x):
        assert isinstance(x, SymbolicArray)
        return basis.evaluate(x)

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-12)


def test_static_and_dynamic_derivative_agree():
    basis = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5)
    x = np.linspace(-0.8, 0.8, 7)
    static_dphi = basis.evaluate(x, deriv=1)

    @arc.compile
    def traced(x):
        return basis.evaluate(x, deriv=1)

    dynamic_dphi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_dphi, dynamic_dphi, atol=1e-12)
