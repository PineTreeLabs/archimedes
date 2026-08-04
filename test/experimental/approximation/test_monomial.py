import numpy as np
import pytest

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import MonomialBasis
from archimedes.measure import UnitInterval

# -- construction validation --


def test_n_basis_must_be_positive():
    with pytest.raises(ValueError):
        MonomialBasis(0)
    with pytest.raises(ValueError):
        MonomialBasis(-1)


def test_negative_deriv_rejected():
    basis = MonomialBasis(3)
    with pytest.raises(ValueError):
        basis.evaluate(np.array([0.0]), deriv=-1)


def test_invalid_side_rejected():
    basis = MonomialBasis(3)
    with pytest.raises(ValueError, match="side must be"):
        basis.evaluate(np.array([0.0]), side="up")


def test_density_is_not_a_field():
    # Unlike OrthogonalPolynomialBasis/FourierBasis, this family has no
    # probability-density normalization -- it inherits `Basis.density`'s
    # fixed False default rather than exposing its own field.
    basis = MonomialBasis(3)
    assert basis.density is False
    with pytest.raises(TypeError):
        MonomialBasis(3, density=True)


# -- values on the reference domain [-1, 1] --


@pytest.mark.parametrize("n_basis", [1, 2, 6])
def test_values_match_vandermonde(n_basis):
    basis = MonomialBasis(n_basis)
    x = np.linspace(-1, 1, 11)
    phi = basis.evaluate(x)
    assert phi.shape == (len(x), n_basis)
    np.testing.assert_allclose(phi, np.vander(x, n_basis, increasing=True), atol=1e-12)


def test_side_left_and_right_agree():
    basis = MonomialBasis(4)
    x = np.linspace(-1, 1, 7)
    np.testing.assert_allclose(
        basis.evaluate(x, side="left"), basis.evaluate(x, side="right")
    )


# -- values under an affine domain remap --


def test_values_on_mapped_domain():
    a, b = 2.0, 7.0
    basis = MonomialBasis(5)
    scale, shift = UnitInterval().affine_params(a, b)
    x = np.linspace(a, b, 9)
    t = (x - shift) / scale
    phi = basis.evaluate(x, a=a, b=b)
    np.testing.assert_allclose(phi, np.vander(t, 5, increasing=True), atol=1e-10)


# -- derivatives --


@pytest.mark.parametrize("deriv", [1, 2, 3])
def test_derivative_matches_finite_difference(deriv):
    basis = MonomialBasis(7)
    x = np.linspace(-0.9, 0.9, 13)
    h = 1e-6
    dphi = basis.evaluate(x, deriv=deriv)
    lo = basis.evaluate(x - h, deriv=deriv - 1)
    hi = basis.evaluate(x + h, deriv=deriv - 1)
    dphi_fd = (hi - lo) / (2 * h)
    np.testing.assert_allclose(dphi, dphi_fd, atol=1e-3)


@pytest.mark.parametrize("deriv", [1, 2, 3])
def test_derivative_matches_closed_form(deriv):
    # d^m/dx^m x^k = k! / (k - m)! * x^(k - m), zero for k < m.
    basis = MonomialBasis(6)
    x = np.linspace(-0.9, 0.9, 9)
    dphi = basis.evaluate(x, deriv=deriv)
    for k in range(6):
        if k < deriv:
            expected = np.zeros_like(x)
        else:
            coeff = 1.0
            for j in range(k - deriv + 1, k + 1):
                coeff *= j
            expected = coeff * x ** (k - deriv)
        np.testing.assert_allclose(dphi[:, k], expected, atol=1e-10)


def test_derivative_under_domain_remap_matches_finite_difference():
    a, b = 2.0, 7.0
    basis = MonomialBasis(5)
    x = np.linspace(a + 0.1, b - 0.1, 9)
    h = 1e-4
    dphi = basis.evaluate(x, deriv=1, a=a, b=b)
    dphi_fd = (basis.evaluate(x + h, a=a, b=b) - basis.evaluate(x - h, a=a, b=b)) / (
        2 * h
    )
    np.testing.assert_allclose(dphi, dphi_fd, atol=1e-6)


def test_deriv_at_or_past_degree_is_zero():
    basis = MonomialBasis(4)
    x = np.linspace(-1, 1, 5)
    for deriv in (4, 5, 8):
        phi = basis.evaluate(x, deriv=deriv)
        np.testing.assert_allclose(phi, 0.0, atol=1e-12)


# -- _product_basis / _derivative_basis / _integral_basis degree arithmetic --


def test_product_basis_size():
    b1, b2 = MonomialBasis(3), MonomialBasis(4)
    product = b1._product_basis(b2)
    assert isinstance(product, MonomialBasis)
    assert product.n_basis == 6


def test_product_basis_rejects_other_family():
    from archimedes.experimental.approximation import FourierBasis

    with pytest.raises(ValueError):
        MonomialBasis(3)._product_basis(FourierBasis(3, kind="cosine"))


def test_derivative_basis_size():
    basis = MonomialBasis(5)
    assert basis._derivative_basis(0) is basis
    assert basis._derivative_basis(2).n_basis == 3


def test_derivative_basis_rejects_invalid_order():
    basis = MonomialBasis(3)
    with pytest.raises(ValueError):
        basis._derivative_basis(-1)
    with pytest.raises(ValueError, match="identically zero"):
        basis._derivative_basis(3)
    with pytest.raises(ValueError):
        basis._derivative_basis(5)


def test_integral_basis_size():
    basis = MonomialBasis(4)
    assert basis._integral_basis(0) is basis
    assert basis._integral_basis(2).n_basis == 6


def test_integral_basis_rejects_negative_order():
    with pytest.raises(ValueError):
        MonomialBasis(3)._integral_basis(-1)


# -- round-trip through Function.derivative()/.integral() --


def test_function_derivative_and_integral_round_trip():
    from archimedes.experimental.approximation import FunctionSpace

    space = FunctionSpace.monomial(6, a=-2.0, b=3.0)

    def f(x):
        return 2.0 + 3.0 * x - x**2 + 0.5 * x**3

    def df(x):
        return 3.0 - 2.0 * x + 1.5 * x**2

    u = space.project(f)
    x = np.linspace(-2.0, 3.0, 11)
    np.testing.assert_allclose(u(x), f(x), atol=1e-8)

    du = u.derivative()
    np.testing.assert_allclose(du(x), df(x), atol=1e-6)

    di = du.integral()
    # Integration pins the antiderivative to vanish at the left endpoint,
    # so it recovers df up to the constant f(a) that differentiation lost.
    np.testing.assert_allclose(di(x), f(x) - f(-2.0), atol=1e-6)


# -- static (NumPy) vs. dynamic (symbolic, via arc.compile) equivalence --


def test_static_and_dynamic_evaluation_agree():
    basis = MonomialBasis(5)
    x = np.linspace(-1, 1, 9)
    static_phi = basis.evaluate(x)

    @arc.compile
    def traced(x):
        assert isinstance(x, SymbolicArray)
        return basis.evaluate(np.atleast_1d(x))

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-12)


def test_static_and_dynamic_derivative_agree():
    basis = MonomialBasis(5)
    x = np.linspace(-0.8, 0.8, 7)
    static_dphi = basis.evaluate(x, deriv=1)

    @arc.compile
    def traced(x):
        return basis.evaluate(np.atleast_1d(x), deriv=1)

    dynamic_dphi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_dphi, dynamic_dphi, atol=1e-12)
