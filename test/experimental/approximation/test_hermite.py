import numpy as np
import pytest

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import CubicHermiteBasis, LagrangeBasis
from archimedes.measure import UnitInterval


@pytest.fixture
def basis():
    return CubicHermiteBasis()


def test_construction_takes_no_arguments(basis):
    assert basis.n_basis == 4


def test_equal_and_hashable(basis):
    other = CubicHermiteBasis()
    assert basis == other
    assert hash(basis) == hash(other)


def test_dof_order(basis):
    np.testing.assert_array_equal(basis._dof_order, [0, 1, 0, 1])


# -- cardinal-like properties at the endpoints --


def test_endpoint_values_and_derivatives(basis):
    # phi_00, phi_01 are value-type: 1/0 (or 0/1) at the owning endpoint,
    # zero derivative at both. phi_10, phi_11 are derivative-type: zero
    # value at both endpoints, unit derivative at the owning one.
    t = np.array([-1.0, 1.0])
    phi = basis.evaluate(t, deriv=0)
    np.testing.assert_allclose(
        phi, np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]), atol=1e-12
    )

    dphi = basis.evaluate(t, deriv=1)
    np.testing.assert_allclose(
        dphi, np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]), atol=1e-12
    )


def test_reproduces_exact_cubic_on_reference_domain(basis):
    # Any cubic is spanned exactly by value + slope at both endpoints.
    def f(t):
        return 1.0 - 2.0 * t + 3.0 * t**2 - 4.0 * t**3

    def df(t):
        return -2.0 + 6.0 * t - 12.0 * t**2

    coeffs = np.array([f(-1.0), df(-1.0), f(1.0), df(1.0)])
    t = np.linspace(-1.0, 1.0, 13)
    np.testing.assert_allclose(basis.evaluate(t) @ coeffs, f(t), atol=1e-10)
    np.testing.assert_allclose(basis.evaluate(t, deriv=1) @ coeffs, df(t), atol=1e-9)


def test_negative_derivative_order_rejected(basis):
    with pytest.raises(ValueError, match="deriv must be >= 0"):
        basis.evaluate(np.array([0.0]), deriv=-1)


def test_vanishes_beyond_polynomial_degree(basis):
    # Degree 3, so the 4th derivative and beyond are identically zero.
    x = np.linspace(-0.9, 0.9, 5)
    for deriv in (4, 7):
        got = basis.evaluate(x, deriv=deriv)
        assert got.shape == (len(x), 4)
        np.testing.assert_array_equal(got, 0.0)


# -- physical-domain scale factor: the central new piece of math --


class TestDomainMapping:
    """Regression coverage for the per-column `scale**(_dof_order - deriv)`
    factor (see the class docstring): a naive uniform `scale**deriv`, as
    every homogeneous family uses, would be wrong for the derivative-type
    columns as soon as the physical element width isn't 2 (i.e. `scale !=
    1`)."""

    @pytest.fixture
    def poly(self):
        # An arbitrary cubic and its derivatives through order 3.
        c = np.array([1.0, -2.0, 3.0, -4.0])  # 1 - 2x + 3x^2 - 4x^3
        return [np.polynomial.Polynomial(c).deriv(k) for k in range(4)]

    @pytest.mark.parametrize("a,b", [(2.0, 9.0), (-5.0, -1.0), (0.0, 0.25)])
    @pytest.mark.parametrize("deriv", [0, 1, 2, 3])
    def test_exact_on_mapped_domain(self, basis, poly, a, b, deriv):
        coeffs = np.array([poly[0](a), poly[1](a), poly[0](b), poly[1](b)])
        x = np.linspace(a, b, 11)
        got = basis.evaluate(x, deriv=deriv, a=a, b=b) @ coeffs
        np.testing.assert_allclose(got, poly[deriv](x), atol=1e-8)

    def test_domain_mapping_preserves_endpoint_pattern(self, basis):
        a, b = 3.0, 11.0
        scale, shift = UnitInterval().affine_params(a, b)
        x = scale * np.array([-1.0, 1.0]) + shift
        phi = basis.evaluate(x, deriv=0, a=a, b=b)
        np.testing.assert_allclose(
            phi, np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]), atol=1e-10
        )


# -- boundary DOFs --


def test_boundary_dofs(basis):
    assert basis.boundary_dofs(0) == (0, 2)
    assert basis.boundary_dofs(1) == (1, 3)
    assert basis.boundary_dofs(2) == (None, None)
    # Default order is 0, matching the value-DOF convention every other
    # family uses.
    assert basis.boundary_dofs() == (0, 2)


# -- derivative basis: the one family that crosses into a different kind --


class TestDerivativeBasis:
    def test_zeroth_derivative_is_self(self, basis):
        assert basis._derivative_basis(0) is basis

    @pytest.mark.parametrize("deriv,expected_n", [(1, 3), (2, 2), (3, 1)])
    def test_crosses_into_lagrange(self, basis, deriv, expected_n):
        derived = basis._derivative_basis(deriv)
        assert isinstance(derived, LagrangeBasis)
        assert derived.n_basis == expected_n

    def test_negative_derivative_order_rejected(self, basis):
        with pytest.raises(ValueError, match="deriv must be >= 0"):
            basis._derivative_basis(-1)

    def test_rejects_derivative_at_or_past_degree(self, basis):
        with pytest.raises(ValueError, match="at or past the degree"):
            basis._derivative_basis(4)

    def test_derivative_reproduces_polynomial_derivative(self, basis):
        # The crossed-into Lagrange basis should still exactly represent
        # the cubic's first derivative (a quadratic) at its own nodes.
        def f(t):
            return 1.0 - 2.0 * t + 3.0 * t**2 - 4.0 * t**3

        def df(t):
            return -2.0 + 6.0 * t - 12.0 * t**2

        coeffs = np.array([f(-1.0), df(-1.0), f(1.0), df(1.0)])
        derived = basis._derivative_basis(1)
        t = np.linspace(-1.0, 1.0, 9)
        deriv_values = basis.evaluate(derived.reference_nodes, deriv=1) @ coeffs
        got = derived.evaluate(t) @ deriv_values
        np.testing.assert_allclose(got, df(t), atol=1e-9)


def test_product_basis_not_implemented(basis):
    with pytest.raises(NotImplementedError):
        basis._product_basis(basis)


# -- static (NumPy) vs. dynamic (symbolic, via arc.compile) equivalence --


@pytest.mark.parametrize("deriv", [0, 1, 2, 3])
def test_static_and_dynamic_evaluation_agree(basis, deriv):
    x = np.array([-1.0, -0.4, 0.0, 0.55, 1.0])
    static_phi = basis.evaluate(x, deriv=deriv)

    @arc.compile
    def traced(xi):
        assert isinstance(xi, SymbolicArray)
        return basis.evaluate(np.atleast_1d(xi), deriv=deriv)

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-10)
