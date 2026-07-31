# ruff: noqa: N806  (M is the conventional name for a Gram matrix)
import numpy as np
import pytest

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import (
    FourierBasis,
    FunctionSpace,
    OrthogonalPolynomialBasis,
    ProductParameters,
    TensorBasis,
)
from archimedes.measure import LegendreMeasure, UnitInterval

# -- construction validation --


def test_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind must be"):
        FourierBasis(3, kind="triangle")


def test_n_basis_must_be_positive():
    with pytest.raises(ValueError):
        FourierBasis(0)
    with pytest.raises(ValueError):
        FourierBasis(-1, kind="cosine")


def test_full_requires_odd_n_basis():
    with pytest.raises(ValueError, match="odd"):
        FourierBasis(4, kind="full")
    FourierBasis(5, kind="full")  # does not raise


def test_negative_deriv_rejected():
    basis = FourierBasis(5)
    with pytest.raises(ValueError):
        basis.evaluate(np.array([0.0]), deriv=-1)


@pytest.mark.parametrize(
    "kind,n_basis,expected",
    [
        ("full", 1, 0),
        ("full", 5, 2),
        ("cosine", 1, 0),
        ("cosine", 4, 3),
        ("sine", 3, 3),
    ],
)
def test_max_mode(kind, n_basis, expected):
    assert FourierBasis(n_basis, kind=kind).max_mode == expected


def test_measures_is_legendre():
    basis = FourierBasis(5)
    assert basis.measures == (LegendreMeasure(),)


# -- values at hand-computed points, on the reference domain [-1, 1] --
# (mass = LegendreMeasure().mass() = 2, so const = 1/sqrt(2), mode factor
# sqrt(2/2) = 1; theta = pi * x, so x=0,0.5,1 give theta=0,pi/2,pi.)


def test_full_values_at_known_points():
    basis = FourierBasis(5, kind="full")  # N=2: {1, cos, sin, cos2, sin2}
    x = np.array([0.0, 0.5, 1.0])
    phi = basis.evaluate(x)
    c = 1.0 / np.sqrt(2.0)
    expected = np.array(
        [
            [c, 1.0, 0.0, 1.0, 0.0],
            [c, 0.0, 1.0, -1.0, 0.0],
            [c, -1.0, 0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(phi, expected, atol=1e-12)


def test_cosine_values_at_known_points():
    basis = FourierBasis(3, kind="cosine")  # N=2: {1, cos, cos2}
    x = np.array([0.0, 0.5, 1.0])
    phi = basis.evaluate(x)
    c = 1.0 / np.sqrt(2.0)
    expected = np.array([[c, 1.0, 1.0], [c, 0.0, -1.0], [c, -1.0, 1.0]])
    np.testing.assert_allclose(phi, expected, atol=1e-12)


def test_sine_values_at_known_points():
    basis = FourierBasis(2, kind="sine")  # N=2: {sin, sin2}
    x = np.array([0.0, 0.5, 1.0])
    phi = basis.evaluate(x)
    expected = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(phi, expected, atol=1e-12)


def test_evaluate_shape():
    for kind, n_basis in [("full", 7), ("cosine", 4), ("sine", 4)]:
        basis = FourierBasis(n_basis, kind=kind)
        x = np.linspace(-1, 1, 11)
        assert basis.evaluate(x).shape == (len(x), n_basis)


# -- derivatives vs. finite differences --


@pytest.mark.parametrize("kind,n_basis", [("full", 7), ("cosine", 4), ("sine", 4)])
@pytest.mark.parametrize("deriv", [1, 2, 3])
def test_derivative_matches_finite_difference(kind, n_basis, deriv):
    basis = FourierBasis(n_basis, kind=kind)
    x = np.linspace(-0.9, 0.9, 13)
    h = 1e-6
    dphi = basis.evaluate(x, deriv=deriv)
    lo = basis.evaluate(x - h, deriv=deriv - 1)
    hi = basis.evaluate(x + h, deriv=deriv - 1)
    dphi_fd = (hi - lo) / (2 * h)
    np.testing.assert_allclose(dphi, dphi_fd, atol=1e-4)


# -- orthonormality via the basis's own default quadrature --


@pytest.mark.parametrize("kind,n_basis", [("full", 7), ("cosine", 4), ("sine", 4)])
@pytest.mark.parametrize("density", [False, True])
def test_orthonormal_on_reference_domain(kind, n_basis, density):
    basis = FourierBasis(n_basis, kind=kind, density=density)
    rule = basis.default_quadrature()
    phi = basis.evaluate(rule.nodes)
    w = rule.scaled_weights(density=density)
    M = phi.T @ (w[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(n_basis), atol=1e-10)


@pytest.mark.parametrize("kind,n_basis", [("full", 7), ("cosine", 4), ("sine", 4)])
def test_orthonormal_on_mapped_domain(kind, n_basis):
    a, b = 2.0, 7.0
    basis = FourierBasis(n_basis, kind=kind)
    rule = basis.default_quadrature()
    x = rule.scaled_points(a, b)
    w = rule.scaled_weights(a, b)
    phi = basis.evaluate(x, a=a, b=b)
    M = phi.T @ (w[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(n_basis), atol=1e-10)


def test_density_orthonormal_against_probability_measure():
    a, b = -3.0, 4.0
    basis = FourierBasis(5, kind="full", density=True)
    rule = basis.default_quadrature()
    x = rule.scaled_points(a, b)
    w = rule.scaled_weights(a, b, density=True)
    phi = basis.evaluate(x, a=a, b=b)
    M = phi.T @ (w[:, None] * phi)
    np.testing.assert_allclose(M, np.eye(5), atol=1e-10)


def test_density_rescales_by_sqrt_mass_relative_to_raw():
    a, b = -1.0, 5.0
    raw = FourierBasis(5, kind="full")
    density = FourierBasis(5, kind="full", density=True)
    x = np.linspace(a, b, 11)
    phi_raw = raw.evaluate(x, a=a, b=b)
    phi_density = density.evaluate(x, a=a, b=b)
    mass = LegendreMeasure().mass(a, b)
    np.testing.assert_allclose(phi_density, phi_raw * np.sqrt(mass), atol=1e-10)


def test_density_defaults_to_false():
    assert FourierBasis(3).density is False


# -- static (NumPy) vs. dynamic (symbolic, via arc.compile) equivalence --


def test_static_and_dynamic_evaluation_agree():
    basis = FourierBasis(5, kind="full")
    x = np.linspace(-1, 1, 9)
    static_phi = basis.evaluate(x)

    @arc.compile
    def traced(x):
        assert isinstance(x, SymbolicArray)
        return basis.evaluate(np.atleast_1d(x))

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-12)


def test_static_and_dynamic_derivative_agree():
    basis = FourierBasis(5, kind="full")
    x = np.linspace(-0.8, 0.8, 7)
    static_dphi = basis.evaluate(x, deriv=1)

    @arc.compile
    def traced(x):
        return basis.evaluate(np.atleast_1d(x), deriv=1)

    dynamic_dphi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_dphi, dynamic_dphi, atol=1e-12)


# -- tensor-product composition: FourierBasis needs no special-casing --


class TestTensorComposition:
    def _space(self):
        fourier = FunctionSpace.fourier(5, a=-1.0, b=1.0)
        legendre = FunctionSpace.legendre(4, a=0.0, b=2.0)
        return FunctionSpace.tensor(fourier, legendre)

    def test_fourier_x_legendre_projects_a_separable_function(self):
        space = self._space()

        def f(x):
            return np.cos(np.pi * x[:, 0]) * x[:, 1] ** 2

        u = space.project(f)
        x = np.stack([np.linspace(-0.9, 0.9, 9), np.linspace(0.1, 1.9, 9)], axis=-1)
        np.testing.assert_allclose(u(x), f(x), atol=1e-8)

    def test_fourier_partial_derivative_in_tensor_product(self):
        # deriv=(1, 0): only the Fourier factor differentiates. Its own
        # `_derivative_basis` never raises (unlike a polynomial factor's),
        # so this exercises TensorBasis's per-dimension delegation against
        # that atypical, parity-based logic.
        space = self._space()

        def f(x):
            return np.cos(np.pi * x[:, 0]) * x[:, 1] ** 2

        def df_dx0(x):
            return -np.pi * np.sin(np.pi * x[:, 0]) * x[:, 1] ** 2

        u = space.project(f)
        du = u.derivative((1, 0))
        x = np.stack([np.linspace(-0.9, 0.9, 9), np.linspace(0.1, 1.9, 9)], axis=-1)
        np.testing.assert_allclose(du(x), df_dx0(x), atol=1e-7)
        np.testing.assert_allclose(du(x), u(x, deriv=(1, 0)), atol=1e-7)

    def test_manual_tensor_basis_construction_matches_sugar(self):
        fourier_basis = FourierBasis(5, kind="full")
        legendre_basis = OrthogonalPolynomialBasis(LegendreMeasure(), 4)
        manual = FunctionSpace(
            TensorBasis((fourier_basis, legendre_basis)),
            ProductParameters(
                dims=(
                    UnitInterval.Parameters(a=-1.0, b=1.0),
                    UnitInterval.Parameters(a=0.0, b=2.0),
                )
            ),
        )
        sugar = self._space()
        assert sugar.basis == manual.basis
        assert sugar.domain == manual.domain
