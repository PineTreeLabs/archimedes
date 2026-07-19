import math

import numpy as np
import pytest
from scipy.special import beta as beta_fn
from scipy.special import roots_hermite, roots_jacobi, roots_laguerre

import archimedes as arc
from archimedes.experimental.quadrature import QuadratureRule
from archimedes.experimental.quadrature._quadrature_rule import (
    _HermiteFamily,
    _JacobiFamily,
    _LaguerreFamily,
    _LegendreFamily,
    gauss_legendre,
    gauss_lobatto,
    gauss_radau,
)

# -- QuadratureRule machinery --


def test_len_and_shape_validation():
    rule = gauss_legendre(5)
    assert len(rule) == 5

    with pytest.raises(ValueError):
        QuadratureRule(
            nodes=np.array([1.0, 2.0]),
            weights=np.array([1.0]),
            name="bad",
            family=_LegendreFamily(),
        )


def test_scaled_points_weights_identity():
    rule = gauss_legendre(5)
    np.testing.assert_array_equal(rule.scaled_points(), rule.nodes)
    np.testing.assert_array_equal(rule.scaled_weights(), rule.weights)


def test_scaled_points_weights_interval():
    rule = gauss_legendre(5)
    a, b = -2.0, 5.0
    x = rule.scaled_points(a, b)
    w = rule.scaled_weights(a, b)
    assert x[0] > a
    assert x[-1] < b
    assert np.isclose(np.sum(w), b - a)


def test_dot_matches_integrate():
    rule = gauss_legendre(4)
    values = np.cos(rule.nodes)
    assert np.isclose(rule.dot(values), rule.integrate(np.cos))


def test_dot_axis():
    rule = gauss_legendre(4)
    values = np.cos(rule.nodes)
    scalar = rule.dot(values)

    # 2D values, nodes along the last axis (default)
    stacked = np.stack([values, 2 * values])
    result_last = rule.dot(stacked, axis=-1)
    np.testing.assert_allclose(result_last, [scalar, 2 * scalar])

    # nodes along the first axis
    result_first = rule.dot(stacked.T, axis=0)
    np.testing.assert_allclose(result_first, [scalar, 2 * scalar])


def test_dot_errors():
    rule = gauss_legendre(4)
    with pytest.raises(ValueError):
        rule.dot(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        rule.dot(np.zeros(len(rule) + 1))


# -- _QuadratureFamily implementations --


def test_legendre_family():
    family = _LegendreFamily()
    assert family.reference_domain == (-1.0, 1.0)
    np.testing.assert_array_equal(family.weight(np.array([-0.5, 0.5])), [1.0, 1.0])

    assert family.affine_params() == (1.0, 0.0)

    with pytest.raises(ValueError):
        family.affine_params(a=0.0)
    with pytest.raises(ValueError):
        family.affine_params(b=1.0)
    with pytest.raises(ValueError):
        family.affine_params(a=-np.inf, b=1.0)


@pytest.mark.parametrize("alpha,beta", [(-1.0, 0.0), (0.0, -1.0)])
def test_jacobi_family_invalid_parameters(alpha, beta):
    with pytest.raises(ValueError):
        _JacobiFamily(alpha=alpha, beta=beta)


def test_jacobi_family_weight_and_shared_affine_params():
    family = _JacobiFamily(alpha=1.0, beta=2.0)
    assert family.reference_domain == (-1.0, 1.0)

    x = np.array([0.0, 0.5])
    expected = (1 - x) ** 1.0 * (1 + x) ** 2.0
    np.testing.assert_allclose(family.weight(x), expected)

    # Shares the interval mapping with _LegendreFamily
    assert family.affine_params(0.0, 2.0) == _LegendreFamily().affine_params(0.0, 2.0)


def test_laguerre_family():
    family = _LaguerreFamily()
    assert family.reference_domain == (0.0, np.inf)
    np.testing.assert_allclose(family.weight(np.array([0.0, 1.0])), [1.0, np.exp(-1.0)])

    assert family.affine_params() == (1.0, 0.0)

    scale, shift = family.affine_params(rate=2.0, start=1.0)
    assert np.isclose(scale, 0.5)
    assert np.isclose(shift, 1.0)

    with pytest.raises(ValueError):
        family.affine_params(rate=-1.0)


def test_hermite_family():
    family = _HermiteFamily()
    assert family.reference_domain == (-np.inf, np.inf)
    np.testing.assert_allclose(family.weight(np.array([0.0, 1.0])), [1.0, np.exp(-1.0)])

    assert family.affine_params() == (1.0, 0.0)

    scale, shift = family.affine_params(mean=1.0, std=2.0)
    assert np.isclose(scale, 2.0)
    assert np.isclose(shift, 1.0)

    with pytest.raises(ValueError):
        family.affine_params(std=-1.0)


# -- Known rules: node generation and exact integration --


def test_gauss_legendre():
    rule = gauss_legendre(5)
    assert rule.degree == 9
    assert len(rule) == 5
    assert np.isclose(np.sum(rule.weights), 2.0)

    # Exact for polynomials up to degree 2n - 1 = 9
    assert np.isclose(rule.integrate(lambda x: x**8), 2 / 9)
    assert np.isclose(rule.integrate(lambda x: x**9), 0.0, atol=1e-12)


def test_gauss_legendre_scaled_domain():
    rule = gauss_legendre(5)
    a, b = -2.0, 5.0
    integral = rule.integrate(lambda x: x**2, a, b)
    expected = (b**3 - a**3) / 3
    assert np.isclose(integral, expected)


@pytest.mark.parametrize("endpoint,fixed_node", [("left", -1.0), ("right", 1.0)])
def test_gauss_radau(endpoint, fixed_node):
    rule = gauss_radau(5, endpoint=endpoint)
    assert len(rule) == 5
    assert np.isclose(np.sum(rule.weights), 2.0)
    idx = 0 if endpoint == "left" else -1
    assert np.isclose(rule.nodes[idx], fixed_node)

    # Exact for polynomials up to degree 2n - 2 = 8
    assert np.isclose(rule.integrate(lambda x: x**8), 2 / 9)


def test_gauss_radau_edge_cases():
    with pytest.raises(ValueError):
        gauss_radau(0)
    with pytest.raises(ValueError):
        gauss_radau(5, endpoint="middle")

    rule = gauss_radau(1)
    assert len(rule) == 1
    assert rule.nodes[0] == -1.0
    assert np.isclose(np.sum(rule.weights), 2.0)


def test_gauss_lobatto():
    rule = gauss_lobatto(5)
    assert len(rule) == 5
    assert np.isclose(np.sum(rule.weights), 2.0)
    assert rule.nodes[0] == -1.0
    assert rule.nodes[-1] == 1.0

    # Exact for polynomials up to degree 2n - 3 = 7
    assert np.isclose(rule.integrate(lambda x: x**6), 2 / 7)


def test_gauss_lobatto_edge_cases():
    with pytest.raises(ValueError):
        gauss_lobatto(1)

    rule = gauss_lobatto(2)
    assert len(rule) == 2
    np.testing.assert_array_equal(rule.nodes, [-1.0, 1.0])
    np.testing.assert_array_equal(rule.weights, [1.0, 1.0])


def test_gauss_jacobi_exact_moment():
    n, alpha, beta = 5, 1.0, 2.0
    x, w = roots_jacobi(n, alpha, beta)
    rule = QuadratureRule(
        x, w, name="gauss_jacobi_5", family=_JacobiFamily(alpha=alpha, beta=beta)
    )

    # Zeroth moment of the Jacobi weight has a closed form in the Beta function
    expected = 2 ** (alpha + beta + 1) * beta_fn(alpha + 1, beta + 1)
    assert np.isclose(rule.integrate(lambda x: np.ones_like(x)), expected)


def test_gauss_laguerre_exact_moments():
    n = 5
    x, w = roots_laguerre(n)
    rule = QuadratureRule(x, w, name="gauss_laguerre_5", family=_LaguerreFamily())

    # Exact for polynomials up to degree 2n - 1
    for k in range(2 * n):
        integral = rule.integrate(lambda x, k=k: x**k)
        assert np.isclose(integral, math.factorial(k))


def test_gauss_laguerre_rate_scaling():
    n = 5
    x, w = roots_laguerre(n)
    rule = QuadratureRule(x, w, name="gauss_laguerre_5", family=_LaguerreFamily())

    rate = 2.0
    integral = rule.integrate(lambda x: np.ones_like(x), rate=rate)
    assert np.isclose(integral, 1 / rate)


def test_gauss_hermite_exact_moments():
    n = 5
    x, w = roots_hermite(n)
    rule = QuadratureRule(x, w, name="gauss_hermite_5", family=_HermiteFamily())

    assert np.isclose(rule.integrate(lambda x: np.ones_like(x)), np.sqrt(np.pi))
    assert np.isclose(rule.integrate(lambda x: x**2), np.sqrt(np.pi) / 2)
    assert np.isclose(rule.integrate(lambda x: x**3), 0.0, atol=1e-10)


def test_gauss_hermite_mean_std_scaling():
    n = 5
    x, w = roots_hermite(n)
    rule = QuadratureRule(x, w, name="gauss_hermite_5", family=_HermiteFamily())

    mean, std = 1.0, 2.0
    integral = rule.integrate(lambda x: np.ones_like(x), mean=mean, std=std)
    assert np.isclose(integral, std * np.sqrt(np.pi))


# -- arc.compile integration: symbolic domain/measure parameters --


def test_compile_symbolic_interval():
    rule = gauss_legendre(5)

    @arc.compile
    def quad(a, b):
        return rule.integrate(lambda x: x**2, a, b)

    result = quad(0.0, 2.0)
    assert np.isclose(float(result), 8 / 3)


def test_compile_symbolic_rate():
    x, w = roots_laguerre(5)
    rule = QuadratureRule(x, w, name="gauss_laguerre_5", family=_LaguerreFamily())

    @arc.compile
    def quad(rate):
        return rule.dot(np.ones_like(rule.nodes), rate=rate)

    result = quad(2.0)
    assert np.isclose(float(result), 0.5)


def test_compile_vector_integrand():
    rule = gauss_legendre(5)

    @arc.compile
    def quad(a, b):
        def f(x):
            return np.stack([x, x**2])

        return rule.integrate(f, a, b, axis=-1)

    result = np.asarray(quad(-1.0, 1.0))
    np.testing.assert_allclose(result, [0.0, 2 / 3], atol=1e-10)
