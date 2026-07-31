import math

import numpy as np
import pytest
from scipy.special import (
    roots_hermite,
    roots_hermitenorm,
    roots_jacobi,
    roots_laguerre,
    roots_legendre,
)

from archimedes.measure import (
    JacobiMeasure,
    LaguerreMeasure,
    LegendreMeasure,
    PhysicistsHermiteMeasure,
    ProbabilistsHermiteMeasure,
)
from archimedes.quadrature import golub_welsch, golub_welsch_rule

N_VALUES = [1, 2, 3, 5, 10]

# -- consistency with scipy.special.roots_* --


@pytest.mark.parametrize("n", N_VALUES)
def test_golub_welsch_rule_legendre_matches_scipy(n):
    rule = golub_welsch_rule(LegendreMeasure(), n)
    nodes, weights = roots_legendre(n)
    np.testing.assert_allclose(rule.nodes, nodes, atol=1e-8)
    np.testing.assert_allclose(rule.weights, weights, atol=1e-8)


@pytest.mark.parametrize(
    "alpha,beta",
    [
        (0.0, 0.0),  # Legendre special case
        (-0.5, -0.5),  # Chebyshev 1st kind
        (0.5, 0.5),  # Chebyshev 2nd kind
        (1.0, 2.0),  # generic asymmetric
    ],
)
@pytest.mark.parametrize("n", N_VALUES)
def test_golub_welsch_rule_jacobi_matches_scipy(alpha, beta, n):
    rule = golub_welsch_rule(JacobiMeasure(alpha=alpha, beta=beta), n)
    nodes, weights = roots_jacobi(n, alpha, beta)
    np.testing.assert_allclose(rule.nodes, nodes, atol=1e-8)
    np.testing.assert_allclose(rule.weights, weights, atol=1e-8)


@pytest.mark.parametrize("n", N_VALUES)
def test_golub_welsch_rule_laguerre_matches_scipy(n):
    rule = golub_welsch_rule(LaguerreMeasure(), n)
    nodes, weights = roots_laguerre(n)
    np.testing.assert_allclose(rule.nodes, nodes, atol=1e-8)
    np.testing.assert_allclose(rule.weights, weights, atol=1e-8)


@pytest.mark.parametrize("n", N_VALUES)
def test_golub_welsch_rule_hermite_matches_scipy(n):
    rule = golub_welsch_rule(PhysicistsHermiteMeasure(), n)
    nodes, weights = roots_hermite(n)
    np.testing.assert_allclose(rule.nodes, nodes, atol=1e-8)
    np.testing.assert_allclose(rule.weights, weights, atol=1e-8)


@pytest.mark.parametrize("n", N_VALUES)
def test_golub_welsch_rule_hermitenorm_matches_scipy(n):
    rule = golub_welsch_rule(ProbabilistsHermiteMeasure(), n)
    nodes, weights = roots_hermitenorm(n)
    np.testing.assert_allclose(rule.nodes, nodes, atol=1e-8)
    np.testing.assert_allclose(rule.weights, weights, atol=1e-8)


# -- independent exactness checks (not just "matches scipy") --


def test_golub_welsch_rule_legendre_exact_to_2n_minus_1():
    rule = golub_welsch_rule(LegendreMeasure(), 5)
    # Exact for polynomials up to degree 2n - 1 = 9
    assert np.isclose(rule.integrate(lambda x: x**8), 2 / 9)
    assert np.isclose(rule.integrate(lambda x: x**9), 0.0, atol=1e-12)


def test_golub_welsch_rule_jacobi_exact_to_2n_minus_1():
    # rule.integrate(f) approximates int f(x) w(x) dx, so the weight is
    # already folded into the quadrature weights -- don't multiply by it
    # again here. int_{-1}^1 (1-x)(1+x)^2 x^5 dx, evaluated analytically.
    measure = JacobiMeasure(alpha=1.0, beta=2.0)
    rule = golub_welsch_rule(measure, 3)
    expected = 4 / 63
    assert np.isclose(rule.integrate(lambda x: x**5), expected)


def test_golub_welsch_rule_laguerre_exact_to_2n_minus_1():
    # int_0^inf e^-x x^k dx = k!
    rule = golub_welsch_rule(LaguerreMeasure(), 3)
    assert np.isclose(rule.integrate(lambda x: x**5), math.factorial(5))


def test_golub_welsch_rule_hermite_exact_to_2n_minus_1():
    # int_{-inf}^{inf} e^{-x^2} x^4 dx = 3 sqrt(pi) / 4
    rule = golub_welsch_rule(PhysicistsHermiteMeasure(), 3)
    assert np.isclose(rule.integrate(lambda x: x**4), 3 * np.sqrt(np.pi) / 4)


# -- golub_welsch / golub_welsch_rule machinery --


def test_golub_welsch_matches_golub_welsch_rule():
    measure = LegendreMeasure()
    alpha, beta = measure.recurrence_coeffs(6)
    nodes, weights = golub_welsch(alpha, beta)
    rule = golub_welsch_rule(measure, 6)
    np.testing.assert_allclose(rule.nodes, nodes)
    np.testing.assert_allclose(rule.weights, weights)


def test_golub_welsch_rule_default_name():
    rule = golub_welsch_rule(LegendreMeasure(), 4)
    assert rule.name == "golub_welsch_LegendreMeasure"


def test_golub_welsch_rule_custom_name():
    rule = golub_welsch_rule(LegendreMeasure(), 4, name="my_rule")
    assert rule.name == "my_rule"


def test_golub_welsch_rule_invalid_n():
    with pytest.raises(ValueError):
        golub_welsch_rule(LegendreMeasure(), 0)
