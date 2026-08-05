import math

import numpy as np
import pytest
from scipy.integrate import quad
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
    Measure,
    PhysicistsHermiteMeasure,
    ProbabilistsHermiteMeasure,
    RealLine,
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


# -- discretized Stieltjes fallback (no closed form) --


class _QuarticMeasure(Measure):
    """A weight with no closed-form recursion -- relies entirely on
    Measure's default (discretized Stieltjes) recurrence_coeffs."""

    domain = RealLine()

    def weight(self, x):
        return np.exp(-(x**4))

    @property
    def reference_mass(self):
        return 1.812804954110954  # quad(weight, -inf, inf)


@pytest.mark.parametrize("n", [5, 10])
def test_golub_welsch_rule_custom_measure_exact_even_moments(n):
    rule = golub_welsch_rule(_QuarticMeasure(), n)
    for deg in range(0, 2 * n, 2):
        expected, _ = quad(lambda x, deg=deg: x**deg * np.exp(-(x**4)), -np.inf, np.inf)
        assert np.isclose(
            rule.integrate(lambda x, deg=deg: x**deg), expected, atol=1e-8
        )


@pytest.mark.parametrize("n", [5, 10])
def test_golub_welsch_rule_custom_measure_odd_moments_vanish(n):
    rule = golub_welsch_rule(_QuarticMeasure(), n)
    for deg in range(1, 2 * n, 2):
        assert np.isclose(rule.integrate(lambda x, deg=deg: x**deg), 0.0, atol=1e-8)
