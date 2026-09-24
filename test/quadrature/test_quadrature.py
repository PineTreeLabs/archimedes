import dataclasses
import math

import numpy as np
import pytest
from scipy.special import beta as beta_fn
from scipy.special import (
    comb,
    roots_hermite,
    roots_hermitenorm,
    roots_jacobi,
    roots_laguerre,
    roots_legendre,
)

import archimedes as arc
from archimedes.measure import (
    JacobiMeasure,
    LaguerreMeasure,
    LegendreMeasure,
    PhysicistsHermiteMeasure,
    ProbabilistsHermiteMeasure,
)
from archimedes.quadrature import (
    QuadratureReferenceData,
    QuadratureRule,
    clenshaw_curtis,
    composite_quad,
    gauss_hermite,
    gauss_jacobi,
    gauss_laguerre,
    gauss_legendre,
    gauss_lobatto,
    gauss_radau,
    quadint,
    simpson,
    trapezoidal,
)

# -- QuadratureRule machinery --


def test_len_and_shape_validation():
    rule = gauss_legendre(5)
    assert len(rule) == 5

    with pytest.raises(ValueError):
        QuadratureRule.from_arrays(
            nodes=np.array([1.0, 2.0]),
            weights=np.array([1.0]),
            name="bad",
            measure=LegendreMeasure(),
        )


def test_map_to_no_args_is_identity():
    rule = gauss_legendre(5)
    np.testing.assert_array_equal(rule.map_to().nodes, rule.nodes)
    np.testing.assert_array_equal(rule.map_to().weights, rule.weights)


def test_map_to_interval():
    rule = gauss_legendre(5)
    a, b = -2.0, 5.0
    mapped = rule.map_to(a, b)
    assert mapped.nodes[0] > a
    assert mapped.nodes[-1] < b
    assert np.isclose(np.sum(mapped.weights), b - a)


def test_map_to_does_not_compose():
    rule = gauss_legendre(5)
    composed = rule.map_to(0.0, 1.0).map_to(2.0, 3.0)
    direct = rule.map_to(2.0, 3.0)
    np.testing.assert_allclose(composed.nodes, direct.nodes)
    np.testing.assert_allclose(composed.weights, direct.weights)
    # Resets all the way back to the reference domain, not just one level.
    reset = rule.map_to(0.0, 1.0).map_to()
    np.testing.assert_allclose(reset.nodes, rule.nodes)


def test_dot_matches_integrate():
    rule = gauss_legendre(4)
    values = np.cos(rule.nodes)
    assert np.isclose(rule.sum(values), rule.integrate(np.cos))


def test_integrate_args_forwarding():
    def f(x, k):
        return x**k

    rule = gauss_legendre(5)
    assert np.isclose(rule.integrate(f, args=(2,)), 2 / 3)
    # Default (no args) still calls f with just the nodes
    assert np.isclose(rule.integrate(lambda x: x**2), 2 / 3)


def test_integrate_args_with_scaled_domain():
    def f(x, k):
        return x**k

    rule = gauss_legendre(5)
    a, b = -2.0, 5.0
    result = rule.map_to(a, b).integrate(f, args=(2,))
    expected = (b**3 - a**3) / 3
    assert np.isclose(result, expected)


def test_dot_axis():
    rule = gauss_legendre(4)
    values = np.cos(rule.nodes)
    scalar = rule.sum(values)

    # 2D values, nodes along the last axis (default)
    stacked = np.stack([values, 2 * values])
    result_last = rule.sum(stacked, axis=-1)
    np.testing.assert_allclose(result_last, [scalar, 2 * scalar])

    # nodes along the first axis
    result_first = rule.sum(stacked.T, axis=0)
    np.testing.assert_allclose(result_first, [scalar, 2 * scalar])


def test_dot_errors():
    rule = gauss_legendre(4)
    with pytest.raises(ValueError):
        rule.sum(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        rule.sum(np.zeros(len(rule) + 1))


# -- Known rules: node generation and exact integration --


def test_gauss_legendre():
    rule = gauss_legendre(5)
    assert len(rule) == 5
    assert np.isclose(np.sum(rule.weights), 2.0)

    # Exact for polynomials up to degree 2n - 1 = 9
    assert np.isclose(rule.integrate(lambda x: x**8), 2 / 9)
    assert np.isclose(rule.integrate(lambda x: x**9), 0.0, atol=1e-12)


def test_gauss_legendre_scaled_domain():
    rule = gauss_legendre(5)
    a, b = -2.0, 5.0
    integral = rule.map_to(a, b).integrate(lambda x: x**2)
    expected = (b**3 - a**3) / 3
    assert np.isclose(integral, expected)


def test_gauss_legendre_a_b_matches_map_to():
    a, b = -2.0, 5.0
    assert gauss_legendre(5, a, b) == gauss_legendre(5).map_to(a, b)


def test_gauss_legendre_no_domain_args_is_unmapped():
    assert gauss_legendre(5, a=None, b=None) == gauss_legendre(5)


@pytest.mark.parametrize("endpoint,fixed_node", [("left", -1.0), ("right", 1.0)])
def test_gauss_radau(endpoint, fixed_node):
    rule = gauss_radau(5, endpoint=endpoint)
    assert len(rule) == 5
    assert np.isclose(np.sum(rule.weights), 2.0)
    idx = 0 if endpoint == "left" else -1
    assert np.isclose(rule.nodes[idx], fixed_node)

    # Exact for polynomials up to degree 2n - 2 = 8
    assert np.isclose(rule.integrate(lambda x: x**8), 2 / 9)


def test_gauss_radau_a_b_matches_map_to():
    a, b = -2.0, 5.0
    assert gauss_radau(5, a=a, b=b) == gauss_radau(5).map_to(a, b)


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


def test_gauss_lobatto_a_b_matches_map_to():
    a, b = -2.0, 5.0
    assert gauss_lobatto(5, a=a, b=b) == gauss_lobatto(5).map_to(a, b)


def test_gauss_lobatto_edge_cases():
    with pytest.raises(ValueError):
        gauss_lobatto(1)

    rule = gauss_lobatto(2)
    assert len(rule) == 2
    np.testing.assert_array_equal(rule.nodes, [-1.0, 1.0])
    np.testing.assert_array_equal(rule.weights, [1.0, 1.0])


def _classical_clenshaw_curtis_weights(d):
    """Direct O(d^2) evaluation of Waldvogel's Equs. (2.4)-(2.5), ascending
    node order, as an independent check on the FFT-based implementation."""
    k = np.arange(d + 1)
    theta = np.pi * k / d
    jmax = d // 2
    w = np.zeros(d + 1)
    for kk in range(d + 1):
        s = sum(
            (1.0 if j == d / 2 else 2.0) / (4 * j**2 - 1) * np.cos(2 * j * theta[kk])
            for j in range(1, jmax + 1)
        )
        c = 1.0 if kk % d == 0 else 2.0
        w[kk] = c / d * (1 - s)
    return w[::-1]


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 11, 16])
def test_clenshaw_curtis_matches_classical_formula(n):
    rule = clenshaw_curtis(n)
    expected = _classical_clenshaw_curtis_weights(n - 1)
    np.testing.assert_allclose(rule.weights, expected, atol=1e-10)


def test_clenshaw_curtis_matches_simpsons_rule():
    # 3-node Clenshaw-Curtis coincides with Simpson's rule
    rule = clenshaw_curtis(3)
    np.testing.assert_allclose(rule.nodes, [-1.0, 0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(rule.weights, [1 / 3, 4 / 3, 1 / 3])


def test_clenshaw_curtis_two_nodes():
    rule = clenshaw_curtis(2)
    np.testing.assert_array_equal(rule.nodes, [-1.0, 1.0])
    np.testing.assert_array_equal(rule.weights, [1.0, 1.0])


def test_clenshaw_curtis_invalid_n():
    with pytest.raises(ValueError):
        clenshaw_curtis(1)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9])
def test_clenshaw_curtis_properties(n):
    rule = clenshaw_curtis(n)
    assert len(rule) == n
    assert np.isclose(rule.nodes[0], -1.0)
    assert np.isclose(rule.nodes[-1], 1.0)
    assert np.all(np.diff(rule.nodes) > 0)
    assert np.all(rule.weights > 0)
    assert np.isclose(np.sum(rule.weights), 2.0)

    # Exact for polynomials up to degree n - 1
    for k in range(n):
        expected = 0.0 if k % 2 == 1 else 2 / (k + 1)
        assert np.isclose(rule.integrate(lambda x, k=k: x**k), expected, atol=1e-10)


def test_clenshaw_curtis_shares_legendre_measure():
    rule = clenshaw_curtis(5)
    assert isinstance(rule.measure, LegendreMeasure)

    a, b = -2.0, 5.0
    integral = rule.map_to(a, b).integrate(lambda x: x**2)
    expected = (b**3 - a**3) / 3
    assert np.isclose(integral, expected)


def test_clenshaw_curtis_a_b_matches_map_to():
    a, b = -2.0, 5.0
    assert clenshaw_curtis(5, a=a, b=b) == clenshaw_curtis(5).map_to(a, b)
    # Also cover the n == 2 special-cased branch.
    assert clenshaw_curtis(2, a=a, b=b) == clenshaw_curtis(2).map_to(a, b)


def test_clenshaw_curtis_composite_quad():
    # Same (uniform-weight) measure as Gauss-Legendre, so it tiles the same way
    rule = composite_quad(clenshaw_curtis(5), [-1.0, 0.0, 1.0])
    assert len(rule) == 10
    assert np.isclose(rule.integrate(lambda x: x**2), 2 / 3)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_trapezoidal_periodic_properties(n):
    rule = trapezoidal(n, periodic=True)
    assert len(rule) == n
    assert rule.nodes[0] == -1.0
    assert rule.nodes[-1] < 1.0  # half-open: no node at the identified +1 endpoint
    assert np.all(np.diff(rule.nodes) > 0)
    assert np.all(rule.weights > 0)
    np.testing.assert_allclose(rule.weights, 2.0 / n)
    assert np.isclose(np.sum(rule.weights), 2.0)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_trapezoidal_periodic_exact_for_trig_polynomials(n):
    rule = trapezoidal(n, periodic=True)
    # Exact (to machine precision) for cos(k*pi*t)/sin(k*pi*t), 1 <= k <= n-1
    for k in range(n):
        expected_cos = 2.0 if k == 0 else 0.0
        assert np.isclose(
            rule.integrate(lambda x, k=k: np.cos(k * np.pi * x)),
            expected_cos,
            atol=1e-10,
        )
        if k >= 1:
            assert np.isclose(
                rule.integrate(lambda x, k=k: np.sin(k * np.pi * x)),
                0.0,
                atol=1e-10,
            )


@pytest.mark.parametrize("n", [3, 5, 8])
def test_trapezoidal_periodic_aliases_at_nyquist(n):
    # At k = n the sampled signal is indistinguishable from the constant, so
    # the rule silently returns the wrong (nonzero) answer instead of 0.
    rule = trapezoidal(n, periodic=True)
    result = rule.integrate(lambda x: np.cos(n * np.pi * x))
    assert not np.isclose(result, 0.0, atol=1e-6)
    # cos(n*pi*x_j) = cos(n*pi*(-1) + 2*pi*j) = cos(n*pi) = (-1)**n at every
    # node, so the (wrong) aliased "integral" is 2*(-1)**n.
    assert np.isclose(result, 2.0 * (-1) ** n)


def test_trapezoidal_periodic_shares_legendre_measure():
    rule = trapezoidal(5, periodic=True)
    assert isinstance(rule.measure, LegendreMeasure)

    a, b = -2.0, 5.0
    integral = rule.map_to(a, b).integrate(lambda x: np.ones_like(x))
    assert np.isclose(integral, b - a)


def test_trapezoidal_periodic_invalid_n():
    with pytest.raises(ValueError):
        trapezoidal(0, periodic=True)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_trapezoidal_properties(n):
    rule = trapezoidal(n)
    assert len(rule) == n
    assert rule.nodes[0] == -1.0
    assert rule.nodes[-1] == 1.0  # closed interval: both endpoints are nodes
    assert np.all(np.diff(rule.nodes) > 0)
    assert np.all(rule.weights > 0)
    assert np.isclose(rule.weights[0], rule.weights[-1])
    # Interior weights are double the (half-weighted) endpoints
    assert np.isclose(2 * rule.weights[0], rule.weights[n // 2])
    assert np.isclose(np.sum(rule.weights), 2.0)


def test_trapezoidal_two_nodes():
    rule = trapezoidal(2)
    np.testing.assert_array_equal(rule.nodes, [-1.0, 1.0])
    np.testing.assert_array_equal(rule.weights, [1.0, 1.0])


def test_trapezoidal_matches_nodes_of_lobatto():
    # Same equally-spaced nodes as Lobatto for n=2 and n=3, only differing
    # from Lobatto once n >= 4 (Lobatto's interior nodes are no longer
    # equally spaced).
    rule = trapezoidal(3)
    lobatto = gauss_lobatto(3)
    np.testing.assert_allclose(rule.nodes, lobatto.nodes)


def test_trapezoidal_exact_for_linear():
    rule = trapezoidal(5)
    assert np.isclose(rule.integrate(lambda x: 3 * x + 2), 4.0)
    # Not exact for a quadratic
    assert not np.isclose(rule.integrate(lambda x: x**2), 2 / 3)


def test_trapezoidal_shares_legendre_measure():
    rule = trapezoidal(5)
    assert isinstance(rule.measure, LegendreMeasure)

    a, b = -2.0, 5.0
    integral = rule.map_to(a, b).integrate(lambda x: np.ones_like(x))
    assert np.isclose(integral, b - a)


def test_trapezoidal_composite_quad():
    # Shares LegendreMeasure's uniform reference weight, so it tiles the
    # same way as Lobatto -- shared endpoint nodes double up per element.
    rule = composite_quad(trapezoidal(3), [-1.0, 0.0, 1.0])
    assert len(rule) == 6
    assert np.isclose(rule.integrate(lambda x: 3 * x + 2), 4.0)


def test_trapezoidal_invalid_n():
    with pytest.raises(ValueError):
        trapezoidal(1)
    with pytest.raises(ValueError):
        trapezoidal(0)


def test_simpson_single_segment_matches_gauss_lobatto_3():
    # Simpson's rule *is* 3-point Gauss-Lobatto -- see
    # test_clenshaw_curtis_matches_simpsons_rule for the other classical
    # coincidence (Clenshaw-Curtis at n=3).
    rule = simpson(1, -1.0, 1.0)
    lobatto = gauss_lobatto(3)
    np.testing.assert_allclose(rule.nodes, lobatto.nodes)
    np.testing.assert_allclose(rule.weights, lobatto.weights)


@pytest.mark.parametrize("n_segments", [1, 2, 3, 5])
def test_simpson_properties(n_segments):
    rule = simpson(n_segments, -1.0, 1.0)
    assert len(rule) == 3 * n_segments
    assert rule.nodes[0] == -1.0
    assert rule.nodes[-1] == 1.0
    assert np.isclose(np.sum(rule.weights), 2.0)


def test_simpson_exact_for_cubic():
    rule = simpson(4, -1.0, 1.0)
    assert np.isclose(rule.integrate(lambda x: x**3 - 2 * x + 1), 2.0)
    # Not exact for a quartic
    assert not np.isclose(rule.integrate(lambda x: x**4), 2 / 5)


def test_simpson_a_b_matches_map_to():
    a, b = -2.0, 5.0
    assert simpson(3, a, b) == simpson(3, -1.0, 1.0).map_to(a, b)


def test_simpson_shares_legendre_measure():
    rule = simpson(4, -1.0, 1.0)
    assert isinstance(rule.measure, LegendreMeasure)


def test_simpson_invalid_n_segments():
    with pytest.raises(ValueError):
        simpson(0, -1.0, 1.0)


def test_gauss_hermite():
    n = 5
    rule = gauss_hermite(n)
    assert len(rule) == n
    assert isinstance(rule.measure, ProbabilistsHermiteMeasure)

    # Exact for polynomials up to degree 2n - 1 = 9
    assert np.isclose(rule.integrate(lambda x: np.ones_like(x)), np.sqrt(2 * np.pi))
    assert np.isclose(rule.integrate(lambda x: x**2), np.sqrt(2 * np.pi))
    assert np.isclose(rule.integrate(lambda x: x**3), 0.0, atol=1e-10)


def test_gauss_hermite_phys():
    n = 5
    rule = gauss_hermite(n, kind="phys")
    assert len(rule) == n
    assert isinstance(rule.measure, PhysicistsHermiteMeasure)

    assert np.isclose(rule.integrate(lambda x: np.ones_like(x)), np.sqrt(np.pi))
    assert np.isclose(rule.integrate(lambda x: x**2), np.sqrt(np.pi) / 2)


def test_gauss_hermite_invalid_kind():
    with pytest.raises(ValueError):
        gauss_hermite(5, kind="norm")


def test_gauss_laguerre():
    n = 5
    rule = gauss_laguerre(n)
    assert len(rule) == n
    assert isinstance(rule.measure, LaguerreMeasure)

    # Exact for polynomials up to degree 2n - 1
    for k in range(2 * n):
        assert np.isclose(rule.integrate(lambda x, k=k: x**k), math.factorial(k))


def test_gauss_jacobi():
    n, alpha, beta = 5, 1.5, 0.5
    rule = gauss_jacobi(n, alpha, beta)
    assert len(rule) == n
    assert isinstance(rule.measure, JacobiMeasure)
    assert rule.measure.alpha == alpha
    assert rule.measure.beta == beta

    # Exact for polynomials up to degree 2n - 1, weighted by
    # (1-x)^alpha (1+x)^beta. Substituting x = 2t - 1 reduces the moment
    # to a sum of Beta functions, giving a closed form to check against.
    def jacobi_moment(k):
        return 2 ** (alpha + beta + 1) * sum(
            comb(k, j) * (-1) ** (k - j) * 2**j * beta_fn(j + beta + 1, alpha + 1)
            for j in range(k + 1)
        )

    for k in range(2 * n):
        assert np.isclose(rule.integrate(lambda x, k=k: x**k), jacobi_moment(k))


def test_gauss_jacobi_invalid_params():
    with pytest.raises(ValueError):
        gauss_jacobi(5, -1.0, 0.5)
    with pytest.raises(ValueError):
        gauss_jacobi(5, 0.5, -1.0)


def test_gauss_jacobi_exact_moment():
    n, alpha, beta = 5, 1.0, 2.0
    x, w = roots_jacobi(n, alpha, beta)
    rule = QuadratureRule.from_arrays(
        x, w, name="gauss_jacobi_5", measure=JacobiMeasure(alpha=alpha, beta=beta)
    )

    # Zeroth moment of the Jacobi weight has a closed form in the Beta function
    expected = 2 ** (alpha + beta + 1) * beta_fn(alpha + 1, beta + 1)
    assert np.isclose(rule.integrate(lambda x: np.ones_like(x)), expected)


def test_gauss_jacobi_a_b_matches_map_to():
    a, b = -2.0, 5.0
    assert gauss_jacobi(5, 1.0, 2.0, a=a, b=b) == gauss_jacobi(5, 1.0, 2.0).map_to(a, b)


def test_gauss_laguerre_exact_moments():
    n = 5
    x, w = roots_laguerre(n)
    rule = QuadratureRule.from_arrays(
        x, w, name="gauss_laguerre_5", measure=LaguerreMeasure()
    )

    # Exact for polynomials up to degree 2n - 1
    for k in range(2 * n):
        integral = rule.integrate(lambda x, k=k: x**k)
        assert np.isclose(integral, math.factorial(k))


def test_gauss_laguerre_rate_scaling():
    n = 5
    x, w = roots_laguerre(n)
    rule = QuadratureRule.from_arrays(
        x, w, name="gauss_laguerre_5", measure=LaguerreMeasure()
    )

    rate = 2.0
    integral = rule.map_to(rate=rate).integrate(lambda x: np.ones_like(x))
    assert np.isclose(integral, 1 / rate)


def test_gauss_laguerre_rate_start_matches_map_to():
    rate, start = 2.0, 1.0
    assert gauss_laguerre(5, rate=rate, start=start) == gauss_laguerre(5).map_to(
        rate=rate, start=start
    )


def test_gauss_hermite_exact_moments():
    n = 5
    x, w = roots_hermite(n)
    rule = QuadratureRule.from_arrays(
        x, w, name="gauss_hermite_5", measure=PhysicistsHermiteMeasure()
    )

    assert np.isclose(rule.integrate(lambda x: np.ones_like(x)), np.sqrt(np.pi))
    assert np.isclose(rule.integrate(lambda x: x**2), np.sqrt(np.pi) / 2)
    assert np.isclose(rule.integrate(lambda x: x**3), 0.0, atol=1e-10)


def test_gauss_hermite_loc_scale_scaling():
    n = 5
    x, w = roots_hermite(n)
    rule = QuadratureRule.from_arrays(
        x, w, name="gauss_hermite_5", measure=PhysicistsHermiteMeasure()
    )

    loc, scale = 1.0, 2.0
    integral = rule.map_to(loc=loc, scale=scale).integrate(lambda x: np.ones_like(x))
    assert np.isclose(integral, scale * np.sqrt(np.pi))


def test_gauss_hermite_loc_scale_matches_map_to():
    loc, scale = 1.0, 2.0
    assert gauss_hermite(5, loc=loc, scale=scale) == gauss_hermite(5).map_to(
        loc=loc, scale=scale
    )


def test_gauss_hermitenorm_exact_moments():
    n = 5
    x, w = roots_hermitenorm(n)
    rule = QuadratureRule.from_arrays(
        x, w, name="gauss_hermitenorm_5", measure=ProbabilistsHermiteMeasure()
    )

    assert np.isclose(rule.integrate(lambda x: np.ones_like(x)), np.sqrt(2 * np.pi))
    assert np.isclose(rule.integrate(lambda x: x**2), np.sqrt(2 * np.pi))
    assert np.isclose(rule.integrate(lambda x: x**3), 0.0, atol=1e-10)


def test_gauss_hermitenorm_matches_gaussian_expectation():
    # Unlike PhysicistsHermiteMeasure, loc/scale here are exactly the mean and standard
    # deviation of a Gaussian density -- no sqrt(2) correction needed.
    n = 6
    x, w = roots_hermitenorm(n)
    rule = QuadratureRule.from_arrays(
        x, w, name="gauss_hermitenorm_6", measure=ProbabilistsHermiteMeasure()
    )

    loc, scale = 2.0, 3.0
    norm = scale * np.sqrt(2 * np.pi)  # normalizes the weight to a proper PDF
    mapped = rule.map_to(loc=loc, scale=scale)

    def expectation(f):
        return mapped.integrate(f) / norm

    assert np.isclose(expectation(lambda x: np.ones_like(x)), 1.0)
    assert np.isclose(expectation(lambda x: x), loc)
    assert np.isclose(expectation(lambda x: x**2), loc**2 + scale**2)


# -- density=True normalization --


def test_sum_density_sums_to_one():
    rule = gauss_legendre(5)
    a, b = -2.0, 5.0
    mapped = rule.map_to(a, b)
    w_density = mapped.sum(np.ones(len(mapped)), density=True)
    assert np.isclose(w_density, 1.0)
    w_plain = mapped.sum(np.ones(len(mapped)))
    assert np.isclose(w_density, w_plain / (b - a))


def test_sum_density_default_false():
    rule = gauss_legendre(5).map_to(-2.0, 5.0)
    values = np.ones(len(rule))
    np.testing.assert_array_equal(rule.sum(values), rule.sum(values, density=False))


def test_integrate_density_matches_gaussian_expectation():
    n = 6
    rule = gauss_hermite(n)
    loc, scale = 2.0, 3.0
    mapped = rule.map_to(loc=loc, scale=scale)

    assert np.isclose(
        mapped.integrate(lambda x: np.ones_like(x), density=True),
        1.0,
    )
    assert np.isclose(mapped.integrate(lambda x: x, density=True), loc)
    assert np.isclose(
        mapped.integrate(lambda x: x**2, density=True),
        loc**2 + scale**2,
    )


def test_sum_density_forwarded():
    n = 6
    rule = gauss_hermite(n)
    loc, scale = 2.0, 3.0
    mapped = rule.map_to(loc=loc, scale=scale)
    values = np.ones_like(mapped.nodes)
    assert np.isclose(mapped.sum(values, density=True), 1.0)


@pytest.mark.parametrize(
    "rule_factory,params",
    [
        (lambda: gauss_legendre(5), {"a": -2.0, "b": 5.0}),
        (lambda: gauss_jacobi(5, 1.0, 2.0), {}),
        (lambda: gauss_laguerre(5), {"rate": 2.0, "start": 1.0}),
        (lambda: gauss_hermite(5, kind="phys"), {"loc": 1.0, "scale": 2.0}),
        (lambda: gauss_hermite(5, kind="prob"), {"loc": 1.0, "scale": 2.0}),
    ],
)
def test_density_weights_sum_to_one_all_families(rule_factory, params):
    mapped = rule_factory().map_to(**params)
    result = mapped.sum(np.ones(len(mapped)), density=True)
    assert np.isclose(result, 1.0)


# -- composite rules --


def test_composite_matches_exact_integral():
    rule = composite_quad(gauss_legendre(3), [-1.0, 0.0, 1.0])
    assert len(rule) == 6
    assert np.isclose(np.sum(rule.weights), 2.0)

    # Exact for a quadratic (well within each panel's degree-5 exactness)
    assert np.isclose(rule.integrate(lambda x: x**2), 2 / 3)


def test_composite_nonuniform_breakpoints():
    rule = composite_quad(gauss_legendre(4), [-1.0, -0.2, 0.5, 1.0])
    assert len(rule) == 12
    assert np.isclose(rule.integrate(lambda x: x**2), 2 / 3)


def test_composite_scaled_domain():
    # A composite rule is still an ordinary rule on the reference domain, so
    # it maps onto an arbitrary target interval like any other Legendre rule.
    rule = composite_quad(gauss_legendre(3), [-1.0, 0.0, 1.0])
    a, b = -2.0, 5.0
    integral = rule.map_to(a, b).integrate(lambda x: x**2)
    expected = (b**3 - a**3) / 3
    assert np.isclose(integral, expected)


def test_composite_requires_uniform_weight():
    jacobi_rule = gauss_jacobi(5, 1.0, 2.0)
    with pytest.raises(ValueError):
        composite_quad(jacobi_rule, [-1.0, 0.0, 1.0])

    laguerre_rule = QuadratureRule.from_arrays(
        *roots_laguerre(5), name="gauss_laguerre_5", measure=LaguerreMeasure()
    )
    with pytest.raises(ValueError):
        composite_quad(laguerre_rule, [0.0, 1.0, 2.0])


def test_composite_breakpoints_validation():
    rule = gauss_legendre(3)

    with pytest.raises(ValueError):
        composite_quad(rule, [-1.0])  # too few entries

    with pytest.raises(ValueError):
        composite_quad(rule, [-1.0, 0.5, 0.0, 1.0])  # not strictly increasing

    with pytest.raises(ValueError):
        composite_quad(rule, [-0.5, 0.0, 1.0])  # doesn't span the reference domain

    with pytest.raises(ValueError):
        composite_quad(rule, [-1.0, 0.0, 0.5])  # doesn't span the reference domain


# -- composite rules: per-element (varying order) --


def test_composite_per_element_rules_varying_order():
    rules = [gauss_legendre(2), gauss_legendre(4), gauss_legendre(3)]
    bp = np.linspace(-1.0, 1.0, 4)  # 3 elements
    rule = composite_quad(rules, bp)

    assert len(rule) == 9
    np.testing.assert_array_equal(np.bincount(rule.elements), [2, 4, 3])
    np.testing.assert_array_equal(rule.breakpoints, bp)

    # Each panel is exact for a quadratic, well within every panel's own
    # (much lower) exactness degree.
    assert np.isclose(rule.integrate(lambda x: x**2), 2 / 3)


def test_composite_per_element_elements_provenance_with_lobatto():
    # Differently-sized Lobatto sub-rules each place a node on the shared
    # interior breakpoint; coordinate lookup can't tell the two copies
    # apart, so `elements` provenance must still resolve correctly here.
    rules = [gauss_lobatto(2), gauss_lobatto(4)]
    bp = np.array([-1.0, 0.0, 1.0])
    rule = composite_quad(rules, bp)

    on_knot = np.isclose(rule.nodes, 0.0)
    assert on_knot.sum() == 2
    assert set(rule.elements[on_knot]) == {0, 1}
    np.testing.assert_array_equal(np.bincount(rule.elements), [2, 4])


def test_composite_rejects_wrong_rule_count():
    rules = [gauss_legendre(2), gauss_legendre(3)]  # 2 rules
    bp = np.linspace(-1.0, 1.0, 4)  # 3 elements

    with pytest.raises(ValueError, match="3 elements"):
        composite_quad(rules, bp)


def test_composite_rejects_mismatched_measures():
    laguerre_rule = QuadratureRule.from_arrays(
        *roots_laguerre(3), name="gauss_laguerre_3", measure=LaguerreMeasure()
    )
    with pytest.raises(ValueError, match="same measure"):
        composite_quad([gauss_legendre(3), laguerre_rule], [-1.0, 0.0, 1.0])


def test_composite_per_element_name_field():
    # Same family (and hence same fixed rule name), different order per
    # element: the common name is kept.
    same_name = composite_quad([gauss_legendre(2), gauss_legendre(4)], [-1.0, 0.0, 1.0])
    assert same_name.name == "gauss_legendre"

    # Different families sharing a measure (Legendre and Lobatto both use
    # `LegendreMeasure`) give mismatched rule names, so the composite rule
    # falls back to a generic name rather than silently picking one.
    mixed_name = composite_quad([gauss_legendre(3), gauss_lobatto(4)], [-1.0, 0.0, 1.0])
    assert mixed_name.name == "composite"


# -- arc.compile integration: symbolic domain/measure parameters --


def test_compile_symbolic_interval():
    rule = gauss_legendre(5)

    @arc.compile
    def quad(a, b):
        return rule.map_to(a, b).integrate(lambda x: x**2)

    result = quad(0.0, 2.0)
    assert np.isclose(float(result), 8 / 3)


def test_map_to_symbolic_interval():
    # Direct regression test that a `map_to`'d rule's `params` field
    # correctly holds and propagates a symbolic value through the
    # @tree.struct/tracing machinery.
    rule = gauss_legendre(5)

    @arc.compile
    def quad(a, b):
        mapped = rule.map_to(a, b)
        return mapped.integrate(lambda x: x**2)

    result = quad(0.0, 2.0)
    assert np.isclose(float(result), 8 / 3)


def test_compile_symbolic_rate():
    x, w = roots_laguerre(5)
    rule = QuadratureRule.from_arrays(
        x, w, name="gauss_laguerre_5", measure=LaguerreMeasure()
    )

    @arc.compile
    def quad(rate):
        return rule.map_to(rate=rate).sum(np.ones_like(rule.nodes))

    result = quad(2.0)
    assert np.isclose(float(result), 0.5)


def test_compile_symbolic_loc_scale():
    x, w = roots_hermitenorm(5)
    rule = QuadratureRule.from_arrays(
        x, w, name="gauss_hermitenorm_5", measure=ProbabilistsHermiteMeasure()
    )

    @arc.compile
    def quad(loc, scale):
        return rule.map_to(loc=loc, scale=scale).sum(np.ones_like(rule.nodes))

    result = quad(1.0, 2.0)
    assert np.isclose(float(result), 2.0 * np.sqrt(2 * np.pi))


def test_compile_vector_integrand():
    rule = gauss_legendre(5)

    @arc.compile
    def quad(a, b):
        def f(x):
            return np.stack([x, x**2])

        return rule.map_to(a, b).integrate(f, axis=-1)

    result = np.asarray(quad(-1.0, 1.0))
    np.testing.assert_allclose(result, [0.0, 2 / 3], atol=1e-10)


# -- quadint --


def test_quadint_default_rule():
    a, b = -3.0, 3.0
    expected = np.exp(b) - np.exp(a)
    assert np.isclose(quadint(np.exp, a, b), expected)


@pytest.mark.parametrize(
    "rule", ["legendre", "radau_left", "radau_right", "lobatto", "clenshaw_curtis"]
)
def test_quadint_rule_dispatch(rule):
    a, b = 0.0, np.pi
    result = quadint(np.sin, a, b, n=10, rule=rule)
    assert np.isclose(result, 2.0, atol=1e-8)


def test_quadint_unknown_rule():
    with pytest.raises(ValueError):
        quadint(np.sin, 0.0, 1.0, rule="simpson")


@pytest.mark.parametrize("a,b", [(-np.inf, 1.0), (0.0, np.inf), (-np.inf, np.inf)])
def test_quadint_infinite_bounds(a, b):
    with pytest.raises(ValueError):
        quadint(lambda x: np.exp(-(x**2)), a, b)


def test_quadint_args_forwarding():
    def f(x, k):
        return x**k

    result = quadint(f, 0.0, 1.0, args=(3,))
    assert np.isclose(result, 0.25)


def test_quadint_vector_integrand():
    def f(x):
        return np.stack([x, x**2])

    result = quadint(f, -1.0, 1.0, n=5, axis=-1)
    np.testing.assert_allclose(result, [0.0, 2 / 3], atol=1e-10)


def test_quadint_compile_symbolic_interval():
    @arc.compile
    def quad(a, b):
        return quadint(lambda x: x**2, a, b, n=5)

    result = quad(0.0, 2.0)
    assert np.isclose(float(result), 8 / 3)


# -- equality / hashing --


def test_quadrature_rule_equality_is_elementwise():
    # The dataclass-generated __eq__ would compare the node/weight arrays with
    # `==` and raise "truth value of an array is ambiguous", so QuadratureRule
    # defines its own.
    assert gauss_legendre(5) == gauss_legendre(5)
    assert gauss_legendre(5) != gauss_legendre(6)
    assert gauss_legendre(5) != gauss_lobatto(5)


def test_quadrature_rule_equality_distinguishes_mapping():
    rule = gauss_legendre(5)
    mapped = rule.map_to(0.0, 1.0)
    assert rule != mapped
    assert mapped == rule.map_to(0.0, 1.0)
    assert mapped != rule.map_to(0.0, 2.0)


def test_quadrature_rule_is_hashable():
    assert hash(gauss_legendre(5)) == hash(gauss_legendre(5))
    assert len({gauss_legendre(5), gauss_legendre(5), gauss_legendre(6)}) == 2


def test_quadrature_rule_equality_against_other_types_is_not_implemented():
    assert gauss_legendre(5).__eq__(object()) is NotImplemented
    assert gauss_legendre(5) != object()


def test_quadrature_reference_data_equality_against_other_types_is_not_implemented():
    ref = gauss_legendre(5).reference
    assert ref.__eq__(object()) is NotImplemented
    assert ref != object()


def test_quadrature_rule_ndim_and_measures():
    rule = gauss_legendre(5)
    assert rule.ndim == 1
    assert rule.measures == (rule.measure,)


def test_params_equal_rejects_mismatched_types():
    from archimedes.measure import RealLine, UnitInterval

    a = UnitInterval.Parameters(a=0.0, b=1.0)
    b = RealLine.Parameters(loc=0.0, scale=1.0)
    interval_rule = gauss_legendre(5).replace(params=a)
    # Same reference payload as `interval_rule` but a mismatched-domain
    # `params` bypassing `map_to` -- exercises the type-mismatch guard in
    # `_params_equal` directly, since `interval_rule`'s own `.replace` can't
    # otherwise produce a `RealLine.Parameters` to compare against.
    mismatched = interval_rule.replace(params=b)
    assert interval_rule != mismatched


def test_from_arrays_matches_direct_construction():
    x, w = roots_laguerre(4)
    from_arrays = QuadratureRule.from_arrays(
        x, w, name="gauss_laguerre_4", measure=LaguerreMeasure()
    )
    direct = QuadratureRule(
        QuadratureReferenceData(x, w, LaguerreMeasure()), "gauss_laguerre_4"
    )
    assert from_arrays == direct


# -- Measure.affine_invariant enforcement --


class _StieltjesLegendre(LegendreMeasure):
    """A Legendre-alike that hasn't opted into `affine_invariant`, to
    exercise the enforcement path independent of any real Stieltjes-fallback
    measure (see `test/measure/test_measure.py` for that one)."""

    affine_invariant = False


def test_map_to_no_args_skips_affine_invariant_check():
    rule = QuadratureRule.from_arrays(
        *roots_legendre(5), name="test", measure=_StieltjesLegendre()
    )
    # No arguments -> always allowed, regardless of affine_invariant.
    assert rule.map_to() == rule.map_to()
    with pytest.raises(ValueError, match="affine_invariant"):
        rule.map_to(0.0, 1.0)


def test_scaled_read_raises_for_non_affine_invariant_measure_with_bypassed_params():
    # Enforcement lives at read time (`nodes`/`weights`), not only at
    # `map_to`-call time, so bypassing `map_to` via `.replace(params=...)`
    # directly is still caught.
    from archimedes.measure import UnitInterval

    rule = QuadratureRule.from_arrays(
        *roots_legendre(5), name="test", measure=_StieltjesLegendre()
    )
    bypassed = rule.replace(params=UnitInterval.Parameters(a=0.0, b=1.0))
    with pytest.raises(ValueError, match="affine_invariant"):
        bypassed.nodes
    with pytest.raises(ValueError, match="affine_invariant"):
        bypassed.weights


class TestCompositeBreakpoints:
    """A composite rule records the breakpoints it was tiled across, so that
    consumers can tell whether it aligns with a piecewise integrand."""

    def test_plain_rule_has_no_breakpoints(self):
        assert gauss_legendre(5).breakpoints is None

    def test_composite_records_breakpoints(self):
        bp = np.array([-1.0, -0.2, 1.0])
        rule = composite_quad(gauss_legendre(3), bp)
        np.testing.assert_array_equal(rule.breakpoints, bp)

    def test_equality_distinguishes_breakpoints(self):
        bp = np.array([-1.0, 0.0, 1.0])
        a = composite_quad(gauss_legendre(3), bp)
        b = composite_quad(gauss_legendre(3), bp.copy())
        c = composite_quad(gauss_legendre(3), np.array([-1.0, 0.5, 1.0]))
        assert a == b
        assert a != c

    def test_composite_not_equal_to_plain_rule(self):
        # A single-element composite has the same nodes and weights as its
        # base, but is still a different kind of object to a consumer that
        # cares about alignment.
        base = gauss_legendre(4)
        tiled = composite_quad(base, np.array([-1.0, 1.0]))
        np.testing.assert_allclose(tiled.nodes, base.nodes, atol=1e-14)
        assert tiled != base
        assert base != tiled


class TestCompositeElements:
    """A composite rule also records *which* element each node came from.

    Coordinates cannot recover this: a Lobatto sub-rule places a node on each
    interior breakpoint from both sides, so the same coordinate appears twice
    with different provenance. A basis that is discontinuous there needs to
    tell them apart.
    """

    def test_plain_rule_has_no_elements(self):
        assert gauss_legendre(5).elements is None

    def test_composite_records_one_element_index_per_node(self):
        base = gauss_legendre(3)
        rule = composite_quad(base, np.linspace(-1.0, 1.0, 4))
        np.testing.assert_array_equal(rule.elements, np.repeat([0, 1, 2], len(base)))

    def test_boundary_nodes_are_duplicated_with_distinct_owners(self):
        # The case that made coordinate lookup wrong: one coordinate, two
        # nodes, two different elements.
        rule = composite_quad(gauss_lobatto(3), np.linspace(-1.0, 1.0, 3))
        on_knot = np.isclose(rule.nodes, 0.0)
        assert on_knot.sum() == 2
        assert set(rule.elements[on_knot]) == {0, 1}

    def test_equality_distinguishes_elements(self):
        bp = np.array([-1.0, 0.0, 1.0])
        rule = composite_quad(gauss_legendre(2), bp)
        shuffled_reference = dataclasses.replace(
            rule.reference, elements=rule.elements[::-1]
        )
        shuffled = dataclasses.replace(rule, reference=shuffled_reference)
        assert rule != shuffled

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"breakpoints": np.array([-1.0, 1.0])}, "must be given together"),
            ({"elements": np.zeros(2, dtype=int)}, "must be given together"),
            (
                {
                    "breakpoints": np.array([-1.0, 1.0]),
                    "elements": np.zeros(3, dtype=int),
                },
                "same shape as nodes",
            ),
            (
                {"breakpoints": np.array([-1.0, 1.0]), "elements": np.array([0, 5])},
                "must index the 1 intervals",
            ),
        ],
    )
    def test_breakpoints_and_elements_must_be_consistent(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            QuadratureRule.from_arrays(
                nodes=np.zeros(2),
                weights=np.zeros(2),
                name="test",
                measure=LegendreMeasure(),
                **kwargs,
            )
