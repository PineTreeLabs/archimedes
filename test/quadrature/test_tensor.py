"""Tensor-product quadrature.

The correctness criterion is exactness: an ``n``-point Gauss rule per
dimension integrates any polynomial of degree ``<= 2n - 1`` *in each
variable separately* exactly, so these check against analytic integrals
rather than against a reimplementation of the construction.
"""

import itertools

import numpy as np
import pytest

import archimedes as arc
from archimedes.measure import (
    HermiteNormMeasure,
    LegendreMeasure,
    RealLine,
    UnitInterval,
)
from archimedes.quadrature import (
    QuadratureRule,
    TensorQuadratureRule,
    composite,
    gauss_hermite,
    gauss_legendre,
    tensor,
)


def _gl(n):
    return gauss_legendre(n)


def _gh(n):
    return gauss_hermite(n, kind="prob")


# -- structure --


def test_shapes_and_counts():
    rule = tensor(_gl(3), _gl(4), _gl(2))
    assert rule.ndim == 3
    assert len(rule) == 24
    assert rule.nodes.shape == (24, 3)
    assert rule.weights.shape == (24,)
    assert rule.scaled_points().shape == (24, 3)
    assert rule.scaled_weights().shape == (24,)


def test_node_ordering_is_first_dimension_slowest():
    a, b = _gl(2), _gl(3)
    rule = tensor(a, b)
    expected = np.array(list(itertools.product(a.nodes, b.nodes)))
    np.testing.assert_allclose(rule.nodes, expected)


def test_weights_are_the_outer_product():
    a, b = _gl(2), _gl(3)
    rule = tensor(a, b)
    expected = np.array([wa * wb for wa in a.weights for wb in b.weights])
    np.testing.assert_allclose(rule.weights, expected)


def test_reference_weights_sum_to_product_of_masses():
    # Legendre reference mass is 2 per dimension.
    rule = tensor(_gl(3), _gl(4), _gl(2))
    assert rule.weights.sum() == pytest.approx(8.0)


def test_measures_are_per_dimension():
    rule = tensor(_gh(3), _gl(3))
    assert rule.measures == (HermiteNormMeasure(), LegendreMeasure())
    # There is deliberately no single `.measure`.
    assert not hasattr(rule, "measure")


def test_one_dimensional_tensor_is_allowed():
    # Degenerate but valid: differs from the underlying rule only in
    # presenting (n, 1) points rather than (n,).
    base = _gl(4)
    rule = tensor(base)
    assert rule.ndim == 1
    assert len(rule) == 4
    np.testing.assert_allclose(rule.nodes[:, 0], base.nodes)
    np.testing.assert_allclose(rule.weights, base.weights)


def test_repr():
    rule = tensor(_gh(3), _gl(4))
    text = repr(rule)
    assert "ndim=2" in text and "n=12" in text
    assert "HermiteNormMeasure" in text and "LegendreMeasure" in text


def test_equality_and_hash():
    assert tensor(_gl(3), _gl(4)) == tensor(_gl(3), _gl(4))
    assert tensor(_gl(3), _gl(4)) != tensor(_gl(4), _gl(3))
    assert tensor(_gl(3), _gl(4)) != tensor(_gl(3), _gh(4))
    assert tensor(_gl(3)) != _gl(3)
    assert hash(tensor(_gl(3), _gl(4))) == hash(tensor(_gl(3), _gl(4)))


def test_quadrature_rule_reports_ndim_one():
    # The 1-D rule participates in the same `Quadrature` interface.
    assert _gl(5).ndim == 1


def test_quadrature_rule_measures_is_a_length_one_tuple():
    # Uniform with the tensor rule, so consumers can zip against a basis's
    # per-dimension measures without dispatching on ndim.
    rule = _gl(5)
    assert rule.measures == (rule.measure,)
    assert len(rule.measures) == rule.ndim


# -- exactness --


@pytest.mark.parametrize("px,py", [(0, 0), (1, 2), (3, 3), (2, 5)])
def test_exact_for_tensor_product_polynomials(px, py):
    # n-point Gauss is exact through degree 2n - 1 in its own variable.
    rule = tensor(_gl(3), _gl(3))
    a, b = (0.0, 2.0), (-1.0, 1.5)
    value = rule.integrate(lambda x: x[:, 0] ** px * x[:, 1] ** py, dims=[a, b])
    expected = ((a[1] ** (px + 1) - a[0] ** (px + 1)) / (px + 1)) * (
        (b[1] ** (py + 1) - b[0] ** (py + 1)) / (py + 1)
    )
    assert value == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_exact_beyond_the_total_degree_space():
    # x^3 y^3 has total degree 6, which a total-degree rule of this size
    # could not integrate; the tensor rule handles it because the degree in
    # *each* variable is only 3 <= 2*2 - 1.
    rule = tensor(_gl(2), _gl(2))
    value = rule.integrate(
        lambda x: x[:, 0] ** 3 * x[:, 1] ** 3, dims=[(0.0, 1.0), (0.0, 1.0)]
    )
    assert value == pytest.approx(1.0 / 16.0)


def test_mixed_measures_give_gaussian_moments():
    # xi ~ N(0, 2) on axis 0, u ~ U(0, 1) on axis 1.
    # E[xi^2 * u] = var * mean(u) = 4 * 0.5
    rule = tensor(_gh(4), _gl(4))
    value = rule.integrate(
        lambda x: x[:, 0] ** 2 * x[:, 1],
        dims=[(0.0, 2.0), (0.0, 1.0)],
        density=True,
    )
    assert value == pytest.approx(2.0)


def test_density_weights_sum_to_one():
    rule = tensor(_gh(4), _gl(4), _gh(3))
    w = rule.scaled_weights(dims=[(0.0, 2.0), (-1.0, 3.0), (1.0, 0.5)], density=True)
    assert w.sum() == pytest.approx(1.0)


def test_four_dimensional_gaussian_moments():
    # The PCE-style case: independent Gaussian inputs, one Gauss-Hermite
    # rule per dimension. Sum of squares has mean equal to the sum of the
    # variances, whatever the correlation structure of the (independent)
    # nodes.
    sigmas = [1.0, 2.0, 0.5, 3.0]
    rule = tensor(*[_gh(3) for _ in sigmas])
    dims = [(0.0, s) for s in sigmas]
    mean_sq = rule.integrate(lambda x: np.sum(x**2, axis=1), dims=dims, density=True)
    assert mean_sq == pytest.approx(sum(s**2 for s in sigmas))


def test_composite_per_dimension():
    # Tiling within a dimension and tensoring across dimensions commute, so
    # a rectilinear mesh is `tensor(composite(...), composite(...))`.
    breaks = np.linspace(-1.0, 1.0, 4)
    rule = tensor(composite(_gl(2), breaks), _gl(3))
    assert len(rule) == (3 * 2) * 3
    np.testing.assert_allclose(rule.breakpoints[0], breaks)
    assert rule.breakpoints[1] is None
    value = rule.integrate(
        lambda x: x[:, 0] ** 3 * x[:, 1] ** 2, dims=[(0.0, 2.0), (0.0, 1.0)]
    )
    assert value == pytest.approx(4.0 / 3.0)


def test_breakpoints_is_always_a_full_length_tuple():
    rule = tensor(_gl(2), _gl(3), _gl(2))
    assert rule.breakpoints == (None, None, None)


def test_elements_is_none_without_composite_factors():
    assert tensor(_gl(2), _gl(3)).elements is None


def test_elements_mirrors_nodes():
    # Column d indexes dimension d's elements, row-aligned with `nodes`, so a
    # univariate factor can be handed a 1-D view of both.
    breaks = np.linspace(-1.0, 1.0, 3)
    composite_rule = composite(_gl(2), breaks)  # 2 elements x 2 nodes
    rule = tensor(composite_rule, _gl(3))
    assert rule.elements.shape == (len(rule), 2)
    # Dimension 0 varies slowest, matching the node ordering.
    np.testing.assert_array_equal(
        rule.elements[:, 0], np.repeat(composite_rule.elements, 3)
    )
    # A non-composite dimension has no elements of its own.
    np.testing.assert_array_equal(rule.elements[:, 1], 0)


def test_elements_zero_fills_non_composite_dimensions():
    breaks = np.linspace(-1.0, 1.0, 3)
    rule = tensor(_gl(2), composite(_gl(2), breaks))
    np.testing.assert_array_equal(rule.elements[:, 0], 0)
    assert set(rule.elements[:, 1]) == {0, 1}


# -- domain parameter forms --


def test_all_parameter_forms_agree():
    rule = tensor(_gl(3), _gh(3))
    as_tuples = rule.scaled_points(dims=[(0.0, 2.0), (1.0, 0.5)])
    as_params = rule.scaled_points(
        dims=[
            UnitInterval.Parameters(a=0.0, b=2.0),
            RealLine.Parameters(mean=1.0, std=0.5),
        ]
    )
    as_dicts = rule.scaled_points(
        dims=[{"a": 0.0, "b": 2.0}, {"mean": 1.0, "std": 0.5}]
    )
    positional = rule.scaled_points([(0.0, 2.0), (1.0, 0.5)])
    np.testing.assert_allclose(as_params, as_tuples)
    np.testing.assert_allclose(as_dicts, as_tuples)
    np.testing.assert_allclose(positional, as_tuples)


def test_omitted_parameters_give_the_reference_domain():
    rule = tensor(_gl(3), _gl(4))
    np.testing.assert_allclose(rule.scaled_points(), rule.nodes)
    np.testing.assert_allclose(rule.scaled_weights(), rule.weights)
    # An explicit None per dimension is the same thing.
    np.testing.assert_allclose(rule.scaled_points(dims=[None, None]), rule.nodes)


# -- integrate / sum --


def test_integrate_matches_sum_at_the_nodes():
    rule = tensor(_gl(3), _gl(3))
    dims = [(0.0, 2.0), (0.0, 1.0)]

    def f(x):
        return np.exp(x[:, 0]) * x[:, 1] ** 2

    x = rule.scaled_points(dims=dims)
    assert rule.integrate(f, dims=dims) == pytest.approx(rule.sum(f(x), dims=dims))


def test_integrate_passes_extra_args():
    rule = tensor(_gl(2), _gl(2))
    dims = [(0.0, 1.0), (0.0, 1.0)]
    value = rule.integrate(lambda x, k: k * x[:, 0] * x[:, 1], dims=dims, args=(6.0,))
    assert value == pytest.approx(6.0 * 0.25)


def test_vector_valued_integrand():
    rule = tensor(_gl(3), _gl(3))
    dims = [(0.0, 2.0), (0.0, 1.0)]
    value = rule.integrate(
        lambda x: np.stack([x[:, 0], x[:, 1] ** 2], axis=-1), dims=dims
    )
    # int_0^2 int_0^1 x dy dx = 2 ; int_0^2 int_0^1 y^2 dy dx = 2/3
    np.testing.assert_allclose(value, [2.0, 2.0 / 3.0])


def test_vector_valued_integrand_nodes_last():
    rule = tensor(_gl(3), _gl(3))
    dims = [(0.0, 2.0), (0.0, 1.0)]
    x = rule.scaled_points(dims=dims)
    values = np.stack([x[:, 0], x[:, 1] ** 2], axis=0)  # (2, n)
    np.testing.assert_allclose(rule.sum(values, dims=dims, axis=-1), [2.0, 2.0 / 3.0])


def test_sum_rejects_wrong_shapes():
    rule = tensor(_gl(2), _gl(2))
    with pytest.raises(ValueError, match="0-D, 1-D, or 2-D"):
        rule.sum(np.zeros((4, 1, 1)))
    with pytest.raises(ValueError, match="to match the quadrature nodes"):
        rule.sum(np.zeros(3))


# -- symbolic --


def test_traces_and_differentiates_through_domain_parameters():
    rule = tensor(_gl(3), _gl(3))

    @arc.compile
    def area_moment(p):
        dims = [(0.0, p[0]), (0.0, p[1])]
        x = rule.scaled_points(dims=dims)
        w = rule.scaled_weights(dims=dims)
        return np.dot(w, x[:, 0] * x[:, 1])

    p = np.array([2.0, 3.0])
    # int_0^a int_0^b x y dy dx = a^2 b^2 / 4
    assert area_moment(p) == pytest.approx(9.0)
    np.testing.assert_allclose(
        arc.grad(area_moment)(p), [2 * p[0] * p[1] ** 2 / 4, p[0] ** 2 * 2 * p[1] / 4]
    )


def test_traces_with_density():
    rule = tensor(_gh(3), _gh(3))

    @arc.compile
    def second_moment(s):
        dims = [(0.0, s[0]), (0.0, s[1])]
        x = rule.scaled_points(dims=dims)
        w = rule.scaled_weights(dims=dims, density=True)
        return np.dot(w, x[:, 0] ** 2 + x[:, 1] ** 2)

    s = np.array([2.0, 3.0])
    assert second_moment(s) == pytest.approx(4.0 + 9.0)


# -- errors --


def test_requires_at_least_one_dimension():
    with pytest.raises(ValueError, match="at least one dimension"):
        tensor()


def test_rejects_non_quadrature_rules():
    with pytest.raises(TypeError, match=r"rules\[1\] must be a QuadratureRule"):
        tensor(_gl(3), "not a rule")


def test_rejects_wrong_number_of_dimensions():
    rule = tensor(_gl(3), _gl(3))
    with pytest.raises(ValueError, match="expected 2 per-dimension parameters"):
        rule.scaled_points(dims=[(0.0, 1.0)])


def test_rejects_mixing_positional_and_keyword_parameters():
    rule = tensor(_gl(3), _gl(3))
    with pytest.raises(TypeError, match="a single sequence"):
        rule.scaled_points([(0.0, 1.0), (0.0, 1.0)], dims=[None, None])


def test_rejects_unknown_keyword():
    rule = tensor(_gl(3), _gl(3))
    with pytest.raises(TypeError, match="a single sequence"):
        rule.scaled_points(a=0.0, b=1.0)


def test_rejects_spread_out_positional_parameters():
    # The 1-D `scaled_points(a, b)` spelling is ambiguous across dimensions.
    rule = tensor(_gl(3), _gl(3))
    with pytest.raises(TypeError, match="not 2 positional arguments"):
        rule.scaled_points((0.0, 1.0), (0.0, 1.0))


def test_rejects_unrecognized_parameter_spec():
    rule = tensor(_gl(3), _gl(3))
    with pytest.raises(TypeError, match="per-dimension parameters must be"):
        rule.scaled_points(dims=[3.0, None])


def test_constructor_accepts_an_explicit_tuple():
    assert TensorQuadratureRule((_gl(3), _gl(2))) == tensor(_gl(3), _gl(2))


def test_rules_are_normalized_to_a_tuple():
    rule = TensorQuadratureRule([_gl(3), _gl(2)])
    assert isinstance(rule.rules, tuple)
    assert all(isinstance(r, QuadratureRule) for r in rule.rules)
