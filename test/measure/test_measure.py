import numpy as np
import pytest
from scipy.special import beta as beta_fn

from archimedes import tree
from archimedes.measure import (
    HermiteMeasure,
    HermiteNormMeasure,
    JacobiMeasure,
    LaguerreMeasure,
    LegendreMeasure,
)

# -- Measure implementations --


def test_legendre_measure():
    measure = LegendreMeasure()
    assert measure.uniform_weight is True
    assert measure.support == (-1.0, 1.0)
    np.testing.assert_array_equal(measure.weight(np.array([-0.5, 0.5])), [1.0, 1.0])
    assert measure.reference_mass == 2.0

    assert measure.affine_params() == (1.0, 0.0)

    with pytest.raises(ValueError):
        measure.affine_params(a=0.0)
    with pytest.raises(ValueError):
        measure.affine_params(b=1.0)
    with pytest.raises(ValueError):
        measure.affine_params(a=-np.inf, b=1.0)


@pytest.mark.parametrize("alpha,beta", [(-1.0, 0.0), (0.0, -1.0)])
def test_jacobi_measure_invalid_parameters(alpha, beta):
    with pytest.raises(ValueError):
        JacobiMeasure(alpha=alpha, beta=beta)


def test_jacobi_measure_weight_and_shared_affine_params():
    measure = JacobiMeasure(alpha=1.0, beta=2.0)
    assert measure.uniform_weight is False
    assert measure.support == (-1.0, 1.0)

    x = np.array([0.0, 0.5])
    expected = (1 - x) ** 1.0 * (1 + x) ** 2.0
    np.testing.assert_allclose(measure.weight(x), expected)

    expected_mass = 2 ** (1.0 + 2.0 + 1) * beta_fn(1.0 + 1, 2.0 + 1)
    assert np.isclose(measure.reference_mass, expected_mass)

    # Shares the interval mapping with LegendreMeasure
    assert measure.affine_params(0.0, 2.0) == LegendreMeasure().affine_params(0.0, 2.0)


def test_laguerre_measure():
    measure = LaguerreMeasure()
    assert measure.uniform_weight is False
    assert measure.support == (0.0, np.inf)
    np.testing.assert_allclose(
        measure.weight(np.array([0.0, 1.0])), [1.0, np.exp(-1.0)]
    )
    assert measure.reference_mass == 1.0

    assert measure.affine_params() == (1.0, 0.0)

    scale, shift = measure.affine_params(rate=2.0, start=1.0)
    assert np.isclose(scale, 0.5)
    assert np.isclose(shift, 1.0)

    # rate defaults to 1.0 when only start is given
    scale, shift = measure.affine_params(start=1.0)
    assert np.isclose(scale, 1.0)
    assert np.isclose(shift, 1.0)

    with pytest.raises(ValueError):
        measure.affine_params(rate=-1.0)


def test_hermite_measure():
    measure = HermiteMeasure()
    assert measure.uniform_weight is False
    assert measure.support == (-np.inf, np.inf)
    np.testing.assert_allclose(
        measure.weight(np.array([0.0, 1.0])), [1.0, np.exp(-1.0)]
    )
    assert np.isclose(measure.reference_mass, np.sqrt(np.pi))

    assert measure.affine_params() == (1.0, 0.0)

    scale, shift = measure.affine_params(mean=1.0, std=2.0)
    assert np.isclose(scale, 2.0)
    assert np.isclose(shift, 1.0)

    # std defaults to 1.0 when only mean is given
    scale, shift = measure.affine_params(mean=1.0)
    assert np.isclose(scale, 1.0)
    assert np.isclose(shift, 1.0)

    with pytest.raises(ValueError):
        measure.affine_params(std=-1.0)


# -- Measure.Parameters structs --


def test_legendre_parameters_flatten_and_replace():
    params = LegendreMeasure.Parameters(a=0.0, b=2.0)
    assert params.a == 0.0
    assert params.b == 2.0
    assert tree.is_struct(params)

    flat, _ = tree.flatten(params)
    assert flat == [0.0, 2.0]

    updated = params.replace(b=4.0)
    assert updated.a == 0.0
    assert updated.b == 4.0


def test_legendre_parameters_defaults_and_validation():
    identity = LegendreMeasure.Parameters()
    assert identity.a is None
    assert identity.b is None

    with pytest.raises(ValueError):
        LegendreMeasure.Parameters(a=0.0)
    with pytest.raises(ValueError):
        LegendreMeasure.Parameters(b=1.0)
    with pytest.raises(ValueError):
        LegendreMeasure.Parameters(a=-np.inf, b=1.0)


def test_jacobi_parameters_shares_legendre_parameters_type():
    # Jacobi doesn't override affine_params, so it shares Legendre's Parameters
    assert JacobiMeasure.Parameters is LegendreMeasure.Parameters


def test_laguerre_parameters_defaults_and_validation():
    identity = LaguerreMeasure.Parameters()
    assert identity.rate == 1.0
    assert identity.start == 0.0

    params = LaguerreMeasure.Parameters(rate=2.0, start=1.0)
    assert tree.is_struct(params)
    flat, _ = tree.flatten(params)
    assert flat == [2.0, 1.0]

    with pytest.raises(ValueError):
        LaguerreMeasure.Parameters(rate=-1.0)


@pytest.mark.parametrize("measure_cls", [HermiteMeasure, HermiteNormMeasure])
def test_hermite_parameters_defaults_and_validation(measure_cls):
    identity = measure_cls.Parameters()
    assert identity.mean == 0.0
    assert identity.std == 1.0

    params = measure_cls.Parameters(mean=1.0, std=2.0)
    assert tree.is_struct(params)
    flat, _ = tree.flatten(params)
    assert flat == [1.0, 2.0]

    with pytest.raises(ValueError):
        measure_cls.Parameters(std=-1.0)


def test_measure_mass_reference_domain():
    # mass() with no args is just reference_mass (scale == 1)
    assert LegendreMeasure().mass() == LegendreMeasure().reference_mass
    assert HermiteMeasure().mass() == HermiteMeasure().reference_mass
    assert LaguerreMeasure().mass() == LaguerreMeasure().reference_mass


def test_measure_mass_mapped_domain():
    # Legendre: mapping [-1, 1] (mass 2) onto [0, 4] (width 4) scales mass by 2
    assert LegendreMeasure().mass(0.0, 4.0) == 4.0

    # Hermite (prob.): mass scales by std, matching the affine Jacobian
    measure = HermiteNormMeasure()
    assert np.isclose(measure.mass(mean=1.0, std=2.0), 2.0 * measure.reference_mass)

    # mass() is exactly what scaled_weights(density=True) divides by
    measure = LaguerreMeasure()
    scale, _ = measure.affine_params(rate=2.0)
    assert np.isclose(measure.mass(rate=2.0), scale * measure.reference_mass)


def test_hermite_and_hermitenorm_parameters_are_distinct_types():
    # Same field shape, but kept as separate types since the measures are
    # separate (mirrors HermiteMeasure vs. HermiteNormMeasure not sharing an
    # `affine_params` implementation).
    assert HermiteMeasure.Parameters is not HermiteNormMeasure.Parameters


def test_hermite_norm_measure():
    measure = HermiteNormMeasure()
    assert measure.uniform_weight is False
    assert measure.support == (-np.inf, np.inf)
    np.testing.assert_allclose(
        measure.weight(np.array([0.0, 1.0])), [1.0, np.exp(-0.5)]
    )
    assert np.isclose(measure.reference_mass, np.sqrt(2 * np.pi))

    assert measure.affine_params() == (1.0, 0.0)

    scale, shift = measure.affine_params(mean=1.0, std=2.0)
    assert np.isclose(scale, 2.0)
    assert np.isclose(shift, 1.0)

    # std defaults to 1.0 when only mean is given
    scale, shift = measure.affine_params(mean=1.0)
    assert np.isclose(scale, 1.0)
    assert np.isclose(shift, 1.0)

    with pytest.raises(ValueError):
        measure.affine_params(std=-1.0)
