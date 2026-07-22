import math

import numpy as np
import pytest
from scipy.special import beta as beta_fn
from scipy.special import roots_hermite, roots_hermitenorm, roots_jacobi, roots_laguerre

import archimedes as arc
from archimedes.experimental.polynomial.orthogonal import (
    LegendreMeasure,
    HermiteMeasure,
    HermiteNormMeasure,
    JacobiMeasure,
    LaguerreMeasure,
)



# -- _QuadratureFamily implementations --


def test_legendre_measure():
    measure = LegendreMeasure()
    assert measure.uniform_weight is True
    assert measure.support == (-1.0, 1.0)
    np.testing.assert_array_equal(measure.weight(np.array([-0.5, 0.5])), [1.0, 1.0])

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

    # Shares the interval mapping with LegendreMeasure
    assert measure.affine_params(0.0, 2.0) == LegendreMeasure().affine_params(0.0, 2.0)


def test_laguerre_measure():
    measure = LaguerreMeasure()
    assert measure.uniform_weight is False
    assert measure.support == (0.0, np.inf)
    np.testing.assert_allclose(measure.weight(np.array([0.0, 1.0])), [1.0, np.exp(-1.0)])

    assert measure.affine_params() == (1.0, 0.0)

    scale, shift = measure.affine_params(rate=2.0, start=1.0)
    assert np.isclose(scale, 0.5)
    assert np.isclose(shift, 1.0)

    with pytest.raises(ValueError):
        measure.affine_params(rate=-1.0)


def test_hermite_measure():
    measure = HermiteMeasure()
    assert measure.uniform_weight is False
    assert measure.support == (-np.inf, np.inf)
    np.testing.assert_allclose(measure.weight(np.array([0.0, 1.0])), [1.0, np.exp(-1.0)])

    assert measure.affine_params() == (1.0, 0.0)

    scale, shift = measure.affine_params(mean=1.0, std=2.0)
    assert np.isclose(scale, 2.0)
    assert np.isclose(shift, 1.0)

    with pytest.raises(ValueError):
        measure.affine_params(std=-1.0)


def test_hermite_norm_measure():
    measure = HermiteNormMeasure()
    assert measure.uniform_weight is False
    assert measure.support == (-np.inf, np.inf)
    np.testing.assert_allclose(measure.weight(np.array([0.0, 1.0])), [1.0, np.exp(-0.5)])

    assert measure.affine_params() == (1.0, 0.0)

    scale, shift = measure.affine_params(mean=1.0, std=2.0)
    assert np.isclose(scale, 2.0)
    assert np.isclose(shift, 1.0)

    with pytest.raises(ValueError):
        measure.affine_params(std=-1.0)