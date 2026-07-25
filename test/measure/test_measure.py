import numpy as np
import pytest
from scipy.special import beta as beta_fn

from archimedes import tree
from archimedes.measure import (
    HalfLine,
    HermiteMeasure,
    HermiteNormMeasure,
    JacobiMeasure,
    LaguerreMeasure,
    LegendreMeasure,
    RealLine,
    UnitInterval,
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


# -- ReferenceDomain.Parameters structs --


def test_legendre_parameters_flatten_and_replace():
    params = UnitInterval.Parameters(a=0.0, b=2.0)
    assert params.a == 0.0
    assert params.b == 2.0
    assert tree.is_struct(params)

    flat, _ = tree.flatten(params)
    assert flat == [0.0, 2.0]

    updated = params.replace(b=4.0)
    assert updated.a == 0.0
    assert updated.b == 4.0


def test_legendre_parameters_defaults_and_validation():
    identity = UnitInterval.Parameters()
    assert identity.a is None
    assert identity.b is None

    with pytest.raises(ValueError):
        UnitInterval.Parameters(a=0.0)
    with pytest.raises(ValueError):
        UnitInterval.Parameters(b=1.0)
    with pytest.raises(ValueError):
        UnitInterval.Parameters(a=-np.inf, b=1.0)


def test_families_sharing_a_domain_share_its_parameters_type():
    # Legendre and Jacobi differ only in weight, not domain, so both are
    # UnitInterval -- as are both Hermite conventions on RealLine. The
    # parameters belong to the domain, so sharing one means sharing them.
    assert type(JacobiMeasure(alpha=1.0, beta=2.0).domain) is UnitInterval
    assert type(LegendreMeasure().domain) is UnitInterval
    assert type(HermiteMeasure().domain) is RealLine
    assert type(HermiteNormMeasure().domain) is RealLine
    assert type(LaguerreMeasure().domain) is HalfLine


def test_laguerre_parameters_defaults_and_validation():
    identity = HalfLine.Parameters()
    assert identity.rate == 1.0
    assert identity.start == 0.0

    params = HalfLine.Parameters(rate=2.0, start=1.0)
    assert tree.is_struct(params)
    flat, _ = tree.flatten(params)
    assert flat == [2.0, 1.0]

    with pytest.raises(ValueError):
        HalfLine.Parameters(rate=-1.0)


@pytest.mark.parametrize("domain_cls", [RealLine])
def test_hermite_parameters_defaults_and_validation(domain_cls):
    identity = domain_cls.Parameters()
    assert identity.mean == 0.0
    assert identity.std == 1.0

    params = domain_cls.Parameters(mean=1.0, std=2.0)
    assert tree.is_struct(params)
    flat, _ = tree.flatten(params)
    assert flat == [1.0, 2.0]

    with pytest.raises(ValueError):
        domain_cls.Parameters(std=-1.0)


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


def test_hermite_and_hermitenorm_share_a_domain_parameters_type():
    # These were separate types before the domain refactor. They're now one:
    # both measures live on the same location-scaled RealLine and had
    # byte-identical affine_params. Only the *weight* differs (and hence the
    # interpretation of `std` relative to it -- see the measure docstrings),
    # which is a Measure concern, not a domain one.
    assert HermiteMeasure().domain.Parameters is HermiteNormMeasure().domain.Parameters


# -- recurrence_coeffs --


def test_legendre_recurrence_coeffs():
    measure = LegendreMeasure()
    alpha, beta = measure.recurrence_coeffs(4)
    np.testing.assert_array_equal(alpha, [0.0, 0.0, 0.0, 0.0])
    assert beta[0] == measure.reference_mass == 2.0
    np.testing.assert_allclose(beta[1:], [1 / 3, 4 / 15, 9 / 35])


@pytest.mark.parametrize(
    "alpha,beta,expected_beta1",
    [
        (0.0, 0.0, 1 / 3),  # Legendre special case
        (-0.5, -0.5, 0.5),  # Chebyshev 1st kind
        (0.5, 0.5, 0.25),  # Chebyshev 2nd kind
        (1.0, 2.0, 0.16),  # generic asymmetric
    ],
)
def test_jacobi_recurrence_coeffs(alpha, beta, expected_beta1):
    measure = JacobiMeasure(alpha=alpha, beta=beta)
    a, b = measure.recurrence_coeffs(5)
    assert b[0] == measure.reference_mass
    assert np.isclose(b[1], expected_beta1)

    # alpha_0 = (beta - alpha) / (alpha + beta + 2), directly from the
    # (simplified) closed form -- not the k >= 1 general formula
    assert np.isclose(a[0], (beta - alpha) / (alpha + beta + 2))


def test_jacobi_recurrence_coeffs_matches_legendre():
    # Legendre is the alpha=beta=0 special case of Jacobi
    jacobi_a, jacobi_b = JacobiMeasure(alpha=0.0, beta=0.0).recurrence_coeffs(5)
    legendre_a, legendre_b = LegendreMeasure().recurrence_coeffs(5)
    np.testing.assert_allclose(jacobi_a, legendre_a)
    np.testing.assert_allclose(jacobi_b, legendre_b)


def test_jacobi_recurrence_coeffs_chebyshev_first_kind():
    # Known monic recurrence for Chebyshev T: alpha_k = 0, beta_1 = 1/2,
    # beta_k = 1/4 for k >= 2
    _, beta = JacobiMeasure(alpha=-0.5, beta=-0.5).recurrence_coeffs(5)
    np.testing.assert_allclose(beta[2:], [0.25, 0.25, 0.25])


def test_laguerre_recurrence_coeffs():
    measure = LaguerreMeasure()
    alpha, beta = measure.recurrence_coeffs(4)
    np.testing.assert_array_equal(alpha, [1.0, 3.0, 5.0, 7.0])
    assert beta[0] == measure.reference_mass == 1.0
    np.testing.assert_array_equal(beta[1:], [1.0, 4.0, 9.0])


def test_hermite_recurrence_coeffs():
    measure = HermiteMeasure()
    alpha, beta = measure.recurrence_coeffs(4)
    np.testing.assert_array_equal(alpha, [0.0, 0.0, 0.0, 0.0])
    assert beta[0] == measure.reference_mass
    np.testing.assert_array_equal(beta[1:], [0.5, 1.0, 1.5])


def test_hermite_norm_recurrence_coeffs():
    measure = HermiteNormMeasure()
    alpha, beta = measure.recurrence_coeffs(4)
    np.testing.assert_array_equal(alpha, [0.0, 0.0, 0.0, 0.0])
    assert beta[0] == measure.reference_mass
    np.testing.assert_array_equal(beta[1:], [1.0, 2.0, 3.0])


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
