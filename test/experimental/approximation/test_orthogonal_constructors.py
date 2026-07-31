"""Tests for the ``OrthogonalPolynomialBasis``-backed ``FunctionSpace``
classmethod constructors: ``.legendre``, ``.chebyshev``, ``.jacobi``,
``.hermite``, ``.hermite_norm``, ``.laguerre``. Each is thin sugar over
``FunctionSpace(OrthogonalPolynomialBasis(measure, n_basis), domain)``, so
these mostly check that the right measure/domain-parameter type is built
and that ``density``/``quad_rule`` are forwarded correctly.
"""

import numpy as np
import pytest

from archimedes.experimental.approximation import (
    FunctionSpace,
    OrthogonalPolynomialBasis,
)
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
from archimedes.quadrature import gauss_legendre


def test_legendre_matches_manual_construction():
    manual = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), 8),
        UnitInterval.Parameters(a=-2.0, b=3.0),
    )
    sugar = FunctionSpace.legendre(8, a=-2.0, b=3.0)
    assert sugar.n_basis == manual.n_basis
    assert sugar.domain == manual.domain
    np.testing.assert_allclose(
        sugar.basis_matrix().matrix, manual.basis_matrix().matrix
    )


@pytest.mark.parametrize("second_kind, expected_exponent", [(False, -0.5), (True, 0.5)])
def test_chebyshev_matches_jacobi_special_case(second_kind, expected_exponent):
    sugar = FunctionSpace.chebyshev(6, second_kind=second_kind)
    assert sugar.basis.measure == JacobiMeasure(expected_exponent, expected_exponent)


def test_jacobi_forwards_alpha_beta_and_domain():
    sugar = FunctionSpace.jacobi(0.5, 1.5, 5, a=0.0, b=2.0)
    assert sugar.basis.measure == JacobiMeasure(0.5, 1.5)
    assert sugar.domain == UnitInterval.Parameters(a=0.0, b=2.0)
    assert sugar.n_basis == 5


def test_hermite_uses_physicists_measure_and_real_line_domain():
    sugar = FunctionSpace.hermite(4, mean=1.0, std=2.0)
    assert isinstance(sugar.basis.measure, HermiteMeasure)
    assert sugar.domain == RealLine.Parameters(mean=1.0, std=2.0)


def test_hermite_norm_uses_probabilists_measure():
    sugar = FunctionSpace.hermite_norm(4)
    assert isinstance(sugar.basis.measure, HermiteNormMeasure)


def test_hermite_norm_forwards_density():
    sugar = FunctionSpace.hermite_norm(4, density=True)
    assert sugar.basis.density is True
    default = FunctionSpace.hermite_norm(4)
    assert default.basis.density is False


def test_laguerre_uses_half_line_domain():
    sugar = FunctionSpace.laguerre(4, rate=2.0, start=1.0)
    assert isinstance(sugar.basis.measure, LaguerreMeasure)
    assert sugar.domain == HalfLine.Parameters(rate=2.0, start=1.0)


def test_quad_rule_forwarded():
    rule = gauss_legendre(10)
    sugar = FunctionSpace.legendre(5, quad_rule=rule)
    assert sugar.quad_rule is rule
