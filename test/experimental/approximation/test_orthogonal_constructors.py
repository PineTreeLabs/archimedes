"""Tests for the ``OrthogonalPolynomialBasis``-backed ``FunctionSpace``
classmethod constructors: ``.legendre``, ``.chebyshev``, ``.jacobi``,
``.hermite``, ``.laguerre``. Each is thin sugar over
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
    JacobiMeasure,
    LaguerreMeasure,
    LegendreMeasure,
    PhysicistsHermiteMeasure,
    ProbabilistsHermiteMeasure,
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


def test_hermite_defaults_to_physicists_measure():
    sugar = FunctionSpace.hermite(4, mean=1.0, std=2.0)
    assert isinstance(sugar.basis.measure, PhysicistsHermiteMeasure)
    assert sugar.domain == RealLine.Parameters(mean=1.0, std=2.0)


def test_hermite_kind_phys_is_explicit_default():
    assert FunctionSpace.hermite(4, kind="phys") == FunctionSpace.hermite(4)


def test_hermite_kind_prob_uses_probabilists_measure():
    sugar = FunctionSpace.hermite(4, kind="prob")
    assert isinstance(sugar.basis.measure, ProbabilistsHermiteMeasure)


def test_hermite_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Hermite kind must be"):
        FunctionSpace.hermite(4, kind="bogus")


def test_hermite_forwards_density_regardless_of_kind():
    # `density=True` is what normalizes to a probability measure, for
    # either `kind` -- neither measure's own weight integrates to 1 on
    # its own.
    for kind in ("phys", "prob"):
        assert FunctionSpace.hermite(4, kind=kind, density=True).basis.density is True
        assert FunctionSpace.hermite(4, kind=kind).basis.density is False


def test_laguerre_uses_half_line_domain():
    sugar = FunctionSpace.laguerre(4, rate=2.0, start=1.0)
    assert isinstance(sugar.basis.measure, LaguerreMeasure)
    assert sugar.domain == HalfLine.Parameters(rate=2.0, start=1.0)


def test_quad_rule_forwarded():
    rule = gauss_legendre(10)
    sugar = FunctionSpace.legendre(5, quad_rule=rule)
    assert sugar.quad_rule is rule
