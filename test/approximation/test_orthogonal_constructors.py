import numpy as np
import pytest

from archimedes.approximation import (
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


def test_legendre():
    manual = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), 8),
        UnitInterval.Parameters(a=-2.0, b=3.0),
    )
    space = FunctionSpace.legendre(8, a=-2.0, b=3.0)
    assert space.n_basis == manual.n_basis
    assert space.domain == manual.domain
    np.testing.assert_allclose(
        space.basis_matrix().matrix, manual.basis_matrix().matrix
    )


@pytest.mark.parametrize("second_kind, expected_exponent", [(False, -0.5), (True, 0.5)])
def test_chebyshev(second_kind, expected_exponent):
    space = FunctionSpace.chebyshev(6, second_kind=second_kind)
    assert space.basis.measure == JacobiMeasure(expected_exponent, expected_exponent)


def test_jacobi():
    space = FunctionSpace.jacobi(0.5, 1.5, 5, a=0.0, b=2.0)
    assert space.basis.measure == JacobiMeasure(0.5, 1.5)
    assert space.domain == UnitInterval.Parameters(a=0.0, b=2.0)
    assert space.n_basis == 5


def test_hermite():
    space = FunctionSpace.hermite(4, loc=1.0, scale=2.0)
    assert isinstance(space.basis.measure, ProbabilistsHermiteMeasure)
    assert space.domain == RealLine.Parameters(loc=1.0, scale=2.0)

    space = FunctionSpace.hermite(4, kind="phys")
    assert isinstance(space.basis.measure, PhysicistsHermiteMeasure)

    with pytest.raises(ValueError, match="Hermite kind must be"):
        FunctionSpace.hermite(4, kind="bogus")

    # density=True normalizes to a probability measure, for either kind.
    for kind in ("phys", "prob"):
        assert FunctionSpace.hermite(4, kind=kind, density=True).basis.density is True
        assert FunctionSpace.hermite(4, kind=kind).basis.density is False


def test_laguerre():
    space = FunctionSpace.laguerre(4, rate=2.0, start=1.0)
    assert isinstance(space.basis.measure, LaguerreMeasure)
    assert space.domain == HalfLine.Parameters(rate=2.0, start=1.0)
