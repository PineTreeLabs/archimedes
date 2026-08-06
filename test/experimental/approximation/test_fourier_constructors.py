"""Tests for the ``FourierBasis``-backed ``FunctionSpace.fourier``
classmethod constructor: thin sugar over
``FunctionSpace(FourierBasis(n_basis, kind=kind, density=density), domain)``,
so these mostly check that the right basis/domain-parameter type is built
and that ``kind``/``density``/``quad_rule`` are forwarded correctly.
"""

import numpy as np
import pytest

from archimedes.experimental.approximation import FourierBasis, FunctionSpace
from archimedes.measure import UnitInterval
from archimedes.quadrature import trapezoidal


def test_fourier_matches_manual_construction():
    manual = FunctionSpace(
        FourierBasis(7, kind="full"), UnitInterval.Parameters(a=-2.0, b=3.0)
    )
    sugar = FunctionSpace.fourier(7, a=-2.0, b=3.0)
    assert sugar.n_basis == manual.n_basis
    assert sugar.domain == manual.domain
    np.testing.assert_allclose(
        sugar.basis_matrix().matrix, manual.basis_matrix().matrix
    )


def test_fourier_kind_full_is_default():
    assert FunctionSpace.fourier(5, kind="full") == FunctionSpace.fourier(5)


def test_fourier_kind_cosine_and_sine_dispatch():
    assert FunctionSpace.fourier(4, kind="cosine").basis.kind == "cosine"
    assert FunctionSpace.fourier(4, kind="sine").basis.kind == "sine"


def test_fourier_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind must be"):
        FunctionSpace.fourier(5, kind="bogus")


def test_fourier_forwards_density():
    assert FunctionSpace.fourier(5, density=True).basis.density is True
    assert FunctionSpace.fourier(5).basis.density is False


def test_fourier_default_domain_is_reference_interval():
    sugar = FunctionSpace.fourier(5)
    assert sugar.domain == UnitInterval.Parameters(a=-1.0, b=1.0)


def test_fourier_quad_rule_forwarded():
    rule = trapezoidal(11, periodic=True)
    sugar = FunctionSpace.fourier(5, quad_rule=rule)
    assert sugar.quad_rule is rule
