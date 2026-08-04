"""Tests for the ``MonomialBasis``-backed ``FunctionSpace.monomial``
classmethod constructor: thin sugar over
``FunctionSpace(MonomialBasis(n_basis), UnitInterval.Parameters(a=a, b=b))``,
so these mostly check that the right basis/domain-parameter type is built
and that ``quad_rule`` is forwarded correctly.
"""

import numpy as np

from archimedes.experimental.approximation import FunctionSpace, MonomialBasis
from archimedes.measure import UnitInterval
from archimedes.quadrature import gauss_legendre


def test_monomial_matches_manual_construction():
    manual = FunctionSpace(MonomialBasis(6), UnitInterval.Parameters(a=-2.0, b=3.0))
    sugar = FunctionSpace.monomial(6, a=-2.0, b=3.0)
    assert sugar.n_basis == manual.n_basis
    assert sugar.domain == manual.domain
    np.testing.assert_allclose(
        sugar.basis_matrix().matrix, manual.basis_matrix().matrix
    )


def test_monomial_default_domain_is_reference_interval():
    sugar = FunctionSpace.monomial(5)
    assert sugar.domain == UnitInterval.Parameters(a=-1.0, b=1.0)


def test_monomial_default_quadrature_is_gauss_legendre():
    sugar = FunctionSpace.monomial(5)
    rule = sugar.basis.default_quadrature()
    expected = gauss_legendre(5)
    np.testing.assert_allclose(rule.nodes, expected.nodes)
    np.testing.assert_allclose(rule.weights, expected.weights)


def test_monomial_quad_rule_forwarded():
    rule = gauss_legendre(9)
    sugar = FunctionSpace.monomial(5, quad_rule=rule)
    assert sugar.quad_rule is rule
