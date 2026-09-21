import numpy as np

from archimedes.approximation import FunctionSpace, MonomialBasis
from archimedes.measure import UnitInterval
from archimedes.quadrature import gauss_legendre


def test_monomial():
    manual = FunctionSpace(MonomialBasis(6), UnitInterval.Parameters(a=-2.0, b=3.0))
    space = FunctionSpace.monomial(6, a=-2.0, b=3.0)
    assert space.n_basis == manual.n_basis
    assert space.domain == manual.domain
    np.testing.assert_allclose(
        space.basis_matrix().matrix, manual.basis_matrix().matrix
    )

    # Default domain is the reference interval, with Gauss-Legendre
    # quadrature at n_basis points.
    space = FunctionSpace.monomial(5)
    assert space.domain == UnitInterval.Parameters(a=-1.0, b=1.0)
    rule = space.basis.default_quadrature()
    expected = gauss_legendre(5)
    np.testing.assert_allclose(rule.nodes, expected.nodes)
    np.testing.assert_allclose(rule.weights, expected.weights)

    # quad_rule overrides the default.
    rule = gauss_legendre(9)
    space = FunctionSpace.monomial(5, quad_rule=rule)
    assert space.reference_quad_rule is rule
