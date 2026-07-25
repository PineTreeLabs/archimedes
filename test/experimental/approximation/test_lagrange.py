import numpy as np
import pytest

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import LagrangeBasis
from archimedes.measure import UnitInterval
from archimedes.quadrature import gauss_lobatto


@pytest.fixture
def nodes():
    # Gauss-Lobatto nodes: well-conditioned, and include both endpoints
    # (useful for exercising the exact-node code path at the domain edges).
    return gauss_lobatto(6).nodes


def test_rejects_out_of_range_nodes():
    with pytest.raises(ValueError):
        LagrangeBasis(reference_nodes=np.array([-1.0, 0.0, 1.5]))


def test_rejects_duplicate_nodes():
    with pytest.raises(ValueError):
        LagrangeBasis(reference_nodes=np.array([-1.0, 0.0, 0.0, 1.0]))


def test_cardinal_property(nodes):
    # ell_i(x_j) = delta_ij, including at the endpoint nodes (the 0/0 case).
    basis = LagrangeBasis(reference_nodes=nodes)
    phi = basis.evaluate(nodes)
    np.testing.assert_allclose(phi, np.eye(len(nodes)), atol=1e-10)


def test_partition_of_unity(nodes):
    basis = LagrangeBasis(reference_nodes=nodes)
    x = np.linspace(-1, 1, 25)
    phi = basis.evaluate(x)
    np.testing.assert_allclose(phi.sum(axis=1), 1.0, atol=1e-10)


def test_interpolates_polynomial_within_degree_exactly(nodes):
    # n_basis=6 nodes -> exact reproduction through degree 5.
    basis = LagrangeBasis(reference_nodes=nodes)

    def f(x):
        return 3 * x**5 - 2 * x**3 + x - 1

    phi_at_nodes = basis.evaluate(nodes)
    np.testing.assert_allclose(phi_at_nodes, np.eye(len(nodes)), atol=1e-10)

    yp = f(nodes)
    x = np.linspace(-1, 1, 17)
    interp = basis.evaluate(x) @ yp
    np.testing.assert_allclose(interp, f(x), atol=1e-8)


def test_domain_mapping_preserves_cardinal_property(nodes):
    a, b = 2.0, 7.0
    basis = LagrangeBasis(reference_nodes=nodes)
    scale, shift = UnitInterval().affine_params(a, b)
    mapped_nodes = scale * nodes + shift

    phi = basis.evaluate(mapped_nodes, a=a, b=b)
    np.testing.assert_allclose(phi, np.eye(len(nodes)), atol=1e-10)


def test_domain_mapping_interpolates_exactly(nodes):
    a, b = 2.0, 7.0
    basis = LagrangeBasis(reference_nodes=nodes)
    scale, shift = UnitInterval().affine_params(a, b)
    mapped_nodes = scale * nodes + shift

    def f(x):
        return 3 * x**5 - 2 * x**3 + x - 1

    yp = f(mapped_nodes)
    x = np.linspace(a, b, 17)
    interp = basis.evaluate(x, a=a, b=b) @ yp
    np.testing.assert_allclose(interp, f(x), atol=1e-6)


def test_unsupported_derivative_order(nodes):
    basis = LagrangeBasis(reference_nodes=nodes)
    with pytest.raises(NotImplementedError):
        basis.evaluate(np.array([0.0]), deriv=2)


# -- derivatives --


def test_derivative_matches_finite_difference(nodes):
    basis = LagrangeBasis(reference_nodes=nodes)
    # Deliberately off-node points; the at-node branch is covered separately.
    x = np.linspace(-0.93, 0.91, 17)
    h = 1e-6
    dphi = basis.evaluate(x, deriv=1)
    dphi_fd = (basis.evaluate(x + h) - basis.evaluate(x - h)) / (2 * h)
    np.testing.assert_allclose(dphi, dphi_fd, atol=1e-5)


def test_derivative_at_nodes_matches_finite_difference(nodes):
    # At a node the general barycentric formula is 0/0, so this exercises the
    # differentiation-matrix branch instead.
    basis = LagrangeBasis(reference_nodes=nodes)
    h = 1e-6
    dphi = basis.evaluate(nodes, deriv=1)
    dphi_fd = (basis.evaluate(nodes + h) - basis.evaluate(nodes - h)) / (2 * h)
    np.testing.assert_allclose(dphi, dphi_fd, atol=1e-5)


def test_derivative_reproduces_polynomial_derivative(nodes):
    basis = LagrangeBasis(reference_nodes=nodes)

    def f(x):
        return 3 * x**5 - 2 * x**3 + x - 1

    def df(x):
        return 15 * x**4 - 6 * x**2 + 1

    x = np.linspace(-1, 1, 21)
    np.testing.assert_allclose(basis.evaluate(x, deriv=1) @ f(nodes), df(x), atol=1e-8)


def test_derivatives_sum_to_zero(nodes):
    # d/dx of the partition of unity.
    basis = LagrangeBasis(reference_nodes=nodes)
    x = np.linspace(-1, 1, 21)
    np.testing.assert_allclose(basis.evaluate(x, deriv=1).sum(axis=1), 0.0, atol=1e-9)


def test_derivative_on_mapped_domain(nodes):
    a, b = 2.0, 7.0
    basis = LagrangeBasis(reference_nodes=nodes)
    scale, shift = UnitInterval().affine_params(a, b)
    mapped = scale * nodes + shift

    def f(x):
        return 3 * x**2 - 2 * x + 1

    def df(x):
        return 6 * x - 2

    # Includes the mapped nodes, to cover the chain-rule factor on the
    # differentiation-matrix branch as well as the generic one.
    x = np.concatenate([np.linspace(a, b, 15), mapped])
    np.testing.assert_allclose(
        basis.evaluate(x, deriv=1, a=a, b=b) @ f(mapped), df(x), atol=1e-8
    )


# -- static (NumPy) vs. dynamic (symbolic, via arc.compile) equivalence --


def test_static_and_dynamic_evaluation_agree_including_at_nodes(nodes):
    basis = LagrangeBasis(reference_nodes=nodes)
    # Include exact node points -- the tricky 0/0 branch -- alongside
    # ordinary interior points.
    x = np.concatenate([nodes, np.array([-0.3, 0.0, 0.55])])
    static_phi = basis.evaluate(x)

    @arc.compile
    def traced(x):
        assert isinstance(x, SymbolicArray)
        return basis.evaluate(np.atleast_1d(x))

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-10)
