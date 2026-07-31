import numpy as np
import pytest

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import LagrangeBasis, PiecewiseBasis
from archimedes.measure import UnitInterval
from archimedes.quadrature import gauss_legendre as gauss_legendre_rule
from archimedes.quadrature import gauss_lobatto, gauss_radau


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


def test_negative_derivative_order_rejected(nodes):
    basis = LagrangeBasis(reference_nodes=nodes)
    with pytest.raises(ValueError, match="deriv must be >= 0"):
        basis.evaluate(np.array([0.0]), deriv=-1)


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


# -- higher derivatives --


class TestHigherDerivatives:
    """``Phi^(k) = Phi @ D**k`` is exact, not approximate: each cardinal
    polynomial's k-th derivative has degree <= n-1 and so lies in the span
    of the basis itself. These check against exact polynomial derivatives
    rather than finite differences, which are far too noisy above k=1."""

    N = 7

    @pytest.fixture
    def basis(self):
        return LagrangeBasis(reference_nodes=gauss_lobatto(self.N).nodes)

    @pytest.fixture
    def poly(self):
        # Degree n-1: exactly representable, and its k-th derivative is a
        # nonzero polynomial for every k <= n-1.
        rng = np.random.default_rng(0)
        return np.polynomial.Polynomial(rng.normal(size=self.N))

    @pytest.mark.parametrize("deriv", [0, 1, 2, 3, 4])
    def test_exact_on_reference_domain(self, basis, poly, deriv):
        x = np.linspace(-0.95, 0.95, 17)
        values = poly(basis.reference_nodes)
        got = basis.evaluate(x, deriv=deriv) @ values
        np.testing.assert_allclose(got, poly.deriv(deriv)(x), atol=1e-10)

    @pytest.mark.parametrize("deriv", [1, 2, 3])
    def test_exact_on_mapped_domain(self, basis, poly, deriv):
        # The chain-rule factor is 1/scale per derivative order.
        a, b = 0.0, 3.0
        scale, shift = UnitInterval().affine_params(a, b)
        x = scale * np.linspace(-0.95, 0.95, 17) + shift
        values = poly(scale * basis.reference_nodes + shift)
        got = basis.evaluate(x, deriv=deriv, a=a, b=b) @ values
        np.testing.assert_allclose(got, poly.deriv(deriv)(x), atol=1e-9)

    @pytest.mark.parametrize("deriv", [1, 2, 3])
    def test_exact_at_nodes(self, basis, poly, deriv):
        # At a node phi is a unit vector, so phi @ D**k is a row of D**k --
        # the 0/0 case is subsumed rather than special-cased.
        x = basis.reference_nodes
        got = basis.evaluate(x, deriv=deriv) @ poly(basis.reference_nodes)
        np.testing.assert_allclose(got, poly.deriv(deriv)(x), atol=1e-10)

    def test_first_derivative_matches_barycentric_formula(self, basis):
        # Cross-check against the analytic barycentric derivative, which the
        # matrix-power identity replaced.
        x = np.linspace(-0.9, 0.9, 11)
        w = basis._weights
        xp = basis.reference_nodes
        xdiff = x[:, None] - xp[None, :]
        temp = w[None, :] / xdiff
        den = np.sum(temp, axis=1)
        phi = temp / den[:, None]
        t_sum = np.sum(temp / xdiff, axis=1)
        expected = phi * (t_sum[:, None] / den[:, None] - 1.0 / xdiff)
        np.testing.assert_allclose(basis.evaluate(x, deriv=1), expected, atol=1e-11)

    def test_vanishes_beyond_polynomial_degree(self, basis):
        # Degree is n-1, so the n-th derivative and beyond are identically 0.
        x = np.linspace(-0.9, 0.9, 5)
        for deriv in (self.N, self.N + 3):
            got = basis.evaluate(x, deriv=deriv)
            assert got.shape == (len(x), self.N)
            np.testing.assert_array_equal(got, 0.0)

    @pytest.mark.parametrize("deriv", [1, 2, 3])
    def test_symbolic_matches_numeric(self, basis, deriv):
        x = np.concatenate([np.array([-0.62, 0.31]), basis.reference_nodes])
        expected = basis.evaluate(x, deriv=deriv)

        @arc.compile
        def traced(xi):
            assert isinstance(xi, SymbolicArray)
            return basis.evaluate(np.atleast_1d(xi), deriv=deriv)

        actual = np.array([np.asarray(traced(xi)).ravel() for xi in x])
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize("deriv", [2, 3])
    def test_piecewise_inherits_higher_derivatives(self, basis, deriv):
        # The whole point of fixing this: stiffness-like operators on a
        # piecewise space need element derivatives above first order.
        bp = np.linspace(-1.0, 1.0, 3)
        pw = PiecewiseBasis(basis, bp, continuity=0)
        x = np.array([-0.7, -0.2, 0.35, 0.8])
        got = pw.evaluate(x, deriv=deriv)
        assert got.shape == (len(x), pw.n_basis)
        assert np.isfinite(got).all()


# -- named node-family constructors --


class TestNodeFamilyConstructors:
    """Each classmethod's ``reference_nodes`` match the corresponding
    ``archimedes.quadrature`` rule (or ``np.linspace``) directly, and its
    ``node_family`` regenerates the same points at a different size --
    which is also what :meth:`_derivative_basis`/:meth:`_product_basis`
    rely on to stay in the same family.
    """

    def test_gauss_lobatto_nodes(self):
        basis = LagrangeBasis.gauss_lobatto(6)
        np.testing.assert_array_equal(basis.reference_nodes, gauss_lobatto(6).nodes)
        assert basis.node_family is None  # the family default, left unset

    def test_gauss_legendre_nodes(self):
        basis = LagrangeBasis.gauss_legendre(6)
        np.testing.assert_array_equal(
            basis.reference_nodes, gauss_legendre_rule(6).nodes
        )
        assert basis.boundary_dofs() == (None, None)  # no endpoint nodes

    @pytest.mark.parametrize("endpoint", ["left", "right"])
    def test_gauss_radau_nodes(self, endpoint):
        basis = LagrangeBasis.gauss_radau(6, endpoint=endpoint)
        np.testing.assert_array_equal(
            basis.reference_nodes, gauss_radau(6, endpoint=endpoint).nodes
        )
        # Radau fixes exactly one endpoint -- the other is interior.
        left, right = basis.boundary_dofs()
        assert (left is not None) != (right is not None)

    def test_gauss_radau_rejects_bad_endpoint(self):
        with pytest.raises(ValueError, match="endpoint must be"):
            LagrangeBasis.gauss_radau(6, endpoint="middle")

    def test_equispaced_nodes(self):
        basis = LagrangeBasis.equispaced(5)
        np.testing.assert_array_equal(basis.reference_nodes, np.linspace(-1.0, 1.0, 5))
        assert basis.boundary_dofs() == (0, 4)  # both endpoints included

    @pytest.mark.parametrize(
        "ctor", [LagrangeBasis.gauss_legendre, LagrangeBasis.equispaced]
    )
    def test_node_family_equal_across_independent_instances(self, ctor):
        # The subtlety this whole test class exists to pin down: an inline
        # `lambda` (or a `functools.partial`, which -- perhaps surprisingly
        # -- has no value-based `__eq__` either) would make two otherwise-
        # identical instances compare unequal, silently breaking
        # `_product_basis` and basis equality between two spaces built the
        # same way.
        a, b = ctor(6), ctor(6)
        assert a == b
        assert hash(a) == hash(b)

    def test_gauss_radau_node_family_differs_by_endpoint(self):
        left = LagrangeBasis.gauss_radau(6, endpoint="left")
        right = LagrangeBasis.gauss_radau(6, endpoint="right")
        assert left != right
        assert left.node_family != right.node_family

    def test_gauss_radau_node_family_equal_for_same_endpoint(self):
        a = LagrangeBasis.gauss_radau(6, endpoint="left")
        b = LagrangeBasis.gauss_radau(6, endpoint="left")
        assert a == b
        assert hash(a) == hash(b)

    @pytest.mark.parametrize(
        "ctor",
        [
            LagrangeBasis.gauss_legendre,
            LagrangeBasis.equispaced,
            lambda n: LagrangeBasis.gauss_radau(n, endpoint="left"),
        ],
    )
    def test_derivative_basis_regenerates_from_same_family(self, ctor):
        basis = ctor(6)
        derived = basis._derivative_basis(2)
        assert derived.node_family == basis.node_family
        assert derived.n_basis == 4

    def test_product_basis_regenerates_from_same_family(self):
        a = LagrangeBasis.gauss_legendre(4)
        b = LagrangeBasis.gauss_legendre(5)
        prod = a._product_basis(b)
        assert prod.node_family == a.node_family
        assert prod.n_basis == 8

    def test_product_basis_rejects_mismatched_family(self):
        a = LagrangeBasis.gauss_radau(5, endpoint="left")
        b = LagrangeBasis.gauss_radau(5, endpoint="right")
        with pytest.raises(ValueError, match="node_family"):
            a._product_basis(b)
