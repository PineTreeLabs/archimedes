# ruff: noqa: N806  (M is the conventional name for a mass matrix)
import numpy as np
import pytest
from _helpers import mass_matrix, stiffness_matrix

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.approximation import (
    CubicHermiteBasis,
    FunctionSpace,
    LagrangeBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
)
from archimedes.approximation._function_space import _normalize_breakpoints
from archimedes.measure import LegendreMeasure, UnitInterval
from archimedes.quadrature import composite_quad, gauss_legendre, gauss_lobatto


@pytest.fixture
def local():
    # Gauss-Lobatto includes both endpoints, so it has boundary DOFs.
    return LagrangeBasis(reference_nodes=gauss_lobatto(3).nodes)


@pytest.fixture
def breakpoints():
    return np.linspace(-1.0, 1.0, 4)  # 3 elements


@pytest.fixture
def hermite():
    return CubicHermiteBasis()


# -- construction / validation --


def test_construction_validation(local, breakpoints):
    with pytest.raises(ValueError):
        PiecewiseBasis(local, np.array([-1.0, 0.0, 0.5]))  # wrong right endpoint
    with pytest.raises(ValueError):
        PiecewiseBasis(local, np.array([-0.5, 0.0, 1.0]))  # wrong left endpoint
    with pytest.raises(ValueError):
        PiecewiseBasis(local, np.array([-1.0, 0.5, 0.0, 1.0]))  # not increasing
    with pytest.raises(ValueError):
        PiecewiseBasis(local, np.array([-1.0]))  # too few breakpoints
    with pytest.raises(ValueError):
        PiecewiseBasis(local, breakpoints, continuity=-2)  # continuity below -1
    with pytest.raises(ValueError):
        PiecewiseBasis((local, local), breakpoints, continuity=0)  # 3 elements, not 2


def test_c1_requires_order_one_boundary_dofs(hermite, local, breakpoints):
    # continuity=1 (matching values *and* first derivatives) is only
    # rejected when the element basis can't supply an order-1 boundary DOF
    # -- it's not a blanket "not yet implemented" any more.
    basis = PiecewiseBasis(hermite, breakpoints, continuity=1)
    assert basis.continuity == 1

    # LagrangeBasis has no derivative-type DOF (boundary_dofs(1) is always
    # (None, None)), so continuity=1 must still be rejected for it even
    # though continuity=0 works fine for the same element basis.
    assert local.boundary_dofs(1) == (None, None)
    with pytest.raises(ValueError, match=r"order 0\.\.1"):
        PiecewiseBasis(local, breakpoints, continuity=1)


def test_c0_requires_boundary_dofs(breakpoints):
    # A modal basis has no endpoint DOF to identify across elements.
    modal = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=3)
    with pytest.raises(ValueError):
        PiecewiseBasis(modal, breakpoints, continuity=0)
    # ... but it tiles fine discontinuously.
    basis = PiecewiseBasis(modal, breakpoints, continuity=-1)
    assert basis.n_basis == 9

    # Gauss-Legendre nodes are all interior -> no boundary DOFs at all either.
    interior_only = LagrangeBasis(reference_nodes=gauss_legendre(3).nodes)
    assert interior_only.boundary_dofs() == (None, None)
    with pytest.raises(ValueError):
        PiecewiseBasis(interior_only, breakpoints, continuity=0)


# -- per-element bases (tuple form) --


def test_uniform_tuple_equals_scalar(local, breakpoints):
    scalar = PiecewiseBasis(local, breakpoints, continuity=0)
    tupled = PiecewiseBasis((local, local, local), breakpoints, continuity=0)

    assert tupled == scalar
    assert hash(tupled) == hash(scalar)


# -- global boundary DOFs --


def test_boundary_dofs(local, breakpoints):
    # The left end belongs to element 0's local `left` DOF and the right end
    # to the last element's local `right` DOF, mapped through assembly.
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    assert basis.boundary_dofs() == (0, basis.n_basis - 1)

    # No shared/global endpoint identity when elements are independent.
    basis = PiecewiseBasis(local, breakpoints, continuity=-1)
    assert basis.boundary_dofs() == (None, None)

    # A discontinuous tiling of an element basis with no boundary DOFs of its
    # own (e.g. Gauss-Legendre nodes) has none globally either.
    interior_only = LagrangeBasis(reference_nodes=gauss_legendre(3).nodes)
    basis = PiecewiseBasis(interior_only, breakpoints, continuity=-1)
    assert basis.boundary_dofs() == (None, None)


def test_boundary_dofs_agree_with_evaluation(local, breakpoints):
    # The DOF identified as the boundary must be the one whose coefficient
    # equals the endpoint value. A unit coefficient there and zero
    # elsewhere must evaluate to 1 at that end and 0 at the other.
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    left, right = basis.boundary_dofs()

    c_left = np.zeros(basis.n_basis)
    c_left[left] = 1.0
    np.testing.assert_allclose(
        basis._evaluate_expansion(c_left, np.array([-1.0, 1.0])), [1.0, 0.0], atol=1e-10
    )

    c_right = np.zeros(basis.n_basis)
    c_right[right] = 1.0
    np.testing.assert_allclose(
        basis._evaluate_expansion(c_right, np.array([-1.0, 1.0])), [0.0, 1.0], atol=1e-10
    )


# -- degree-of-freedom counting --


def test_dof_counts(local, breakpoints):
    dg = PiecewiseBasis(local, breakpoints, continuity=-1)
    cg = PiecewiseBasis(local, breakpoints, continuity=0)

    assert dg.n_elements == cg.n_elements == 3
    assert dg._n_broken == cg._n_broken == 9
    assert dg.n_basis == 9  # identity assembly
    assert cg.n_basis == 9 - 2  # one shared DOF per interior breakpoint

    np.testing.assert_array_equal(dg._assembly, np.eye(9))
    assert cg._assembly.shape == (9, 7)


def test_no_dead_dofs(local, breakpoints):
    # Every global DOF must have support somewhere -- the C0 merge sums two
    # half-supports into one hat function rather than discarding either.
    x = np.linspace(-1, 1, 401)
    for continuity in (-1, 0):
        basis = PiecewiseBasis(local, breakpoints, continuity=continuity)
        phi = basis.evaluate(x)
        assert np.all(np.abs(phi).max(axis=0) > 1e-10)


def test_merged_vertex_dof(local, breakpoints):
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    x = np.linspace(-1, 1, 801)
    phi = basis.evaluate(x)

    # The DOF at the first interior breakpoint is shared by elements 0 and 1.
    knot = breakpoints[1]
    shared = np.argmax(np.abs(phi[np.argmin(np.abs(x - knot))]))
    col = phi[:, shared]
    assert np.abs(col[x < knot]).max() > 1e-6
    assert np.abs(col[x > knot]).max() > 1e-6


# -- continuity behavior --


@pytest.mark.parametrize("continuity", [-1, 0])
@pytest.mark.parametrize("side", ["left", "right"])
def test_partition_of_unity(local, breakpoints, continuity, side):
    # Coordinate-only evaluation: whichever `side` is chosen, exactly one
    # element claims each breakpoint. With closed intervals both neighbours
    # would claim it and the assembled basis would sum to 2 there.
    basis = PiecewiseBasis(local, breakpoints, continuity=continuity)
    x = np.concatenate([np.linspace(-1, 1, 401), breakpoints])
    phi = basis.evaluate(x, side=side)
    np.testing.assert_allclose(phi.sum(axis=1), 1.0, atol=1e-10)


def test_continuity_at_breakpoints(local, breakpoints):
    eps = 1e-9

    # C0 values must agree across every interior breakpoint.
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    for knot in breakpoints[1:-1]:
        left = basis.evaluate(np.array([knot - eps]))
        right = basis.evaluate(np.array([knot + eps]))
        np.testing.assert_allclose(left, right, atol=1e-6)

    # C-1 (independent elements) must jump instead.
    basis = PiecewiseBasis(local, breakpoints, continuity=-1)
    knot = breakpoints[1]
    left = basis.evaluate(np.array([knot - eps]))
    right = basis.evaluate(np.array([knot + eps]))
    assert not np.allclose(left, right, atol=1e-6)

    # C0 constrains values, not slopes: deriv=1 is one-sided at a breakpoint.
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    left = basis.evaluate(np.array([knot - eps]), deriv=1)
    right = basis.evaluate(np.array([knot + eps]), deriv=1)
    assert not np.allclose(left, right, atol=1e-3)


# -- C1 continuity (cubic Hermite) --


class TestC1Continuity:
    """continuity=1 merges both the value and the first-derivative DOF at
    each interior breakpoint, using ``CubicHermiteBasis`` as the per-element
    basis."""

    BP = np.array([-1.0, -0.3, 0.4, 1.0])  # deliberately uneven, 3 elements

    @pytest.fixture
    def basis(self, hermite):
        return PiecewiseBasis(hermite, self.BP, continuity=1)

    def test_dof_counts(self, basis):
        # 3 elements * 4 local DOFs = 12 broken; 2 interior breakpoints, each
        # merging 2 DOFs (value + slope) -> 12 - 2*2 = 8.
        assert basis.n_elements == 3
        assert basis._n_broken == 12
        assert basis.n_basis == 8
        assert basis._assembly.shape == (12, 8)

    def test_boundary_dofs(self, basis):
        assert basis.boundary_dofs(0) == (0, basis.n_basis - 2)
        assert basis.boundary_dofs(1) == (1, basis.n_basis - 1)
        assert basis.boundary_dofs(2) == (None, None)

    def test_value_and_slope_continuity(self, basis):
        # C1 constrains value and first derivative, not second: deriv=0,1
        # must agree across a breakpoint, deriv=2 need not.
        eps = 1e-9
        rng = np.random.default_rng(0)
        coefficients = rng.normal(size=basis.n_basis)
        for knot in self.BP[1:-1]:
            left0 = basis._evaluate_expansion(
                coefficients, np.array([knot - eps]), side="left"
            )
            right0 = basis._evaluate_expansion(
                coefficients, np.array([knot + eps]), side="right"
            )
            np.testing.assert_allclose(left0, right0, atol=1e-4)

            left1 = basis._evaluate_expansion(
                coefficients, np.array([knot - eps]), deriv=1, side="left"
            )
            right1 = basis._evaluate_expansion(
                coefficients, np.array([knot + eps]), deriv=1, side="right"
            )
            np.testing.assert_allclose(left1, right1, atol=1e-3)

            left2 = basis._evaluate_expansion(
                coefficients, np.array([knot - eps]), deriv=2, side="left"
            )
            right2 = basis._evaluate_expansion(
                coefficients, np.array([knot + eps]), deriv=2, side="right"
            )
            assert not np.allclose(left2, right2, atol=1e-2)

    def test_merged_value_dof(self, basis):
        x = np.linspace(-1, 1, 801)
        phi = basis.evaluate(x)
        knot = self.BP[1]
        shared = np.argmax(np.abs(phi[np.argmin(np.abs(x - knot))]))
        col = phi[:, shared]
        assert np.abs(col[x < knot]).max() > 1e-6
        assert np.abs(col[x > knot]).max() > 1e-6

    def test_merged_slope_dof(self, basis):
        x = np.linspace(-1, 1, 801)
        dphi = basis.evaluate(x, deriv=1)
        knot = self.BP[1]
        shared = np.argmax(np.abs(dphi[np.argmin(np.abs(x - knot))]))
        col = dphi[:, shared]
        assert np.abs(col[x < knot]).max() > 1e-6
        assert np.abs(col[x > knot]).max() > 1e-6

    @pytest.mark.parametrize("deriv", [0, 1, 2, 3])
    def test_evaluate_expansion_matches_dense(self, basis, deriv):
        # `evaluate_expansion`'s fused path (taken because every element
        # shares one `CubicHermiteBasis` instance) must agree with the dense
        # `evaluate(...) @ coefficients` path on a non-uniform mesh whose
        # physical domain isn't (-1, 1). Value and derivative DOFs need
        # different Jacobian scaling, so a single scalar factor isn't enough
        # once the element width differs from the reference width of 2.
        a, b = 2.0, 9.0
        scale, shift = UnitInterval().affine_params(a, b)
        x = np.concatenate(
            [np.linspace(-1.2, 1.2, 37) * scale + shift, self.BP * scale + shift]
        )
        rng = np.random.default_rng(1)
        coefficients = rng.normal(size=basis.n_basis)

        dense = basis.evaluate(x, deriv=deriv, a=a, b=b) @ coefficients
        fused = basis._evaluate_expansion(coefficients, x, deriv=deriv, a=a, b=b)
        np.testing.assert_allclose(fused, dense, atol=1e-8)

    def test_evaluate_expansion_symbolic_matches_numeric(self, basis):
        a, b = 2.0, 9.0
        rng = np.random.default_rng(2)
        coefficients = rng.normal(size=basis.n_basis)
        x = np.linspace(a, b, 11)
        expected = basis._evaluate_expansion(coefficients, x, deriv=1, a=a, b=b)

        @arc.compile
        def traced(xx, cc):
            assert isinstance(xx, SymbolicArray)
            return basis._evaluate_expansion(cc, xx, deriv=1, a=a, b=b)

        np.testing.assert_allclose(
            np.asarray(traced(x, coefficients)).ravel(), expected, atol=1e-9
        )

    def test_project_and_reconstruct_derivatives(self, hermite):
        # End-to-end: project a smooth function onto a C1 Hermite space on a
        # non-uniform mesh and physical domain. Check that evaluating
        # deriv=0..3 (not just deriv=0) reconstructs a sane, finite field.
        a, b = -2.0, 6.0
        basis = PiecewiseBasis(hermite, self.BP, continuity=1)
        space = FunctionSpace(
            basis,
            domain=UnitInterval.Parameters(a=a, b=b),
            reference_quad_rule=composite_quad(gauss_legendre(4), self.BP),
        )

        def f(x):
            return np.sin(0.5 * x) + 0.1 * x**2

        fn = space.project(f)
        x = np.linspace(a, b, 41)
        np.testing.assert_allclose(fn(x), f(x), atol=1e-2)
        for deriv in (0, 1, 2, 3):
            dense = basis.evaluate(x, deriv=deriv, a=a, b=b) @ fn.coefficients
            fused = fn(x, deriv=deriv)
            np.testing.assert_allclose(fused, dense, atol=1e-8)


class TestModalDiscontinuous:
    """A modal (`OrthogonalPolynomialBasis`) element under `continuity=-1`,
    the one continuity level a modal basis supports. A modal basis has no
    boundary DOFs to merge across elements, so higher continuity is
    rejected.

    Unlike a nodal or Hermite element, its `evaluate` normalizes by
    `measure.mass(a, b)` (`Basis._reference_scale_exponent`). The fused fast
    path must therefore forward each element's own physical domain to the
    shared per-element basis rather than the reference interval. A single
    element spanning the whole physical domain can't distinguish the two,
    so the tests below use a mesh with more than one element."""

    BP = np.array([-1.0, -0.3, 0.4, 1.0])  # deliberately uneven, 3 elements

    @pytest.fixture
    def modal(self):
        return OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=4)

    @pytest.fixture
    def basis(self, modal):
        return PiecewiseBasis(modal, self.BP, continuity=-1)

    @pytest.mark.parametrize("deriv", [0, 1, 2, 3])
    def test_evaluate_expansion_matches_dense(self, basis, deriv):
        # Non-uniform mesh and a physical domain far from (-1, 1): the
        # element width must differ from the reference width of 2 to
        # distinguish per-element from reference-interval normalization.
        a, b = 2.0, 9.0
        scale, shift = UnitInterval().affine_params(a, b)
        x = np.concatenate(
            [np.linspace(-1.2, 1.2, 37) * scale + shift, self.BP * scale + shift]
        )
        rng = np.random.default_rng(1)
        coefficients = rng.normal(size=basis.n_basis)

        dense = basis.evaluate(x, deriv=deriv, a=a, b=b) @ coefficients
        fused = basis._evaluate_expansion(coefficients, x, deriv=deriv, a=a, b=b)
        np.testing.assert_allclose(fused, dense, atol=1e-8)

    @pytest.mark.parametrize("deriv", [0, 1, 2])
    def test_evaluate_expansion_matches_dense_normalized(self, deriv):
        # `density=True` folds the mass out of the normalization entirely, a
        # different branch of `_reference_scale_exponent` (0.0 rather than
        # 0.5) -- must also agree with the dense path.
        modal = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=4, density=True)
        basis = PiecewiseBasis(modal, self.BP, continuity=-1)
        a, b = 2.0, 9.0
        scale, shift = UnitInterval().affine_params(a, b)
        x = np.linspace(-1.2, 1.2, 37) * scale + shift
        rng = np.random.default_rng(2)
        coefficients = rng.normal(size=basis.n_basis)

        dense = basis.evaluate(x, deriv=deriv, a=a, b=b) @ coefficients
        fused = basis._evaluate_expansion(coefficients, x, deriv=deriv, a=a, b=b)
        np.testing.assert_allclose(fused, dense, atol=1e-8)

    def test_project_and_reconstruct(self, modal):
        # End-to-end via the public `FunctionSpace.piecewise` API: project a
        # polynomial (exactly representable per element) onto a
        # multi-element discontinuous Legendre space. Check that it
        # round-trips.
        a, b = 0.0, 2 * np.pi
        space = FunctionSpace.piecewise(
            "legendre", 3, np.linspace(a, b, 9), continuity=-1
        )

        def f(x):
            return x**2 - 3 * x + 1

        fn = space.project(f)
        x = np.linspace(a, b, 41)
        np.testing.assert_allclose(fn(x), f(x), atol=1e-10)


# -- domain mapping --


def test_domain_mapping(local, breakpoints):
    a, b = 0.0, 3.0
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    x_ref = np.linspace(-1, 1, 201)
    scale, shift = UnitInterval().affine_params(a, b)

    np.testing.assert_allclose(
        basis.evaluate(scale * x_ref + shift, a=a, b=b),
        basis.evaluate(x_ref),
        atol=1e-10,
    )


# -- composition with FunctionSpace --


def test_mass_and_stiffness_matrices(local, breakpoints):
    # The DOF-merge assembly must not introduce rank deficiency (mass matrix
    # nonsingular) or break the classic FEM/Neumann structure: the constant
    # function lies in the C0 space and has zero derivative, so the
    # stiffness matrix must have the all-ones coefficient vector in its
    # kernel and rank exactly n_basis - 1.
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    space = FunctionSpace(
        basis,
        domain=UnitInterval.Parameters(a=0.0, b=3.0),
        reference_quad_rule=composite_quad(gauss_legendre(4), breakpoints),
    )

    M = mass_matrix(space)
    np.testing.assert_allclose(M, M.T, atol=1e-12)
    assert np.linalg.matrix_rank(M) == space.n_basis

    K = stiffness_matrix(space)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    np.testing.assert_allclose(K @ np.ones(space.n_basis), 0.0, atol=1e-10)
    assert np.linalg.matrix_rank(K) == space.n_basis - 1


# -- equality / hashing --


def test_equality_and_hash(local, breakpoints):
    a = PiecewiseBasis(local, breakpoints, continuity=0)
    b = PiecewiseBasis(local, breakpoints.copy(), continuity=0)
    c = PiecewiseBasis(local, breakpoints, continuity=-1)
    d = PiecewiseBasis(local, np.linspace(-1.0, 1.0, 5), continuity=0)

    assert a == b
    assert a != c
    assert a != d
    assert a != "not a basis"
    assert len({a, b, c, d}) == 3


# -- static vs. dynamic (symbolic) evaluation --


@pytest.mark.parametrize("continuity", [-1, 0])
def test_static_and_dynamic_evaluation(local, breakpoints, continuity):
    basis = PiecewiseBasis(local, breakpoints, continuity=continuity)
    # Interior points plus exact breakpoints, where the `side` convention
    # decides ownership (default "right").
    x = np.concatenate([np.array([-0.7, -0.1, 0.42, 0.9]), breakpoints])
    static_phi = basis.evaluate(x)

    @arc.compile
    def traced(x):
        assert isinstance(x, SymbolicArray)
        return basis.evaluate(np.atleast_1d(x))

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-10)


# -- quadrature compatibility --


class TestQuadratureCompatibility:
    """A piecewise basis can only be integrated exactly by a rule whose
    elements do not straddle its kinks. The basis supplies such a rule, and
    an explicit one is checked rather than silently accepted."""

    BP = np.array([-1.0, -0.3, 0.4, 1.0])  # deliberately uneven
    DOMAIN = UnitInterval.Parameters(a=0.0, b=3.0)

    @pytest.fixture
    def basis(self, local):
        return PiecewiseBasis(local, self.BP, continuity=0)

    def test_required_breakpoints(self, basis, local):
        np.testing.assert_array_equal(basis._required_breakpoints, self.BP)
        # A globally smooth family imposes no constraint
        assert local._required_breakpoints is None

    def test_default_quadrature(self, basis):
        # Check that the default is a composite
        rule = basis._default_quadrature()
        np.testing.assert_array_equal(rule.breakpoints, self.BP)
        # `element_basis` is always normalized to a per-element tuple, even
        # in the (here, uniform) scalar-constructor case.
        assert len(rule) == sum(len(b.reference_nodes) for b in basis.element_basis)

        # Check that the default integrates the mass matrix exactly
        default = FunctionSpace(basis, domain=self.DOMAIN)
        # A much higher-order aligned rule must give the same mass matrix.
        exact = FunctionSpace(
            basis,
            domain=self.DOMAIN,
            reference_quad_rule=composite_quad(gauss_legendre(8), self.BP),
        )
        np.testing.assert_allclose(mass_matrix(default), mass_matrix(exact), atol=1e-12)

    @pytest.mark.parametrize("continuity", [-1, 0])
    def test_piecewise_default_quadrature(self, local, continuity):
        # The mass matrix is singular if the rule has fewer points than
        # n_basis; the default must never trip that guard.
        basis = PiecewiseBasis(local, self.BP, continuity=continuity)
        assert len(basis._default_quadrature()) >= basis.n_basis

    def test_compatible_rules_accepted(self, basis):
        rule = composite_quad(gauss_legendre(4), self.BP)
        assert (
            FunctionSpace(basis, domain=self.DOMAIN, reference_quad_rule=rule)
            is not None
        )

        # A superset is fine too: each sub-element still lies inside one
        # element of the basis, so the integrand is a polynomial there.
        midpoints = (self.BP[:-1] + self.BP[1:]) / 2
        refined = np.unique(np.concatenate([self.BP, midpoints]))
        rule = composite_quad(gauss_legendre(4), refined)
        assert (
            FunctionSpace(basis, domain=self.DOMAIN, reference_quad_rule=rule)
            is not None
        )

    def test_incompatible_rules_rejected(self, basis):
        # Same number of elements, different partition: the rule's element
        # boundaries don't align with the basis's own breakpoints.
        rule = composite_quad(gauss_legendre(4), np.linspace(-1.0, 1.0, 4))
        with pytest.raises(ValueError, match="must not straddle"):
            FunctionSpace(basis, domain=self.DOMAIN, reference_quad_rule=rule)

        # Node count is beside the point: a 24-point global rule is still
        # wrong, while an aligned 12-point one is exact.
        with pytest.raises(ValueError, match="must not straddle"):
            FunctionSpace(
                basis, domain=self.DOMAIN, reference_quad_rule=gauss_legendre(24)
            )

    def test_smooth_basis_accepts_any_rule(self, local):
        # required_breakpoints is None, so there is nothing to enforce.
        space = FunctionSpace(
            local, domain=self.DOMAIN, reference_quad_rule=gauss_legendre(7)
        )
        assert len(space.quad_rule) == 7


# -- per-element order (heterogeneous element_basis) --


class TestPerElementOrder:
    """``element_basis`` may be a tuple with one entry per element, each of a
    different order (or even family), not just a single shared instance."""

    BP = np.linspace(-1.0, 1.0, 4)  # 3 elements
    DOMAIN = UnitInterval.Parameters(a=0.0, b=3.0)

    @staticmethod
    def _lobatto(n):
        return LagrangeBasis(reference_nodes=gauss_lobatto(n).nodes)

    @pytest.fixture
    def element_bases(self):
        return (self._lobatto(3), self._lobatto(5), self._lobatto(4))

    @pytest.fixture
    def basis(self, element_bases):
        return PiecewiseBasis(element_bases, self.BP, continuity=0)

    def test_dof_counts(self, basis):
        assert basis._n_broken == 3 + 5 + 4
        assert basis.n_basis == (3 + 5 + 4) - 2  # one merged DOF per interior knot
        assert basis._assembly.shape == (12, 10)

    def test_boundary_dofs(self, basis):
        assert basis.boundary_dofs() == (0, basis.n_basis - 1)

    def test_construction_validation(self):
        # Gauss-Legendre nodes are all interior -> no boundary DOFs, even
        # though the other two elements have them.
        interior_only = LagrangeBasis(reference_nodes=gauss_legendre(3).nodes)
        with pytest.raises(ValueError):
            PiecewiseBasis(
                (self._lobatto(3), interior_only, self._lobatto(4)),
                self.BP,
                continuity=0,
            )

        # A modal basis has a real measure; a nodal one has None -- mixing
        # them leaves the assembled basis with no single coherent weight.
        # continuity=-1 so this isn't rejected by the boundary-dof check
        # first (a modal basis has no boundary DOFs either).
        modal = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=3)
        with pytest.raises(ValueError, match="measure"):
            PiecewiseBasis(
                (self._lobatto(3), modal, self._lobatto(4)), self.BP, continuity=-1
            )

    def test_default_quadrature_matches_composite_quad(self, basis, element_bases):
        rule = basis._default_quadrature()
        expected = composite_quad(
            [b._default_quadrature() for b in element_bases], self.BP
        )
        assert rule == expected

    def test_properties(self, basis):
        # The properties checked at module level for a uniform element_basis
        # -- exact mass matrix via the default quadrature, partition of
        # unity from either side, and C0 continuity at interior breakpoints
        # -- must all still hold once elements genuinely differ in order.
        default = FunctionSpace(basis, domain=self.DOMAIN)
        exact = FunctionSpace(
            basis,
            domain=self.DOMAIN,
            reference_quad_rule=composite_quad(gauss_legendre(8), self.BP),
        )
        np.testing.assert_allclose(mass_matrix(default), mass_matrix(exact), atol=1e-12)

        x = np.concatenate([np.linspace(-1, 1, 401), self.BP])
        for side in ("left", "right"):
            phi = basis.evaluate(x, side=side)
            np.testing.assert_allclose(phi.sum(axis=1), 1.0, atol=1e-10)

        eps = 1e-9
        for knot in self.BP[1:-1]:
            left = basis.evaluate(np.array([knot - eps]))
            right = basis.evaluate(np.array([knot + eps]))
            np.testing.assert_allclose(left, right, atol=1e-6)

    def test_evaluate_expansion_matches_dense(self, basis):
        # Heterogeneous element_basis falls back to the dense
        # evaluate(x) @ coefficients path -- no fused fast path is possible
        # once elements genuinely differ.
        rng = np.random.default_rng(0)
        coefficients = rng.normal(size=basis.n_basis)
        x = np.concatenate([np.linspace(-1.3, 1.3, 23), self.BP])
        dense = basis.evaluate(x) @ coefficients
        fused = basis._evaluate_expansion(coefficients, x)
        np.testing.assert_allclose(fused, dense, atol=1e-9)

    def test_evaluate_expansion_symbolic_matches_numeric(self, basis):
        rng = np.random.default_rng(1)
        coefficients = rng.normal(size=basis.n_basis)
        x = np.concatenate([np.linspace(-1.3, 1.3, 23), self.BP])
        expected = basis._evaluate_expansion(coefficients, x)

        @arc.compile
        def traced(xx, cc):
            assert isinstance(xx, SymbolicArray)
            return basis._evaluate_expansion(cc, xx)

        np.testing.assert_allclose(
            np.asarray(traced(x, coefficients)).ravel(), expected, atol=1e-10
        )

    def test_evaluate_expansion_uniform_tuple(self):
        # A uniform *tuple* (not just the scalar form) must still agree with
        # the scalar-constructed basis -- the fast path is keyed on value
        # equality across elements, not on how the basis was constructed.
        uniform = PiecewiseBasis(
            (self._lobatto(3), self._lobatto(3), self._lobatto(3)),
            self.BP,
            continuity=0,
        )
        scalar = PiecewiseBasis(self._lobatto(3), self.BP, continuity=0)
        rng = np.random.default_rng(2)
        coefficients = rng.normal(size=scalar.n_basis)
        x = np.linspace(-1.0, 1.0, 11)
        np.testing.assert_allclose(
            uniform._evaluate_expansion(coefficients, x),
            scalar._evaluate_expansion(coefficients, x),
            atol=1e-12,
        )


# -- fused evaluation (locate-and-gather) --


class TestEvaluateExpansion:
    """``evaluate_expansion`` locates each point's element and gathers only
    that element's coefficients, so its cost is independent of the number of
    elements. It must agree exactly with the dense
    ``evaluate(...) @ coefficients`` path it replaces."""

    A, B = 0.0, 3.0

    def _points(self, breakpoints):
        # Interior points, exact breakpoints, exact endpoints, and points
        # outside the domain on both sides.
        scale, shift = UnitInterval().affine_params(self.A, self.B)
        return np.concatenate(
            [np.linspace(-1.3, 1.3, 23) * scale + shift, breakpoints * scale + shift]
        )

    @pytest.mark.parametrize(
        "continuity,n_elements,deriv,n_components",
        [
            (-1, 1, 0, None),
            (-1, 1, 2, None),
            (-1, 5, 0, None),
            (-1, 5, 2, None),
            (0, 1, 0, None),
            (0, 1, 2, None),
            (0, 5, 0, None),
            (0, 5, 2, None),
            (0, 5, 1, 3),  # one vector-valued case, so that path isn't dropped
        ],
    )
    def test_matches_dense(self, local, continuity, n_elements, deriv, n_components):
        bp = np.linspace(-1.0, 1.0, n_elements + 1)
        basis = PiecewiseBasis(local, bp, continuity=continuity)
        shape = (
            (basis.n_basis,)
            if n_components is None
            else (
                basis.n_basis,
                n_components,
            )
        )
        coefficients = np.random.default_rng(0).normal(size=shape)
        x = self._points(bp)

        dense = basis.evaluate(x, deriv=deriv, a=self.A, b=self.B) @ coefficients
        fused = basis._evaluate_expansion(
            coefficients, x, deriv=deriv, a=self.A, b=self.B
        )
        np.testing.assert_allclose(fused, dense, atol=1e-9)

    def test_zero_outside_domain(self, local, breakpoints):
        # `evaluate` masks every element, so a point outside contributes
        # nothing; `low`/`searchsorted` clamp instead, which would
        # extrapolate. The two must agree.
        basis = PiecewiseBasis(local, breakpoints, continuity=0)
        coefficients = np.ones(basis.n_basis)
        outside = np.array([self.A - 1.0, self.B + 1.0])
        got = basis._evaluate_expansion(coefficients, outside, a=self.A, b=self.B)
        np.testing.assert_array_equal(got, 0.0)

    @pytest.mark.parametrize("deriv", [0, 1, 2])
    def test_symbolic_points(self, local, breakpoints, deriv):
        basis = PiecewiseBasis(local, breakpoints, continuity=0)
        coefficients = np.random.default_rng(1).normal(size=basis.n_basis)
        x = self._points(breakpoints)
        expected = basis._evaluate_expansion(
            coefficients, x, deriv=deriv, a=self.A, b=self.B
        )

        @arc.compile
        def traced(xx, cc):
            assert isinstance(xx, SymbolicArray)
            return basis._evaluate_expansion(cc, xx, deriv=deriv, a=self.A, b=self.B)

        np.testing.assert_allclose(
            np.asarray(traced(x, coefficients)).ravel(), expected, atol=1e-10
        )

    def test_symbolic_coefficients(self, local, breakpoints):
        # The gather must also work the other way round: numeric element
        # indices into a symbolic coefficient vector.
        basis = PiecewiseBasis(local, breakpoints, continuity=0)
        coefficients = np.random.default_rng(2).normal(size=basis.n_basis)
        x = np.array([0.4, 1.5, 2.7])
        expected = basis._evaluate_expansion(coefficients, x, a=self.A, b=self.B)

        @arc.compile
        def traced(cc):
            assert isinstance(cc, SymbolicArray)
            return basis._evaluate_expansion(cc, x, a=self.A, b=self.B)

        np.testing.assert_allclose(
            np.asarray(traced(coefficients)).ravel(), expected, atol=1e-10
        )

    def test_graph_size(self, local):
        # The whole point: graph size must not grow with n_elements.
        sizes = []
        for n_elements in (4, 16, 64):
            basis = PiecewiseBasis(
                local, np.linspace(-1.0, 1.0, n_elements + 1), continuity=0
            )
            coefficients = np.zeros(basis.n_basis)
            x = np.array([1.7])

            def fused(xx, cc, basis=basis):
                return basis._evaluate_expansion(cc, xx, a=self.A, b=self.B)

            compiled = arc.compile(fused, static_argnames=("basis",))
            sizes.append(compiled._specialize(x, coefficients)[0].func.n_nodes())

        assert len(set(sizes)) == 1, f"graph size varied with n_elements: {sizes}"

    def test_function_call_uses_fused_path(self, local, breakpoints):
        # FunctionSpace.evaluate routes through evaluate_expansion, so a
        # Function's __call__ gets this for free.
        basis = PiecewiseBasis(local, breakpoints, continuity=0)
        space = FunctionSpace(basis, domain=UnitInterval.Parameters(a=self.A, b=self.B))

        def f(x):
            return 3 * x**2 - 2 * x + 1

        fn = space.project(f)
        x = np.linspace(self.A, self.B, 31)
        np.testing.assert_allclose(fn(x), f(x), atol=1e-9)
        np.testing.assert_allclose(
            fn(x), basis.evaluate(x, a=self.A, b=self.B) @ fn.coefficients, atol=1e-9
        )


# -- FunctionSpace.piecewise() constructor --


def test_matches_manual_construction():
    # Today's hand-built Problem-B-style space (quadratic Lagrange, Lobatto
    # nodes, C0, 10 elements on [0, 1]) vs. the classmethod.
    element = LagrangeBasis(reference_nodes=np.array([-1.0, 0.0, 1.0]))
    manual = FunctionSpace(
        PiecewiseBasis(element, np.linspace(-1.0, 1.0, 11), continuity=0),
        UnitInterval.Parameters(a=0.0, b=1.0),
    )
    space = FunctionSpace.piecewise("lagrange", 2, np.linspace(0.0, 1.0, 11))

    assert space.n_basis == manual.n_basis
    phi_m, phi_s = manual.basis_matrix(), space.basis_matrix()
    np.testing.assert_allclose(phi_m.matrix, phi_s.matrix)
    np.testing.assert_allclose(phi_m.weights, phi_s.weights)


@pytest.mark.parametrize(
    "family, continuity",
    [
        ("lobatto", 0),
        ("equispaced", 0),
        ("radau_left", -1),  # no endpoint nodes -- C0 is invalid, see below
    ],
)
def test_nodes_family_matches_lagrange_classmethod(family, continuity):
    a, b = 2.0, 7.0  # off-reference domain, to exercise the affine mapping
    breakpoints = np.array([a, 4.0, b])
    space = FunctionSpace.piecewise(
        "lagrange", 3, breakpoints, nodes=family, continuity=continuity
    )

    if family == "radau_left":
        expected_element = LagrangeBasis.gauss_radau(4, endpoint="left")
    elif family == "radau_right":
        expected_element = LagrangeBasis.gauss_radau(4, endpoint="right")
    else:
        ctor = {
            "lobatto": LagrangeBasis.gauss_lobatto,
            "equispaced": LagrangeBasis.equispaced,
            "legendre": LagrangeBasis.gauss_legendre,
        }[family]
        expected_element = ctor(4)

    expected = FunctionSpace(
        PiecewiseBasis(
            expected_element, _normalize_breakpoints(breakpoints)[2], continuity
        ),
        UnitInterval.Parameters(a=a, b=b),
    )
    np.testing.assert_allclose(
        space.basis_matrix().matrix, expected.basis_matrix().matrix
    )


def test_nodes_as_callable_or_array():
    family = LagrangeBasis.gauss_legendre
    space = FunctionSpace.piecewise(
        "lagrange",
        3,
        np.linspace(0.0, 1.0, 3),
        nodes=lambda n: family(n).reference_nodes,
        continuity=-1,  # Gauss-Legendre nodes have no endpoint DOFs
    )
    assert (
        space.n_basis
        == PiecewiseBasis(
            LagrangeBasis.gauss_legendre(4), np.linspace(-1.0, 1.0, 3), continuity=-1
        ).n_basis
    )

    space = FunctionSpace.piecewise(
        "lagrange", 2, np.linspace(0.0, 1.0, 3), nodes=np.array([-1.0, 0.0, 1.0])
    )
    assert space.n_basis == 5  # 2 elements * 3 nodes - 1 shared


def test_legendre_kind():
    space = FunctionSpace.piecewise(
        "legendre", 3, np.linspace(-1.0, 1.0, 4), continuity=-1
    )
    assert isinstance(space.basis, PiecewiseBasis)
    assert all(
        isinstance(eb, OrthogonalPolynomialBasis) for eb in space.basis.element_basis
    )
    assert all(eb._measures[0] == LegendreMeasure() for eb in space.basis.element_basis)
    assert space.n_basis == 12


def test_legendre_kind_validation():
    # OrthogonalPolynomialBasis has no boundary DOF, so C0 assembly is
    # incoherent -- PiecewiseBasis itself rejects this; no new check needed.
    with pytest.raises(ValueError, match="continuity=0"):
        FunctionSpace.piecewise("legendre", 3, np.linspace(-1.0, 1.0, 4), continuity=0)

    with pytest.raises(ValueError, match="nodes is only meaningful"):
        FunctionSpace.piecewise(
            "legendre", 3, np.linspace(-1.0, 1.0, 4), nodes="lobatto", continuity=-1
        )


def test_classmethod_validation():
    with pytest.raises(ValueError, match="kind must be"):
        FunctionSpace.piecewise("bogus", 3, np.linspace(-1.0, 1.0, 4))

    with pytest.raises(ValueError, match="unknown nodes family"):
        FunctionSpace.piecewise(
            "lagrange", 3, np.linspace(0.0, 1.0, 4), nodes="bogus_family"
        )

    with pytest.raises(ValueError, match="nodes has 3 points but degree=3 needs 4"):
        FunctionSpace.piecewise(
            "lagrange", 3, np.linspace(0.0, 1.0, 3), nodes=np.array([-1.0, 0.0, 1.0])
        )

    for breakpoints in (
        np.array([1.0]),  # too few points
        np.array([1.0, 0.5, 2.0]),  # not increasing
    ):
        with pytest.raises(ValueError):
            FunctionSpace.piecewise("lagrange", 2, breakpoints)

    with pytest.raises(ValueError, match="degree has 2 entries"):
        FunctionSpace.piecewise("lagrange", (1, 2), np.linspace(0.0, 1.0, 4))


def test_normalize_breakpoints():
    a, b, ref = _normalize_breakpoints([0.0, 0.3, 1.0])
    assert a == 0.0
    assert b == 1.0
    assert ref[0] == -1.0
    assert ref[-1] == 1.0
    np.testing.assert_allclose(ref, np.array([-1.0, -0.4, 1.0]))


def test_degree_as_per_element_tuple():
    space = FunctionSpace.piecewise("lagrange", (1, 2, 3), np.linspace(0.0, 1.0, 4))
    manual = PiecewiseBasis(
        (
            LagrangeBasis.gauss_lobatto(2),
            LagrangeBasis.gauss_lobatto(3),
            LagrangeBasis.gauss_lobatto(4),
        ),
        np.linspace(-1.0, 1.0, 4),
        continuity=0,
    )
    assert space.n_basis == manual.n_basis


def test_quad_rule_passthrough():
    breakpoints = np.linspace(0.0, 1.0, 4)
    _, _, ref = _normalize_breakpoints(breakpoints)
    rule = composite_quad(gauss_legendre(5), ref)
    space = FunctionSpace.piecewise("lagrange", 2, breakpoints, quad_rule=rule)
    assert space.reference_quad_rule is rule

    # A plain (non-composite) rule can't integrate a piecewise-smooth basis
    # exactly.
    with pytest.raises(ValueError, match="only piecewise smooth"):
        FunctionSpace.piecewise(
            "lagrange", 2, np.linspace(0.0, 1.0, 4), quad_rule=gauss_legendre(5)
        )


# -- kind="hermite" --


@pytest.mark.parametrize("continuity", [-1, 0, 1])
def test_hermite_kind_matches_manual_construction(continuity):
    breakpoints = np.linspace(0.0, 1.0, 6)
    _, _, ref = _normalize_breakpoints(breakpoints)
    manual = FunctionSpace(
        PiecewiseBasis(CubicHermiteBasis(), ref, continuity=continuity),
        UnitInterval.Parameters(a=0.0, b=1.0),
    )
    space = FunctionSpace.piecewise("hermite", 3, breakpoints, continuity=continuity)
    assert space.n_basis == manual.n_basis
    np.testing.assert_allclose(
        space.basis_matrix().matrix, manual.basis_matrix().matrix
    )


def test_hermite_kind_validation():
    # A tuple must be validated entry by entry, not just at index 0.
    for degree, breakpoints in [
        (2, np.linspace(0.0, 1.0, 6)),  # scalar degree, wrong value
        ((3, 3, 2), np.linspace(0.0, 1.0, 4)),  # per-element tuple, one bad entry
    ]:
        with pytest.raises(ValueError, match="degree must be 3"):
            FunctionSpace.piecewise("hermite", degree, breakpoints, continuity=1)

    with pytest.raises(ValueError, match="nodes is only meaningful"):
        FunctionSpace.piecewise(
            "hermite", 3, np.linspace(0.0, 1.0, 6), nodes="lobatto", continuity=1
        )
