# ruff: noqa: N806  (M is the conventional name for a mass matrix)
import numpy as np
import pytest
from _helpers import mass_matrix, stiffness_matrix

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import (
    FunctionSpace,
    LagrangeBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
)
from archimedes.measure import LegendreMeasure, UnitInterval
from archimedes.quadrature import composite, gauss_legendre, gauss_lobatto


@pytest.fixture
def local():
    # Gauss-Lobatto includes both endpoints, so it has boundary DOFs.
    return LagrangeBasis(reference_nodes=gauss_lobatto(3).nodes)


@pytest.fixture
def breakpoints():
    return np.linspace(-1.0, 1.0, 4)  # 3 elements


# -- construction / validation --


def test_rejects_non_spanning_breakpoints(local):
    with pytest.raises(ValueError):
        PiecewiseBasis(local, np.array([-1.0, 0.0, 0.5]))
    with pytest.raises(ValueError):
        PiecewiseBasis(local, np.array([-0.5, 0.0, 1.0]))


def test_rejects_non_increasing_breakpoints(local):
    with pytest.raises(ValueError):
        PiecewiseBasis(local, np.array([-1.0, 0.5, 0.0, 1.0]))


def test_rejects_too_few_breakpoints(local):
    with pytest.raises(ValueError):
        PiecewiseBasis(local, np.array([-1.0]))


def test_rejects_unsupported_continuity(local, breakpoints):
    with pytest.raises(ValueError):
        PiecewiseBasis(local, breakpoints, continuity=1)


def test_c0_requires_element_basis_with_boundary_dofs(breakpoints):
    # A modal basis has no endpoint DOF to identify across elements.
    modal = OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=3)
    with pytest.raises(ValueError):
        PiecewiseBasis(modal, breakpoints, continuity=0)
    # ... but it tiles fine discontinuously.
    basis = PiecewiseBasis(modal, breakpoints, continuity=-1)
    assert basis.n_basis == 9


def test_c0_requires_both_endpoints(breakpoints):
    # Gauss-Legendre nodes are all interior -> no boundary DOFs at all.
    interior_only = LagrangeBasis(reference_nodes=gauss_legendre(3).nodes)
    assert interior_only.boundary_dofs() == (None, None)
    with pytest.raises(ValueError):
        PiecewiseBasis(interior_only, breakpoints, continuity=0)


def test_boundary_dofs_lobatto(local):
    assert local.boundary_dofs() == (0, local.n_basis - 1)


# -- global boundary DOFs --


def test_piecewise_boundary_dofs_c0(local, breakpoints):
    # The left end belongs to element 0's local `left` DOF and the right end
    # to the last element's local `right` DOF, mapped through assembly.
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    assert basis.boundary_dofs() == (0, basis.n_basis - 1)


def test_piecewise_boundary_dofs_discontinuous(local, breakpoints):
    # No shared/global endpoint identity when elements are independent.
    basis = PiecewiseBasis(local, breakpoints, continuity=-1)
    assert basis.boundary_dofs() == (None, None)


def test_piecewise_boundary_dofs_no_element_endpoint_dofs(breakpoints):
    # A discontinuous tiling of an element basis with no boundary DOFs of its
    # own (e.g. Gauss-Legendre nodes) has none globally either.
    interior_only = LagrangeBasis(reference_nodes=gauss_legendre(3).nodes)
    basis = PiecewiseBasis(interior_only, breakpoints, continuity=-1)
    assert basis.boundary_dofs() == (None, None)


def test_piecewise_boundary_dofs_agree_with_evaluation(local, breakpoints):
    # The DOF identified as the boundary must actually be the one whose
    # coefficient equals the endpoint value: a unit coefficient there and
    # zero elsewhere should evaluate to 1 at that end and 0 at the other.
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    left, right = basis.boundary_dofs()

    c_left = np.zeros(basis.n_basis)
    c_left[left] = 1.0
    np.testing.assert_allclose(
        basis.evaluate_expansion(c_left, np.array([-1.0, 1.0])), [1.0, 0.0], atol=1e-10
    )

    c_right = np.zeros(basis.n_basis)
    c_right[right] = 1.0
    np.testing.assert_allclose(
        basis.evaluate_expansion(c_right, np.array([-1.0, 1.0])), [0.0, 1.0], atol=1e-10
    )


# -- degree-of-freedom counting --


def test_dof_counts(local, breakpoints):
    dg = PiecewiseBasis(local, breakpoints, continuity=-1)
    cg = PiecewiseBasis(local, breakpoints, continuity=0)

    assert dg.n_elements == cg.n_elements == 3
    assert dg.n_broken == cg.n_broken == 9
    assert dg.n_basis == 9  # identity assembly
    assert cg.n_basis == 9 - 2  # one shared DOF per interior breakpoint

    np.testing.assert_array_equal(dg.assembly_matrix, np.eye(9))
    assert cg.assembly_matrix.shape == (9, 7)


def test_no_dead_dofs(local, breakpoints):
    # Every global DOF must have support somewhere -- the C0 merge sums two
    # half-supports into one hat function rather than discarding either.
    x = np.linspace(-1, 1, 401)
    for continuity in (-1, 0):
        basis = PiecewiseBasis(local, breakpoints, continuity=continuity)
        phi = basis.evaluate(x)
        assert np.all(np.abs(phi).max(axis=0) > 1e-10)


def test_merged_vertex_dof_spans_both_elements(local, breakpoints):
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


def test_c0_is_continuous_at_breakpoints(local, breakpoints):
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    eps = 1e-9
    for knot in breakpoints[1:-1]:
        left = basis.evaluate(np.array([knot - eps]))
        right = basis.evaluate(np.array([knot + eps]))
        np.testing.assert_allclose(left, right, atol=1e-6)


def test_discontinuous_basis_jumps_at_breakpoints(local, breakpoints):
    basis = PiecewiseBasis(local, breakpoints, continuity=-1)
    eps = 1e-9
    knot = breakpoints[1]
    left = basis.evaluate(np.array([knot - eps]))
    right = basis.evaluate(np.array([knot + eps]))
    assert not np.allclose(left, right, atol=1e-6)


def test_c0_derivative_is_discontinuous_at_breakpoints(local, breakpoints):
    # C0 constrains values, not slopes: deriv=1 is one-sided at a breakpoint.
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    eps = 1e-9
    knot = breakpoints[1]
    left = basis.evaluate(np.array([knot - eps]), deriv=1)
    right = basis.evaluate(np.array([knot + eps]), deriv=1)
    assert not np.allclose(left, right, atol=1e-3)


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


def test_projection_of_elementwise_representable_function(local, breakpoints):
    a, b = 0.0, 3.0
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    space = FunctionSpace(
        basis,
        domain=UnitInterval.Parameters(a=a, b=b),
        # Composite rule so quadrature resolves the element structure.
        quad_rule=composite(gauss_legendre(4), breakpoints),
    )

    # Globally quadratic: degree <= 2 on each element and continuous, so it
    # lies exactly in the C0 space.
    def f(x):
        return 3 * x**2 - 2 * x + 1

    fn = space.project(f)
    x = np.linspace(a, b, 61)
    np.testing.assert_allclose(fn(x), f(x), atol=1e-8)


def test_mass_matrix_is_nonsingular(local, breakpoints):
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    space = FunctionSpace(
        basis,
        domain=UnitInterval.Parameters(a=0.0, b=3.0),
        quad_rule=composite(gauss_legendre(4), breakpoints),
    )
    M = mass_matrix(space)
    np.testing.assert_allclose(M, M.T, atol=1e-12)
    assert np.linalg.matrix_rank(M) == space.n_basis


def test_stiffness_matrix_annihilates_constants(local, breakpoints):
    # The constant function lies in the C0 space and has zero derivative, so
    # the stiffness matrix must have the all-ones coefficient vector in its
    # kernel and rank exactly n_basis - 1 (the classic FEM/Neumann structure).
    basis = PiecewiseBasis(local, breakpoints, continuity=0)
    space = FunctionSpace(
        basis,
        domain=UnitInterval.Parameters(a=0.0, b=3.0),
        quad_rule=composite(gauss_legendre(4), breakpoints),
    )
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
    assert len({a, b, c, d}) == 3


# -- static vs. dynamic (symbolic) evaluation --


@pytest.mark.parametrize("continuity", [-1, 0])
def test_static_and_dynamic_evaluation_agree(local, breakpoints, continuity):
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
        np.testing.assert_array_equal(basis.required_breakpoints, self.BP)
        # A globally smooth family imposes no constraint
        assert local.required_breakpoints is None

    def test_default_quadrature_is_composite_over_own_breakpoints(self, basis):
        rule = basis.default_quadrature()
        np.testing.assert_array_equal(rule.breakpoints, self.BP)
        assert len(rule) == basis.n_elements * len(basis.element_basis.reference_nodes)

    def test_default_quadrature_integrates_mass_matrix_exactly(self, basis):
        default = FunctionSpace(basis, domain=self.DOMAIN)
        # A much higher-order aligned rule must give the same mass matrix.
        exact = FunctionSpace(
            basis, domain=self.DOMAIN, quad_rule=composite(gauss_legendre(8), self.BP)
        )
        np.testing.assert_allclose(mass_matrix(default), mass_matrix(exact), atol=1e-12)

    def test_space_without_quad_rule_projects_correctly(self, basis):
        space = FunctionSpace(basis, domain=self.DOMAIN)

        def f(x):
            return 3 * x**2 - 2 * x + 1

        x = np.linspace(0.0, 3.0, 41)
        np.testing.assert_allclose(space.project(f)(x), f(x), atol=1e-9)

    def test_aligned_rule_accepted(self, basis):
        rule = composite(gauss_legendre(4), self.BP)
        assert FunctionSpace(basis, domain=self.DOMAIN, quad_rule=rule) is not None

    def test_refinement_accepted(self, basis):
        # A superset is fine: each sub-element still lies inside one element
        # of the basis, so the integrand is a polynomial there.
        midpoints = (self.BP[:-1] + self.BP[1:]) / 2
        refined = np.unique(np.concatenate([self.BP, midpoints]))
        rule = composite(gauss_legendre(4), refined)
        assert FunctionSpace(basis, domain=self.DOMAIN, quad_rule=rule) is not None

    def test_misaligned_composite_rejected(self, basis):
        # Same number of elements, different partition -- silently produced a
        # ~2.5% error in the mass matrix before this was checked.
        rule = composite(gauss_legendre(4), np.linspace(-1.0, 1.0, 4))
        with pytest.raises(ValueError, match="must not straddle"):
            FunctionSpace(basis, domain=self.DOMAIN, quad_rule=rule)

    def test_plain_rule_rejected_however_fine(self, basis):
        # Node count is beside the point: a 24-point global rule is still
        # wrong, while an aligned 12-point one is exact.
        with pytest.raises(ValueError, match="must not straddle"):
            FunctionSpace(basis, domain=self.DOMAIN, quad_rule=gauss_legendre(24))

    def test_smooth_basis_accepts_any_rule(self, local):
        # required_breakpoints is None, so there is nothing to enforce.
        space = FunctionSpace(local, domain=self.DOMAIN, quad_rule=gauss_legendre(7))
        assert len(space.quad_rule) == 7

    @pytest.mark.parametrize("continuity", [-1, 0])
    def test_default_quadrature_satisfies_n_basis_guard(self, local, continuity):
        # The mass matrix is singular if the rule has fewer points than
        # n_basis; the default must never trip that guard.
        basis = PiecewiseBasis(local, self.BP, continuity=continuity)
        assert len(basis.default_quadrature()) >= basis.n_basis


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

    @pytest.mark.parametrize("continuity", [-1, 0])
    @pytest.mark.parametrize("n_elements", [1, 2, 5])
    @pytest.mark.parametrize("deriv", [0, 1, 2])
    @pytest.mark.parametrize("n_components", [None, 3])
    def test_matches_dense_path(
        self, local, continuity, n_elements, deriv, n_components
    ):
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
        fused = basis.evaluate_expansion(
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
        got = basis.evaluate_expansion(coefficients, outside, a=self.A, b=self.B)
        np.testing.assert_array_equal(got, 0.0)

    @pytest.mark.parametrize("deriv", [0, 1, 2])
    def test_symbolic_matches_numeric(self, local, breakpoints, deriv):
        basis = PiecewiseBasis(local, breakpoints, continuity=0)
        coefficients = np.random.default_rng(1).normal(size=basis.n_basis)
        x = self._points(breakpoints)
        expected = basis.evaluate_expansion(
            coefficients, x, deriv=deriv, a=self.A, b=self.B
        )

        @arc.compile
        def traced(xx, cc):
            assert isinstance(xx, SymbolicArray)
            return basis.evaluate_expansion(cc, xx, deriv=deriv, a=self.A, b=self.B)

        np.testing.assert_allclose(
            np.asarray(traced(x, coefficients)).ravel(), expected, atol=1e-10
        )

    def test_symbolic_coefficients_with_numeric_points(self, local, breakpoints):
        # The gather must also work the other way round: numeric element
        # indices into a symbolic coefficient vector.
        basis = PiecewiseBasis(local, breakpoints, continuity=0)
        coefficients = np.random.default_rng(2).normal(size=basis.n_basis)
        x = np.array([0.4, 1.5, 2.7])
        expected = basis.evaluate_expansion(coefficients, x, a=self.A, b=self.B)

        @arc.compile
        def traced(cc):
            assert isinstance(cc, SymbolicArray)
            return basis.evaluate_expansion(cc, x, a=self.A, b=self.B)

        np.testing.assert_allclose(
            np.asarray(traced(coefficients)).ravel(), expected, atol=1e-10
        )

    def test_cost_is_independent_of_element_count(self, local):
        # The whole point: graph size must not grow with n_elements.
        sizes = []
        for n_elements in (4, 16, 64):
            basis = PiecewiseBasis(
                local, np.linspace(-1.0, 1.0, n_elements + 1), continuity=0
            )
            coefficients = np.zeros(basis.n_basis)
            x = np.array([1.7])

            def fused(xx, cc, basis=basis):
                return basis.evaluate_expansion(cc, xx, a=self.A, b=self.B)

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
