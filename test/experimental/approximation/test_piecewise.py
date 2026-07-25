# ruff: noqa: N806  (M is the conventional name for a mass matrix)
import numpy as np
import pytest

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
def test_partition_of_unity(local, breakpoints, continuity):
    # Half-open element ownership: with closed intervals both elements would
    # claim an interior breakpoint and the assembled basis would sum to 2.
    basis = PiecewiseBasis(local, breakpoints, continuity=continuity)
    x = np.concatenate([np.linspace(-1, 1, 401), breakpoints])
    phi = basis.evaluate(x)
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
    M = space.mass_matrix()
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
    K = space.stiffness_matrix()  # noqa: N806
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
    # Interior points plus exact breakpoints, where element ownership decides.
    x = np.concatenate([np.array([-0.7, -0.1, 0.42, 0.9]), breakpoints])
    static_phi = basis.evaluate(x)

    @arc.compile
    def traced(x):
        assert isinstance(x, SymbolicArray)
        return basis.evaluate(np.atleast_1d(x))

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-10)
