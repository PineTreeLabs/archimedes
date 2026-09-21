"""Tests for ``ConcatBasis``: the direct sum of several bases' functions."""

import numpy as np
import pytest

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.approximation import (
    ConcatBasis,
    ConstrainedBasis,
    FunctionSpace,
    LagrangeBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
)
from archimedes.measure import LegendreMeasure, PhysicistsHermiteMeasure, UnitInterval
from archimedes.quadrature import gauss_legendre


@pytest.fixture
def legendre6():
    return OrthogonalPolynomialBasis(LegendreMeasure(), 6)


@pytest.fixture
def vertex():
    # Two linear "hat" functions: the classical SEM vertex modes.
    return LagrangeBasis(reference_nodes=np.array([-1.0, 1.0]))


@pytest.fixture
def bubble(legendre6):
    return ConstrainedBasis.dirichlet(legendre6)  # 4 functions, vanish at +-1


@pytest.fixture
def rule():
    return gauss_legendre(6)


@pytest.fixture
def combo(vertex, bubble, rule):
    return ConcatBasis((vertex, bubble), quad_rule=rule)


# -- construction --


def test_construction_validation(vertex, rule):
    with pytest.raises(ValueError, match="non-empty"):
        ConcatBasis((), quad_rule=rule)

    hermite_piece = OrthogonalPolynomialBasis(PhysicistsHermiteMeasure(), 4)
    with pytest.raises(TypeError, match="Parameters type"):
        ConcatBasis((vertex, hermite_piece), quad_rule=rule)


def test_properties(combo, vertex, bubble, rule):
    assert combo.n_basis == vertex.n_basis + bubble.n_basis
    assert combo.Parameters is vertex.Parameters
    assert combo.measures == (None,)
    assert combo.default_quadrature() is rule


# -- evaluate --


def test_evaluate_concatenates_pieces(combo, vertex, bubble):
    x = np.linspace(-1.0, 1.0, 9)
    got = combo.evaluate(x)
    expected = np.concatenate(
        [vertex.evaluate(x, a=None, b=None), bubble.evaluate(x, a=None, b=None)],
        axis=-1,
    )
    np.testing.assert_allclose(got, expected)

    got = combo.evaluate(x, deriv=1)
    expected = np.concatenate(
        [
            vertex.evaluate(x, deriv=1, a=None, b=None),
            bubble.evaluate(x, deriv=1, a=None, b=None),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(got, expected)


def test_side_argument_is_validated(combo):
    with pytest.raises(ValueError, match="side must be"):
        combo.evaluate(np.array([0.0]), side="up")


# -- boundary_dofs --


def test_boundary_dofs(combo, vertex):
    left, right = combo.boundary_dofs(0)
    vertex_left, vertex_right = vertex.boundary_dofs(0)
    assert (left, right) == (vertex_left, vertex_right)

    # Order 1 (derivative DOF) isn't claimed by either the linear vertex
    # functions or the modal bubble space.
    assert combo.boundary_dofs(1) == (None, None)


def test_boundary_dofs_conflict_rejected(rule):
    left_claimer = LagrangeBasis(reference_nodes=np.array([-1.0, 0.0]))
    other_left_claimer = LagrangeBasis(reference_nodes=np.array([-1.0, 0.5]))
    combo = ConcatBasis((left_claimer, other_left_claimer), quad_rule=rule)
    with pytest.raises(ValueError, match="more than one piece claims the left"):
        combo.boundary_dofs(0)

    right_claimer = LagrangeBasis(reference_nodes=np.array([0.0, 1.0]))
    other_right_claimer = LagrangeBasis(reference_nodes=np.array([0.5, 1.0]))
    combo = ConcatBasis((right_claimer, other_right_claimer), quad_rule=rule)
    with pytest.raises(ValueError, match="more than one piece claims the right"):
        combo.boundary_dofs(0)


# -- required_breakpoints --


def test_required_breakpoints_none_when_no_piece_has_any(combo):
    assert combo.required_breakpoints is None


def test_required_breakpoints_is_union(vertex, rule):
    a = PiecewiseBasis(vertex, np.array([-1.0, 0.0, 1.0]), continuity=-1)
    b = PiecewiseBasis(vertex, np.array([-1.0, 0.5, 1.0]), continuity=-1)
    combo = ConcatBasis((a, b), quad_rule=rule)
    np.testing.assert_array_equal(combo.required_breakpoints, [-1.0, 0.0, 0.5, 1.0])


# -- _dof_order --


def test_dof_order_concatenates_pieces(vertex, bubble, combo):
    np.testing.assert_array_equal(
        combo._dof_order,
        np.concatenate([vertex._dof_order, bubble._dof_order]),
    )


# -- end-to-end: vertex + bubble reproduces the full polynomial space --


def test_vertex_bubble_reproduces_full_space_projection(legendre6, combo):
    def target(x):
        return 2 - 3 * x + x**2 - 0.5 * x**3 + 0.2 * x**4 - 0.1 * x**5

    space_full = FunctionSpace(legendre6, UnitInterval.Parameters(a=-1.0, b=1.0))
    space_combo = FunctionSpace(combo, UnitInterval.Parameters(a=-1.0, b=1.0))

    f_full = space_full.project(target)
    f_combo = space_combo.project(target)

    xs = np.linspace(-1.0, 1.0, 25)
    np.testing.assert_allclose(f_combo(xs), f_full(xs), atol=1e-10)
    np.testing.assert_allclose(f_combo(xs), target(xs), atol=1e-10)


def test_vertex_coefficients_equal_boundary_values(legendre6, combo):
    def target(x):
        return 2 - 3 * x + x**2 - 0.5 * x**3 + 0.2 * x**4 - 0.1 * x**5

    space_combo = FunctionSpace(combo, UnitInterval.Parameters(a=-1.0, b=1.0))
    f_combo = space_combo.project(target)
    np.testing.assert_allclose(f_combo.coefficients[0], target(-1.0), atol=1e-10)
    np.testing.assert_allclose(f_combo.coefficients[1], target(1.0), atol=1e-10)


# -- static (NumPy) vs. dynamic (symbolic) evaluation agreement --


@pytest.mark.parametrize("deriv", [0, 1])
def test_static_and_dynamic_evaluation_agree(combo, deriv):
    x = np.array([-1.0, -0.4, 0.0, 0.55, 1.0])
    static_phi = combo.evaluate(x, deriv=deriv)

    @arc.compile
    def traced(xi):
        assert isinstance(xi, SymbolicArray)
        return combo.evaluate(np.atleast_1d(xi), deriv=deriv)

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-10)
