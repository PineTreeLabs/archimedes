"""Tests for ``ConstrainedBasis``: a basis recombined by a fixed matrix,
typically the null space of a boundary-condition constraint.
"""

import numpy as np
import pytest

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import ConstrainedBasis, FunctionSpace
from archimedes.measure import LegendreMeasure, UnitInterval


@pytest.fixture
def legendre8():
    from archimedes.experimental.approximation import OrthogonalPolynomialBasis

    return OrthogonalPolynomialBasis(LegendreMeasure(), 8)


# -- from_constraints: the general mechanism --


def test_from_constraints_builds_null_space(legendre8):
    kwargs = {"a": None, "b": None}
    constraint_matrix = np.stack(
        [
            legendre8.evaluate(np.array([-1.0]), **kwargs)[0],
            legendre8.evaluate(np.array([1.0]), **kwargs)[0],
        ]
    )
    constrained = ConstrainedBasis.from_constraints(
        legendre8, lambda base: constraint_matrix
    )
    assert constrained.n_basis == legendre8.n_basis - 2
    np.testing.assert_allclose(constraint_matrix @ constrained.matrix, 0.0, atol=1e-10)


def test_from_constraints_rejects_wrong_shape(legendre8):
    def bad_constraints(base):
        return np.ones((2, base.n_basis + 1))

    with pytest.raises(ValueError, match="constraints must return shape"):
        ConstrainedBasis.from_constraints(legendre8, bad_constraints)


def test_from_constraints_rejects_dependent_rows(legendre8):
    def dependent_constraints(base):
        row = legendre8.evaluate(np.array([-1.0]), a=None, b=None)[0]
        return np.stack([row, 2.0 * row])  # same row twice over, up to scale

    with pytest.raises(ValueError, match="not linearly independent"):
        ConstrainedBasis.from_constraints(legendre8, dependent_constraints)


def test_matrix_row_count_checked_at_construction(legendre8):
    with pytest.raises(ValueError, match="matrix has"):
        ConstrainedBasis(base=legendre8, matrix=np.zeros((legendre8.n_basis + 1, 3)))


def test_square_invertible_matrix_is_a_plain_change_of_basis(legendre8):
    # No constraint at all: a change of basis that preserves n_basis, built
    # directly rather than through from_constraints (see the class docstring).
    matrix = np.eye(legendre8.n_basis)[:, ::-1]  # reverse the columns
    reordered = ConstrainedBasis(base=legendre8, matrix=matrix)
    assert reordered.n_basis == legendre8.n_basis
    x = np.linspace(-1.0, 1.0, 5)
    np.testing.assert_allclose(reordered.evaluate(x), legendre8.evaluate(x)[:, ::-1])


# -- dirichlet / neumann convenience constructors --


def test_dirichlet_vanishes_at_reference_endpoints(legendre8):
    dirichlet = ConstrainedBasis.dirichlet(legendre8)
    assert dirichlet.n_basis == legendre8.n_basis - 2
    endpoints = dirichlet.evaluate(np.array([-1.0, 1.0]))
    np.testing.assert_allclose(endpoints, 0.0, atol=1e-10)


def test_dirichlet_vanishes_on_a_rescaled_target_domain(legendre8):
    # The constraint is built at the reference domain, but must still hold
    # after mapping onto an arbitrary target [a, b].
    dirichlet = ConstrainedBasis.dirichlet(legendre8)
    endpoints = dirichlet.evaluate(np.array([0.0, 3.0]), a=0.0, b=3.0)
    np.testing.assert_allclose(endpoints, 0.0, atol=1e-10)


def test_neumann_derivative_vanishes_at_reference_endpoints(legendre8):
    neumann = ConstrainedBasis.neumann(legendre8)
    assert neumann.n_basis == legendre8.n_basis - 2
    endpoint_derivs = neumann.evaluate(np.array([-1.0, 1.0]), deriv=1)
    np.testing.assert_allclose(endpoint_derivs, 0.0, atol=1e-8)


def test_dirichlet_and_neumann_span_different_subspaces(legendre8):
    dirichlet = ConstrainedBasis.dirichlet(legendre8)
    neumann = ConstrainedBasis.neumann(legendre8)
    # Same size, but not the same functions: the Dirichlet space's own
    # derivative doesn't vanish at the endpoints (generically), and vice
    # versa.
    d_at_end = dirichlet.evaluate(np.array([-1.0]), deriv=1)[0]
    n_at_end = neumann.evaluate(np.array([-1.0]))[0]
    assert not np.allclose(d_at_end, 0.0)
    assert not np.allclose(n_at_end, 0.0)


# -- delegated Basis interface --


def test_delegates_parameters_measures_and_breakpoints(legendre8):
    dirichlet = ConstrainedBasis.dirichlet(legendre8)
    assert dirichlet.Parameters is legendre8.Parameters
    assert dirichlet.measures == legendre8.measures
    assert dirichlet.required_breakpoints == legendre8.required_breakpoints


def test_default_quadrature_delegates_to_base(legendre8):
    dirichlet = ConstrainedBasis.dirichlet(legendre8)
    assert dirichlet.default_quadrature() == legendre8.default_quadrature()


def test_side_argument_is_validated(legendre8):
    dirichlet = ConstrainedBasis.dirichlet(legendre8)
    with pytest.raises(ValueError, match="side must be"):
        dirichlet.evaluate(np.array([0.0]), side="up")


# -- composes with FunctionSpace / project --


def test_projection_onto_dirichlet_space_vanishes_at_boundary(legendre8):
    dirichlet = ConstrainedBasis.dirichlet(legendre8)
    space = FunctionSpace(dirichlet, UnitInterval.Parameters(a=-1.0, b=1.0))
    f = space.project(lambda x: (1 - x**2) * np.sin(3 * x))
    xs = np.array([-1.0, 1.0])
    np.testing.assert_allclose(f(xs), 0.0, atol=1e-10)


def test_mass_matrix_is_well_conditioned():
    # SVD null-space columns are orthonormal, and the base is already
    # orthonormal, so the combination should stay perfectly conditioned
    # rather than drifting with n like a hand-derived combination would.
    from archimedes.experimental.approximation import OrthogonalPolynomialBasis

    base32 = OrthogonalPolynomialBasis(LegendreMeasure(), 32)
    dirichlet = ConstrainedBasis.dirichlet(base32)
    space = FunctionSpace(dirichlet, UnitInterval.Parameters(a=-1.0, b=1.0))
    phi = space.basis_matrix()
    mass_matrix = phi.matrix.T @ (phi.weights[:, None] * phi.matrix)
    assert np.linalg.cond(mass_matrix) < 1.1


# -- static (NumPy) vs. dynamic (symbolic) evaluation agreement --


@pytest.mark.parametrize("deriv", [0, 1])
def test_static_and_dynamic_evaluation_agree(legendre8, deriv):
    dirichlet = ConstrainedBasis.dirichlet(legendre8)
    x = np.array([-1.0, -0.4, 0.0, 0.55, 1.0])
    static_phi = dirichlet.evaluate(x, deriv=deriv)

    @arc.compile
    def traced(xi):
        assert isinstance(xi, SymbolicArray)
        return dirichlet.evaluate(np.atleast_1d(xi), deriv=deriv)

    dynamic_phi = np.array([np.asarray(traced(xi)).ravel() for xi in x])
    np.testing.assert_allclose(static_phi, dynamic_phi, atol=1e-10)
