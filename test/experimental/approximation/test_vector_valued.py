"""Vector-valued (scalar -> R^m) function approximation.

The vector-valuedness lives entirely in the coefficients: a ``Basis``
evaluates to ``(npts, n_basis)`` regardless, so a ``FunctionSpace`` built
for scalars carries ``(n_basis, m)`` coefficients unchanged. These tests
sweep every basis family to confirm that holds, and that the operations
which *do* need to know about components -- ``project`` and
``inner_product`` -- agree with doing each component separately.
"""

import numpy as np
import pytest
from _helpers import mass_matrix

import archimedes as arc
from archimedes import tree
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import (
    Function,
    FunctionSpace,
    LagrangeBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
)
from archimedes.measure import LegendreMeasure, UnitInterval
from archimedes.quadrature import composite, gauss_legendre, gauss_lobatto

A, B = 0.0, 2.0  # target domain
BREAKPOINTS = np.linspace(-1.0, 1.0, 3)


def _modal():
    return FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), n_basis=5),
        domain=UnitInterval.Parameters(a=A, b=B),
        quad_rule=gauss_legendre(10),
    )


def _nodal():
    return FunctionSpace(
        LagrangeBasis(reference_nodes=gauss_lobatto(4).nodes),
        domain=UnitInterval.Parameters(a=A, b=B),
        quad_rule=gauss_legendre(10),
    )


def _piecewise():
    return FunctionSpace(
        PiecewiseBasis(
            LagrangeBasis(reference_nodes=gauss_lobatto(4).nodes),
            BREAKPOINTS,
            continuity=0,
        ),
        domain=UnitInterval.Parameters(a=A, b=B),
        # Composite rule so quadrature resolves the element structure.
        quad_rule=composite(gauss_legendre(5), BREAKPOINTS),
    )


SPACE_BUILDERS = {
    "modal": _modal,
    "nodal": _nodal,
    "piecewise": _piecewise,
}


@pytest.fixture(params=sorted(SPACE_BUILDERS))
def space(request):
    """Every basis family, so nothing here is specific to one of them."""
    return SPACE_BUILDERS[request.param]()


# Components chosen to be exactly representable in all three spaces: each is
# a global polynomial of degree <= 2, which is in the modal (degree 4), the
# nodal (degree 3), and the C0 piecewise-cubic space alike.
def f_vec(x):
    return np.stack([x**2, 2 * x, 1.0 - x], axis=-1)  # (npts, 3)


def f0(x):
    return x**2


def f1(x):
    return 2 * x


def f2(x):
    return 1.0 - x


COMPONENTS = (f0, f1, f2)
M_COMPONENTS = len(COMPONENTS)


# -- shapes --


def test_evaluate_shapes(space):
    n = space.n_basis
    x = np.linspace(A, B, 7)

    scalar = space.evaluate(np.ones(n), x)
    assert scalar.shape == (7,)

    vector = space.evaluate(np.ones((n, M_COMPONENTS)), x)
    assert vector.shape == (7, M_COMPONENTS)


def test_project_preserves_shape(space):
    assert space.project(f0).coefficients.shape == (space.n_basis,)
    assert space.project(f_vec).coefficients.shape == (space.n_basis, M_COMPONENTS)


# -- projection --


def test_project_vector_is_exact_for_representable_components(space):
    fn = space.project(f_vec)
    x = np.linspace(A, B, 41)
    np.testing.assert_allclose(fn(x), f_vec(x), atol=1e-9)


def test_project_vector_matches_per_component_projection(space):
    """The whole point of the shared-space design: projecting all components
    at once must give exactly what projecting each one separately gives."""
    joint = space.project(f_vec).coefficients
    separate = np.stack([space.project(f).coefficients for f in COMPONENTS], axis=-1)
    np.testing.assert_allclose(joint, separate, atol=1e-12)


def test_project_vector_respects_quad_rule_override(space):
    # Same override path as the scalar case; just confirm it's plumbed
    # through the vector branch too.
    rule = gauss_legendre(12)
    fn = space.project(f_vec, quad_rule=rule)
    x = np.linspace(A, B, 21)
    np.testing.assert_allclose(fn(x), f_vec(x), atol=1e-9)


def test_project_single_column_is_not_the_scalar_case(space):
    """An ``(npts, 1)`` target stays 2-D rather than collapsing to scalar."""

    def f(x):
        return np.stack([x**2], axis=-1)

    fn = space.project(f)
    assert fn.coefficients.shape == (space.n_basis, 1)
    assert fn(np.linspace(A, B, 5)).shape == (5, 1)


def test_derivative_of_vector_function(space):
    fn = space.project(f_vec)
    x = np.linspace(A + 0.1, B - 0.1, 11)
    expected = np.stack([2 * x, 2 * np.ones_like(x), -np.ones_like(x)], axis=-1)
    np.testing.assert_allclose(fn(x, deriv=1), expected, atol=1e-8)


# -- inner product / norm contract over components --


def test_inner_product_is_scalar_and_contracts(space):
    """<f, g> = int f . g w dx -- a scalar, equal to the sum of the
    per-component inner products."""
    fn = space.project(f_vec)
    gn = space.project(lambda x: f_vec(x) * np.array([2.0, -1.0, 0.5]))

    joint = fn.dot(gn)
    assert np.ndim(joint) == 0

    per_component = sum(
        space.inner_product(fn.coefficients[:, k], gn.coefficients[:, k])
        for k in range(M_COMPONENTS)
    )
    np.testing.assert_allclose(joint, per_component, rtol=1e-10)


def test_norm_is_scalar_l2_norm_of_whole_function(space):
    fn = space.project(f_vec)
    norm = fn.norm()
    assert np.ndim(norm) == 0

    component_norms = [
        np.sqrt(space.inner_product(fn.coefficients[:, k], fn.coefficients[:, k]))
        for k in range(M_COMPONENTS)
    ]
    # Contraction means the whole-function norm is the root-sum-square of
    # the component norms, not an array of them.
    np.testing.assert_allclose(
        norm, np.sqrt(sum(n**2 for n in component_norms)), rtol=1e-10
    )


def test_inner_product_matches_mass_matrix_contraction(space):
    fn = space.project(f_vec)
    gn = space.project(lambda x: f_vec(x) + 1.0)
    M = mass_matrix(space)  # noqa: N806
    expected = np.sum(fn.coefficients * (M @ gn.coefficients))
    np.testing.assert_allclose(fn.dot(gn), expected, rtol=1e-9)


def test_scalar_inner_product_still_scalar(space):
    # The contraction branch must not perturb the scalar path.
    fn = space.project(f0)
    assert np.ndim(fn.dot(fn)) == 0
    np.testing.assert_allclose(fn.norm() ** 2, fn.dot(fn), rtol=1e-12)


# -- arithmetic --


def test_add_and_scale_vector_functions(space):
    fn = space.project(f_vec)
    gn = space.project(lambda x: 3.0 * f_vec(x))
    total = fn + 2.0 * gn

    x = np.linspace(A, B, 13)
    np.testing.assert_allclose(total(x), f_vec(x) + 6.0 * f_vec(x), atol=1e-8)


# -- pytree / symbolic --


def test_vector_function_is_a_pytree(space):
    fn = space.project(f_vec)
    flat, unravel = tree.ravel(fn)
    # coefficients (n_basis * m) + the two domain endpoints
    assert flat.shape == (space.n_basis * M_COMPONENTS + 2,)
    restored = unravel(flat)
    np.testing.assert_allclose(restored.coefficients, fn.coefficients, atol=1e-12)


def test_vector_evaluation_traces(space):
    fn = space.project(f_vec)
    x = np.linspace(A, B, 5)

    @arc.compile
    def traced(coefficients):
        assert isinstance(coefficients, SymbolicArray)
        return Function(coefficients, space)(x)

    np.testing.assert_allclose(traced(fn.coefficients), fn(x), atol=1e-10)


def test_vector_projection_traces(space):
    """``project`` unrolls the per-column solves at trace time; check the
    traced result matches the numeric one."""
    expected = space.project(f_vec).coefficients

    @arc.compile
    def traced(scale):
        assert isinstance(scale, SymbolicArray)
        return space.project(lambda x: scale * f_vec(x)).coefficients

    np.testing.assert_allclose(traced(np.array(1.0)), expected, atol=1e-10)
    np.testing.assert_allclose(traced(np.array(2.0)), 2.0 * expected, atol=1e-10)


def test_vector_inner_product_traces(space):
    """Exercises the symbolic contraction inside ``inner_product``."""
    fn = space.project(f_vec)

    @arc.compile
    def traced(coefficients):
        assert isinstance(coefficients, SymbolicArray)
        return Function(coefficients, space).dot(Function(coefficients, space))

    np.testing.assert_allclose(traced(fn.coefficients), fn.dot(fn), rtol=1e-10)


def test_gradient_of_vector_norm(space):
    """``norm`` is a scalar for vector-valued coefficients, so it can be
    differentiated -- which would fail outright if it returned an array.

    ``arc.grad`` differentiates with respect to a *vector* argument, so the
    coefficient matrix is passed flat and reshaped inside the objective;
    this is also the shape a solver would hand back.
    """
    fn = space.project(f_vec)
    shape = fn.coefficients.shape

    def objective(flat):
        return Function(np.reshape(flat, shape), space).norm()

    grad = arc.grad(objective)(fn.coefficients.ravel())
    assert grad.shape == (shape[0] * shape[1],)

    # d/dC sqrt(sum_k C_k . M . C_k) = (M @ C) / ||f||
    M = mass_matrix(space)  # noqa: N806
    expected = ((M @ fn.coefficients) / fn.norm()).ravel()
    # atol: some entries are structurally zero (a component orthogonal to a
    # basis function), where a relative tolerance is meaningless.
    np.testing.assert_allclose(grad, expected, rtol=1e-7, atol=1e-12)
