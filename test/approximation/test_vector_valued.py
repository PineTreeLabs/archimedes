"""Vector-valued (scalar -> R^m) function approximation.

The vector-valuedness lives entirely in the coefficients: a ``Basis``
evaluates to ``(npts, n_basis)`` regardless, so a ``FunctionSpace`` built
for scalars carries ``(n_basis, m)`` coefficients unchanged. The plumbing
this exercises (``project``, ``_evaluate``, ``_inner_product``'s component
contraction, pytree flattening, symbolic tracing) does not branch on which
family's ``Basis`` is underneath, so most tests below use a single
representative family (``nodal``) rather than sweeping all three. The two
tests that exercise a family's own ``evaluate``/``derivative`` path
directly also sweep ``piecewise``, the one family with element-tiled
evaluation.
"""

import numpy as np
from _helpers import mass_matrix
from conftest import family_space

import archimedes as arc
from archimedes import tree
from archimedes._core._array_impl import SymbolicArray
from archimedes.approximation import Function
from archimedes.quadrature import composite_quad, gauss_legendre

A, B = 0.0, 2.0  # target domain
BREAKPOINTS = np.linspace(-1.0, 1.0, 3)

# One representative family for the generic (family-agnostic) vector-valued
# plumbing -- see the module docstring.
space = family_space(
    "nodal",
    nodal_n_basis=4,
    quad_rule=gauss_legendre(10),
)

# nodal + piecewise, for the two tests that actually touch a family's own
# evaluate/derivative implementation rather than just the generic plumbing
# around it.
space2 = family_space(
    "nodal",
    "piecewise",
    nodal_n_basis=4,
    element_n_basis=4,
    breakpoints=BREAKPOINTS,
    quad_rule=gauss_legendre(10),
    piecewise_quad_rule=composite_quad(gauss_legendre(5), BREAKPOINTS),
)


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


# -- projection --
# (Shape-only checks -- `test_evaluate_shapes`/`test_project_preserves_shape`
# -- are cut: every exactness test below already implies the shape is right,
# since a wrong shape would fail the value comparison too.)


def test_project_vector_is_exact_for_representable_components(space2):
    fn = space2.project(f_vec)
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
    # through the vector branch too. An override rule is used exactly as
    # given (no further domain mapping -- see `FunctionSpace.quad_rule`),
    # so it must already be mapped onto this space's domain.
    rule = gauss_legendre(12, A, B)
    fn = space.project(f_vec, quad_rule=rule)
    x = np.linspace(A, B, 21)
    np.testing.assert_allclose(fn(x), f_vec(x), atol=1e-9)


def test_project_single_column_is_not_scalar_case(space):
    """An ``(npts, 1)`` target stays 2-D rather than collapsing to scalar."""

    def f(x):
        return np.stack([x**2], axis=-1)

    fn = space.project(f)
    assert fn.coefficients.shape == (space.n_basis, 1)
    assert fn(np.linspace(A, B, 5)).shape == (5, 1)


def test_derivative_of_vector_function(space2):
    fn = space2.project(f_vec)
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
        space._inner_product(fn.coefficients[:, k], gn.coefficients[:, k])
        for k in range(M_COMPONENTS)
    )
    np.testing.assert_allclose(joint, per_component, rtol=1e-10)


def test_norm_is_scalar_l2_norm_of_whole_function(space):
    fn = space.project(f_vec)
    norm = fn.norm()
    assert np.ndim(norm) == 0

    component_norms = [
        np.sqrt(space._inner_product(fn.coefficients[:, k], fn.coefficients[:, k]))
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


def test_vector_function_is_pytree(space):
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
