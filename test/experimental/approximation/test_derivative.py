"""Derivatives of BasisExpansions.

A derivative lowers the degree, so the exact result lives in a *smaller*
space -- the mirror of a product, which needs a larger one. Both follow the
same rule: return the tightest space in which the operation is exact. These
check against analytic derivatives, and against ``f(x, deriv=k)``, which is
the independent path to the same values.
"""

import numpy as np
import pytest
from _helpers import mass_matrix, stiffness_matrix

import archimedes as arc
from archimedes.experimental.approximation import (
    Basis,
    BasisExpansion,
    CubicHermiteBasis,
    FunctionSpace,
    LagrangeBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
    ProductParameters,
    TensorBasis,
)
from archimedes.measure import (
    JacobiMeasure,
    LegendreMeasure,
    ProbabilistsHermiteMeasure,
    UnitInterval,
)
from archimedes.quadrature import gauss_lobatto, gauss_radau

A, B = 0.0, 2.0
DOMAIN = UnitInterval.Parameters(a=A, b=B)
BREAKS = np.linspace(-1.0, 1.0, 4)
# Off the breakpoints: a C0 basis has a genuinely two-valued derivative there.
X = np.linspace(0.07, 1.93, 15)


def _lobatto(n):
    return LagrangeBasis(reference_nodes=gauss_lobatto(n).nodes)


SPACE_BUILDERS = {
    "modal": lambda: FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), 6), domain=DOMAIN
    ),
    "jacobi": lambda: FunctionSpace(
        OrthogonalPolynomialBasis(JacobiMeasure(1.5, 0.5), 6), domain=DOMAIN
    ),
    "nodal": lambda: FunctionSpace(_lobatto(6), domain=DOMAIN),
    "piecewise": lambda: FunctionSpace(
        PiecewiseBasis(_lobatto(4), BREAKS, continuity=0), domain=DOMAIN
    ),
}


@pytest.fixture(params=sorted(SPACE_BUILDERS))
def space(request):
    return SPACE_BUILDERS[request.param]()


def f_(x):
    return 2 * x**3 - x**2 + 5 * x - 1


def df_(x):
    return 6 * x**2 - 2 * x + 5


def d2f_(x):
    return 12 * x - 2


# -- exactness --


def test_derivative_is_exact(space):
    du = space.project(f_).derivative()
    np.testing.assert_allclose(du(X), df_(X), atol=1e-11)


def test_derivative_agrees_with_pointwise_evaluation(space):
    # The defining invariant: the same values reached two independent ways.
    u = space.project(f_)
    np.testing.assert_allclose(u.derivative()(X), u(X, deriv=1), atol=1e-11)


@pytest.mark.parametrize("order,exact", [(1, df_), (2, d2f_)])
def test_higher_order_derivatives(space, order, exact):
    du = space.project(f_).derivative(order)
    np.testing.assert_allclose(du(X), exact(X), atol=1e-10)


def test_repeated_differentiation_matches_a_single_call(space):
    u = space.project(f_)
    np.testing.assert_allclose(
        u.derivative().derivative()(X), u.derivative(2)(X), atol=1e-10
    )


def test_derivative_down_to_a_constant_is_allowed():
    # deriv == n_basis - 1 leaves the constants, which is a real space.
    space = FunctionSpace(OrthogonalPolynomialBasis(LegendreMeasure(), 4), DOMAIN)
    du = space.project(f_).derivative(3)
    assert du.space.n_basis == 1
    np.testing.assert_allclose(du(X), 12.0, atol=1e-10)


@pytest.mark.parametrize("order", [4, 9])
def test_derivative_past_the_degree_is_an_error(order):
    # Identically zero, so there is no space to put it in -- better to say so
    # than to invent a one-dimensional space holding zeros.
    space = FunctionSpace(OrthogonalPolynomialBasis(LegendreMeasure(), 4), DOMAIN)
    with pytest.raises(ValueError, match="at or past the degree"):
        space.project(f_).derivative(order)


@pytest.mark.parametrize(
    "basis,order",
    [
        (OrthogonalPolynomialBasis(LegendreMeasure(), 4), 4),
        (LagrangeBasis(reference_nodes=gauss_lobatto(4).nodes), 4),
        (
            PiecewiseBasis(
                LagrangeBasis(reference_nodes=gauss_lobatto(3).nodes),
                BREAKS,
                continuity=0,
            ),
            3,
        ),
    ],
    ids=["modal", "nodal", "piecewise"],
)
def test_every_family_rejects_a_derivative_past_the_degree(basis, order):
    # Piecewise delegates to its element basis, so the bound is the element's.
    with pytest.raises(ValueError, match="at or past the degree"):
        basis._derivative_basis(order)


def test_derivative_past_the_degree_names_the_tensor_dimension():
    basis = TensorBasis(
        (
            OrthogonalPolynomialBasis(LegendreMeasure(), 5),
            OrthogonalPolynomialBasis(LegendreMeasure(), 3),
        )
    )
    with pytest.raises(ValueError, match="in dimension 1: .*at or past the degree"):
        basis._derivative_basis((1, 3))


def test_pointwise_evaluation_past_the_degree_still_gives_zero():
    # `evaluate` is asking for values, where zero is the right answer; only
    # `derivative` needs a space and so has nowhere to put it.
    space = FunctionSpace(OrthogonalPolynomialBasis(LegendreMeasure(), 4), DOMAIN)
    np.testing.assert_allclose(space.project(f_)(X, deriv=4), 0.0, atol=1e-10)


# -- the space is the minimal one --


def test_derivative_space_is_smaller(space):
    u = space.project(f_)
    du = u.derivative()
    if isinstance(space.basis, PiecewiseBasis):
        # Sizing is per element, then reassembled under continuity.
        assert du.space.basis.element_basis[0].n_basis == (
            space.basis.element_basis[0].n_basis - 1
        )
    else:
        assert du.space.n_basis == space.n_basis - 1


def test_polynomial_families_keep_their_measure_and_normalization():
    for measure in (
        LegendreMeasure(),
        JacobiMeasure(1.5, 0.5),
        ProbabilistsHermiteMeasure(),
    ):
        basis = OrthogonalPolynomialBasis(measure, 5, density=True)
        derived = basis._derivative_basis()
        assert derived.measure == measure
        assert derived.density is True
        assert derived.n_basis == 4


def test_jacobi_derivative_stays_orthonormal():
    # The classical identity shifts (alpha, beta), but that is a statement
    # about sparsity, not span: the derivative is still exactly representable
    # in the same family, so the result's basis is orthonormal as before.
    space = FunctionSpace(
        OrthogonalPolynomialBasis(JacobiMeasure(1.5, 0.5), 6), domain=DOMAIN
    )
    derived = space._derivative_space()
    np.testing.assert_allclose(mass_matrix(derived), np.eye(5), atol=1e-12)


def test_projecting_back_up_is_exact():
    # The documented way to get `f + f.derivative()`: the derivative lives in
    # a subspace, so projecting it up loses nothing.
    space = SPACE_BUILDERS["modal"]()
    u = space.project(f_)
    lifted = space.project(u.derivative())
    assert lifted.space.n_basis == space.n_basis
    np.testing.assert_allclose(lifted(X), df_(X), atol=1e-11)
    np.testing.assert_allclose((u + lifted)(X), f_(X) + df_(X), atol=1e-11)


# -- piecewise --


def test_piecewise_derivative_becomes_discontinuous():
    basis = PiecewiseBasis(_lobatto(4), BREAKS, continuity=0)
    derived = basis._derivative_basis()
    assert derived.continuity == -1
    np.testing.assert_allclose(derived.breakpoints, BREAKS)


def test_piecewise_derivative_with_varying_order():
    # Per-element bases of different order each shrink by their own local
    # degree, and the result is still forced discontinuous.
    basis = PiecewiseBasis(
        (_lobatto(3), _lobatto(5), _lobatto(4)), BREAKS, continuity=0
    )
    derived = basis._derivative_basis()
    assert derived.continuity == -1
    assert [b.n_basis for b in derived.element_basis] == [2, 4, 3]


def test_piecewise_derivative_of_p1_elements_is_dg_p0():
    # The concrete FEM case: P1 continuous -> piecewise constant, one DOF per
    # element, and products of those stay at one DOF per element.
    breaks = np.linspace(-1.0, 1.0, 5)
    space = FunctionSpace(
        PiecewiseBasis(_lobatto(2), breaks, continuity=0), domain=DOMAIN
    )
    assert space.n_basis == 5
    du = space.project(lambda x: x**2).derivative()
    assert du.space.n_basis == 4
    assert du.space.basis.continuity == -1
    assert (du * du).space.n_basis == 4


def test_already_discontinuous_stays_discontinuous():
    basis = PiecewiseBasis(_lobatto(4), BREAKS, continuity=-1)
    assert basis._derivative_basis().continuity == -1


def test_zeroth_derivative_returns_the_same_basis():
    basis = PiecewiseBasis(_lobatto(4), BREAKS, continuity=0)
    assert basis._derivative_basis(0) is basis


# -- C1 (Hermite): continuity drops by exactly one order per differentiation --


def test_c1_hermite_first_derivative_is_c0():
    # Continuity through order q means differentiating `deriv` times can only
    # be relied on for continuity `q - deriv`: a C1 (Hermite) function's
    # *first* derivative is exactly C0 (matching slopes is the Hermite DOF
    # itself), not forced all the way down to discontinuous.
    basis = PiecewiseBasis(CubicHermiteBasis(), BREAKS, continuity=1)
    derived = basis._derivative_basis(1)
    assert derived.continuity == 0
    np.testing.assert_allclose(derived.breakpoints, BREAKS)


def test_c1_hermite_second_derivative_is_discontinuous():
    # Curvature/moment need not match across elements -- the familiar
    # "moment jump" in Hermite beam finite elements.
    basis = PiecewiseBasis(CubicHermiteBasis(), BREAKS, continuity=1)
    assert basis._derivative_basis(2).continuity == -1


def test_hermite_function_derivative_matches_closed_form():
    # End-to-end: f_ is a global cubic, exactly representable (and C1 at the
    # breakpoints) in a piecewise Hermite space, so both the derivative
    # BasisExpansion and the pointwise deriv=1 evaluation should reproduce df_
    # to numerical precision.
    space = FunctionSpace(
        PiecewiseBasis(CubicHermiteBasis(), BREAKS, continuity=1), domain=DOMAIN
    )
    u = space.project(f_)
    du = u.derivative()
    assert du.space.basis.continuity == 0
    np.testing.assert_allclose(du(X), df_(X), atol=1e-8)
    np.testing.assert_allclose(du(X), u(X, deriv=1), atol=1e-8)


# -- diff_matrix --


def test_square_diff_matrix_reproduces_the_classical_one():
    # For a Lagrange basis the generic projection formula collapses to the
    # barycentric differentiation matrix, up to the domain's chain rule.
    basis = _lobatto(6)
    space = FunctionSpace(basis, domain=DOMAIN)
    scale = (B - A) / 2
    np.testing.assert_allclose(
        space._diff_matrix(), basis._diff_matrix / scale, atol=1e-11
    )


def test_square_diff_matrix_is_exact_in_the_same_space(space):
    # The same-space form collocation wants: coefficients keep their meaning.
    u = space.project(f_)
    du = BasisExpansion(space._diff_matrix() @ u.coefficients, space)
    np.testing.assert_allclose(du(X), df_(X), atol=1e-10)


def test_diff_matrix_shape_follows_the_target(space):
    target = space._derivative_space()
    assert space._diff_matrix().shape == (space.n_basis, space.n_basis)
    assert space._diff_matrix(space=target).shape == (target.n_basis, space.n_basis)


def test_diff_matrix_matches_function_derivative(space):
    u = space.project(f_)
    target = space._derivative_space()
    np.testing.assert_allclose(
        space._diff_matrix(space=target) @ u.coefficients,
        u.derivative().coefficients,
        atol=1e-10,
    )


def test_explicit_result_space_is_honored(space):
    u = space.project(f_)
    du = u.derivative(space=space)
    assert du.space is space
    np.testing.assert_allclose(du(X), df_(X), atol=1e-10)


def test_undersized_explicit_space_projects_rather_than_failing():
    space = SPACE_BUILDERS["modal"]()
    small = FunctionSpace(OrthogonalPolynomialBasis(LegendreMeasure(), 2), DOMAIN)
    du = space.project(f_).derivative(space=small)
    assert du.space.n_basis == 2
    assert np.abs(du(X) - df_(X)).max() > 1e-6


# -- weak forms --


def test_stiffness_matrix_agrees_with_derivative_inner_products():
    # The end-to-end check that matters for FEM: assembling <u', v'> from
    # derivative BasisExpansions gives the same answer as the stiffness matrix.
    space = SPACE_BUILDERS["nodal"]()
    u = space.project(f_)
    v = space.project(lambda x: x**2 - 3 * x)
    assert u.derivative().dot(v.derivative()) == pytest.approx(
        u.coefficients @ stiffness_matrix(space) @ v.coefficients, rel=1e-10
    )


# -- tensor --


@pytest.mark.parametrize(
    "deriv,shape,exact",
    [
        ((1, 0), (4, 4), lambda x: 3 * x[:, 0] ** 2 * x[:, 1] ** 2 + 1.0),
        ((0, 1), (5, 3), lambda x: 2 * x[:, 0] ** 3 * x[:, 1] - 1.0),
        ((1, 1), (4, 3), lambda x: 6 * x[:, 0] ** 2 * x[:, 1]),
        ((2, 0), (3, 4), lambda x: 6 * x[:, 0] * x[:, 1] ** 2),
    ],
)
def test_tensor_partial_derivatives(deriv, shape, exact):
    basis = TensorBasis(
        (
            OrthogonalPolynomialBasis(LegendreMeasure(), 5),
            OrthogonalPolynomialBasis(LegendreMeasure(), 4),
        )
    )
    domain = ProductParameters(dims=(UnitInterval.Parameters(A, B),) * 2)
    space = FunctionSpace(basis, domain=domain)
    u = space.project(lambda x: x[:, 0] ** 3 * x[:, 1] ** 2 + x[:, 0] - x[:, 1])
    du = u.derivative(deriv)
    # Each factor shrinks by its own order; there is no cross-dimension
    # coupling.
    assert du.space.basis.shape == shape
    x = np.stack([np.linspace(0.1, 1.9, 9), np.linspace(0.2, 1.8, 9)], axis=-1)
    np.testing.assert_allclose(du(x), exact(x), atol=1e-10)


def test_tensor_derivative_rejects_a_bare_integer():
    basis = TensorBasis((OrthogonalPolynomialBasis(LegendreMeasure(), 4),) * 2)
    with pytest.raises(ValueError, match="must be a multi-index"):
        basis._derivative_basis(1)


# -- vector-valued --


def test_vector_valued_derivative(space):
    def fv(x):
        return np.stack([x**3, 2 * x**2], axis=-1)

    du = space.project(fv).derivative()
    assert du.coefficients.shape == (du.space.n_basis, 2)
    np.testing.assert_allclose(du(X), np.stack([3 * X**2, 4 * X], axis=-1), atol=1e-10)


# -- symbolic --


def test_derivative_traces(space):
    u = space.project(f_)
    expected = u.derivative()(X)

    @arc.compile
    def traced(c):
        return BasisExpansion(c, space).derivative()(X)

    np.testing.assert_allclose(
        np.asarray(traced(u.coefficients)).ravel(), expected, atol=1e-10
    )


def test_gradient_through_a_traced_domain_parameter():
    basis = _lobatto(5)
    coefficients = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    @arc.compile
    def energy(p):
        u = BasisExpansion(
            coefficients, FunctionSpace(basis, UnitInterval.Parameters(A, p[0]))
        )
        du = u.derivative()
        return du.dot(du)

    p = np.array([2.0])
    step = 1e-6
    fd = (energy(p + step) - energy(p)) / step
    assert arc.grad(energy)(p)[0] == pytest.approx(fd, rel=1e-5)


# -- errors --


@pytest.mark.parametrize(
    "basis",
    [
        OrthogonalPolynomialBasis(LegendreMeasure(), 4),
        LagrangeBasis(reference_nodes=gauss_lobatto(4).nodes),
        PiecewiseBasis(
            LagrangeBasis(reference_nodes=gauss_lobatto(3).nodes), BREAKS, continuity=0
        ),
    ],
    ids=["modal", "nodal", "piecewise"],
)
def test_negative_derivative_order_rejected(basis):
    with pytest.raises(ValueError, match="deriv must be >= 0"):
        basis._derivative_basis(-1)


# -- node families --


def _radau_left(n):
    return gauss_radau(n, endpoint="left").nodes


def test_default_node_family_is_lobatto():
    derived = _lobatto(5)._derivative_basis()
    np.testing.assert_allclose(derived.reference_nodes, gauss_lobatto(4).nodes)


def test_node_family_is_used_and_propagated():
    basis = LagrangeBasis(reference_nodes=_radau_left(5), node_family=_radau_left)
    derived = basis._derivative_basis()
    np.testing.assert_allclose(derived.reference_nodes, _radau_left(4))
    # Preserved, so repeated operations don't drift back to the default.
    assert derived.node_family is _radau_left
    np.testing.assert_allclose(
        derived._derivative_basis().reference_nodes, _radau_left(3)
    )
    np.testing.assert_allclose(
        basis._product_basis(basis).reference_nodes, _radau_left(9)
    )


def test_node_family_does_not_change_exactness():
    # Any n distinct nodes span the same P_{n-1}, so the choice cannot affect
    # whether the derivative is exact -- only conditioning and which DOFs are
    # nodal.
    basis = LagrangeBasis(reference_nodes=_radau_left(6), node_family=_radau_left)
    space = FunctionSpace(basis, domain=DOMAIN)
    np.testing.assert_allclose(space.project(f_).derivative()(X), df_(X), atol=1e-10)


def test_zeroth_derivative_keeps_explicit_nodes():
    # Regenerating from the family would silently move nodes that were given
    # explicitly; TensorBasis asks for order 0 on undifferentiated factors.
    nodes = np.array([-1.0, -0.3, 0.4, 1.0])
    basis = LagrangeBasis(reference_nodes=nodes)
    assert basis._derivative_basis(0) is basis


def test_tensor_derivative_preserves_undifferentiated_factor_nodes():
    nodes = np.array([-1.0, -0.3, 0.4, 1.0])
    basis = TensorBasis((LagrangeBasis(reference_nodes=nodes), _lobatto(4)))
    derived = basis._derivative_basis((0, 1))
    np.testing.assert_allclose(derived.bases[0].reference_nodes, nodes)


def test_bases_with_different_node_families_are_unequal():
    a = LagrangeBasis(reference_nodes=_radau_left(4))
    b = LagrangeBasis(reference_nodes=_radau_left(4), node_family=_radau_left)
    assert a != b
    assert hash(a) != hash(b)


def test_product_requires_matching_node_families():
    a = LagrangeBasis(reference_nodes=_radau_left(4), node_family=_radau_left)
    b = _lobatto(4)
    with pytest.raises(ValueError, match="same node_family"):
        a._product_basis(b)


def test_node_family_returning_the_wrong_count_is_rejected():
    basis = LagrangeBasis(
        reference_nodes=_radau_left(5), node_family=lambda n: np.zeros(2)
    )
    with pytest.raises(ValueError, match=r"node_family\(4\) returned shape"):
        basis._derivative_basis()


def test_basis_without_derivative_support_raises():
    class Constant(Basis):
        n_basis = 1

        @property
        def Parameters(self):  # noqa: N802
            return UnitInterval.Parameters

        def default_quadrature(self):
            return gauss_lobatto(2)

        def evaluate(self, x, deriv=0, a=None, b=None):
            column = np.ones_like(x) if deriv == 0 else np.zeros_like(x)
            return column[:, None]

    with pytest.raises(NotImplementedError, match="does not define a derivative basis"):
        Constant()._derivative_basis()
