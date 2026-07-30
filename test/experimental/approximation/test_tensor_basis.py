"""Tensor-product bases: multivariate function spaces built from univariate factors.

The correctness criterion throughout is exactness on *non-separable*
polynomials -- a tensor-product space is not a restriction to separable
functions, and checking only on products like ``f(x) g(y)`` would miss the
whole point (and would pass even if the Khatri-Rao ordering were wrong).
"""

import numpy as np
import pytest
from _helpers import mass_matrix, stiffness_matrix

import archimedes as arc
from archimedes.experimental.approximation import (
    Function,
    FunctionSpace,
    LagrangeBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
    ProductParameters,
    TensorBasis,
)
from archimedes.measure import (
    HermiteNormMeasure,
    LegendreMeasure,
    RealLine,
    UnitInterval,
)
from archimedes.quadrature import (
    composite,
    gauss_hermite,
    gauss_legendre,
    gauss_lobatto,
    tensor,
)

BOX = ProductParameters(
    dims=(UnitInterval.Parameters(0.0, 2.0), UnitInterval.Parameters(-1.0, 1.0))
)
BREAKS = np.linspace(-1.0, 1.0, 3)


def _modal(n):
    return OrthogonalPolynomialBasis(LegendreMeasure(), n)


def _nodal(n):
    return LagrangeBasis(reference_nodes=gauss_lobatto(n).nodes)


def _piecewise(n):
    return PiecewiseBasis(_nodal(n), BREAKS, continuity=0)


FACTORIES = {"modal": _modal, "nodal": _nodal, "piecewise": _piecewise}


@pytest.fixture(params=sorted(FACTORIES))
def space(request):
    make = FACTORIES[request.param]
    return FunctionSpace(TensorBasis((make(4), make(4))), domain=BOX)


def f_(x):
    """Non-separable: no product of a function of x and a function of y."""
    return x[:, 0] ** 2 * x[:, 1] + 3.0 * x[:, 0] - x[:, 1] ** 3 + 1.0


def _grid(nx=7, ny=5):
    gx, gy = np.meshgrid(
        np.linspace(0.0, 2.0, nx), np.linspace(-1.0, 1.0, ny), indexing="ij"
    )
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


# -- structure --


def test_sizes_and_shape():
    basis = TensorBasis((_modal(4), _modal(3), _modal(2)))
    assert basis.ndim == 3
    assert basis.n_basis == 24
    assert basis.shape == (4, 3, 2)


def test_measures_are_per_dimension():
    basis = TensorBasis(
        (OrthogonalPolynomialBasis(HermiteNormMeasure(), 3), _modal(3), _nodal(3))
    )
    assert basis.measures == (HermiteNormMeasure(), LegendreMeasure(), None)


def test_measures_default_to_none_for_weightless_families():
    assert _nodal(3).measures == (None,)
    # A piecewise basis reports its element basis's weight.
    assert _piecewise(3).measures == (None,)
    assert PiecewiseBasis(_modal(3), BREAKS, continuity=-1).measures == (
        LegendreMeasure(),
    )


def test_density_is_shared_across_factors():
    plain = TensorBasis((_modal(3), _modal(3)))
    assert plain.density is False
    normalized = TensorBasis(
        tuple(
            OrthogonalPolynomialBasis(HermiteNormMeasure(), 3, density=True)
            for _ in range(2)
        )
    )
    assert normalized.density is True


def test_parameters_type_and_domain_validation():
    basis = TensorBasis((_modal(3), _modal(3)))
    assert basis.Parameters is ProductParameters
    assert FunctionSpace(basis, domain=BOX).n_basis == 9


def test_equality_and_hash():
    assert TensorBasis((_modal(3), _modal(4))) == TensorBasis((_modal(3), _modal(4)))
    assert TensorBasis((_modal(3), _modal(4))) != TensorBasis((_modal(4), _modal(3)))
    assert hash(TensorBasis((_modal(3), _modal(4)))) == hash(
        TensorBasis((_modal(3), _modal(4)))
    )


def test_default_quadrature_is_the_tensor_of_the_factors_rules():
    basis = TensorBasis((_modal(3), _modal(4)))
    rule = basis.default_quadrature()
    assert rule.ndim == 2
    assert len(rule) == 12


# -- ordering --


def test_multi_index_is_flattened_in_c_order():
    # The claim the docstring makes: coefficients.reshape(shape) recovers the
    # natural array layout, with the last dimension varying fastest.
    bx, by = _modal(4), _modal(3)
    basis = TensorBasis((bx, by))
    x = _grid()
    phi = basis.evaluate(x)
    phi_x = bx.evaluate(x[:, 0])
    phi_y = by.evaluate(x[:, 1])
    for i in range(4):
        for j in range(3):
            flat = np.ravel_multi_index((i, j), basis.shape)
            np.testing.assert_allclose(phi[:, flat], phi_x[:, i] * phi_y[:, j])


def test_node_and_basis_orderings_agree():
    # Both the quadrature nodes and the basis multi-index use C order, which
    # is what lets a coefficient array be reshaped against either.
    basis = TensorBasis((_modal(3), _modal(4)))
    rule = basis.default_quadrature()
    assert rule.nodes.shape == (12, 2)
    assert basis.evaluate(rule.nodes).shape == (12, 12)


# -- exactness --


def test_projection_is_exact_for_non_separable_polynomials(space):
    u = space.project(f_)
    x = _grid()
    np.testing.assert_allclose(u(x), f_(x), atol=1e-11)


def test_orthonormal_basis_has_identity_mass_matrix():
    space = FunctionSpace(TensorBasis((_modal(4), _modal(3))), domain=BOX)
    np.testing.assert_allclose(mass_matrix(space), np.eye(12), atol=1e-12)


def test_separable_functions_are_representable_too():
    space = FunctionSpace(TensorBasis((_modal(4), _modal(4))), domain=BOX)
    x = _grid()

    def separable(x):
        return (x[:, 0] ** 2 - 1.0) * (x[:, 1] + 2.0)

    np.testing.assert_allclose(space.project(separable)(x), separable(x), atol=1e-11)


def test_three_dimensional_space():
    basis = TensorBasis((_modal(3), _modal(3), _modal(3)))
    domain = ProductParameters(dims=(UnitInterval.Parameters(0.0, 1.0),) * 3)
    space = FunctionSpace(basis, domain=domain)
    assert space.n_basis == 27

    def g(x):
        return x[:, 0] * x[:, 1] * x[:, 2] + x[:, 0] ** 2 - x[:, 2]

    x = np.random.default_rng(0).uniform(0.0, 1.0, size=(20, 3))
    np.testing.assert_allclose(space.project(g)(x), g(x), atol=1e-11)


def test_coefficients_reshape_to_the_multi_index_grid(space):
    u = space.project(f_)
    assert u.coefficients.reshape(space.basis.shape).shape == (
        space.basis.bases[0].n_basis,
        space.basis.bases[1].n_basis,
    )


# -- derivatives --


@pytest.mark.parametrize(
    "deriv,exact",
    [
        ((1, 0), lambda x: 2.0 * x[:, 0] * x[:, 1] + 3.0),
        ((0, 1), lambda x: x[:, 0] ** 2 - 3.0 * x[:, 1] ** 2),
        ((1, 1), lambda x: 2.0 * x[:, 0]),
        ((2, 0), lambda x: 2.0 * x[:, 1]),
        ((0, 3), lambda x: -6.0 * np.ones_like(x[:, 0])),
    ],
)
def test_partial_derivatives_including_mixed(space, deriv, exact):
    u = space.project(f_)
    # Stay off the element boundaries: a C0 piecewise basis has a genuinely
    # one-sided derivative there.
    x = np.stack([np.linspace(0.15, 1.85, 11), np.linspace(-0.85, 0.85, 11)], axis=-1)
    np.testing.assert_allclose(u(x, deriv=deriv), exact(x), atol=1e-9)


def test_derivative_beyond_the_degree_vanishes():
    basis = TensorBasis((_modal(3), _modal(3)))
    x = _grid()
    np.testing.assert_allclose(basis.evaluate(x, deriv=(3, 0)), 0.0, atol=1e-12)


def test_scalar_zero_is_shorthand_for_no_derivative():
    basis = TensorBasis((_modal(3), _modal(3)))
    x = _grid()
    np.testing.assert_allclose(basis.evaluate(x, deriv=0), basis.evaluate(x))
    np.testing.assert_allclose(basis.evaluate(x, deriv=(0, 0)), basis.evaluate(x))


def test_stiffness_matrix_is_the_gradient_form():
    # In >1D the stiffness matrix is the Laplacian's bilinear form, i.e. the
    # sum of the per-direction stiffnesses, not any single partial.
    basis = TensorBasis((_modal(4), _modal(4)))
    space = FunctionSpace(basis, domain=BOX)
    blocks = []
    for d in [(1, 0), (0, 1)]:
        dphi = space.basis_matrix(deriv=d)
        blocks.append(dphi.T @ dphi.matrix)
    np.testing.assert_allclose(stiffness_matrix(space), sum(blocks), atol=1e-12)


def test_stiffness_matrix_matches_a_known_laplacian_entry():
    # <grad u, grad u> for u = x*y on [0,1]^2 is int (y^2 + x^2) = 2/3.
    basis = TensorBasis((_modal(3), _modal(3)))
    domain = ProductParameters(dims=(UnitInterval.Parameters(0.0, 1.0),) * 2)
    space = FunctionSpace(basis, domain=domain)
    c = space.project(lambda x: x[:, 0] * x[:, 1]).coefficients
    assert c @ stiffness_matrix(space) @ c == pytest.approx(2.0 / 3.0)


# -- mixed measures / PCE --


def test_mixed_measures_with_density_give_moments():
    # Two independent Gaussians; with density=True the leading coefficient of
    # an orthonormal expansion is the mean and sum(c[1:]**2) the variance.
    basis = TensorBasis(
        tuple(
            OrthogonalPolynomialBasis(HermiteNormMeasure(), 4, density=True)
            for _ in range(2)
        )
    )
    sigmas = (2.0, 3.0)
    domain = ProductParameters(dims=tuple(RealLine.Parameters(0.0, s) for s in sigmas))
    space = FunctionSpace(basis, domain=domain)
    u = space.project(lambda x: x[:, 0] ** 2 + x[:, 1] ** 2)
    assert u.coefficients[0] == pytest.approx(sigmas[0] ** 2 + sigmas[1] ** 2)
    variance = 2 * sigmas[0] ** 4 + 2 * sigmas[1] ** 4
    assert np.sum(u.coefficients[1:] ** 2) == pytest.approx(variance)


def test_gaussian_and_uniform_dimensions_together():
    basis = TensorBasis(
        (
            OrthogonalPolynomialBasis(HermiteNormMeasure(), 4, density=True),
            OrthogonalPolynomialBasis(LegendreMeasure(), 4, density=True),
        )
    )
    domain = ProductParameters(
        dims=(RealLine.Parameters(0.0, 2.0), UnitInterval.Parameters(0.0, 1.0))
    )
    space = FunctionSpace(basis, domain=domain)
    u = space.project(lambda x: x[:, 0] ** 2 * x[:, 1])
    # E[xi^2 * u] = 4 * 0.5
    assert u.coefficients[0] == pytest.approx(2.0)


# -- vector-valued and products --


def test_vector_valued_projection(space):
    def fv(x):
        return np.stack([x[:, 0] * x[:, 1], x[:, 0] ** 2 - x[:, 1]], axis=-1)

    u = space.project(fv)
    assert u.coefficients.shape == (space.n_basis, 2)
    x = _grid()
    np.testing.assert_allclose(u(x), fv(x), atol=1e-11)


def test_products_factorize_dimension_by_dimension(space):
    f = space.project(lambda x: x[:, 0] * x[:, 1])
    g = space.project(lambda x: x[:, 0] + x[:, 1])
    product = f * g
    per_dim = [b.n_basis for b in product.space.basis.bases]
    assert product.space.n_basis == per_dim[0] * per_dim[1]
    x = _grid()
    np.testing.assert_allclose(
        product(x), (x[:, 0] * x[:, 1]) * (x[:, 0] + x[:, 1]), atol=1e-10
    )


def test_product_basis_sizes_per_dimension():
    a = TensorBasis((_modal(3), _modal(4)))
    b = TensorBasis((_modal(2), _modal(2)))
    assert a._product_basis(b).shape == (3 + 2 - 1, 4 + 2 - 1)


# -- domain parameter forms --


def test_dim_parameter_forms_agree():
    basis = TensorBasis((_modal(3), _modal(3)))
    x = _grid()
    as_params = basis.evaluate(
        x, dims=(UnitInterval.Parameters(0.0, 2.0), UnitInterval.Parameters(-1.0, 1.0))
    )
    as_dicts = basis.evaluate(x, dims=({"a": 0.0, "b": 2.0}, {"a": -1.0, "b": 1.0}))
    as_tuples = basis.evaluate(x, dims=((0.0, 2.0), (-1.0, 1.0)))
    np.testing.assert_allclose(as_dicts, as_params)
    np.testing.assert_allclose(as_tuples, as_params)


def test_omitted_dims_use_the_reference_domain():
    basis = TensorBasis((_modal(3), _modal(3)))
    x = np.stack([np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)], axis=-1)
    np.testing.assert_allclose(basis.evaluate(x, dims=(None, None)), basis.evaluate(x))


def test_too_many_positional_dim_parameters_rejected():
    basis = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(ValueError, match="positional domain parameters"):
        basis.evaluate(_grid(), dims=((0.0, 1.0, 2.0), (0.0, 1.0)))


def test_unrecognized_dim_parameter_spec_rejected():
    basis = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(TypeError, match="per-dimension parameters must be"):
        basis.evaluate(_grid(), dims=(3.0, None))


def test_wrong_number_of_dim_parameters_rejected():
    basis = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(ValueError, match="expected 2 per-dimension parameters"):
        basis.evaluate(_grid(), dims=((0.0, 1.0),))


# -- symbolic --


def test_traces_and_differentiates(space):
    u = space.project(f_)
    x = _grid(3, 3)
    expected = u(x)

    @arc.compile
    def evaluate(c):
        return Function(c, space)(x)

    np.testing.assert_allclose(
        np.asarray(evaluate(u.coefficients)).ravel(), expected, atol=1e-11
    )


def test_gradient_through_domain_parameters():
    basis = TensorBasis((_modal(3), _modal(3)))
    x = _grid(3, 3)

    @arc.compile
    def total(p):
        domain = ProductParameters(
            dims=(
                UnitInterval.Parameters(0.0, p[0]),
                UnitInterval.Parameters(0.0, p[1]),
            )
        )
        space = FunctionSpace(basis, domain=domain)
        return np.sum(space._basis_eval(x))

    p = np.array([2.0, 3.0])
    grad = arc.grad(total)(p)
    step = 1e-6
    for i in range(2):
        shifted = p.copy()
        shifted[i] += step
        fd = (total(shifted) - total(p)) / step
        assert grad[i] == pytest.approx(fd, rel=1e-4, abs=1e-6)


# -- quadrature compatibility --


def test_rule_of_the_wrong_dimension_rejected():
    basis = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(ValueError, match="but the quadrature rule is 1-dimensional"):
        FunctionSpace(basis, domain=BOX, quad_rule=gauss_legendre(4))


def test_mismatched_weight_rejected_per_dimension():
    basis = TensorBasis((_modal(3), _modal(3)))
    rule = tensor(gauss_legendre(4), gauss_hermite(4))
    with pytest.raises(ValueError, match="does not match the basis in dimension 1"):
        FunctionSpace(basis, domain=BOX, quad_rule=rule)


def test_mismatched_weight_rejected_in_one_dimension():
    # The same check applies to a plain 1-D space, where it was previously
    # possible to integrate a Legendre basis against a Hermite rule.
    basis = _modal(3)
    with pytest.raises(ValueError, match="quadrature weight does not match"):
        FunctionSpace(
            basis, domain=UnitInterval.Parameters(0.0, 1.0), quad_rule=gauss_hermite(4)
        )


def test_weightless_basis_imposes_no_weight_constraint():
    # A nodal basis reports no measure, so any rule is structurally allowed.
    basis = TensorBasis((_nodal(3), _nodal(3)))
    space = FunctionSpace(
        basis, domain=BOX, quad_rule=tensor(gauss_legendre(5), gauss_legendre(5))
    )
    assert space.n_basis == 9


def test_misaligned_breakpoints_rejected_per_dimension():
    basis = TensorBasis((_piecewise(3), _modal(3)))
    rule = tensor(gauss_legendre(6), gauss_legendre(4))
    with pytest.raises(ValueError, match="piecewise smooth in dimension 0"):
        FunctionSpace(basis, domain=BOX, quad_rule=rule)


def test_aligned_composite_rule_accepted():
    basis = TensorBasis((_piecewise(3), _modal(3)))
    rule = tensor(composite(gauss_legendre(3), BREAKS), gauss_legendre(4))
    space = FunctionSpace(basis, domain=BOX, quad_rule=rule)
    np.testing.assert_allclose(
        space.project(lambda x: x[:, 0] * x[:, 1])(_grid()),
        _grid()[:, 0] * _grid()[:, 1],
        atol=1e-11,
    )


def test_required_breakpoints_is_a_per_dimension_tuple():
    basis = TensorBasis((_piecewise(3), _modal(3)))
    required = basis.required_breakpoints
    np.testing.assert_allclose(required[0], BREAKS)
    assert required[1] is None


# -- errors --


def test_requires_at_least_one_dimension():
    with pytest.raises(ValueError, match="at least one dimension"):
        TensorBasis(())


def test_rejects_non_basis_factors():
    with pytest.raises(TypeError, match=r"bases\[1\] must be a Basis"):
        TensorBasis((_modal(3), "nope"))


def test_rejects_nested_tensor_factors():
    inner = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(ValueError, match="must be univariate"):
        TensorBasis((_modal(3), inner))


def test_rejects_disagreeing_density():
    with pytest.raises(ValueError, match="must agree on `density`"):
        TensorBasis(
            (
                OrthogonalPolynomialBasis(LegendreMeasure(), 3),
                OrthogonalPolynomialBasis(LegendreMeasure(), 3, density=True),
            )
        )


def test_rejects_nonzero_integer_deriv():
    basis = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(ValueError, match="must be a multi-index"):
        basis.evaluate(_grid(), deriv=1)


def test_rejects_wrong_length_multi_index():
    basis = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(ValueError, match="one entry per dimension"):
        basis.evaluate(_grid(), deriv=(1, 0, 0))


def test_rejects_negative_derivative_order():
    basis = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(ValueError, match="orders must be >= 0"):
        basis.evaluate(_grid(), deriv=(-1, 0))


def test_rejects_wrong_x_shape():
    basis = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(ValueError, match=r"x must have shape \(npts, 2\)"):
        basis.evaluate(np.linspace(0.0, 1.0, 5))
    with pytest.raises(ValueError, match=r"x must have shape \(npts, 2\)"):
        basis.evaluate(np.zeros((5, 3)))


def test_rejects_mismatched_product_dimensions():
    a = TensorBasis((_modal(3), _modal(3)))
    b = TensorBasis((_modal(3),))
    with pytest.raises(ValueError, match="same number of dimensions"):
        a._product_basis(b)


def test_rejects_product_with_a_non_tensor_basis():
    a = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(ValueError, match="cannot form a product basis"):
        a._product_basis(_modal(3))


def test_product_parameters_validation():
    with pytest.raises(ValueError, match="at least one dimension"):
        ProductParameters(dims=())
    with pytest.raises(TypeError, match=r"dims\[0\] must be a"):
        ProductParameters(dims=(1.0,))


def test_domain_must_be_product_parameters():
    basis = TensorBasis((_modal(3), _modal(3)))
    with pytest.raises(TypeError, match="must be a ProductParameters"):
        FunctionSpace(basis, domain=UnitInterval.Parameters(0.0, 1.0))


def test_piecewise_rejects_a_multivariate_element_basis():
    inner = TensorBasis((_nodal(3), _nodal(3)))
    with pytest.raises(ValueError, match="element_basis must be univariate"):
        PiecewiseBasis(inner, BREAKS, continuity=-1)
