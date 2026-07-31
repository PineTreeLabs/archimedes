"""Pointwise products of BasisExpansions.

The product of two basis expansions does not lie in either operand's space,
but for polynomial families the space it *does* lie in is known statically:
``n_1 + n_2 - 1``. So the product can be exact rather than approximate,
which is what these check -- against the analytic product, not against a
reimplementation.
"""

import numpy as np
import pytest

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray
from archimedes.experimental.approximation import (
    Basis,
    BasisExpansion,
    CubicHermiteBasis,
    FourierBasis,
    FunctionSpace,
    LagrangeBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
)
from archimedes.measure import (
    LegendreMeasure,
    PhysicistsHermiteMeasure,
    RealLine,
    UnitInterval,
)
from archimedes.quadrature import gauss_lobatto

A, B = 0.0, 2.0
DOMAIN = UnitInterval.Parameters(a=A, b=B)
BREAKPOINTS = np.linspace(-1.0, 1.0, 3)


def _lobatto(n):
    return LagrangeBasis(reference_nodes=gauss_lobatto(n).nodes)


SPACE_BUILDERS = {
    "modal": lambda: FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), 4), domain=DOMAIN
    ),
    "nodal": lambda: FunctionSpace(_lobatto(4), domain=DOMAIN),
    "piecewise": lambda: FunctionSpace(
        PiecewiseBasis(_lobatto(4), BREAKPOINTS, continuity=0), domain=DOMAIN
    ),
}


@pytest.fixture(params=sorted(SPACE_BUILDERS))
def space(request):
    return SPACE_BUILDERS[request.param]()


def f_(x):
    return 2 * x**2 - x + 1


def g_(x):
    return x**3 - 3 * x


# -- exactness --


def test_product_is_exact(space):
    f, g = space.project(f_), space.project(g_)
    x = np.linspace(A, B, 61)
    np.testing.assert_allclose(
        (f * g)(x), f_(x) * g_(x), atol=1e-12, err_msg="product is not exact"
    )


def test_product_space_size(space):
    f, g = space.project(f_), space.project(g_)
    n1, n2 = f.space.n_basis, g.space.n_basis
    product = f * g
    if isinstance(space.basis, PiecewiseBasis):
        # Sizing is per element, then reassembled under continuity.
        n_local = space.basis.element_basis[0].n_basis
        expected_local = 2 * n_local - 1
        assert product.space.basis.element_basis[0].n_basis == expected_local
    else:
        assert product.space.n_basis == n1 + n2 - 1


def test_multiply_is_commutative(space):
    f, g = space.project(f_), space.project(g_)
    x = np.linspace(A, B, 41)
    np.testing.assert_allclose((f * g)(x), (g * f)(x), atol=1e-12)


def test_repeated_products_stay_exact_and_grow(space):
    f = space.project(f_)
    x = np.linspace(A, B, 41)
    squared = f * f
    cubed = squared * f
    np.testing.assert_allclose(squared(x), f_(x) ** 2, atol=1e-11)
    np.testing.assert_allclose(cubed(x), f_(x) ** 3, atol=1e-10)
    # Degree grows: this is why reducing back down is an explicit project().
    assert cubed.space.n_basis > squared.space.n_basis > f.space.n_basis


def test_project_back_down_after_product(space):
    # The documented way to control growth.
    f, g = space.project(f_), space.project(g_)
    product = f * g
    reduced = space.project(product)
    assert reduced.space.n_basis == space.n_basis
    x = np.linspace(A, B, 41)
    # Truncation loses information, but the low-order content is retained.
    assert np.abs(reduced(x) - product(x)).max() > 1e-6


# -- scalars still work --


def test_scalar_multiplication_stays_in_space(space):
    f = space.project(f_)
    x = np.linspace(A, B, 21)
    for scaled in (3.0 * f, f * 3.0):
        assert scaled.space is f.space
        np.testing.assert_allclose(scaled(x), 3.0 * f_(x), atol=1e-10)


# -- vector-valued --


def test_vector_times_vector_is_elementwise(space):
    def fv(x):
        return np.stack([x**2, 1.0 - x], axis=-1)

    def gv(x):
        return np.stack([x, x**3], axis=-1)

    f, g = space.project(fv), space.project(gv)
    x = np.linspace(A, B, 41)
    np.testing.assert_allclose((f * g)(x), fv(x) * gv(x), atol=1e-11)


def test_scalar_valued_times_vector_valued_broadcasts(space):
    def gv(x):
        return np.stack([x, x**3], axis=-1)

    f, g = space.project(f_), space.project(gv)
    x = np.linspace(A, B, 41)
    expected = f_(x)[:, None] * gv(x)
    np.testing.assert_allclose((f * g)(x), expected, atol=1e-11)
    np.testing.assert_allclose((g * f)(x), expected, atol=1e-11)


# -- explicit result space --


def test_explicit_space_override(space):
    f, g = space.project(f_), space.project(g_)
    big = f.space._product_space(g.space)
    product = f.multiply(g, space=big)
    x = np.linspace(A, B, 41)
    np.testing.assert_allclose(product(x), f_(x) * g_(x), atol=1e-11)


def test_undersized_explicit_space_projects_rather_than_failing(space):
    # Documented behavior: a too-small space gives the projection of the
    # product, which is a well-defined approximation but not exact.
    f, g = space.project(f_), space.project(g_)
    product = f.multiply(g, space=space)
    assert product.space.n_basis == space.n_basis
    x = np.linspace(A, B, 41)
    assert np.abs(product(x) - f_(x) * g_(x)).max() > 1e-6


# -- incompatibility --


def test_different_measures_rejected():
    legendre = OrthogonalPolynomialBasis(LegendreMeasure(), 3)
    hermite = OrthogonalPolynomialBasis(PhysicistsHermiteMeasure(), 3)
    with pytest.raises(ValueError, match="same measure"):
        legendre._product_basis(hermite)


def test_different_density_rejected():
    raw = OrthogonalPolynomialBasis(PhysicistsHermiteMeasure(), 3)
    density = OrthogonalPolynomialBasis(PhysicistsHermiteMeasure(), 3, density=True)
    with pytest.raises(ValueError, match="same normalization"):
        raw._product_basis(density)


def test_matching_density_forwarded_to_product():
    left = OrthogonalPolynomialBasis(PhysicistsHermiteMeasure(), 3, density=True)
    right = OrthogonalPolynomialBasis(PhysicistsHermiteMeasure(), 4, density=True)
    product = left._product_basis(right)
    assert product.density is True


def test_different_basis_families_rejected():
    modal = OrthogonalPolynomialBasis(LegendreMeasure(), 3)
    nodal = _lobatto(3)
    with pytest.raises(ValueError, match="cannot form a product basis"):
        modal._product_basis(nodal)
    with pytest.raises(ValueError, match="cannot form a product basis"):
        nodal._product_basis(modal)
    with pytest.raises(ValueError, match="cannot form a product basis"):
        PiecewiseBasis(nodal, BREAKPOINTS, continuity=0)._product_basis(nodal)


def test_different_breakpoints_rejected():
    left = PiecewiseBasis(_lobatto(3), np.linspace(-1.0, 1.0, 3), continuity=0)
    right = PiecewiseBasis(_lobatto(3), np.linspace(-1.0, 1.0, 4), continuity=0)
    with pytest.raises(ValueError, match="identical breakpoints"):
        left._product_basis(right)


def test_structurally_different_domains_rejected():
    interval = FunctionSpace(
        OrthogonalPolynomialBasis(LegendreMeasure(), 3), domain=DOMAIN
    )
    line = FunctionSpace(
        OrthogonalPolynomialBasis(PhysicistsHermiteMeasure(), 3),
        domain=RealLine.Parameters(),
    )
    with pytest.raises(ValueError, match="structurally identical domains"):
        interval._product_space(line)


def test_basis_without_product_support_raises():
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

    with pytest.raises(NotImplementedError, match="does not define a product basis"):
        Constant()._product_basis(Constant())


def test_hermite_has_no_product_basis():
    # A cubic times a cubic is degree 6, not representable in this 4-dim
    # family; CubicHermiteBasis doesn't override `_product_basis`, so this
    # is the base class's NotImplementedError, propagated unchanged through
    # PiecewiseBasis's own per-element `_product_basis`.
    hermite = CubicHermiteBasis()
    with pytest.raises(NotImplementedError, match="does not define a product basis"):
        hermite._product_basis(hermite)

    basis = PiecewiseBasis(hermite, BREAKPOINTS, continuity=1)
    with pytest.raises(NotImplementedError, match="does not define a product basis"):
        basis._product_basis(basis)


# -- continuity of piecewise products --


@pytest.mark.parametrize(
    "left,right,expected",
    [(0, 0, 0), (-1, -1, -1), (0, -1, -1), (-1, 0, -1)],
)
def test_piecewise_product_takes_weaker_continuity(left, right, expected):
    # A product is only as smooth as its least smooth factor.
    a = PiecewiseBasis(_lobatto(3), BREAKPOINTS, continuity=left)
    b = PiecewiseBasis(_lobatto(3), BREAKPOINTS, continuity=right)
    assert a._product_basis(b).continuity == expected


def test_product_basis_with_varying_order():
    # Per-element product sizes follow the same n_1 + n_2 - 1 rule as the
    # scalar case, applied element by element.
    left = PiecewiseBasis((_lobatto(3), _lobatto(5)), BREAKPOINTS, continuity=-1)
    right = PiecewiseBasis((_lobatto(2), _lobatto(4)), BREAKPOINTS, continuity=-1)
    product = left._product_basis(right)
    assert [b.n_basis for b in product.element_basis] == [3 + 2 - 1, 5 + 4 - 1]


def test_discontinuous_product_is_exact():
    basis = PiecewiseBasis(_lobatto(4), BREAKPOINTS, continuity=-1)
    space = FunctionSpace(basis, domain=DOMAIN)
    f, g = space.project(f_), space.project(g_)
    x = np.linspace(A, B, 51)
    np.testing.assert_allclose((f * g)(x), f_(x) * g_(x), atol=1e-11)


def test_single_node_lagrange_product():
    # n == 1 for both operands: gauss_lobatto is undefined below 2 points,
    # so the constant case takes its own branch.
    constant = LagrangeBasis(reference_nodes=np.zeros(1))
    product = constant._product_basis(constant)
    assert product.n_basis == 1
    x = np.linspace(-1.0, 1.0, 5)
    np.testing.assert_allclose(product.evaluate(x), np.ones((5, 1)), atol=1e-12)


# -- FourierBasis: product-to-sum closure table --
# Fourier's own basis functions aren't polynomials, so this doesn't fit the
# generic `space`/`f_`/`g_` fixture above -- a dedicated section, same as
# the piecewise-specific tests above it.


@pytest.mark.parametrize(
    "kind1,n1,kind2,n2,expected_kind,expected_n",
    [
        ("full", 5, "full", 3, "full", 7),  # N=2+1=3 -> 2*3+1=7
        ("full", 5, "cosine", 3, "full", 9),  # N=2+2=4 -> 2*4+1=9
        ("full", 5, "sine", 2, "full", 9),  # N=2+2=4 -> 2*4+1=9
        ("cosine", 3, "full", 5, "full", 9),  # order shouldn't matter
        ("cosine", 3, "cosine", 4, "cosine", 6),  # N=2+3=5 -> 5+1=6
        ("sine", 2, "sine", 3, "cosine", 6),  # N=2+3=5 -> lands in cosine, 5+1=6
        ("cosine", 3, "sine", 3, "sine", 5),  # N=2+3=5 -> sine, n_basis=5
        ("sine", 3, "cosine", 3, "sine", 5),  # order shouldn't matter
    ],
)
def test_fourier_product_closure_table(kind1, n1, kind2, n2, expected_kind, expected_n):
    left = FourierBasis(n1, kind=kind1)
    right = FourierBasis(n2, kind=kind2)
    product = left._product_basis(right)
    assert product.kind == expected_kind
    assert product.n_basis == expected_n


def test_fourier_product_is_exact():
    domain = UnitInterval.Parameters(a=-1.0, b=1.0)
    space = FunctionSpace(FourierBasis(5, kind="full"), domain=domain)

    def f(x):
        return 1.0 + 2 * np.cos(np.pi * x) - np.sin(np.pi * x)

    def g(x):
        return np.cos(2 * np.pi * x) + 3 * np.sin(np.pi * x)

    fexp, gexp = space.project(f), space.project(g)
    product = fexp * gexp
    assert product.space.basis.kind == "full"
    x = np.linspace(-1.0, 1.0, 41)
    np.testing.assert_allclose((fexp * gexp)(x), f(x) * g(x), atol=1e-10)


def test_fourier_sine_squared_lands_in_cosine():
    domain = UnitInterval.Parameters(a=-1.0, b=1.0)
    space = FunctionSpace(FourierBasis(2, kind="sine"), domain=domain)

    def f(x):
        return np.sin(np.pi * x)

    fexp = space.project(f)
    squared = fexp * fexp
    assert squared.space.basis.kind == "cosine"
    x = np.linspace(-1.0, 1.0, 41)
    np.testing.assert_allclose(squared(x), f(x) ** 2, atol=1e-10)


def test_fourier_different_density_rejected():
    raw = FourierBasis(5, kind="full", density=False)
    density = FourierBasis(5, kind="full", density=True)
    with pytest.raises(ValueError, match="same normalization"):
        raw._product_basis(density)


def test_fourier_matching_density_forwarded_to_product():
    left = FourierBasis(3, kind="cosine", density=True)
    right = FourierBasis(4, kind="cosine", density=True)
    product = left._product_basis(right)
    assert product.density is True


def test_fourier_different_basis_families_rejected():
    fourier = FourierBasis(5, kind="full")
    modal = OrthogonalPolynomialBasis(LegendreMeasure(), 3)
    with pytest.raises(ValueError, match="cannot form a product basis"):
        fourier._product_basis(modal)
    with pytest.raises(ValueError, match="cannot form a product basis"):
        modal._product_basis(fourier)


# -- symbolic --


def test_product_traces(space):
    f, g = space.project(f_), space.project(g_)
    x = np.linspace(A, B, 5)
    expected = (f * g)(x)

    @arc.compile
    def traced(cf, cg):
        assert isinstance(cf, SymbolicArray)
        product = BasisExpansion(cf, f.space) * BasisExpansion(cg, g.space)
        return product(x)

    np.testing.assert_allclose(
        np.asarray(traced(f.coefficients, g.coefficients)).ravel(),
        expected,
        atol=1e-10,
    )
