"""Tests for ``FunctionSpace.piecewise()``, the classmethod sugar over
``PiecewiseBasis`` + ``UnitInterval.Parameters`` that lets a caller give a
mesh directly on the physical domain instead of separately naming the
reference-domain breakpoints and the target ``a``/``b``.
"""

import numpy as np
import pytest

from archimedes.experimental.approximation import (
    CubicHermiteBasis,
    FunctionSpace,
    LagrangeBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
)
from archimedes.experimental.approximation._function_space import (
    _normalize_breakpoints,
)
from archimedes.measure import LegendreMeasure, UnitInterval
from archimedes.quadrature import composite_quad, gauss_legendre


def test_matches_manual_construction():
    # Today's hand-built Problem-B-style space (quadratic Lagrange, Lobatto
    # nodes, C0, 10 elements on [0, 1]) vs. the classmethod.
    element = LagrangeBasis(reference_nodes=np.array([-1.0, 0.0, 1.0]))
    manual = FunctionSpace(
        PiecewiseBasis(element, np.linspace(-1.0, 1.0, 11), continuity=0),
        UnitInterval.Parameters(a=0.0, b=1.0),
    )
    sugar = FunctionSpace.piecewise("lagrange", 2, np.linspace(0.0, 1.0, 11))

    assert sugar.n_basis == manual.n_basis
    phi_m, phi_s = manual.basis_matrix(), sugar.basis_matrix()
    np.testing.assert_allclose(phi_m.matrix, phi_s.matrix)
    np.testing.assert_allclose(phi_m.weights, phi_s.weights)


@pytest.mark.parametrize(
    "family, continuity",
    [
        ("lobatto", 0),
        ("equispaced", 0),
        ("legendre", -1),  # no endpoint nodes -- C0 is invalid, see below
        ("radau_left", -1),
        ("radau_right", -1),
    ],
)
def test_nodes_family_matches_lagrange_classmethod(family, continuity):
    a, b = 2.0, 7.0  # off-reference domain, to exercise the affine mapping
    breakpoints = np.array([a, 4.0, b])
    sugar = FunctionSpace.piecewise(
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
        sugar.basis_matrix().matrix, expected.basis_matrix().matrix
    )


def test_nodes_as_callable():
    family = LagrangeBasis.gauss_legendre
    sugar = FunctionSpace.piecewise(
        "lagrange",
        3,
        np.linspace(0.0, 1.0, 3),
        nodes=lambda n: family(n).reference_nodes,
        continuity=-1,  # Gauss-Legendre nodes have no endpoint DOFs
    )
    assert (
        sugar.n_basis
        == PiecewiseBasis(
            LagrangeBasis.gauss_legendre(4), np.linspace(-1.0, 1.0, 3), continuity=-1
        ).n_basis
    )


def test_nodes_as_explicit_array():
    sugar = FunctionSpace.piecewise(
        "lagrange", 2, np.linspace(0.0, 1.0, 3), nodes=np.array([-1.0, 0.0, 1.0])
    )
    assert sugar.n_basis == 5  # 2 elements * 3 nodes - 1 shared


def test_legendre_kind_builds_modal_dg_space():
    sugar = FunctionSpace.piecewise(
        "legendre", 3, np.linspace(-1.0, 1.0, 4), continuity=-1
    )
    assert isinstance(sugar.basis, PiecewiseBasis)
    assert all(
        isinstance(eb, OrthogonalPolynomialBasis) for eb in sugar.basis.element_basis
    )
    assert all(eb.measures[0] == LegendreMeasure() for eb in sugar.basis.element_basis)
    assert sugar.n_basis == 12


def test_legendre_kind_rejects_continuity_zero():
    # OrthogonalPolynomialBasis has no boundary DOF, so C0 assembly is
    # incoherent -- PiecewiseBasis itself rejects this; no new check needed.
    with pytest.raises(ValueError, match="continuity=0"):
        FunctionSpace.piecewise("legendre", 3, np.linspace(-1.0, 1.0, 4), continuity=0)


def test_legendre_kind_rejects_nodes():
    with pytest.raises(ValueError, match="nodes is only meaningful"):
        FunctionSpace.piecewise(
            "legendre", 3, np.linspace(-1.0, 1.0, 4), nodes="lobatto", continuity=-1
        )


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="kind must be"):
        FunctionSpace.piecewise("bogus", 3, np.linspace(-1.0, 1.0, 4))


def test_unknown_nodes_family_rejected():
    with pytest.raises(ValueError, match="unknown nodes family"):
        FunctionSpace.piecewise(
            "lagrange", 3, np.linspace(0.0, 1.0, 4), nodes="bogus_family"
        )


def test_nodes_array_length_mismatch_rejected():
    with pytest.raises(ValueError, match="nodes has 3 points but degree=3 needs 4"):
        FunctionSpace.piecewise(
            "lagrange", 3, np.linspace(0.0, 1.0, 3), nodes=np.array([-1.0, 0.0, 1.0])
        )


@pytest.mark.parametrize(
    "breakpoints",
    [
        np.array([1.0]),  # too few points
        np.array([1.0, 0.5, 2.0]),  # not increasing
    ],
)
def test_invalid_breakpoints_rejected(breakpoints):
    with pytest.raises(ValueError):
        FunctionSpace.piecewise("lagrange", 2, breakpoints)


def test_normalize_breakpoints_pins_reference_endpoints_exactly():
    a, b, ref = _normalize_breakpoints([0.0, 0.3, 1.0])
    assert a == 0.0
    assert b == 1.0
    assert ref[0] == -1.0
    assert ref[-1] == 1.0
    np.testing.assert_allclose(ref, np.array([-1.0, -0.4, 1.0]))


def test_degree_as_per_element_tuple():
    sugar = FunctionSpace.piecewise("lagrange", (1, 2, 3), np.linspace(0.0, 1.0, 4))
    manual = PiecewiseBasis(
        (
            LagrangeBasis.gauss_lobatto(2),
            LagrangeBasis.gauss_lobatto(3),
            LagrangeBasis.gauss_lobatto(4),
        ),
        np.linspace(-1.0, 1.0, 4),
        continuity=0,
    )
    assert sugar.n_basis == manual.n_basis


def test_degree_tuple_length_mismatch_rejected():
    with pytest.raises(ValueError, match="degree has 2 entries"):
        FunctionSpace.piecewise("lagrange", (1, 2), np.linspace(0.0, 1.0, 4))


def test_quad_rule_passthrough_accepts_compatible_rule():
    breakpoints = np.linspace(0.0, 1.0, 4)
    _, _, ref = _normalize_breakpoints(breakpoints)
    rule = composite_quad(gauss_legendre(5), ref)
    sugar = FunctionSpace.piecewise("lagrange", 2, breakpoints, quad_rule=rule)
    assert sugar.quad_rule is rule


def test_quad_rule_passthrough_rejects_incompatible_rule():
    # A plain (non-composite) rule can't integrate a piecewise-smooth basis
    # exactly -- FunctionSpace's own validation catches this through the new
    # constructor path, same as it always has.
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
    sugar = FunctionSpace.piecewise("hermite", 3, breakpoints, continuity=continuity)
    assert sugar.n_basis == manual.n_basis
    np.testing.assert_allclose(
        sugar.basis_matrix().matrix, manual.basis_matrix().matrix
    )


def test_hermite_kind_requires_degree_three():
    with pytest.raises(ValueError, match="degree must be 3"):
        FunctionSpace.piecewise("hermite", 2, np.linspace(0.0, 1.0, 6), continuity=1)


def test_hermite_kind_rejects_nodes():
    with pytest.raises(ValueError, match="nodes is only meaningful"):
        FunctionSpace.piecewise(
            "hermite", 3, np.linspace(0.0, 1.0, 6), nodes="lobatto", continuity=1
        )


def test_hermite_kind_per_element_tuple_degree_validates_each_entry():
    with pytest.raises(ValueError, match="degree must be 3"):
        FunctionSpace.piecewise(
            "hermite", (3, 3, 2), np.linspace(0.0, 1.0, 4), continuity=1
        )
