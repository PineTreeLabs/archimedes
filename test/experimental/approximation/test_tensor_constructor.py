"""Tests for the ``FunctionSpace.tensor`` classmethod constructor: thin
sugar over ``FunctionSpace(TensorBasis(bases), ProductParameters(dims))``,
combining independent univariate spaces into one multivariate space.
"""

import numpy as np
import pytest

from archimedes.experimental.approximation import FunctionSpace
from archimedes.experimental.approximation._basis._tensor import (
    ProductParameters,
    TensorBasis,
)
from archimedes.quadrature import gauss_legendre, tensor_quad


def test_matches_manual_tensor_basis_construction():
    space_1 = FunctionSpace.legendre(4, a=-1.0, b=1.0)
    space_2 = FunctionSpace.hermite(3, loc=0.0, scale=2.0, kind="prob")

    sugar = FunctionSpace.tensor(space_1, space_2)
    manual = FunctionSpace(
        TensorBasis((space_1.basis, space_2.basis)),
        ProductParameters(dims=(space_1.domain, space_2.domain)),
    )

    assert sugar.basis == manual.basis
    assert sugar.domain == manual.domain
    np.testing.assert_allclose(
        sugar.basis_matrix().matrix, manual.basis_matrix().matrix
    )


def test_n_basis_is_the_product_of_the_factors():
    space = FunctionSpace.tensor(
        FunctionSpace.legendre(3), FunctionSpace.legendre(4), FunctionSpace.legendre(5)
    )
    assert space.n_basis == 3 * 4 * 5


def test_single_space_is_a_degenerate_tensor_product():
    space_1 = FunctionSpace.legendre(4)
    space = FunctionSpace.tensor(space_1)
    assert space.basis == TensorBasis((space_1.basis,))


def test_rejects_no_spaces():
    with pytest.raises(ValueError, match="at least one space"):
        FunctionSpace.tensor()


def test_rejects_a_multivariate_factor():
    already_tensor = FunctionSpace.tensor(
        FunctionSpace.legendre(3), FunctionSpace.legendre(3)
    )
    with pytest.raises(ValueError, match="must be univariate"):
        FunctionSpace.tensor(already_tensor, FunctionSpace.legendre(3))


def test_quad_rule_forwarded():
    rule = gauss_legendre(10)
    tensor_rule = tensor_quad(rule, rule)
    space = FunctionSpace.tensor(
        FunctionSpace.legendre(5),
        FunctionSpace.legendre(5),
        quad_rule=tensor_rule,
    )
    assert space.quad_rule is tensor_rule


def test_project_matches_manual_construction_for_a_separable_function():
    space_1 = FunctionSpace.hermite(4, loc=0.0, scale=1.0, kind="prob", density=True)
    space_2 = FunctionSpace.hermite(4, loc=0.0, scale=2.0, kind="prob", density=True)

    sugar = FunctionSpace.tensor(space_1, space_2)
    manual = FunctionSpace(
        TensorBasis((space_1.basis, space_2.basis)),
        ProductParameters(dims=(space_1.domain, space_2.domain)),
    )

    def f(x):
        x1, x2 = x[:, 0], x[:, 1]
        return x1**2 + x2

    f_sugar = sugar.project(f)
    f_manual = manual.project(f)
    np.testing.assert_allclose(f_sugar.coefficients, f_manual.coefficients)
