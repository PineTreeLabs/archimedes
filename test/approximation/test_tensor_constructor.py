import numpy as np
import pytest

from archimedes.approximation import FunctionSpace
from archimedes.approximation._basis._tensor import (
    ProductParameters,
    TensorBasis,
)


def test_tensor():
    space_1 = FunctionSpace.legendre(4, a=-1.0, b=1.0)
    space_2 = FunctionSpace.hermite(3, loc=0.0, scale=2.0, kind="prob")
    space = FunctionSpace.tensor(space_1, space_2)
    manual = FunctionSpace(
        TensorBasis((space_1.basis, space_2.basis)),
        ProductParameters(dims=(space_1.domain, space_2.domain)),
    )
    assert space.basis == manual.basis
    assert space.domain == manual.domain
    np.testing.assert_allclose(
        space.basis_matrix().matrix, manual.basis_matrix().matrix
    )

    # n_basis is the product of the factors' sizes, even for a single
    # (degenerate) factor.
    space = FunctionSpace.tensor(
        FunctionSpace.legendre(3), FunctionSpace.legendre(4), FunctionSpace.legendre(5)
    )
    assert space.n_basis == 3 * 4 * 5
    assert FunctionSpace.tensor(space_1).basis == TensorBasis((space_1.basis,))

    with pytest.raises(ValueError, match="at least one space"):
        FunctionSpace.tensor()

    already_tensor = FunctionSpace.tensor(
        FunctionSpace.legendre(3), FunctionSpace.legendre(3)
    )
    with pytest.raises(ValueError, match="must be univariate"):
        FunctionSpace.tensor(already_tensor, FunctionSpace.legendre(3))

    # project() matches the manual construction for a representative
    # function.
    space_1 = FunctionSpace.hermite(4, loc=0.0, scale=1.0, kind="prob", density=True)
    space_2 = FunctionSpace.hermite(4, loc=0.0, scale=2.0, kind="prob", density=True)
    space = FunctionSpace.tensor(space_1, space_2)
    manual = FunctionSpace(
        TensorBasis((space_1.basis, space_2.basis)),
        ProductParameters(dims=(space_1.domain, space_2.domain)),
    )

    def f(x):
        x1, x2 = x[:, 0], x[:, 1]
        return x1**2 + x2

    np.testing.assert_allclose(
        space.project(f).coefficients, manual.project(f).coefficients
    )
