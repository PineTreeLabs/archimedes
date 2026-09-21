import numpy as np

from archimedes.approximation import FourierBasis, FunctionSpace
from archimedes.measure import UnitInterval


def test_fourier():
    manual = FunctionSpace(
        FourierBasis(7, kind="full"), UnitInterval.Parameters(a=-2.0, b=3.0)
    )
    space = FunctionSpace.fourier(7, a=-2.0, b=3.0)
    assert space.n_basis == manual.n_basis
    assert space.domain == manual.domain
    np.testing.assert_allclose(
        space.basis_matrix().matrix, manual.basis_matrix().matrix
    )

    # kind defaults to "full", and dispatches to "cosine"/"sine" otherwise.
    assert FunctionSpace.fourier(5, kind="full") == FunctionSpace.fourier(5)
    assert FunctionSpace.fourier(4, kind="cosine").basis.kind == "cosine"
    assert FunctionSpace.fourier(4, kind="sine").basis.kind == "sine"

    # density is forwarded, and defaults to False.
    assert FunctionSpace.fourier(5, density=True).basis.density is True
    assert FunctionSpace.fourier(5).basis.density is False

    # Default domain is the reference interval.
    space = FunctionSpace.fourier(5)
    assert space.domain == UnitInterval.Parameters(a=-1.0, b=1.0)
