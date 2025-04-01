# ruff: noqa: N802
# ruff: noqa: N803
# ruff: noqa: N806

import numpy as np
import pytest

from archimedes.experimental.coco.discretization import (
    RadauFiniteElements,
    RadauInterval,
)


def test_radau_interval():
    n_nodes = 3
    radau = RadauInterval(n_nodes=n_nodes)

    assert radau.n_nodes == n_nodes
    assert radau.nodes[-1] == 1.0
    assert radau.diff_matrix.shape == (n_nodes, n_nodes + 1)
    assert radau.diff_inv.shape == (n_nodes, n_nodes)

    x = np.array([1, 2, 3, 4])
    t0, tf = 0, 1
    x_fn = radau.create_interpolant(x, t0, tf)
    assert np.allclose(x_fn(t0), x[0])
    assert np.allclose(x_fn(tf), x[-1])


def test_radau_finite_elements():
    N = [3, 4, 5]
    knots = [-0.5, 0.5]
    radau_fe = RadauFiniteElements(N=N, knots=knots)

    assert radau_fe.n_elements == len(N)
    assert radau_fe.n_nodes == sum(N)
    assert len(radau_fe.nodes) == radau_fe.n_nodes + 1  # Includes non-collocated end
    assert len(radau_fe.weights) == radau_fe.n_nodes
    assert radau_fe.diff_matrix.shape == (radau_fe.n_nodes, radau_fe.n_nodes + 1)

    # Invalid number of knots
    with pytest.raises(ValueError):
        RadauFiniteElements(N=[3, 4], knots=[-0.5, 0.5, 0.7])

    # Invalid knot ordering
    with pytest.raises(ValueError):
        RadauFiniteElements(N=[3, 4, 5], knots=[0.5, 0.0])

    # Knots outside of (-1, 1)
    with pytest.raises(ValueError):
        RadauFiniteElements(N=[3, 4, 5], knots=[-1.5, 0.5])

    x = np.random.rand(radau_fe.n_nodes + 1, 2)
    u = np.random.rand(radau_fe.n_nodes, 1)
    t0, tf = 0, 1
    x_fn, u_fn = radau_fe.create_interpolants(x, u, t0, tf)
    t = radau_fe.time_nodes(t0, tf)
    assert np.allclose(x_fn(t), x)
    assert np.allclose(u_fn(t[:-1]), u)

    # Error handling in local_to_global
    with pytest.raises(ValueError):
        radau_fe.local_to_global(1000)


def test_extrapolation():
    N = [3, 4]
    knots = [0]
    radau_fe = RadauFiniteElements(N=N, knots=knots)

    x = np.random.rand(radau_fe.n_nodes + 1, 2)
    u = np.random.rand(radau_fe.n_nodes, 1)
    t0, tf = 0, 1
    x_fn, _u_fn = radau_fe.create_interpolants(x, u, t0, tf, extrapolation="flat")
    assert np.allclose(x_fn(t0 - 1), x[0])
    assert np.allclose(x_fn(tf + 1), x[-1])

    x_fn, _u_fn = radau_fe.create_interpolants(x, u, t0, tf, extrapolation="zero")
    assert np.allclose(x_fn(t0 - 1), 0)
    assert np.allclose(x_fn(tf + 1), 0)

    with pytest.raises(ValueError):
        radau_fe.create_interpolants(x, u, t0, tf, extrapolation="invalid")
