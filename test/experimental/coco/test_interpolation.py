# ruff: noqa: N802, N803, N806

import numpy as np
import pytest

import archimedes as arc
from archimedes.experimental import coco as cc
from archimedes.experimental.coco.interpolation import (
    LagrangePolynomial,
    gauss_legendre,
    gauss_lobatto,
    gauss_radau,
)


def test_gauss_legendre():
    x, w = gauss_legendre(5)
    assert len(x) == 5
    assert len(w) == 5
    assert np.isclose(np.sum(w), 2.0)


def test_gauss_radau():
    with pytest.raises(ValueError):
        gauss_radau(0)

    x, w = gauss_radau(5)
    assert len(x) == 5
    assert len(w) == 5
    assert x[0] == -1.0
    assert np.isclose(np.sum(w), 2.0)


def test_gauss_lobatto():
    with pytest.raises(ValueError):
        gauss_lobatto(1)

    x, w = gauss_lobatto(5)
    assert len(x) == 5
    assert len(w) == 5
    assert x[0] == -1.0
    assert x[-1] == 1.0
    assert np.isclose(np.sum(w), 2.0)


def test_lagrange_polynomial_interpolate():
    nodes = np.array([-1.0, 0.0, 1.0])
    poly = LagrangePolynomial(nodes)

    y0 = np.array([1.0, 0.0, 1.0])
    x = np.linspace(-1, 1, 11)
    y = arc.compile(poly.interpolate)(y0, x)

    assert np.isclose(y[0], 1.0)
    assert np.isclose(y[5], 0.0)
    assert np.isclose(y[-1], 1.0)


def test_lagrange_polynomial_diff_matrix():
    nodes = np.array([-1.0, 0.0, 1.0])
    poly = LagrangePolynomial(nodes)

    D = poly.diff_matrix
    assert D.shape == (3, 3)
    assert np.isclose(np.sum(D, axis=1), 0.0).all()


# Integrated tests for working with the LagrangePolynomial class
@pytest.mark.parametrize("gauss", [gauss_legendre, gauss_radau, gauss_lobatto])
def test_barycentric(gauss):
    N = 16
    nodes, weights = gauss(N)  # Gaussian quadrature weights

    assert len(nodes) == N
    assert len(weights) == N

    # If the endpoints are not included (i.e. Radau or Legendre),
    # add them for the differentiation matrix.  These points are
    # unweighted because they are not part of the quadrature rule.
    if nodes[0] != -1:  # Legendre only
        nodes = np.insert(nodes, 0, -1.0)
        weights = np.insert(weights, 0, 0.0)
    if nodes[-1] != 1:  # Both Radau and Legendre
        nodes = np.append(nodes, 1)
        weights = np.append(weights, 0.0)

    p = cc.LagrangePolynomial(nodes)

    # Shift [-1, 1] -> [a, b]
    a, b = -2, 5  # Interval
    x = (b - a) / 2 * nodes + (a + b) / 2

    D = (2 / (b - a)) * p.diff_matrix
    w = (b - a) / 2 * weights  # weights on shifted interval

    y = np.cos(x)  # Evaluate the function at the nodes

    # Test differentiation
    assert np.allclose(D @ y, -np.sin(x), atol=1e-8)

    # Test quadrature (more accurate than differentiation)
    I1 = np.dot(w, y)
    I2 = np.sin(b) - np.sin(a)

    assert np.allclose(I1, I2, atol=1e-12)


def test_edge_cases():
    with pytest.raises(ValueError):
        cc.gauss_radau(0)

    cc.gauss_radau(1)
    cc.gauss_radau(2)

    with pytest.raises(ValueError):
        cc.gauss_lobatto(0)
        cc.gauss_lobatto(1)

    cc.gauss_lobatto(2)
    cc.gauss_lobatto(3)


def test_interpolation():
    N = 16
    nodes, _weights = cc.gauss_legendre(N)
    p = cc.LagrangePolynomial(nodes)

    f = np.cos
    y0 = f(nodes)

    x = 0.5 * (nodes[5] + nodes[6])
    y = f(x)
    y_interp = arc.compile(p.interpolate)(y0, x)

    assert np.allclose(y, y_interp, atol=1e-8)

    # Incorrect number of nodes in call to interpolate
    with pytest.raises(ValueError):
        p.interpolate(y0[:-1], x)
