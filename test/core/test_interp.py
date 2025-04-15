"""Test N-dimensional interpolation"""

# ruff: noqa: N802
# ruff: noqa: N803
# ruff: noqa: N806

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

import archimedes as arc
from archimedes._core import SymbolicArray, compile, sym


class TestNumPyInterp:
    def test_interp1d(self):
        xp = np.linspace(1, 6, 6)
        fp = np.array([-1, -1, -2, -3, 0, 2])

        @compile(kind="MX")
        def f(x):
            return np.interp(x, xp, fp)

        # Test evaluating symbolically (note must be "MX")
        x = sym("x", kind="MX")
        y = np.interp(x, xp, fp)
        assert isinstance(y, SymbolicArray)
        assert y.shape == ()
        assert y.dtype == x.dtype

        # Test evaluating numerically
        x = 2.5
        assert np.isclose(f(x), np.interp(x, xp, fp))

        # Above and below the range
        assert f(0) == -1
        assert f(7) == 2

        # Test array evaluation
        x = np.array([2.5, 3.5])
        assert np.allclose(f(x), np.interp(x, xp, fp))

    def test_error_handling(self):
        xp = np.linspace(1, 6, 6)
        fp = np.array([-1, -1, -2, -3, 0, 2])

        # Test with SX (should through an error)
        with pytest.raises(ValueError, match=r".*MX.*"):
            x = sym("x", kind="SX")
            np.interp(x, xp, fp)

        x = sym("x", kind="MX")
        with pytest.raises(ValueError, match=r"xp must be 1-dimensional.*"):
            np.interp(x, xp[:, None], fp)

        with pytest.raises(ValueError, match=r"fp must be 1-dimensional.*"):
            np.interp(x, xp, fp[:, None])

        with pytest.raises(ValueError, match=r".*must be NumPy arrays.*"):
            np.interp(x, arc.sym_like(xp, "xp"), fp)

        with pytest.raises(ValueError, match=r".*must be scalars.*"):
            np.interp(x, xp, fp, left=xp)


class TestNDInterp:
    def test_interp1d(self):
        xp = np.linspace(1, 6, 6)
        fp = np.array([-1, -1, -2, -3, 0, 2])

        f_lut = arc.interpolant([xp], fp, arg_names=("x",), ret_name="y", name="f")

        # Check names
        assert f_lut.arg_names == [
            "x",
        ]
        assert f_lut.return_names == [
            "y",
        ]
        assert f_lut.name == "f"

        # Test evaluating symbolically (note must be "MX")
        x = sym("x", kind="MX")
        y = f_lut(x)
        assert isinstance(y, SymbolicArray)
        assert y.shape == ()
        assert y.dtype == x.dtype

        # Test evaluating numerically
        x = 2.5
        assert np.isclose(f_lut(x), np.interp(x, xp, fp))

        # Test array evaluation
        x = np.array([2.5, 3.5, 4.5])
        y = f_lut(x)
        assert y.shape == (3,)
        assert np.allclose(y, np.interp(x, xp, fp))

    def test_interp2d(self):
        # https://web.casadi.org/docs/#using-lookup-tables
        xgrid = np.linspace(-5, 5, 11)
        ygrid = np.linspace(-4, 4, 9)
        X, Y = np.meshgrid(xgrid, ygrid, indexing="ij")
        R = np.sqrt(5 * X**2 + Y**2) + 1
        Z = np.sin(R) / R
        lut = arc.interpolant(
            [xgrid, ygrid],
            Z,
            arg_names=("x", "y"),
            ret_name="z",
            name="f",
            method="bspline",
        )

        # Check names
        assert lut.arg_names == ["x", "y"]
        assert lut.return_names == [
            "z",
        ]
        assert lut.name == "f"

        # Test numeric scalar evaluation
        x, y = 0.5, 1
        z_arc = lut(x, y)
        interp = RectBivariateSpline(xgrid, ygrid, Z)
        z_scipy = interp.ev(x, y)
        assert np.allclose(z_arc, z_scipy)

        # Test symbolic array evaluation
        n = 3
        x = sym("x", shape=(n,), kind="MX")
        y = sym("y", shape=(n,), kind="MX")
        z = lut(x, y)
        assert z.shape == (n,)
        assert z.dtype == x.dtype

        # Test numeric array evaluation
        x, y = [0.5, 0.75], [1.0, 1.25]
        z_arc = lut(x, y)
        interp = RectBivariateSpline(xgrid, ygrid, Z)
        z_scipy = interp.ev(x, y)
        assert np.allclose(z_arc, z_scipy)

    def test_default_args(self):
        # https://web.casadi.org/docs/#using-lookup-tables
        xgrid = np.linspace(-5, 5, 11)
        ygrid = np.linspace(-4, 4, 9)
        X, Y = np.meshgrid(xgrid, ygrid, indexing="ij")
        R = np.sqrt(5 * X**2 + Y**2) + 1
        Z = np.sin(R) / R
        arc.interpolant(
            [xgrid, ygrid],
            Z,
        )

    def test_error_handling(self):
        xgrid = np.linspace(-5, 5, 11)
        ygrid = np.linspace(-4, 4, 9)
        X, Y = np.meshgrid(xgrid, ygrid, indexing="ij")
        R = np.sqrt(5 * X**2 + Y**2) + 1
        Z = np.sin(R) / R
        lut = arc.interpolant(
            [xgrid, ygrid],
            Z,
            arg_names=("x", "y"),
            ret_name="z",
            name="f",
            method="bspline",
        )

        # Wrong number of arguments
        with pytest.raises(TypeError, match=r".*too many positional arguments.*"):
            x = 0.5
            lut(x, x, x)

        # Too many dimensions
        with pytest.raises(ValueError, match=r".*0- or 1-dimensional.*"):
            x = np.ones((2, 2))
            lut(x, 1.0)

        # Inconsistent lengths of point arrays
        with pytest.raises(ValueError, match=r".*must have the same length.*"):
            lut(np.array([0.5, 0.75, 1.0]), np.array([1.0, 1.5]))

        # Too many dimensions for grid points
        with pytest.raises(ValueError, match=r"grid\[1\] must be 1-dimensional.*"):
            lut = arc.interpolant(
                [xgrid, ygrid[:, None]],
                Z,
            )

        # Too many dimensions for data array
        with pytest.raises(ValueError, match=r"data must be 2-dimensional.*"):
            lut = arc.interpolant(
                [xgrid, ygrid],
                Z[:, None],
            )

        # Not enough data points
        with pytest.raises(ValueError, match=r"data must have length.*"):
            lut = arc.interpolant(
                [xgrid, ygrid],
                Z[:4],
            )

        # Invalid method
        with pytest.raises(ValueError, match=r"method must be one of.*"):
            lut = arc.interpolant(
                [xgrid, ygrid],
                Z,
                method="something else",
            )

        # Not enough arg names
        with pytest.raises(ValueError, match=r"arg_names must have length 2.*"):
            lut = arc.interpolant(
                [xgrid, ygrid],
                Z,
                arg_names=["x"],
            )

        # Non-string arg names
        with pytest.raises(ValueError, match=r"arg_names must be a list of strings.*"):
            lut = arc.interpolant(
                [xgrid, ygrid],
                Z,
                arg_names=[0, 1],
            )

        # Non-string return names
        with pytest.raises(ValueError, match=r"ret_name must be a string.*"):
            lut = arc.interpolant(
                [xgrid, ygrid],
                Z,
                ret_name=0,
            )
