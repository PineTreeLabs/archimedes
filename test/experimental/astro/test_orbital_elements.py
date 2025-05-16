# ruff: noqa: N802, N803, N806

import numpy as np

import archimedes as arc
from archimedes.experimental.astro.orbital_elements import (
    KeplerElements,
    cartesian_to_kepler,
    kepler_to_cartesian,
)

# NOTE: All test data generated using MATLAB `kepler2ijk` and similar functions


# Compile this to check that symbolic/numeric evaluation works
@arc.compile
def round_trip(oe):
    cartesian = kepler_to_cartesian(oe)
    oe2 = cartesian_to_kepler(cartesian)
    return cartesian, oe2


# Helper function for testing Kepler -> Cartesian and back
def _test_round_trip(oe, r_ex, v_ex, rtol=1e-3):
    """
    Test the round trip conversion from Kepler elements to Cartesian and back.

    Args:
        oe: KeplerElements instance
        r_ex: Expected position vector
        v_ex: Expected velocity vector
        rtol: Relative tolerance for comparisons
    """
    # Convert Kepler elements to Cartesian and back
    cartesian, oe2 = round_trip(oe)
    r, v = cartesian.r, cartesian.v

    # Check if the Cartesian state matches the expected values
    assert np.allclose(r, r_ex, rtol=rtol)
    assert np.allclose(v, v_ex, rtol=rtol)

    # Check if the original elements match the converted ones
    assert np.isclose(oe.a, oe2.a, rtol=rtol)
    assert np.isclose(oe.e, oe2.e, rtol=rtol)
    assert np.isclose(oe.i, oe2.i, rtol=rtol)
    assert np.isclose(oe.omega, oe2.omega, rtol=rtol)
    assert np.isclose(oe.RAAN, oe2.RAAN, rtol=rtol)
    assert np.isclose(oe.nu, oe2.nu, rtol=rtol)


class TestKeplerElements:
    """Test suite for Keplerian elements conversion."""

    def test_circular_orbit(self):
        """Test conversion for a circular orbit."""
        # Create Kepler elements (a, e, i, omega, RAAN, nu)
        oe = KeplerElements(7000e3, 0.0, 0.0, 0.0, 0.0, np.deg2rad(45.0))

        # Expected position and velocity vectors
        r_ex = np.array([4949.7e3, 4949.7e3, 0.0])
        v_ex = np.array([-5.3359e3, 5.3359e3, 0.0])

        _test_round_trip(oe, r_ex, v_ex)

    def test_elliptical_orbit(self):
        """Test conversion for an elliptical orbit."""
        # KeplerElements(a, e, i, omega, RAAN, nu)
        oe = KeplerElements(
            10.0e6,
            0.5,
            np.deg2rad(30.0),
            np.deg2rad(90.0),
            np.deg2rad(60.0),
            np.deg2rad(180.0),
        )

        r_ex = np.array([11250.0e3, -6495.0e3, -7500.0e3])
        v_ex = np.array([1.8225e3, 3.1567e3, 0.0])

        _test_round_trip(oe, r_ex, v_ex)

        # Check conditional offset when Ω > 180 degrees
        oe = KeplerElements(
            10000.0e3,
            0.5,
            np.deg2rad(30.0),
            np.deg2rad(90.0),
            np.deg2rad(210.0),
            np.deg2rad(180.0),
        )

        r_ex = np.array([-6495.0e3, 11250.0e3, -7500.0e3])
        v_ex = np.array([-3.1567e3, -1.8225e3, 0.0])

        _test_round_trip(oe, r_ex, v_ex)

        # Check conditional offset when ω > 180 degrees
        oe = KeplerElements(
            10000.0e3,
            0.5,
            np.deg2rad(30.0),
            np.deg2rad(210.0),
            np.deg2rad(60.0),
            np.deg2rad(180.0),
        )

        r_ex = np.array([870.0e3, 14498.0e3, 3750.0e3])
        v_ex = np.array([-3.2788e3, -0.2115e3, 1.5784e3])

        _test_round_trip(oe, r_ex, v_ex)

        # Check conditional offset when ν > 180 degrees
        oe = KeplerElements(
            10000.0e3,
            0.5,
            np.deg2rad(30.0),
            np.deg2rad(210.0),
            np.deg2rad(60.0),
            np.deg2rad(200.0),
        )

        r_ex = np.array([-3581.0e3, 12568.0e3, 5419.0e3])
        v_ex = np.array([-3.0280e3, -2.5958e3, 0.7646e3])

        _test_round_trip(oe, r_ex, v_ex)

    def test_circular_polar_orbit(self):
        """Test conversion for a circular polar orbit."""
        oe = KeplerElements(8e6, 0.0, np.pi / 2, 0.0, 0.0, np.deg2rad(270.0))

        r_ex = np.array([0.0, 0.0, -8e6])
        v_ex = np.array([7.0587e3, 0.0, 0.0])

        _test_round_trip(oe, r_ex, v_ex)

        # Test conditional offset when arglat < 180 degrees
        oe = KeplerElements(8e6, 0.0, np.pi / 2, 0.0, 0.0, np.deg2rad(170.0))

        r_ex = np.array([-7.8785e6, 0.0, 1.3892e6])
        v_ex = np.array([-1.2257e3, 0.0, -6.9514e3])

        _test_round_trip(oe, r_ex, v_ex)

    def test_elliptical_equatorial_orbit(self):
        """Test conversion for an elliptical equatorial orbit."""
        oe = KeplerElements(12e6, 0.7, 0.0, np.deg2rad(45.0), 0.0, np.deg2rad(135.0))

        r_ex = np.array([-12.118e6, 0.0, 0.0])
        v_ex = np.array([-3.9946e3, -4.0757e3, 0.0])

        _test_round_trip(oe, r_ex, v_ex)

    def test_parabolic_orbit(self):
        """Test conversion for a parabolic orbit."""
        oe = KeplerElements(
            7e6,
            1.0,
            np.deg2rad(45.0),
            np.deg2rad(60.0),
            np.deg2rad(120.0),
            np.deg2rad(90.0),
        )

        r_ex = np.array([1.776e6, -12.975e6, 4.950e6])
        v_ex = np.array([4.8405e3, -5.6219e3, -1.3810e3])

        _test_round_trip(oe, r_ex, v_ex)

    def test_hyperbolic_orbit(self):
        """Test conversion for a hyperbolic orbit."""
        oe = KeplerElements(
            -15e6,
            1.5,
            np.deg2rad(60.0),
            np.deg2rad(30.0),
            np.deg2rad(300.0),
            np.deg2rad(120.0),
        )

        r_ex = np.array([-16.238e6, 65.625e6, 32.476e6])
        v_ex = np.array([-2.0172e3, 5.4904e3, 1.7290e3])

        _test_round_trip(oe, r_ex, v_ex)
