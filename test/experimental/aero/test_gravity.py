# ruff: noqa: N802, N803, N806

import numpy as np

from archimedes.experimental.aero.gravity import (
    PointGravity,
    lla2eci,
)


def test_gravity_at_origin():
    """Test gravity vector at NED origin points exactly downward."""
    p_EN, R_EN = lla2eci(45.0, -120.0)
    gravity_model = PointGravity(p_EN, R_EN)

    # At origin (0,0,0) in NED frame, gravity should point exactly down
    g_N = gravity_model(np.zeros(3))

    # Expected gravity magnitude at Earth's surface
    g_magnitude = 9.81  # m/s^2

    # Check direction is down (within numerical precision)
    assert abs(g_N[0]) < 1e-10  # North component ≈ 0
    assert abs(g_N[1]) < 1e-10  # East component ≈ 0
    assert abs(g_N[2] - g_magnitude) < 0.05  # Down component ≈ 9.81


def test_gravity_at_altitude():
    """Test gravity decreases with altitude according to inverse square law."""
    p_EN, R_EN = lla2eci(0.0, 0.0)
    gravity_model = PointGravity(p_EN, R_EN)

    # Gravity at origin
    g0 = gravity_model(np.zeros(3))
    g0_magnitude = np.linalg.norm(g0)

    # Gravity at 1000 km altitude (down direction)
    altitude = 1_000_000  # 1000 km in meters
    g_up = gravity_model(np.array([0, 0, -altitude]))
    g_up_magnitude = np.linalg.norm(g_up)

    # Calculate expected ratio using inverse square law
    r0 = 6.378e6  # Earth radius [m]
    r1 = r0 + altitude
    expected_ratio = (r0 / r1) ** 2

    # Check the ratio is correct (within tolerance)
    actual_ratio = g_up_magnitude / g0_magnitude
    assert abs(actual_ratio - expected_ratio) < 1e-3


def test_gravity_direction_at_north():
    """Test gravity points toward Earth center when displaced northward."""
    p_EN, R_EN = lla2eci(0.0, 0.0)
    gravity_model = PointGravity(p_EN, R_EN)

    # Move 1000 km north
    north_displacement = 1_000_000  # meters
    g_N = gravity_model(np.array([north_displacement, 0, 0]))

    # Should have negative north component (pulling back toward equator)
    assert g_N[0] < 0
    # East component should be near zero
    assert abs(g_N[1]) < 1e-10
    # Down component should still be dominant
    assert g_N[2] > 0


def test_gravity_direction_at_east():
    """Test gravity points toward Earth center when displaced eastward."""
    p_EN, R_EN = lla2eci(0.0, 0.0)
    gravity_model = PointGravity(p_EN, R_EN)

    # Move 1000 km east
    east_displacement = 1_000_000  # meters
    g_N = gravity_model(np.array([0, east_displacement, 0]))

    # North component should be near zero
    assert abs(g_N[0]) < 1e-10
    # Should have negative east component (pulling back toward origin)
    assert g_N[1] < 0
    # Down component should still be dominant
    assert g_N[2] > 0


def test_gravity_magnitude():
    """Test the gravity magnitude is approximately 9.81 m/s^2 at surface."""
    p_EN, R_EN = lla2eci(0.0, 0.0)
    gravity_model = PointGravity(p_EN, R_EN)

    g_N = gravity_model(np.zeros(3))
    g_magnitude = np.linalg.norm(g_N)

    assert abs(g_magnitude - 9.81) < 0.05


def test_different_latitudes():
    """Test gravity at different latitudes."""
    # Create models at different latitudes
    p_EN, R_EN = lla2eci(0.0, 0.0)
    equator_model = PointGravity(p_EN, R_EN)
    p_EN, R_EN = lla2eci(90.0, 0.0)
    pole_model = PointGravity(p_EN, R_EN)

    # Get gravity at origin for both
    g_equator = equator_model(np.zeros(3))
    g_pole = pole_model(np.zeros(3))

    # Both should point down in their respective NED frames
    assert abs(g_equator[0]) < 1e-10
    assert abs(g_equator[1]) < 1e-10
    assert g_equator[2] > 0

    assert abs(g_pole[0]) < 1e-10
    assert abs(g_pole[1]) < 1e-10
    assert g_pole[2] > 0


def test_rotation_matrix_properties():
    """Test that R_EN has proper orthogonal properties."""
    p_EN, R_EN = lla2eci(37.7749, -122.4194)  # San Francisco
    gravity_model = PointGravity(p_EN, R_EN)

    # Check orthogonality
    R = gravity_model.R_EN
    identity = np.eye(3)
    assert np.allclose(R.T @ R, identity, atol=1e-10)
    assert np.allclose(R @ R.T, identity, atol=1e-10)

    # Check determinant is 1 (proper rotation)
    assert abs(np.linalg.det(R) - 1.0) < 1e-10
