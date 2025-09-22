# ruff: noqa: N802, N803, N806

import numpy as np
import pytest

from archimedes.experimental.aero.atmosphere import (
    StandardAtmosphere1976,
)


class TestStandardAtmosphere1976:
    """Test suite for the StandardAtmosphere1976 class."""

    @pytest.fixture
    def atm(self) -> StandardAtmosphere1976:
        """Create an instance of StandardAtmosphere1976 for testing."""
        return StandardAtmosphere1976()

    def test_sea_level_values(self, atm: StandardAtmosphere1976):
        """Test that sea level values match standard values."""
        # Standard sea-level values
        T0 = 288.15  # K
        P0 = 101325  # Pa

        assert np.isclose(atm.calc_T(0), T0, rtol=1e-5)
        assert np.isclose(atm.calc_p(0), P0, rtol=1e-5)

    def test_temperature_gradient(self, atm: StandardAtmosphere1976):
        """Test temperature calculations at different altitudes."""
        # Test troposphere
        assert np.isclose(atm.calc_T(5000), 255.65, rtol=1e-4)

        # Test tropopause (isothermal layer)
        assert np.isclose(atm.calc_T(11000), 216.65, rtol=1e-4)
        assert np.isclose(atm.calc_T(15000), 216.65, rtol=1e-4)

        # Test stratosphere
        assert np.isclose(atm.calc_T(25000), 221.65, rtol=1e-4)

    def test_pressure_gradient(self, atm: StandardAtmosphere1976):
        """Test pressure calculations at different altitudes."""
        # Test various altitudes
        assert np.isclose(atm.calc_p(5000), 54048.2, rtol=1e-3)
        assert np.isclose(atm.calc_p(11000), 22632.1, rtol=1e-3)
        assert np.isclose(atm.calc_p(20000), 5474.9, rtol=1e-3)

    def test_density_calculation(self, atm: StandardAtmosphere1976):
        """Test density calculations at different altitudes."""
        # Standard values from tables
        altitudes = [0, 5000, 11000, 20000]
        expected_densities = [1.225, 0.7364, 0.3639, 0.0880]

        for alt, exp_density in zip(altitudes, expected_densities):
            # Calculate density from pressure and temperature
            T = atm.calc_T(alt)
            p = atm.calc_p(alt)
            R = 287.05287  # J/(kg·K)
            density = p / (R * T)
            assert np.isclose(density, exp_density, rtol=1e-3)

    def test_mach_and_qbar(self, atm: StandardAtmosphere1976):
        """Test Mach number and dynamic pressure calculations."""
        # Test at sea level with 100 m/s speed
        mach, qbar = atm(100, 0)

        # Expected values
        # Speed of sound at sea level: sqrt(1.4 * 287.05287 * 288.15) = 340.3 m/s
        exp_mach = 100 / 340.3
        exp_qbar = 0.5 * 1.225 * (100**2)

        assert np.isclose(mach, exp_mach, rtol=1e-3)
        assert np.isclose(qbar, exp_qbar, rtol=1e-3)

        # Test at 11000m with 200 m/s speed
        mach, qbar = atm(200, 11000)

        # Speed of sound at 11000m: sqrt(1.4 * 287.05287 * 216.65) = 295.1 m/s
        # Density at 11000m: 0.3639 kg/m³
        exp_mach_high = 200 / 295.1
        exp_qbar_high = 0.5 * 0.3639 * (200**2)

        assert np.isclose(mach, exp_mach_high, rtol=1e-3)
        assert np.isclose(qbar, exp_qbar_high, rtol=1e-3)

    def test_altitude_range(self, atm: StandardAtmosphere1976):
        """Test handling of altitudes outside the valid range."""
        # Test negative altitude (should use sea level values)
        assert np.isclose(atm.calc_T(-100), atm.calc_T(0), rtol=1e-5)
        assert np.isclose(atm.calc_p(-100), atm.calc_p(0), rtol=1e-5)

        # Test very high altitude (model is valid up to 86km)
        assert atm.calc_T(90000) is not None
        assert atm.calc_p(90000) is not None

    def test_array_inputs(self, atm: StandardAtmosphere1976):
        """Test that array inputs work correctly."""
        altitudes = np.array([0, 5000, 11000])
        temperatures = np.array([288.15, 255.65, 216.65])

        calc_temps = np.array([atm.calc_T(alt) for alt in altitudes])
        assert np.allclose(calc_temps, temperatures, rtol=1e-4)

    def test_reference_data(self, atm: StandardAtmosphere1976):
        """Test against official USSA1976 reference data."""
        # Example reference data (altitude in m, temperature in K, pressure in Pa)
        reference_data = [
            # Alt(m),   Temp(K),    Pressure(Pa),  Density(kg/m³)
            [0, 288.15, 101325, 1.225],
            [1000, 281.65, 89876, 1.112],
            [2000, 275.15, 79501, 1.007],
            [5000, 255.65, 54048, 0.7364],
            [10000, 223.15, 26500, 0.4135],
            [11000, 216.65, 22632, 0.3639],
            [15000, 216.65, 12112, 0.1948],
            [20000, 216.65, 5529, 0.0889],
            [25000, 221.65, 2511, 0.0395],
            [30000, 226.65, 1172, 0.0180],
            [40000, 251.05, 277.5, 0.0038],
            [50000, 270.65, 75.95, 0.001],
        ]

        for alt, temp, press, dens in reference_data:
            print(f"Testing altitude: {alt} m")
            assert np.isclose(atm.calc_T(alt), temp, rtol=1e-3)
            assert np.isclose(atm.calc_p(alt), press, rtol=1e-2)

            # Test Mach and qbar at 100 m/s
            mach, qbar = atm(100, alt)
            speed_of_sound = np.sqrt(1.4 * 287.05287 * temp)
            exp_mach = 100 / speed_of_sound
            exp_qbar = 0.5 * dens * (100**2)

            assert np.isclose(mach, exp_mach, rtol=1e-3)
            assert np.isclose(qbar, exp_qbar, rtol=1e-1)
