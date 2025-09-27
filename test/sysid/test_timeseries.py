# ruff: noqa: N802, N803, N806, E741

import numpy as np
import pytest

import archimedes as arc
from archimedes.sysid import Timeseries


class TestTimeseries:
    """Unit tests for Timeseries class."""

    def test_valid_construction(self):
        """Test successful construction with valid inputs."""
        # Time vector
        ts = np.array([0.0, 0.1, 0.2, 0.3])
        nt = len(ts)

        # Input signals (2 inputs)
        us = np.random.randn(2, nt)

        # Output measurements (3 outputs)
        ys = np.random.randn(3, nt)

        # Should construct successfully
        timeseries = Timeseries(ts=ts, us=us, ys=ys)

        assert np.array_equal(timeseries.ts, ts)
        assert np.array_equal(timeseries.us, us)
        assert np.array_equal(timeseries.ys, ys)

        # Test indexing
        timeseries_slice = timeseries[:2]
        assert timeseries_slice.ts.shape == (2,)
        assert timeseries_slice.us.shape == (2, 2)
        assert timeseries_slice.ys.shape == (3, 2)

        assert np.array_equal(timeseries_slice.ts, ts[:2])
        assert np.array_equal(timeseries_slice.us, us[:, :2])
        assert np.array_equal(timeseries_slice.ys, ys[:, :2])

    def test_minimal_dimensions(self):
        """Test construction with minimal valid dimensions."""
        # Single time point
        ts = np.array([0.0])

        # Single input, single output
        us = np.array([[1.0]])  # shape (1, 1)
        ys = np.array([[2.0]])  # shape (1, 1)

        timeseries = Timeseries(ts=ts, us=us, ys=ys)

        assert timeseries.ts.shape == (1,)
        assert timeseries.us.shape == (1, 1)
        assert timeseries.ys.shape == (1, 1)

    def test_multiple_dimensions(self):
        """Test construction with multiple inputs/outputs."""
        ts = np.linspace(0, 1, 100)
        nt = len(ts)

        # 5 inputs, 3 outputs
        us = np.random.randn(5, nt)
        ys = np.random.randn(3, nt)

        timeseries = Timeseries(ts=ts, us=us, ys=ys)

        assert timeseries.ts.shape == (100,)
        assert timeseries.us.shape == (5, 100)
        assert timeseries.ys.shape == (3, 100)

    def test_time_vector_not_1d_error(self):
        """Test error when time vector is not 1D."""
        # 2D time vector (invalid)
        ts = np.array([[0.0, 0.1], [0.2, 0.3]])
        us = np.random.randn(2, 4)
        ys = np.random.randn(1, 4)

        with pytest.raises(ValueError, match="Time vector must be one-dimensional"):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_time_vector_0d_error(self):
        """Test error when time vector is 0D scalar."""
        ts = np.array(0.0)  # 0D array
        us = np.random.randn(1, 1)
        ys = np.random.randn(1, 1)

        with pytest.raises(ValueError, match="Time vector must be one-dimensional"):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_time_vector_3d_error(self):
        """Test error when time vector is 3D."""
        ts = np.random.randn(2, 2, 2)  # 3D array
        us = np.random.randn(1, 8)
        ys = np.random.randn(1, 8)

        with pytest.raises(ValueError, match="Time vector must be one-dimensional"):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_outputs_not_2d_error(self):
        """Test error when outputs are not 2D."""
        ts = np.array([0.0, 0.1, 0.2])
        us = np.random.randn(2, 3)

        # 1D outputs (invalid)
        ys = np.array([1.0, 2.0, 3.0])

        with pytest.raises(
            ValueError,
            match="Output measurements must be two-dimensional with shape \\(ny, nt\\)",
        ):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_outputs_0d_error(self):
        """Test error when outputs are 0D."""
        ts = np.array([0.0])
        us = np.random.randn(1, 1)
        ys = np.array(5.0)  # 0D array

        with pytest.raises(
            ValueError,
            match="Output measurements must be two-dimensional with shape \\(ny, nt\\)",
        ):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_outputs_3d_error(self):
        """Test error when outputs are 3D."""
        ts = np.array([0.0, 0.1])
        us = np.random.randn(1, 2)
        ys = np.random.randn(2, 2, 2)  # 3D array

        with pytest.raises(
            ValueError,
            match="Output measurements must be two-dimensional with shape \\(ny, nt\\)",
        ):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_inputs_not_2d_error(self):
        """Test error when inputs are not 2D."""
        ts = np.array([0.0, 0.1, 0.2])
        ys = np.random.randn(1, 3)

        # 1D inputs (invalid)
        us = np.array([1.0, 2.0, 3.0])

        with pytest.raises(
            ValueError,
            match="Input signals must be two-dimensional with shape \\(nu, nt\\)",
        ):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_inputs_0d_error(self):
        """Test error when inputs are 0D."""
        ts = np.array([0.0])
        ys = np.random.randn(1, 1)
        us = np.array(3.0)  # 0D array

        with pytest.raises(
            ValueError,
            match="Input signals must be two-dimensional with shape \\(nu, nt\\)",
        ):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_inputs_3d_error(self):
        """Test error when inputs are 3D."""
        ts = np.array([0.0, 0.1])
        ys = np.random.randn(1, 2)
        us = np.random.randn(2, 2, 2)  # 3D array

        with pytest.raises(
            ValueError,
            match="Input signals must be two-dimensional with shape \\(nu, nt\\)",
        ):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_time_outputs_size_mismatch_error(self):
        """Test error when time vector size doesn't match outputs time dimension."""
        ts = np.array([0.0, 0.1, 0.2])  # 3 time points
        us = np.random.randn(2, 3)  # 3 time points (correct)
        ys = np.random.randn(1, 4)  # 4 time points (incorrect)

        with pytest.raises(
            ValueError,
            match="Time vector size must match the number of time points in ys",
        ):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_time_inputs_size_mismatch_error(self):
        """Test error when time vector size doesn't match inputs time dimension."""
        ts = np.array([0.0, 0.1, 0.2])  # 3 time points
        ys = np.random.randn(1, 3)  # 3 time points (correct)
        us = np.random.randn(2, 5)  # 5 time points (incorrect)

        with pytest.raises(
            ValueError,
            match="Time vector size must match the number of time points in us",
        ):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_inputs_outputs_size_mismatch_error(self):
        """Test error when inputs and outputs have different time dimensions."""
        ts = np.array([0.0, 0.1, 0.2])  # 3 time points
        us = np.random.randn(2, 3)  # 3 time points
        ys = np.random.randn(1, 2)  # 2 time points (mismatch)

        # Should fail on outputs check first
        with pytest.raises(
            ValueError,
            match="Time vector size must match the number of time points in ys",
        ):
            Timeseries(ts=ts, us=us, ys=ys)

    def test_tree_functionality(self):
        """Test that Timeseries works with tree operations."""
        ts = np.array([0.0, 0.1, 0.2])
        us = np.random.randn(2, 3)
        ys = np.random.randn(1, 3)

        timeseries = Timeseries(ts=ts, us=us, ys=ys)

        # Test flattening
        flat, _ = arc.tree.flatten(timeseries)
        assert len(flat) == 3  # ts, us, ys

        # Test mapping
        scaled = arc.tree.map(lambda x: x * 2, timeseries)
        assert np.allclose(scaled.ts, ts * 2)
        assert np.allclose(scaled.us, us * 2)
        assert np.allclose(scaled.ys, ys * 2)

    def test_replace_method(self):
        """Test the replace method for creating modified copies."""
        ts = np.array([0.0, 0.1, 0.2])
        us = np.random.randn(2, 3)
        ys = np.random.randn(1, 3)

        timeseries = Timeseries(ts=ts, us=us, ys=ys)

        # Replace time vector
        new_ts = np.array([1.0, 1.1, 1.2])
        new_timeseries = timeseries.replace(ts=new_ts)

        assert np.array_equal(new_timeseries.ts, new_ts)
        assert np.array_equal(new_timeseries.us, us)  # unchanged
        assert np.array_equal(new_timeseries.ys, ys)  # unchanged

        # Original should be unchanged (frozen dataclass)
        assert np.array_equal(timeseries.ts, ts)

    def test_replace_with_validation(self):
        """Test that replace method still validates dimensions."""
        ts = np.array([0.0, 0.1, 0.2])
        us = np.random.randn(2, 3)
        ys = np.random.randn(1, 3)

        timeseries = Timeseries(ts=ts, us=us, ys=ys)

        # Replace with incompatible dimensions should fail
        bad_ts = np.array([1.0, 1.1])  # Wrong size

        with pytest.raises(ValueError, match="Time vector size must match"):
            timeseries.replace(ts=bad_ts)

    def test_immutability(self):
        """Test that Timeseries instances are immutable (frozen)."""
        ts = np.array([0.0, 0.1, 0.2])
        us = np.random.randn(2, 3)
        ys = np.random.randn(1, 3)

        timeseries = Timeseries(ts=ts, us=us, ys=ys)

        # Should not be able to modify fields directly
        with pytest.raises(AttributeError):
            timeseries.ts = np.array([1.0, 1.1, 1.2])

    def test_edge_case_empty_time(self):
        """Test behavior with empty time vectors."""
        ts = np.array([])
        us = np.random.randn(1, 0)  # 0 time points
        ys = np.random.randn(1, 0)  # 0 time points

        # Should construct successfully
        timeseries = Timeseries(ts=ts, us=us, ys=ys)
        assert timeseries.ts.size == 0
        assert timeseries.us.shape == (1, 0)
        assert timeseries.ys.shape == (1, 0)

    def test_realistic_system_id_data(self):
        """Test with realistic system identification data dimensions."""
        # Common scenario: 10 seconds at 100 Hz sampling
        dt = 0.01
        t_end = 10.0
        ts = np.arange(0, t_end, dt)
        nt = len(ts)

        # CartPole system: 1 input (force), 4 outputs (x, θ, ẋ, θ̇)
        nu, ny = 1, 4
        us = np.random.randn(nu, nt)
        ys = np.random.randn(ny, nt)

        timeseries = Timeseries(ts=ts, us=us, ys=ys)

        assert timeseries.ts.shape == (nt,)
        assert timeseries.us.shape == (nu, nt)
        assert timeseries.ys.shape == (ny, nt)
        assert len(ts) == 1000  # 10s / 0.01s
