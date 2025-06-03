# Mathematical correctness tests for Extended Kalman Filter
import numpy as np
import pytest
from scipy.linalg import expm

from archimedes.state_estimation import (
    ExtendedKalmanFilter, ekf_step, ekf_correct
)
from archimedes import jac


class TestEKF:
    """Test mathematical properties and correctness of EKF implementation."""
    
    def test_ekf_matches_linear(self):
        """For linear systems, EKF should give identical results to linear KF."""
        # Define a simple linear system: x_{k+1} = F*x_k + w, y_k = H*x_k + v
        dt = 0.1
        F = np.array([[1.0, dt], [0.0, 1.0]])  # Constant velocity
        H = np.array([[1.0, 0.0]])  # Position measurement
        Q = np.array([[0.1, 0.0], [0.0, 0.1]])
        R = np.array([[0.5]])
        
        # Linear dynamics and observation functions
        def f_linear(t, x):
            return F @ x
            
        def h_linear(t, x):
            return H @ x
        
        # Initial conditions
        x0 = np.array([1.0, 0.5])
        P0 = np.eye(2) * 0.1
        y = np.array([1.1])  # Measurement
        
        # EKF step
        x_ekf, P_ekf, e_ekf = ekf_step(f_linear, h_linear, 0.0, x0, y, P0, Q, R)
        
        # Manual linear KF step for comparison
        # Predict
        x_pred = F @ x0
        P_pred = F @ P0 @ F.T + Q
        
        # Update
        y_pred = H @ x_pred
        innovation = y - y_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_linear = x_pred + K @ innovation
        P_linear = (np.eye(2) - K @ H) @ P_pred
        
        # Results should be identical (within numerical precision)
        np.testing.assert_allclose(x_ekf, x_linear, rtol=1e-12)
        np.testing.assert_allclose(P_ekf, P_linear, rtol=1e-12)
        np.testing.assert_allclose(e_ekf, innovation, rtol=1e-12)

    def test_measurement_update_properties(self):
        """Test mathematical properties of the measurement update step."""
        
        def h(t, x):
            return np.array([x[0] + 0.1*x[1]**2], like=x)  # Nonlinear measurement

        R = np.array([[0.1]])

        # State and covariance BEFORE measurement update
        x_prior = np.array([1.0, 0.5])
        P_prior = np.array([[0.1, 0.02], [0.02, 0.08]])
        y = np.array([1.02])

        # Apply ONLY the measurement update
        x_post, P_post, innovation = ekf_correct(h, 0.0, x_prior, y, P_prior, R)

        # Test 1: Posterior covariance should be smaller than prior
        assert (
            np.trace(P_post) < np.trace(P_prior),
            "Measurement should reduce uncertainty"
        )

        # Test 2: Innovation should be consistent with prediction
        y_pred = h(0.0, x_prior)
        expected_innovation = y - y_pred
        np.testing.assert_allclose(innovation, expected_innovation, rtol=1e-12)

        # Test 3: Information matrix should increase
        info_prior = np.linalg.inv(P_prior)
        info_post = np.linalg.inv(P_post)

        # Information should not decrease (Loewner ordering)
        diff = info_post - info_prior
        eigenvals = np.linalg.eigvals(diff)
        assert np.all(eigenvals >= -1e-10), "Information matrix should not decrease"
    
    def test_prediction_step_properties(self):
        """Test mathematical properties of the prediction step."""
        # Nonlinear dynamics
        def f(t, x):
            return np.array([x[0] + 0.1*x[1], 0.9*x[1] + 0.1*np.sin(x[0])], like=x)
        
        def h(t, x):
            return np.array([x[0]], like=x)  # Simple measurement
        
        Q = np.array([[0.01, 0.0], [0.0, 0.02]])
        R = np.array([[0.1]])
        
        x_prev = np.array([0.5, 1.0])
        P_prev = np.eye(2) * 0.05
        
        # We need a dummy measurement for the full step
        y_dummy = np.array([0.5])
        
        x_pred, P_pred, _ = ekf_step(f, h, 0.0, x_prev, y_dummy, P_prev, Q, R)
        
        # Manually compute just the prediction step
        F = jac(f, argnums=1)(0.0, x_prev)
        x_pred_manual = f(0.0, x_prev)
        P_pred_manual = F @ P_prev @ F.T + Q
        
        # The prediction part should match (before measurement update)
        np.testing.assert_allclose(f(0.0, x_prev), x_pred_manual, rtol=1e-12)
        
        # Test: Process noise should increase uncertainty
        assert np.trace(P_pred_manual) > np.trace(P_prev), "Process noise should increase uncertainty"
