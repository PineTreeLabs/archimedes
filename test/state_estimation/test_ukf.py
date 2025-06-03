# Mathematical correctness tests for Unscented Kalman Filter
import numpy as np
import pytest
from scipy.linalg import sqrtm

from archimedes.experimental.state_estimation import UnscentedKalmanFilter, ukf_step


class TestUKFMathematicalCorrectness:
    """Test mathematical properties and correctness of UKF implementation."""
    
    def test_ukf_matches_linear(self):
        """For linear systems, UKF should give very similar results to linear KF."""
        # Define a simple linear system
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
        
        # UKF step
        x_ukf, P_ukf, e_ukf = ukf_step(f_linear, h_linear, 0.0, x0, y, P0, Q, R)
        
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
        
        # Results should be very close (UKF should be exact for linear systems)
        np.testing.assert_allclose(x_ukf, x_linear, atol=1e-3)
        np.testing.assert_allclose(P_ukf, P_linear, atol=1e-3)
        np.testing.assert_allclose(e_ukf, innovation, atol=1e-3)
    
    def test_sigma_point_generation(self):
        """Test that sigma points have correct statistical properties."""
        # Test parameters
        n = 3  # State dimension
        x_mean = np.array([1.0, -0.5, 2.0])
        P = np.array([[0.1, 0.02, 0.0],
                     [0.02, 0.08, -0.01],
                     [0.0, -0.01, 0.05]])
        
        # Generate sigma points (mimicking UKF internal calculation)
        L = len(x_mean)
        n_sigma = 2 * L + 1
        
        # Compute square root of covariance
        A = np.linalg.cholesky(P)
        
        # Generate sigma points
        sigma_points = [x_mean]  # Central point
        
        for j in range(L):
            sigma_points.append(x_mean + np.sqrt(L) * A[:, j])
            
        for j in range(L):
            sigma_points.append(x_mean - np.sqrt(L) * A[:, j])
        
        sigma_points = np.array(sigma_points)
        
        # Uniform weights (as used in the UKF implementation)
        w = np.ones(n_sigma) / n_sigma
        
        # Test 1: Weighted mean should equal original mean
        computed_mean = np.sum(w[:, np.newaxis] * sigma_points, axis=0)
        np.testing.assert_allclose(computed_mean, x_mean, rtol=1e-12)
        
        # Test 2: Weighted covariance should equal original covariance
        deviations = sigma_points - x_mean[np.newaxis, :]
        computed_cov = np.sum(w[:, np.newaxis, np.newaxis] * 
                             deviations[:, :, np.newaxis] * 
                             deviations[:, np.newaxis, :], axis=0)
        np.testing.assert_allclose(computed_cov, P, rtol=1e-12)
    
    def test_unscented_transform(self):
        """Test that unscented transform is exact for polynomials up to 2nd order."""
        # Define test mean and covariance
        x_mean = np.array([1.0, -0.5])
        P = np.array([[0.1, 0.02], [0.02, 0.08]])
        
        # Test polynomial functions
        def linear_func(x):
            """Linear function: should be exact"""
            return np.array([2*x[0] + 3*x[1], -x[0] + 4*x[1]])
        
        def quadratic_func(x):
            """Quadratic function: should be exact for unscented transform"""
            return np.array([x[0]**2, x[1]**2, x[0]*x[1]])
        
        # Generate sigma points
        L = len(x_mean)
        A = np.linalg.cholesky(P)
        sigma_points = [x_mean]
        
        for j in range(L):
            sigma_points.append(x_mean + np.sqrt(L) * A[:, j])
            sigma_points.append(x_mean - np.sqrt(L) * A[:, j])
        
        w = np.ones(2*L + 1) / (2*L + 1)
        
        # Test linear function
        y_points_linear = np.array([linear_func(sp) for sp in sigma_points])
        y_mean_ut = np.sum(w[:, np.newaxis] * y_points_linear, axis=0)
        
        # Analytical result for linear function
        A_matrix = np.array([[2, 3], [-1, 4]])
        y_mean_analytical = A_matrix @ x_mean
        
        np.testing.assert_allclose(y_mean_ut, y_mean_analytical, rtol=1e-12)
        
        # Test quadratic function - this tests the core UKF property
        y_points_quad = np.array([quadratic_func(sp) for sp in sigma_points])
        y_mean_ut_quad = np.sum(w[:, np.newaxis] * y_points_quad, axis=0)
        
        # Analytical moments for quadratic function
        # E[X²] = Var[X] + E[X]²
        # E[XY] = Cov[X,Y] + E[X]E[Y]
        expected_quad = np.array([
            P[0,0] + x_mean[0]**2,  # E[x₁²]
            P[1,1] + x_mean[1]**2,  # E[x₂²]
            P[0,1] + x_mean[0]*x_mean[1]  # E[x₁x₂]
        ])
        
        np.testing.assert_allclose(y_mean_ut_quad, expected_quad, rtol=1e-10)

    def test_measurement_update_information_increase(self):
        """Test that measurement updates increase information (decrease uncertainty)."""
        def f(t, x):
            return x  # Identity dynamics for simplicity
        
        def h(t, x):
            return np.array([x[0] + 0.1*x[1]**2, x[1]])  # Nonlinear measurement
        
        Q = np.eye(2) * 0.001  # Small process noise
        R = np.eye(2) * 0.1
        
        x_prior = np.array([1.0, 0.5])
        P_prior = np.array([[0.2, 0.05], [0.05, 0.15]])
        y = np.array([1.02, 0.48])
        
        x_post, P_post, _ = ukf_step(f, h, 0.0, x_prior, y, P_prior, Q, R)
        
        # Test: Posterior uncertainty should be smaller
        trace_prior = np.trace(P_prior + Q)  # Include process noise
        trace_post = np.trace(P_post)
        
        assert trace_post < trace_prior, "Measurement should reduce uncertainty"
        
        # Test: Determinant should decrease (volume of uncertainty ellipsoid)
        det_prior = np.linalg.det(P_prior + Q)
        det_post = np.linalg.det(P_post)
        
        assert det_post < det_prior, "Measurement should reduce uncertainty volume"
    
    def test_ukf_symmetry(self):
        """Test that UKF respects symmetry properties of the problem."""
        # Symmetric system
        def f_symmetric(t, x):
            return np.array([0.9*x[0], 0.9*x[1]])  # Symmetric dynamics
        
        def h_symmetric(t, x):
            return np.array([x[0]**2 + x[1]**2])  # Rotationally symmetric measurement
        
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        
        # Symmetric initial conditions
        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 1.0])  # 90-degree rotation
        P = np.eye(2) * 0.1  # Symmetric covariance
        
        y1 = np.array([1.0])  # Same measurement for both (due to symmetry)
        y2 = np.array([1.0])
        
        # UKF steps
        x1_post, P1_post, _ = ukf_step(f_symmetric, h_symmetric, 0.0, x1, y1, P, Q, R)
        x2_post, P2_post, _ = ukf_step(f_symmetric, h_symmetric, 0.0, x2, y2, P, Q, R)
        
        # Test: Posterior covariances should be identical (up to rotation)
        # For this symmetric case, they should actually be identical
        np.testing.assert_allclose(P1_post, P2_post, rtol=1e-6)
        
        # Test: Posterior states should have same magnitude
        assert abs(np.linalg.norm(x1_post) - np.linalg.norm(x2_post)) < 1e-6
    
    def test_degenerate_covariance_handling(self):
        """Test UKF behavior with near-singular covariance matrices."""
        def f(t, x):
            return np.array([0.99*x[0], 0.99*x[1]])
        
        def h(t, x):
            return np.array([x[0]])  # Only observe first state
        
        Q = np.eye(2) * 1e-6  # Very small process noise
        R = np.array([[0.01]])
        
        # Start with well-conditioned covariance
        x = np.array([1.0, 1.0])
        P = np.eye(2) * 0.1
        
        # Run many steps with only partial observations
        # This should make P become nearly singular in unobserved direction
        for i in range(50):
            y = np.array([1.0 + 0.01*np.random.randn()])
            x, P, _ = ukf_step(f, h, i*0.1, x, y, P, Q, R)
            
            # Should not crash even if P becomes ill-conditioned
            assert np.all(np.isfinite(x)), f"State became non-finite at step {i}"
            assert np.all(np.isfinite(P)), f"Covariance became non-finite at step {i}"
            
            # Eigenvalues should remain non-negative
            eigenvals = np.linalg.eigvals(P)
            assert np.all(eigenvals >= -1e-12), f"Negative eigenvalues at step {i}: {eigenvals}"
