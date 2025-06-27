# ruff: noqa: N802, N803, N806, E741

# Mathematical correctness tests for Extended Kalman Filter
import numpy as np
import pytest
from scipy.linalg import cholesky

from archimedes.observers import (
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
)


def get_kf(filter_name):
    return {
        "ekf": ExtendedKalmanFilter,
        "ukf": UnscentedKalmanFilter,
    }[filter_name]


@pytest.mark.parametrize("filter_name", ["ekf", "ukf"])
def test_linear_consistency(filter_name):
    """For linear systems, nonlinear filters should match linear KF."""
    KalmanFilter = get_kf(filter_name)

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

    kf = KalmanFilter(f_linear, h_linear, Q, R)

    # KF step
    x_kf, P_kf, e_kf = kf.step(0.0, x0, y, P0)

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
    np.testing.assert_allclose(x_kf, x_linear, rtol=1e-10)
    np.testing.assert_allclose(P_kf, P_linear, rtol=1e-10)
    np.testing.assert_allclose(e_kf, innovation, rtol=1e-10)


@pytest.mark.parametrize("filter_name", ["ekf", "ukf"])
def test_symmetry(filter_name):
    """Test that the filter respects symmetry properties of the problem."""
    KalmanFilter = get_kf(filter_name)

    # Symmetric system
    def f_symmetric(t, x):
        return np.array([0.9 * x[0], 0.9 * x[1]], like=x)  # Symmetric dynamics

    def h_symmetric(t, x):
        return np.array(
            [x[0] ** 2 + x[1] ** 2], like=x
        )  # Rotationally symmetric measurement

    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])

    kf = KalmanFilter(
        dyn=f_symmetric,
        obs=h_symmetric,
        Q=Q,
        R=R,
    )

    # Symmetric initial conditions
    x1 = np.array([1.0, 0.0])
    x2 = np.array([0.0, 1.0])  # 90-degree rotation
    P = np.eye(2) * 0.1  # Symmetric covariance

    y1 = np.array([1.0])  # Same measurement for both (due to symmetry)
    y2 = np.array([1.0])

    # UKF steps
    x1_post, P1_post, _ = kf.step(0.0, x1, y1, P)
    x2_post, P2_post, _ = kf.step(0.0, x2, y2, P)

    # Posterior states should have same magnitude
    np.testing.assert_allclose(
        np.linalg.norm(x1_post), np.linalg.norm(x2_post), rtol=1e-10
    )

    # Posterior covariances should be related by 90° rotation
    # P2 should equal R90^T * P1 * R90 where R90 is 90° rotation matrix
    R90 = np.array([[0, 1], [-1, 0]])  # 90° counter-clockwise rotation
    P1_rotated = R90.T @ P1_post @ R90
    np.testing.assert_allclose(P2_post, P1_rotated, rtol=1e-10)

    print(
        f"State magnitudes: {np.linalg.norm(x1_post):.10f} == "
        f"{np.linalg.norm(x2_post):.10f}"
    )
    print(
        "Covariance rotation relationship verified with precision: "
        f"{np.max(np.abs(P2_post - P1_rotated)):.2e}"
    )


class TestKFClasses:
    @pytest.mark.parametrize("filter_name", ["ekf", "ukf"])
    def test_class_properties(self, filter_name, simple_nonlinear_system):
        KalmanFilter = get_kf(filter_name)

        kf = KalmanFilter(
            dyn=simple_nonlinear_system["f"],
            obs=simple_nonlinear_system["h"],
            Q=simple_nonlinear_system["Q"],
            R=simple_nonlinear_system["R"],
        )

        nx = simple_nonlinear_system["Q"].shape[0]
        ny = simple_nonlinear_system["R"].shape[0]

        assert kf.nx == nx, f"{filter_name} nx property mismatch"
        assert kf.ny == ny, f"{filter_name} ny property mismatch"


class TestEKF:
    """Test mathematical properties and correctness of EKF implementation."""

    def test_measurement_update_properties(self):
        """Test mathematical properties of the measurement update step."""

        def h(t, x):
            return np.array([x[0] + 0.1 * x[1] ** 2], like=x)  # Nonlinear measurement

        R = np.array([[0.1]])

        ekf = ExtendedKalmanFilter(None, h, None, R)

        # State and covariance BEFORE measurement update
        x_prior = np.array([1.0, 0.5])
        P_prior = np.array([[0.1, 0.02], [0.02, 0.08]])
        y = np.array([1.02])

        # Apply ONLY the measurement update
        x_post, P_post, innovation = ekf.correct(0.0, x_prior, y, P_prior)

        # Test 1: Posterior covariance should be smaller than prior
        assert np.trace(P_post) < np.trace(P_prior), (
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


class TestUKF:
    """Test mathematical properties and correctness of UKF implementation."""

    def test_unscented_transform(self):
        # Define test mean and covariance
        x_mean = np.array([1.0, -0.5])
        P = np.array([[0.1, 0.02], [0.02, 0.08]])

        # Test polynomial functions
        def linear_func(x):
            """Linear function: should be exact"""
            return np.array([2 * x[0] + 3 * x[1], -x[0] + 4 * x[1]])

        def quadratic_func(x):
            """Quadratic function: should be exact for unscented transform"""
            return np.array([x[0] ** 2, x[1] ** 2, x[0] * x[1]])

        # Generate sigma points using Julier method (matches our UKF)
        L = len(x_mean)
        kappa = 0.0

        # FIXED: Use scipy.linalg.cholesky and proper indexing
        U = cholesky((L + kappa) * P)

        sigma_points = []
        sigma_points.append(x_mean)

        for j in range(L):
            # FIXED: Use U[j] (j-th row) instead of A[:, j] (j-th column)
            sigma_points.append(x_mean + U[j])
            sigma_points.append(x_mean - U[j])

        # FIXED: Use proper Julier weights instead of equal weights
        w = np.full(2 * L + 1, 0.5 / (L + kappa))
        w[0] = kappa / (L + kappa)

        # Test linear function
        y_points_linear = np.array([linear_func(sp) for sp in sigma_points])
        y_mean_ut = np.sum(w[:, np.newaxis] * y_points_linear, axis=0)

        # Analytical result for linear function
        A_matrix = np.array([[2, 3], [-1, 4]])
        y_mean_analytical = A_matrix @ x_mean

        np.testing.assert_allclose(y_mean_ut, y_mean_analytical, rtol=1e-12)

        # Test quadratic function
        y_points_quad = np.array([quadratic_func(sp) for sp in sigma_points])
        y_mean_ut_quad = np.sum(w[:, np.newaxis] * y_points_quad, axis=0)

        # Analytical moments for quadratic function
        expected_quad = np.array(
            [
                P[0, 0] + x_mean[0] ** 2,  # E[x₁²]
                P[1, 1] + x_mean[1] ** 2,  # E[x₂²]
                P[0, 1] + x_mean[0] * x_mean[1],  # E[x₁x₂]
            ]
        )

        np.testing.assert_allclose(y_mean_ut_quad, expected_quad, rtol=1e-10)

    def test_measurement_update_information_increase(self):
        """Test that measurement updates increase information (decrease uncertainty)."""

        def f(t, x):
            return x  # Identity dynamics for simplicity

        def h(t, x):
            return np.array([x[0] + 0.1 * x[1] ** 2, x[1]])  # Nonlinear measurement

        Q = np.eye(2) * 0.001  # Small process noise
        R = np.eye(2) * 0.1

        ukf = UnscentedKalmanFilter(f, h, Q, R)

        x_prior = np.array([1.0, 0.5])
        P_prior = np.array([[0.2, 0.05], [0.05, 0.15]])
        y = np.array([1.02, 0.48])

        x_post, P_post, _ = ukf.step(0.0, x_prior, y, P_prior)

        # Test: Posterior uncertainty should be smaller
        trace_prior = np.trace(P_prior + Q)  # Include process noise
        trace_post = np.trace(P_post)

        assert trace_post < trace_prior, "Measurement should reduce uncertainty"

        # Test: Determinant should decrease (volume of uncertainty ellipsoid)
        det_prior = np.linalg.det(P_prior + Q)
        det_post = np.linalg.det(P_post)

        assert det_post < det_prior, "Measurement should reduce uncertainty volume"

    def test_degenerate_covariance_handling(self):
        """Test UKF behavior with near-singular covariance matrices."""

        def f(t, x):
            return np.array([0.99 * x[0], 0.99 * x[1]])

        def h(t, x):
            return np.array([x[0]])  # Only observe first state

        Q = np.eye(2) * 1e-6  # Very small process noise
        R = np.array([[0.01]])

        ukf = UnscentedKalmanFilter(f, h, Q, R)

        # Start with well-conditioned covariance
        x = np.array([1.0, 1.0])
        P = np.eye(2) * 0.1

        # Run many steps with only partial observations
        # This should make P become nearly singular in unobserved direction
        for i in range(50):
            y = np.array([1.0 + 0.01 * np.random.randn()])
            x, P, _ = ukf.step(i * 0.1, x, y, P)

            # Should not crash even if P becomes ill-conditioned
            assert np.all(np.isfinite(x)), f"State became non-finite at step {i}"
            assert np.all(np.isfinite(P)), f"Covariance became non-finite at step {i}"

            # Eigenvalues should remain non-negative
            eigenvals = np.linalg.eigvals(P)
            assert np.all(eigenvals >= -1e-12), (
                f"Negative eigenvalues at step {i}: {eigenvals}"
            )
