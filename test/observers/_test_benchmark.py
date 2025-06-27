# ruff: noqa: N802, N803, N806, E741

# Benchmark problems and robustness tests for Kalman filters
import numpy as np
import pytest
from scipy.stats import chi2

from archimedes.experimental import discretize
from archimedes.observers import (
    ekf_step,
    ukf_step,
)

from .conftest import assert_positive_definite

np.random.seed(0)  # For reproducibility


class TestBenchmarkProblems:
    """Test Kalman filters on standard benchmark problems."""

    @pytest.mark.parametrize("filter_name", ["ekf", "ukf"])
    def test_constant_velocity(self, filter_name):
        dt = 0.1

        kf_step = {
            "ekf": ekf_step,
            "ukf": ukf_step,
        }[filter_name]

        # Define a simple 1D constant velocity model
        # State: [position, velocity]
        def f(t, x):
            F = np.array(
                [
                    [1, dt],
                    [0, 1],
                ],
                like=x,
            )
            return F @ x

        H = np.array([[1, 0]])  # Measure only position

        def h(t, x):
            return H @ x

        # Initial state and covariance
        x0 = np.array([0.0, 1.0])  # Start at origin with 1 m/s velocity
        P0 = np.eye(2) * 0.1

        # Process and measurement noise
        Q = np.array([[0.1 * dt**2, 0], [0, 0.1 * dt]])
        R = np.array([[0.1]])

        # Generate synthetic measurements
        n_steps = 100
        x_true = np.zeros((n_steps, 2))
        z = np.zeros((n_steps, 1))

        x_true[0] = x0
        z[0] = h(0, x0) + np.random.normal(0, np.sqrt(R))
        for t in range(n_steps - 1):
            x_true[t + 1] = f(t * dt, x_true[t])
            z[t + 1] = h(t * dt, x_true[t + 1]) + np.random.normal(0, np.sqrt(R))

        # Run EKF
        P = P0.copy()
        x_hat = np.zeros_like(x_true)
        e = np.zeros_like(z)
        x_hat[0] = x0

        for t in range(n_steps - 1):
            x_hat[t + 1], P, e[t + 1] = kf_step(f, h, t * dt, x_hat[t], z[t], P, Q, R)

        # Test 1: Check if estimation errors are reasonable
        position_rmse = np.sqrt(np.mean((x_true[:, 0] - x_hat[:, 0]) ** 2))
        velocity_rmse = np.sqrt(np.mean((x_true[:, 1] - x_hat[:, 1]) ** 2))
        print(f"Position RMSE: {position_rmse:.3f}")
        print(f"Velocity RMSE: {velocity_rmse:.3f}")

        # Test 2: Check if innovations are consistent with their covariance
        # Normalized innovation squared (NIS) should follow chi-square distribution
        S = H @ P @ H.T + R
        nis = (e**2 / S).flatten()
        alpha = 0.05  # 95% confidence
        dof = 1  # 1D measurement
        chi2_val = chi2.ppf(1 - alpha, df=dof)
        nis_consistency = np.mean(nis < chi2_val)
        print(f"NIS consistency: {nis_consistency:.1%} (should be close to 95%)")

        # Test 3: Check if state covariance remains positive definite
        eigenvals = np.linalg.eigvals(P)
        is_pd = np.all(eigenvals > 0)
        print(f"Final covariance is positive definite: {is_pd}")

        assert position_rmse < 0.5
        assert velocity_rmse < 1.0
        assert abs(nis_consistency - 0.95) < 0.1
        assert is_pd

    def test_van_der_pol_oscillator(self):
        """Test on Van der Pol oscillator - a classic nonlinear benchmark."""
        # Van der Pol parameters
        mu = 0.1  # Damping parameter
        dt = 0.1

        def f(t, x):
            """Van der Pol dynamics: x1' = x2, x2' = mu*(1-x1²)*x2 - x1"""
            x1, x2 = x[0], x[1]
            x1_next = x1 + dt * x2
            x2_next = x2 + dt * (mu * (1 - x1**2) * x2 - x1)
            return np.array([x1_next, x2_next], like=x)

        def h(t, x):
            """Observe position only"""
            return np.array([x[0]], like=x)

        Q = np.array([[0.01, 0.0], [0.0, 0.01]])
        R = np.array([[0.1]])

        # Test both EKF and UKF
        for filter_name, filter_step in [("EKF", ekf_step), ("UKF", ukf_step)]:
            x = np.array([1.0, 0.0])  # Initial condition
            P = np.eye(2) * 0.1

            positions = []
            covariances = []

            # Run for one period of oscillation
            for i in range(50):
                # Simulate measurement
                y = np.array([x[0] + np.random.normal(0, np.sqrt(R[0, 0]))])

                # Filter step
                x, P, _ = filter_step(f, h, i * dt, x, y, P, Q, R)

                positions.append(x[0])
                covariances.append(np.trace(P))

                # Basic sanity checks
                assert_positive_definite(P, f"{filter_name} covariance")
                assert np.all(np.isfinite(x)), f"{filter_name} state became infinite"

            # Test: Should maintain reasonable oscillatory behavior
            positions = np.array(positions)
            assert np.max(positions) > 0.5, (
                f"{filter_name} oscillation amplitude too small"
            )
            assert np.min(positions) < -0.5, (
                f"{filter_name} oscillation amplitude too small"
            )

    def test_coordinated_turn_model(self):
        """Test on coordinated turn model - standard tracking benchmark."""
        # Coordinated turn parameters
        dt = 0.1
        omega = 0.1  # Turn rate (rad/s)

        def f(t, x):
            """Coordinated turn: [x, y, vx, vy] with constant turn rate"""
            cos_wt = np.cos(omega * dt)
            sin_wt = np.sin(omega * dt)

            # State transition matrix for coordinated turn
            F = np.array(
                [
                    [1, 0, sin_wt / omega, -(1 - cos_wt) / omega],
                    [0, 1, (1 - cos_wt) / omega, sin_wt / omega],
                    [0, 0, cos_wt, -sin_wt],
                    [0, 0, sin_wt, cos_wt],
                ],
                like=x,
            )

            return F @ x

        def h(t, x):
            """Observe position only"""
            return np.array([x[0], x[1]], like=x)

        Q = np.eye(4) * 0.01  # Process noise
        R = np.eye(2) * 0.1  # Measurement noise

        # Test both filters
        for filter_name, filter_step in [("EKF", ekf_step), ("UKF", ukf_step)]:
            # Initial state: position (0,0), velocity (1,0)
            x = np.array([0.0, 0.0, 1.0, 0.0])
            P = np.eye(4) * 0.1

            positions = []

            # Simulate coordinated turn
            for i in range(60):  # About one turn
                y = h(i * dt, x) + np.random.multivariate_normal([0, 0], R)
                x, P, _ = filter_step(f, h, i * dt, x, y, P, Q, R)

                positions.append([x[0], x[1]])
                assert_positive_definite(P, f"{filter_name} covariance")

            positions = np.array(positions)

            # Test: Should trace out roughly circular path
            center_x, center_y = np.mean(positions, axis=0)
            radii = np.sqrt(
                (positions[:, 0] - center_x) ** 2 + (positions[:, 1] - center_y) ** 2
            )
            radius_std = np.std(radii)

            assert radius_std < 1.0, f"{filter_name} path not sufficiently circular"

    @pytest.mark.parametrize("filter_name", ["ekf", "ukf"])
    def test_pendulum_system(self, filter_name):
        """Test on nonlinear pendulum - mechanical system benchmark."""
        # Pendulum parameters
        g = 9.81  # gravity
        L = 1.0  # length
        dt = 0.05

        def ode_rhs(t, x, u, p):
            """Pendulum dynamics: theta, theta_dot"""
            theta, theta_dot = x[0], x[1]

            return np.array([theta_dot, -(g / L) * np.sin(theta)], like=x)

        f = discretize(ode_rhs, dt=dt, method="rk4")

        def h(t, x, u, p):
            """Observe angle and angular velocity (with different noise levels)"""
            return np.array([x[0], x[1]], like=x)

        R = np.array([[0.001, 0.0], [0.0, 0.01]])  # Measurement noise
        Q = 0.01 * R  # Small process noise estimate
        args = (None, None)  # No control input or parameters needed

        kf_step = {
            "ekf": ekf_step,
            "ukf": ukf_step,
        }[filter_name]

        # Start with small angle (nearly linear regime)
        x = np.array([0.2, 0.0])  # 0.2 rad ≈ 11 degrees
        P = np.eye(2) * 0.01

        angles = []
        energies = []

        # Simulate pendulum motion
        for i in range(100):
            angles.append(x[0])

            # Compute total energy (should be approximately conserved)
            kinetic = 0.5 * L**2 * x[1] ** 2
            potential = g * L * (1 - np.cos(x[0]))
            energies.append(kinetic + potential)

            y = h(i * dt, x, *args) + np.random.multivariate_normal([0, 0], R)
            x, P, _ = kf_step(f, h, i * dt, x, y, P, Q, R, args=args)

            assert_positive_definite(P, f"{filter_name} covariance")

        angles = np.array(angles)
        energies = np.array(energies)

        # Test: Should oscillate
        assert np.max(angles) > 0.1, f"{filter_name} pendulum not oscillating"
        assert np.min(angles) < -0.1, f"{filter_name} pendulum not oscillating"

        # Test: Energy should be roughly conserved (with some drift due to noise)
        energy_drift = np.abs(energies[-1] - energies[0]) / energies[0]
        assert energy_drift < 0.5, (
            f"{filter_name} energy drift too large: {energy_drift}"
        )


class TestRobustness:
    """Test filter robustness to various challenging conditions."""

    def test_missing_measurements(self):
        """Test filter behavior with intermittent measurements."""

        def f(t, x):
            return np.array([0.9 * x[0] + 0.1 * x[1], 0.8 * x[1]], like=x)

        def h(t, x):
            return np.array([x[0]], like=x)

        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])

        # Test both filters
        for filter_name, filter_step in [("EKF", ekf_step), ("UKF", ukf_step)]:
            x = np.array([1.0, 0.5])
            P = np.eye(2) * 0.1

            for i in range(30):
                # Skip measurements randomly (30% dropout rate)
                if np.random.rand() < 0.3:
                    # Prediction only (no measurement update)
                    # Simulate this by using a very large measurement noise
                    y = h(i * 0.1, x)  # Perfect measurement
                    R_large = R * 1e6  # Very large noise = no information
                    x, P, _ = filter_step(f, h, i * 0.1, x, y, P, Q, R_large)
                else:
                    # Normal measurement update
                    y = h(i * 0.1, x) + np.random.normal(0, np.sqrt(R[0, 0]))
                    x, P, _ = filter_step(f, h, i * 0.1, x, y, P, Q, R)

                # Filter should remain stable
                assert_positive_definite(
                    P, f"{filter_name} covariance with missing data"
                )
                assert np.all(np.isfinite(x)), f"{filter_name} state with missing data"

    def test_measurement_outliers(self):
        """Test filter robustness to measurement outliers."""

        def f(t, x):
            return np.array([x[0] + 0.1 * x[1], 0.95 * x[1]], like=x)

        def h(t, x):
            return np.array([x[0]], like=x)

        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])

        # Test both filters
        for filter_name, filter_step in [("EKF", ekf_step), ("UKF", ukf_step)]:
            x = np.array([1.0, 0.5])
            P = np.eye(2) * 0.1

            states = []
            innovations = []

            for i in range(40):
                # Generate normal measurement or outlier
                if i % 10 == 5:  # Every 10th measurement is an outlier
                    y = np.array([x[0] + 5.0])  # Large outlier
                else:
                    y = h(i * 0.1, x) + np.random.normal(0, np.sqrt(R[0, 0]))

                x, P, innovation = filter_step(f, h, i * 0.1, x, y, P, Q, R)

                states.append(x.copy())
                innovations.append(innovation[0])

                assert_positive_definite(P, f"{filter_name} covariance with outliers")
                assert np.all(np.isfinite(x)), f"{filter_name} state with outliers"

            states = np.array(states)
            innovations = np.array(innovations)

            # Test: State estimates should not be completely corrupted by outliers
            final_state_error = np.linalg.norm(states[-1] - states[0])
            assert final_state_error < 2.0, f"{filter_name} too sensitive to outliers"

            # Test: Innovations should be larger during outlier times
            outlier_innovations = innovations[5::10]  # Outlier times
            normal_innovations = innovations[[i for i in range(40) if i % 10 != 5]]

            assert np.mean(np.abs(outlier_innovations)) > 2 * np.mean(
                np.abs(normal_innovations)
            ), f"{filter_name} not detecting outliers"

    def test_poor_initial_conditions(self):
        """Test filter convergence from poor initial guesses."""

        def f(t, x):
            return np.array([x[0] + 0.1 * x[1], 0.9 * x[1]], like=x)

        def h(t, x):
            return np.array([x[0], 0.5 * x[1]], like=x)  # Observe both states

        Q = np.eye(2) * 0.01
        R = np.eye(2) * 0.1

        # True initial state
        x_true = np.array([1.0, 0.5])

        # Test both filters with very poor initial guess
        for filter_name, filter_step in [("EKF", ekf_step), ("UKF", ukf_step)]:
            x = np.array([5.0, -3.0])  # Very wrong initial guess
            P = np.eye(2) * 10.0  # Large initial uncertainty

            errors = []

            for i in range(50):
                # Generate measurement from true system
                x_true = f(i * 0.1, x_true) + np.random.multivariate_normal([0, 0], Q)
                y = h(i * 0.1, x_true) + np.random.multivariate_normal([0, 0], R)

                # Filter step
                x, P, _ = filter_step(f, h, i * 0.1, x, y, P, Q, R)

                error = np.linalg.norm(x - x_true)
                errors.append(error)

                assert_positive_definite(
                    P, f"{filter_name} covariance during convergence"
                )

            errors = np.array(errors)

            # Test: Should converge eventually
            initial_error = errors[0]
            final_error = np.mean(errors[-10:])  # Average of last 10 steps

            assert final_error < 0.5 * initial_error, (
                f"{filter_name} not converging from poor initial conditions"
            )

    def test_near_singular_covariance(self):
        """Test behavior with nearly singular covariance matrices."""

        def f(t, x):
            # Dynamics that preserve one direction
            return np.array([0.99 * x[0], 0.99 * x[1] + 0.01 * x[0]], like=x)

        def h(t, x):
            # Only observe first component
            return np.array([x[0]], like=x)

        Q = np.array([[1e-8, 0], [0, 1e-8]])  # Very small process noise
        R = np.array([[0.01]])

        # Test both filters
        for filter_name, filter_step in [("EKF", ekf_step), ("UKF", ukf_step)]:
            x = np.array([1.0, 1.0])
            P = np.eye(2) * 0.1

            condition_numbers = []

            for i in range(30):
                y = h(i * 0.1, x) + np.random.normal(0, np.sqrt(R[0, 0]))
                x, P, _ = filter_step(f, h, i * 0.1, x, y, P, Q, R)

                # Track condition number
                cond_num = np.linalg.cond(P)
                condition_numbers.append(cond_num)

                # Should not crash even with poor conditioning
                assert np.all(np.isfinite(x)), (
                    f"{filter_name} state with poor conditioning"
                )
                assert np.all(np.isfinite(P)), (
                    f"{filter_name} covariance with poor conditioning"
                )

                # Eigenvalues should remain non-negative
                eigenvals = np.linalg.eigvals(P)
                assert np.all(eigenvals >= -1e-10), (
                    f"{filter_name} negative eigenvalues: {eigenvals}"
                )

            # Test: Condition number should grow (unobservable direction)
            # but filter should remain stable
            assert condition_numbers[-1] > condition_numbers[0], (
                "Condition number should increase for unobservable system"
            )

    @pytest.mark.parametrize("filter_type", ["ekf", "ukf"])
    def test_consistency_statistics(self, filter_type):
        """Test filter consistency using NEES and NIS statistics."""

        # Simple linear system for predictable behavior
        def f(t, x):
            return np.array([0.9 * x[0] + 0.1 * x[1], 0.8 * x[1]], like=x)

        def h(t, x):
            return np.array([x[0] + 0.1 * x[1]], like=x)

        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])

        filter_step = ekf_step if filter_type == "ekf" else ukf_step

        # Run multiple Monte Carlo trials
        n_trials = 20
        n_steps = 30
        all_nees = []
        all_nis = []

        for trial in range(n_trials):
            np.random.seed(42 + trial)  # Repeatable but different seeds

            x_true = np.array([1.0, 0.5])
            x_est = np.array([0.8, 0.3])  # Slightly wrong initial guess
            P = np.eye(2) * 0.1

            trial_nees = []
            trial_nis = []

            for i in range(n_steps):
                # True system evolution
                x_true = f(i * 0.1, x_true) + np.random.multivariate_normal([0, 0], Q)
                y = h(i * 0.1, x_true) + np.random.normal(0, np.sqrt(R[0, 0]))

                # Filter step
                x_est, P, innovation = filter_step(f, h, i * 0.1, x_est, y, P, Q, R)

                # Compute NEES (Normalized Estimation Error Squared)
                error = x_est - x_true
                nees = error.T @ np.linalg.inv(P) @ error
                trial_nees.append(nees)

                # Compute NIS (Normalized Innovation Squared)
                from archimedes import jac

                H = jac(h, argnums=1)(i * 0.1, x_est)
                H = np.atleast_2d(H)
                S = H @ P @ H.T + R
                nis = innovation.T @ np.linalg.inv(S) @ innovation
                trial_nis.append(nis)

            all_nees.extend(trial_nees)
            all_nis.extend(trial_nis)

        all_nees = np.array(all_nees)
        all_nis = np.array(all_nis)

        # Statistical tests for consistency
        # NEES should follow chi-square distribution with DOF = state dimension
        nees_mean = np.mean(all_nees)
        expected_nees = 2.0  # Expected value for 2D state

        # NIS should follow chi-square distribution with DOF = measurement dimension
        nis_mean = np.mean(all_nis)
        expected_nis = 1.0  # Expected value for 1D measurement

        # Allow some tolerance due to finite sample size and approximations
        assert abs(nees_mean - expected_nees) < 0.5, (
            f"{filter_type.upper()} NEES inconsistent: {nees_mean} vs expected "
            f"{expected_nees}"
        )

        assert abs(nis_mean - expected_nis) < 0.3, (
            f"{filter_type.upper()} NIS inconsistent: {nis_mean} vs expected "
            f"{expected_nis}"
        )
