import pytest

import numpy as np
from scipy.stats import chi2

from archimedes.experimental.state_estimation import ekf_step, ukf_step


np.random.seed(42)


@pytest.mark.parametrize("method", ["ekf", "ukf"])
def test_kf_constant_velocity(method):
    dt = 0.1

    kf_step = {
        "ekf": ekf_step,
        "ukf": ukf_step,
    }[method]

    # Define a simple 1D constant velocity model
    # State: [position, velocity]
    def f(t, x):
        F = np.array([
            [1, dt],
             [0, 1],
        ], like=x)
        return F @ x

    H = np.array([[1, 0]])  # Measure only position
    def h(t, x):
        return H @ x
    
    # Initial state and covariance
    x0 = np.array([0., 1.])  # Start at origin with 1 m/s velocity
    P0 = np.eye(2) * 0.1
    
    # Process and measurement noise
    Q = np.array([[0.1*dt**2, 0],
                  [0, 0.1*dt]])
    R = np.array([[0.1]])
    
    # Generate synthetic measurements
    n_steps = 100
    x_true = np.zeros((n_steps, 2))
    z = np.zeros((n_steps, 1))
    
    x_true[0] = x0
    z[0] = h(0, x0) + np.random.normal(0, np.sqrt(R))
    for t in range(n_steps-1):
        x_true[t+1] = f(t*dt, x_true[t])
        z[t+1] = h(t*dt, x_true[t+1]) + np.random.normal(0, np.sqrt(R))
    
    # Run EKF
    P = P0.copy()
    x_hat = np.zeros_like(x_true)
    e = np.zeros_like(z)
    x_hat[0] = x0

    for t in range(n_steps-1):
        x_hat[t+1], P, e[t+1] = kf_step(f, h, t*dt, x_hat[t], z[t], P, Q, R)
    
    # Test 1: Check if estimation errors are reasonable
    position_rmse = np.sqrt(np.mean((x_true[:, 0] - x_hat[:, 0])**2))
    velocity_rmse = np.sqrt(np.mean((x_true[:, 1] - x_hat[:, 1])**2))
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

