# ruff: noqa: N802, N803, N806, E741

# Test fixtures and utilities for state estimation tests
import numpy as np
import pytest


@pytest.fixture
def simple_linear_system():
    """Fixture providing a simple 2D linear system for testing."""
    dt = 0.1
    F = np.array([[1.0, dt], [0.0, 1.0]])  # Constant velocity model
    H = np.array([[1.0, 0.0]])  # Position measurement only
    Q = np.array([[0.01, 0.0], [0.0, 0.01]])  # Process noise
    R = np.array([[0.1]])  # Measurement noise

    def f(t, x):
        return F @ x

    def h(t, x):
        return H @ x

    return {"f": f, "h": h, "F": F, "H": H, "Q": Q, "R": R, "dt": dt}


@pytest.fixture
def simple_nonlinear_system():
    """Fixture providing a simple nonlinear system for testing."""

    def f(t, x):
        return np.array([x[0] + 0.1 * x[1], 0.9 * x[1] + 0.1 * np.sin(x[0])])

    def h(t, x):
        return np.array([x[0] + 0.05 * x[1] ** 2])

    Q = np.array([[0.01, 0.0], [0.0, 0.02]])
    R = np.array([[0.1]])

    return {"f": f, "h": h, "Q": Q, "R": R}


@pytest.fixture
def random_positive_definite_matrix():
    """Generate a random positive definite matrix for testing."""

    def _generate(n=2, condition_number=10.0):
        """Generate random positive definite matrix with specified condition number."""
        # Generate random matrix
        A = np.random.randn(n, n)

        # Make it symmetric positive definite
        A = A @ A.T

        # Control condition number
        U, s, Vh = np.linalg.svd(A)
        s_new = np.linspace(1.0, condition_number, n)
        A = U @ np.diag(s_new) @ Vh

        return A

    return _generate


@pytest.fixture
def filter_test_data():
    """Generate synthetic test data for filter validation."""

    def _generate(n_steps=50, system="linear", seed=42):
        """Generate test trajectory with known ground truth."""
        np.random.seed(seed)

        if system == "linear":
            # Linear system
            dt = 0.1
            F = np.array([[1.0, dt], [0.0, 0.95]])
            H = np.array([[1.0, 0.0]])
            Q = np.array([[0.01, 0.0], [0.0, 0.01]])
            R = np.array([[0.1]])

            def f(t, x):
                return F @ x

            def h(t, x):
                return H @ x

        elif system == "nonlinear":
            # Nonlinear system
            def f(t, x):
                return np.array([x[0] + 0.1 * x[1], 0.9 * x[1] + 0.1 * np.sin(x[0])])

            def h(t, x):
                return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2)])

            Q = np.array([[0.01, 0.0], [0.0, 0.02]])
            R = np.array([[0.1]])

        # Generate true trajectory
        x_true = np.zeros((n_steps, 2))
        y_meas = np.zeros((n_steps, R.shape[0]))

        x_true[0] = [1.0, 0.5]

        for k in range(1, n_steps):
            # True state evolution with process noise
            w = np.random.multivariate_normal([0, 0], Q)
            x_true[k] = f((k - 1) * 0.1, x_true[k - 1]) + w

            # Measurement with noise
            v = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
            y_meas[k] = h(k * 0.1, x_true[k]) + v

        return {
            "x_true": x_true,
            "y_meas": y_meas,
            "f": f,
            "h": h,
            "Q": Q,
            "R": R,
            "dt": 0.1,
        }

    return _generate


def assert_positive_definite(matrix, name="matrix"):
    """Helper function to assert matrix is positive definite."""
    eigenvals = np.linalg.eigvals(matrix)
    assert np.all(eigenvals > 0), (
        f"{name} is not positive definite. Eigenvalues: {eigenvals}"
    )


def assert_symmetric(matrix, name="matrix", rtol=1e-12):
    """Helper function to assert matrix is symmetric."""
    np.testing.assert_allclose(
        matrix, matrix.T, rtol=rtol, err_msg=f"{name} is not symmetric"
    )


def compute_nees(errors, covariances):
    """Compute Normalized Estimation Error Squared (NEES) statistic."""
    nees = []
    for error, P in zip(errors, covariances):
        nees.append(error.T @ np.linalg.inv(P) @ error)
    return np.array(nees)


def compute_nis(innovations, innovation_covariances):
    """Compute Normalized Innovation Squared (NIS) statistic."""
    nis = []
    for innov, S in zip(innovations, innovation_covariances):
        nis.append(innov.T @ np.linalg.inv(S) @ innov)
    return np.array(nis)
