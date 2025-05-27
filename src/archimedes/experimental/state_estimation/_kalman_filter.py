import numpy as np

from archimedes import jac


def ekf_correct(h, t, x, y, P, R, args=None):
    """Perform the "correct" step of the extended Kalman filter

    Args:
        h: function of (t, x, *args) that computes the measurement function
        t: current time
        x: state vector (typically the prediction from the forward model)
        y: measurement
        P: state covariance (typically the prediction from the forward model)
        R: measurement noise covariance
        args: additional arguments to pass to f and h

    Returns:
        x: updated state vector
        P: updated state covariance
        e: innovation or measurement residual
    """

    if args is None:
        args = ()

    H = jac(h, argnums=1)(t, x, *args)

    e = y - h(t, x, *args)  # Innovation or measurement residual
    S = H @ P @ H.T + R  # Innovation covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    x = x + K @ e  # Updated state estimate
    P = (np.eye(len(x)) - K @ H) @ P  # Updated state covariance

    return x, P, e


def ekf_step(f, h, t, x, y, P, Q, R, args=None):
    """Perform one step of the extended Kalman filter

    Args:
        f: function of (t, x, *args) that computes the state transition function
        h: function of (t, x, *args) that computes the measurement function
        t: current time
        x: state vector
        y: measurement
        P: state covariance
        Q: process noise covariance
        R: measurement noise covariance
        args: additional arguments to pass to f and h

    Returns:
        x: updated state vector
        P: updated state covariance
        e: innovation or measurement residual
    """
    if args is None:
        args = ()

    F = jac(f, argnums=1)(t, x, *args)

    # Predict step
    x = f(t, x, *args)
    P = F @ P @ F.T + Q

    # Update step
    x, P, e = ekf_correct(h, t, x, y, P, R, args)

    return x, P, e


def ukf_step(f, h, t, x, y, P, Q, R, *args):
    """Perform one step of the unscented Kalman filter

    Args:
        f: function of (t, x, *args) that computes the state transition function
        h: function of (t, x, *args) that computes the measurement function
        t: current time
        x: state vector
        y: measurement
        P: state covariance
        Q: process noise covariance
        R: measurement noise covariance
        args: additional arguments to pass to f and h

    Returns:
        x: updated state vector
        P: updated state covariance
        e: innovation or measurement residual
    """
    A = np.linalg.cholesky(P)

    # Construct sigma points
    L = len(x)
    n = 2 * L + 1
    s = [x]

    for j in range(L):
        s.append(x + np.sqrt(L) * A[:, j])

    for j in range(L):
        s.append(x - np.sqrt(L) * A[:, j])

    w_a = np.ones(n) / n  # Weights for means
    w_c = w_a  # Weights for covariance

    # Predict step
    x_pred = []
    for i in range(n):
        x_pred.append(f(t, s[i], *args))

    x_hat = sum([w_a[i] * x_pred[i] for i in range(n)])
    P = Q + sum(
        [w_c[i] * np.outer(x_pred[i] - x_hat, x_pred[i] - x_hat) for i in range(n)]
    )

    # Update step
    y_pred = [h(t, s[i], *args) for i in range(n)]
    y_hat = sum(
        [w_a[i] * y_pred[i] for i in range(n)]
    )  # Empirical mean of measurements
    S_hat = R + sum(
        [w_c[i] * np.outer(y_pred[i] - y_hat, y_pred[i] - y_hat) for i in range(n)]
    )  # Empirical covariance
    Cxz = sum(
        [w_c[i] * np.outer(x_pred[i] - x_hat, y_pred[i] - y_hat) for i in range(n)]
    )  # Cross-covariance

    K = Cxz @ np.linalg.inv(S_hat)  # Kalman gain
    e = y - y_hat  # Innovation or measurement residual
    x = x_hat + K @ e  # Updated state estimate
    P = P - K @ S_hat @ K.T  # Updated state covariance

    return x, P, e
