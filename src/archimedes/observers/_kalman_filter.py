from __future__ import annotations

import abc
from typing import Callable

import numpy as np
from scipy.linalg import cholesky  # For Cholesky decomposition
import casadi as cs

from archimedes import jac, struct, compile


__all__ = [
    "KalmanFilterBase",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ekf_step",
    "ukf_step",
]


# # NOTE: Currently unused, but necessary for MLE calculations
# @compile(kind="SX")
# def _spd_logdet(A):
#     """slogdet for SPD matrices using Cholesky decomposition"""
#     L = np.linalg.cholesky(A)  # A = L @ L.T

#     # det(A) = det(L)^2 = (product of diagonal elements of L)^2
#     # log(det(A)) = 2 * sum(log(diagonal elements of L))
#     return 2.0 * sum(np.log(L[i, i]) for i in range(A.shape[0]))


@struct.pytree_node
class KalmanFilterBase(metaclass=abc.ABCMeta):
    dyn: Callable = struct.field(static=True)
    obs: Callable = struct.field(static=True)
    Q: np.ndarray
    R: np.ndarray

    @abc.abstractmethod
    def step(t, x, y, P, args=None):
        """Perform one step of the filter, combining prediction and update steps.

        Args:
            t: current time
            x: state vector
            y: measurement
            P: state covariance
            args: additional arguments to pass to f and h

        Returns:
            x: updated state vector
            P: updated state covariance
            e: innovation or measurement residual
        """
        pass

    @property
    def nx(self):
        return self.Q.shape[0]

    @property
    def ny(self):
        return self.R.shape[0]


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
    H = np.atleast_2d(H)  # Ensure H is a 2D array

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


@struct.pytree_node
class ExtendedKalmanFilter(KalmanFilterBase):

    def step(self, t, x, y, P, args=None):
        return ekf_step(self.dyn, self.obs, t, x, y, P, self.Q, self.R, args)


def _julier_sigma_points(x_center, P_cov, kappa):
    """Generate Julier sigma points exactly as in filterpy"""
    n = len(x_center)

    # Cholesky decomposition - scipy returns upper triangular
    U = cholesky((n + kappa) * P_cov)
    
    sigmas = np.zeros((2*n + 1, n))
    sigmas[0] = x_center
    
    for k in range(n):
        # Note: U[k] accesses the k-th row of the upper triangular matrix
        sigmas[k + 1] = x_center + U[k]      # x + U[k]
        sigmas[n + k + 1] = x_center - U[k]  # x - U[k]
        
    return sigmas

def _julier_weights(n, kappa):
    """Compute Julier weights exactly as in filterpy"""
    Wm = np.full(2*n + 1, 0.5 / (n + kappa))
    Wm[0] = kappa / (n + kappa)
    Wc = Wm.copy()  # For Julier, Wm and Wc are the same
    return Wm, Wc
    

def ukf_step(f, h, t, x, y, P, Q, R, args=None, kappa=0.0):
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
        kappa: secondary scaling parameter (typically 0 or 3-n)

    Returns:
        x: updated state vector
        P: updated state covariance
        e: innovation or measurement residual
    """
    if args is None:
        args = ()

    n = len(x)

    # Generate weights
    Wm, Wc = _julier_weights(n, kappa)

    # Generate initial sigma points
    sigmas = _julier_sigma_points(x, P, kappa)

    # PREDICT STEP - propagate sigma points through dynamics
    sigmas_f = np.zeros((2*n + 1, n))
    for i in range(2*n + 1):
        sigmas_f[i] = f(t, sigmas[i], *args)

    # Predicted mean (unscented transform)
    x_pred = np.zeros(n)
    for i in range(2*n + 1):
        x_pred += Wm[i] * sigmas_f[i]
    
    # Predicted covariance
    P_pred = Q.copy()
    for i in range(2*n + 1):
        diff = sigmas_f[i] - x_pred
        P_pred += Wc[i] * np.outer(diff, diff)

    # Regenerate sigma points around predicted state (critical for accuracy)
    sigmas_pred = _julier_sigma_points(x_pred, P_pred, kappa)

    # UPDATE STEP - propagate predicted sigma points through measurement function
    dim_z = len(y)
    sigmas_h = np.zeros((2*n + 1, dim_z))
    for i in range(2*n + 1):
        sigmas_h[i] = h(t, sigmas_pred[i], *args)
    
    # Predicted measurement mean
    y_pred = np.zeros(dim_z)
    for i in range(2*n + 1):
        y_pred += Wm[i] * sigmas_h[i]
    
    # Innovation covariance
    S = R.copy()
    for i in range(2*n + 1):
        diff_y = sigmas_h[i] - y_pred
        S += Wc[i] * np.outer(diff_y, diff_y)
    
    # Cross-covariance
    Pxz = np.zeros((n, dim_z))
    for i in range(2*n + 1):
        diff_x = sigmas_pred[i] - x_pred
        diff_y = sigmas_h[i] - y_pred
        Pxz += Wc[i] * np.outer(diff_x, diff_y)

    # Kalman gain
    K = Pxz @ np.linalg.inv(S)
    
    # Innovation
    e = y - y_pred
    
    # Updated state estimate
    x_new = x_pred + K @ e
    
    # Updated covariance
    P_new = P_pred - K @ S @ K.T

    return x_new, P_new, e


@struct.pytree_node
class UnscentedKalmanFilter(KalmanFilterBase):

    def step(self, t, x, y, P, args=None):
        return ukf_step(self.dyn, self.obs, t, x, y, P, self.Q, self.R, args)
