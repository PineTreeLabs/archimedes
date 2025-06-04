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
    """Abstract base class for Kalman filter implementations.

    This class defines the common interface for Kalman filters used in system
    identification and state estimation.

    All Kalman filter implementations follow the discrete-time state-space model:

    .. code-block:: text

        x[k+1] = f(t[k], x[k], *args) + w[k]    (dynamics)
        y[k]   = h(t[k], x[k], *args) + v[k]    (observations)

    where ``w[k] ~ N(0, Q)`` is process noise and ``v[k] ~ N(0, R)`` is
    measurement noise.

    Parameters
    ----------
    dyn : callable
        Dynamics function with signature ``f(t, x, *args)`` that returns the
        predicted state at the next time step. Must be compatible with automatic
        differentiation for gradient-based filters.
    obs : callable  
        Observation function with signature ``h(t, x, *args)`` that maps state
        to expected measurements. Must be compatible with automatic differentiation
        for gradient-based filters.
    Q : array_like
        Process noise covariance matrix of shape ``(nx, nx)`` where ``nx`` is
        the state dimension. Must be positive semi-definite.
    R : array_like
        Measurement noise covariance matrix of shape ``(ny, ny)`` where ``ny`` is
        the measurement dimension. Must be positive definite.

    Attributes
    ----------
    nx : int
        State dimension, derived from the shape of ``Q``.
    ny : int  
        Measurement dimension, derived from the shape of ``R``.

    Notes
    -----
    This class is decorated with ``@struct.pytree_node``, making it compatible
    with function transformations and enabling efficient automatic differentiation
    through the filter operations. The filter parameters can be modified using
    standard PyTree operations.

    Subclasses must implement the abstract ``step`` method that performs one
    complete filtering step (prediction + update). The implementation details
    depend on the specific filtering algorithm (EKF, UKF, etc.).

    The ``args`` parameter in function signatures allows passing additional
    arguments to both dynamics and observation functions, enabling time-varying
    parameters or external inputs.

    Examples
    --------
    >>> # Subclasses implement the filtering algorithm
    >>> class CustomFilter(KalmanFilterBase):
    ...     def step(self, t, x, y, P, args=None):
    ...         # Implementation-specific filtering logic
    ...         return x_new, P_new, innovation
    >>>
    >>> # Define system dynamics and observations  
    >>> def f(t, x):
    ...     return np.array([x[0] + 0.1*x[1], 0.9*x[1]], like=x)
    >>>
    >>> def h(t, x):
    ...     return np.array([x[0]], like=x)
    >>>
    >>> # Create filter instance
    >>> Q = np.eye(2) * 0.01  # Process noise
    >>> R = np.array([[0.1]]) # Measurement noise
    >>> kf = CustomFilter(dyn=f, obs=h, Q=Q, R=R)
    >>>
    >>> # Filter properties
    >>> print(f"State dimension: {kf.nx}")
    >>> print(f"Measurement dimension: {kf.ny}")

    See Also
    --------
    ExtendedKalmanFilter : Extended Kalman Filter implementation
    UnscentedKalmanFilter : Unscented Kalman Filter implementation

    """
    dyn: Callable = struct.field(static=True)
    obs: Callable = struct.field(static=True)
    Q: np.ndarray
    R: np.ndarray

    @abc.abstractmethod
    def step(self, t, x, y, P, args=None):
        """Perform one step of the filter, combining prediction and update steps.

        This abstract method must be implemented by subclasses to define the
        specific filtering algorithm. It should perform both the prediction
        step (using the dynamics model) and the update step (incorporating
        the measurement).

        Parameters
        ----------
        t : float
            Current time step.
        x : array_like
            State vector of shape ``(nx,)``.
        y : array_like
            Measurement vector of shape ``(ny,)``.
        P : array_like
            State covariance matrix of shape ``(nx, nx)``.
        args : tuple, optional
            Additional arguments to pass to the dynamics and observation
            functions. Default is None.

        Returns
        -------
        x_new : ndarray
            Updated state estimate of shape ``(nx,)``.
        P_new : ndarray
            Updated state covariance matrix of shape ``(nx, nx)``.
        innovation : ndarray
            Measurement residual (innovation) of shape ``(ny,)``, computed
            as the difference between the actual measurement and the predicted
            measurement from the current state estimate.

        Notes
        -----
        The innovation sequence should have zero mean and known covariance
        (the innovation covariance) if the filter is performing optimally.
        This can be used for filter consistency checking and parameter tuning.
        """

    @property
    def nx(self):
        """State dimension derived from process noise covariance matrix."""
        return self.Q.shape[0]

    @property
    def ny(self):
        """Measurement dimension derived from measurement noise covariance matrix."""
        return self.R.shape[0]


@struct.pytree_node
class ExtendedKalmanFilter(KalmanFilterBase):
    """Extended Kalman Filter for nonlinear state estimation.

    The Extended Kalman Filter (EKF) handles nonlinear dynamics and observation
    models by linearizing them around the current state estimate using Jacobian
    matrices. This makes it computationally efficient while providing reasonable
    performance for mildly nonlinear systems.

    The EKF uses first-order Taylor series approximations:

    .. code-block:: text

        F[k] ≈ ∂f/∂x |_{x[k|k-1]}    (dynamics Jacobian)
        H[k] ≈ ∂h/∂x |_{x[k|k-1]}    (observation Jacobian)

    These Jacobians are computed automatically using the ``archimedes.jac``
    function, enabling seamless integration with the automatic differentiation
    framework.

    Parameters
    ----------
    Inherits all parameters from :class:`KalmanFilterBase`.

    Notes
    -----
    The EKF is most suitable for systems where:
    
    - Nonlinearities are mild (locally approximately linear)
    - Computational efficiency is important  
    - The system dynamics and observations are smooth and differentiable
    
    For highly nonlinear systems, consider using :class:`UnscentedKalmanFilter`
    which can better capture nonlinear transformations of uncertainty.

    The EKF algorithm consists of two steps:
    
    1. **Prediction**: Propagate state and covariance using linearized dynamics
    2. **Update**: Incorporate measurements using linearized observation model

    The linearization can introduce bias in the state estimates and may cause
    filter divergence if the nonlinearities are too strong or if the initial
    estimate is far from the true state.

    Examples
    --------
    >>> import numpy as np
    >>> from archimedes.observers import ExtendedKalmanFilter
    >>>
    >>> # Define nonlinear dynamics (simple pendulum)
    >>> def f(t, x):
    ...     dt = 0.1
    ...     return np.hstack([
    ...         x[0] + dt * x[1],                # position
    ...         x[1] - dt * 9.81 * np.sin(x[0])  # velocity
    ...     ])
    >>>
    >>> # Define observations
    >>> def h(t, x):
    ...     return x[0]
    >>>
    >>> # Create EKF
    >>> Q = np.eye(2) * 0.01
    >>> R = np.array([[0.1]])
    >>> ekf = ExtendedKalmanFilter(dyn=f, obs=h, Q=Q, R=R)
    >>>
    >>> # Filtering step
    >>> x = np.array([0.1, 0.0])  # Initial state
    >>> P = np.eye(2) * 0.1       # Initial covariance  
    >>> y = np.array([0.12])      # Measurement
    >>> x_new, P_new, innov = ekf.step(0.0, x, y, P)

    See Also
    --------
    KalmanFilterBase : Abstract base class for all Kalman filters
    UnscentedKalmanFilter : Alternative for highly nonlinear systems

    References
    ----------
    .. [1] "Extended Kalman filter", Wikipedia.
           https://en.wikipedia.org/wiki/Extended_Kalman_filter
    .. [2] Gelb, A. "Applied Optimal Estimation." MIT Press, 1974.
    """

    def correct(self, t, x, y, P, args=None):
        """Perform the measurement update (correction) step of the EKF.

        This method applies the measurement update equations to incorporate
        new observations into the state estimate. It computes the observation
        Jacobian automatically and updates both the state estimate and its
        covariance matrix.

        Parameters
        ----------
        t : float
            Current time step.
        x : array_like
            Prior state estimate of shape ``(nx,)``, typically from the
            prediction step.
        y : array_like
            Measurement vector of shape ``(ny,)``.
        P : array_like
            Prior state covariance matrix of shape ``(nx, nx)``, typically
            from the prediction step.
        args : tuple, optional
            Additional arguments to pass to the observation function.
            Default is None.

        Returns
        -------
        x_post : ndarray
            Posterior (updated) state estimate of shape ``(nx,)``.
        P_post : ndarray
            Posterior (updated) state covariance matrix of shape ``(nx, nx)``.
        innovation : ndarray
            Measurement residual of shape ``(ny,)``, computed as the
            difference between the actual measurement and the predicted
            measurement from the prior state estimate.

        Notes
        -----
        This method can be used independently of the prediction step, allowing
        for custom prediction schemes or processing multiple measurements at
        the same time step.

        The method computes the observation Jacobian ``H = ∂h/∂x`` using
        automatic differentiation, then applies the standard Kalman update
        equations:

        .. code-block:: text

            innovation = y - h(x_prior)
            S = H @ P_prior @ H.T + R  
            K = P_prior @ H.T @ S^(-1)
            x_post = x_prior + K @ innovation
            P_post = (I - K @ H) @ P_prior

        Examples
        --------
        >>> # Apply correction with custom prediction
        >>> x_pred = custom_prediction_step(x_prev)
        >>> P_pred = custom_covariance_prediction(P_prev)  
        >>> x_new, P_new, innov = ekf.correct(t, x_pred, y, P_pred)
        """
        h = self.obs
        R = self.R

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


    def step(self, t, x, y, P, args=None):
        """Perform one complete EKF step (prediction + update).

        This method implements the full Extended Kalman Filter algorithm,
        performing both the prediction step (using the dynamics model) and
        the update step (incorporating the measurement) in sequence.

        Parameters
        ----------
        t : float
            Current time step.
        x : array_like
            Previous state estimate of shape ``(nx,)``.
        y : array_like
            Current measurement vector of shape ``(ny,)``.
        P : array_like
            Previous state covariance matrix of shape ``(nx, nx)``.
        args : tuple, optional
            Additional arguments to pass to both dynamics and observation
            functions. Default is None.

        Returns
        -------
        x_new : ndarray
            Updated state estimate of shape ``(nx,)``.
        P_new : ndarray
            Updated state covariance matrix of shape ``(nx, nx)``.
        innovation : ndarray
            Measurement residual of shape ``(ny,)``.

        Notes
        -----
        The method performs the following sequence:

        1. **Prediction Step**: 
           - Compute dynamics Jacobian ``F = ∂f/∂x``
           - Propagate state: ``x_pred = f(t, x)``
           - Propagate covariance: ``P_pred = F @ P @ F.T + Q``

        2. **Update Step**:
           - Apply measurement correction using :meth:`correct`

        The automatic differentiation ensures that Jacobians are computed
        accurately and efficiently, even for complex nonlinear functions.

        Examples
        --------
        >>> # Complete filtering step
        >>> x_k, P_k, innov = ekf.step(t_k, x_prev, y_k, P_prev)
        """
        f = self.dyn
        h = self.obs
        Q = self.Q
        R = self.R

        if args is None:
            args = ()

        F = jac(f, argnums=1)(t, x, *args)

        # Predict step
        x = f(t, x, *args)
        P = F @ P @ F.T + Q

        # Update step
        x, P, e = self.correct(t, x, y, P, args)

        return x, P, e


def _julier_sigma_points(x_center, P_cov, kappa):
    """Generate Julier sigma points"""
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
    """Compute Julier weights"""
    Wm = np.full(2*n + 1, 0.5 / (n + kappa))
    Wm[0] = kappa / (n + kappa)
    Wc = Wm.copy()  # For Julier, Wm and Wc are the same
    return Wm, Wc


@struct.pytree_node
class UnscentedKalmanFilter(KalmanFilterBase):
    kappa: float = 0.0

    def step(self, t, x, y, P, args=None):
        f = self.dyn
        h = self.obs
        Q = self.Q
        R = self.R
        kappa = self.kappa
        
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
