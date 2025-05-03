from typing import Callable
import numpy as np

from archimedes import struct
from archimedes import jac


@struct.pytree_node
class ContinuousEKF:
    dyn: Callable = struct.field(static=True)  # ẋ = dyn(t, x, u)
    obs: Callable = struct.field(static=True)  # y = obs(t, x, u)
    Q: np.ndarray  # Process noise covariance
    R: np.ndarray  # Measurement noise covariance
    unravel_x: Callable = struct.field(static=True, default=None)
    unravel_y: Callable = struct.field(static=True, default=None)

    @struct.pytree_node
    class State:
        x: np.ndarray
        P: np.ndarray

    @property
    def nx(self):
        return self.Q.shape[0]

    @property
    def ny(self):
        return self.R.shape[0]

    @property
    def R_inv(self):
        return np.linalg.inv(self.R)

    def dynamics(self, t, state: State, y, u):
        x = state.x
        P = np.reshape(state.P, (self.nx, self.nx))
        F = jac(self.dyn, argnums=1)(t, x, u)
        H = jac(self.obs, argnums=1)(t, x, u)
        K = P @ H.T @ self.R_inv
        x_t = self.dyn(t, x, u) + K @ (y - self.obs(t, x, u))
        P_t = F @ P + P @ F.T - K @ H @ P + self.Q
        P_t = P_t.flatten()
        return self.State(x=x_t, P=P_t)

