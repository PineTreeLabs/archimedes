from __future__ import annotations
from typing import Callable

import numpy as np

from archimedes import compile, scan, tree, jac, struct
from archimedes.experimental.state_estimation import ekf_step, ukf_step


@struct.pytree_node
class PEMObjective:
    Q: np.ndarray
    R: np.ndarray
    P0: np.ndarray
    kf_method: str = struct.field(static=True)
    dyn: Callable = struct.field(static=True)
    obs: Callable = struct.field(static=True)
    ts: np.ndarray = struct.field(static=True)
    ys: np.ndarray = struct.field(static=True)
    optimize_x0: bool = struct.field(static=True)

    @property
    def kf_step(self) -> Callable:
        """Return the Kalman filter step function based on the method"""
        return {
            "ekf": ekf_step,
            "ukf": ukf_step,
        }[self.kf_method]

    def forward(self, x0: np.ndarray, args: tuple) -> tuple:
        nx = self.Q.shape[0]
        ny = self.R.shape[0]
        if self.P0 is None:
            P0 = np.eye(nx)
        else:
            P0 = self.P0

        ts = self.ts
        ys = self.ys

        args_flat, unravel_args = tree.ravel(args)
        na = args_flat.size

        V = 0.0  # Cost function
        J = np.zeros((nx + na,), like=x0)  # Jacobian-like term: sum_{i=1}^N psi[i]
        H = np.zeros(
            (nx + na, nx + na), like=x0
        )  # Hessian-like term: sum_{i=1}^N psi[i].T @ psi[i]
        init_carry, unravel_carry = tree.ravel((x0, P0, args, V, J, H))

        # Predicted measurement (used for sensitivity analysis)
        def step(t, x, y, P, args_flat):
            args = unravel_args(args_flat)
            x, P, e = self.kf_step(
                self.dyn, self.obs, t, x, y, P, self.Q, self.R, args=args
            )
            y_hat = y - e  # Recover predicted observation
            return y_hat

        calc_psi = jac(step, argnums=(1, 4))  # dy_hat/d[x0, params]
        # Returns tuple of gradients: (∂y/∂x₀, ∂y/∂params)

        @compile(kind="MX")
        def scan_fn(carry_flat, input):
            t, y = input[0], input[1:]
            x, P, args, V, J, H = unravel_carry(carry_flat)
            args_flat = tree.ravel(args)[0]
            psi_x0, psi_params = calc_psi(t, x, y, P, args_flat)
            psi = np.concatenate([psi_x0, psi_params], axis=1)  # shape (ny, nx+na)
            x, P, e = self.kf_step(
                self.dyn, self.obs, t, x, y, P, self.Q, self.R, args=args
            )
            output = np.concatenate([x, e], axis=0)

            # Accumulate cost function, Jacobian, and Hessian
            V += np.sum(e**2)
            J -= psi.T @ e
            H += psi.T @ psi

            carry, _ = tree.ravel((x, P, args, V, J, H))
            return carry, output

        inputs = np.vstack((ts, ys)).T
        carry, scan_output = scan(scan_fn, init_carry, xs=inputs)
        _, _, _, V, J, H = unravel_carry(carry)
        scan_output = scan_output.T
        x_hat, e = scan_output[:nx], scan_output[nx:]

        # Average the function results
        V /= ts.size
        J /= ts.size
        H /= ts.size

        if not self.optimize_x0:
            J = J[nx:]
            H = H[nx:, nx:]

        return {
            "x_hat": x_hat,
            "e": e,
            "V": V,
            "J": J,
            "H": H,
        }

    def __call__(self, x0: np.ndarray, args: tuple) -> tuple:
        """Evaluate the residuals

        Args:
            x0: initial state
            args: additional arguments for the state transition and measurement functions

        Returns:
            tuple of (V, J, H)
            - V: cost function
            - J: Jacobian
            - H: Hessian approximation
        """
        results = self.forward(x0, args)
        V = results["V"]
        J = results["J"]
        H = results["H"]
        return V, J, H


def make_pem(
    dyn,
    obs,
    ts,
    ys,
    Q,
    R,
    P0=None,
    kf_method="ekf",
    optimize_x0=False,
):
    """Create a function to evaluate the residuals

    Args:
        dyn: function of (t, x, *args) that computes the state transition function
        obs: function of (t, x, *args) that computes the measurement function
        ts: time points (nt,)
        ys: measurements (ny, nt)
        Q: process noise covariance
        R: measurement noise covariance
        P0: initial state covariance (optional, defaults to identity)
        kf_method: "ekf" or "ukf" (optional, defaults to "ekf")
        optimize_x0: whether to optimize the initial state (optional,
            defaults to False)

    Returns:
        function of (x0, args) that computes the residuals
    """
    return PEMObjective(
        Q=Q,
        R=R,
        P0=P0,
        kf_method=kf_method,
        dyn=dyn,
        obs=obs,
        ts=ts,
        ys=ys,
        optimize_x0=optimize_x0,
    )
