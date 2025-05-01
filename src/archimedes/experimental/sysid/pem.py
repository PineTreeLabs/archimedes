"""Implementation of a nonlinear prediction error method"""

import numpy as np

from archimedes import compile, scan, tree, jac
from archimedes.experimental.state_estimation import ekf_step, ukf_step


def make_pem(
    dyn,
    obs,
    ts,
    zs,
    Q,
    R,
    P0=None,
    kf_method="ekf",
):
    """Create a function to evaluate the residuals

    Args:
        dyn: function of (t, x, *args) that computes the state transition function
        obs: function of (t, x, *args) that computes the measurement function
        ts: time points (nt,)
        zs: measurements (ny, nt)
        Q: process noise covariance
        R: measurement noise covariance
        P0: initial state covariance (optional, defaults to identity)
        kf_method: "ekf" or "ukf" (optional, defaults to "ekf")

    Returns:
        function of (x0, args) that computes the residuals
    """
    kf_step = {
        "ekf": ekf_step,
        "ukf": ukf_step,
    }[kf_method]

    nx = Q.shape[0]
    ny = R.shape[0]
    if P0 is None:
        P0 = np.eye(nx)

    @compile(kind="MX")
    def kf_fwd(x0, args):
        args_flat, unravel_args = tree.ravel(args)
        na = args_flat.size

        V = 0.0  # Cost function
        J = np.zeros((na,), like=x0)  # Jacobian-like term: sum_{i=1}^N psi[i]
        H = np.zeros((na, na), like=x0)  # Hessian-like term: sum_{i=1}^N psi[i].T @ psi[i]
        init_carry, unravel_carry = tree.ravel((x0, P0, args, V, J, H))

        # Predicted measurement (used for sensitivity analysis)
        def step(t, x, z, P, args_flat):
            args = unravel_args(args_flat)
            x, P, e = kf_step(dyn, obs, t, x, z, P, Q, R, args=args)
            z_hat = z - e  # Recover predicted observation
            return z_hat

        calc_psi = jac(step, argnums=(4,))  # dy_hat/dp: shape (ny, na)

        @compile(kind="MX")
        def scan_fn(carry_flat, input):
            t, z = input[0], input[1:]
            x, P, args, V, J, H = unravel_carry(carry_flat)
            args_flat = tree.ravel(args)[0]
            psi = calc_psi(t, x, z, P, args_flat)
            x, P, e = kf_step(dyn, obs, t, x, z, P, Q, R, args=args)
            output = np.concatenate([x, e], axis=0)

            # Accumulate cost function, Jacobian, and Hessian
            V += np.sum(e**2)
            J += psi.T @ e
            H += psi.T @ psi

            carry, _ = tree.ravel((x, P, args, V, J, H))
            return carry, output

        inputs = np.vstack((ts, zs)).T
        carry, scan_output = scan(scan_fn, init_carry, xs=inputs)
        _, _, _, V, J, H = unravel_carry(carry)
        scan_output = scan_output.T
        x_hat, e = scan_output[:nx], scan_output[nx:]

        # Average the function results
        V /= ts.size
        J /= ts.size
        H /= ts.size

        return x_hat, e, V, J, H

    return kf_fwd
