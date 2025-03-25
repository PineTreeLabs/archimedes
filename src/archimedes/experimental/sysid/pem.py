"""Implementation of a nonlinear prediction error method"""
import numpy as np

from archimedes import tree, scan, sym_function
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
    if P0 is None:
        P0 = np.eye(nx)

    @sym_function(kind="MX")
    def kf_fwd(x0, args):
        @sym_function(kind="MX")
        def scan_fn(carry_flat, input):
            t, z = input[0], input[1:]
            x, P, args = unravel_carry(carry_flat)
            x, P, e = kf_step(dyn, obs, t, x, z, P, Q, R, args=args)
            output = np.concatenate([x, e], axis=0)
            carry, _ = tree.ravel((x, P, args))
            return carry, output

        init_carry, unravel_carry = tree.ravel((x0, P0, args))
        inputs = np.vstack((ts, zs)).T
        _carry, scan_output = scan(scan_fn, init_carry, xs=inputs)
        scan_output = scan_output.T
        x_hat, e = scan_output[:nx], scan_output[nx:]
        return x_hat, e

    return kf_fwd