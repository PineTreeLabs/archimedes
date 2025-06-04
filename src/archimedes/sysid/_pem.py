from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np

from archimedes import compile, scan, tree, jac, struct

from ._lm import lm_solve

if TYPE_CHECKING:
    from archimedes.typing import PyTree
    from archimedes.experimental.state_estimation import KalmanFilterBase
    from .timeseries import Timeseries
    from ._lm import LMResult

    T = TypeVar("T", bound=PyTree)


__all__ = ["pem"]


@struct.pytree_node
class PEMObjective:
    predictor: KalmanFilterBase
    data: Timeseries
    P0: np.ndarray
    x0: np.ndarray = struct.field(static=True, default=None)

    def forward(self, x0: np.ndarray, params: tuple) -> tuple:
        ts = self.data.ts
        us = self.data.us
        ys = self.data.ys

        nx = self.predictor.nx
        nu = us.shape[0]
        ny = ys.shape[0]

        if self.P0 is None:
            P0 = np.eye(nx)
        else:
            P0 = self.P0

        params_flat, unravel_params = tree.ravel(params)
        na = params_flat.size

        V = 0.0  # Cost function
        J = np.zeros((nx + na,), like=x0)  # Jacobian-like term: sum_{i=1}^N psi[i]
        H = np.zeros(
            (nx + na, nx + na), like=x0
        )  # Hessian-like term: sum_{i=1}^N psi[i].T @ psi[i]
        init_carry, unravel_carry = tree.ravel((x0, P0, params, V, J, H))

        # Predicted measurement (used for sensitivity analysis)
        def step(t, x, u, y, P, params_flat):
            params = unravel_params(params_flat)
            x, P, e = self.predictor.step(t, x, y, P, args=(u, params))
            y_hat = y - e  # Recover predicted observation
            return y_hat

        # Returns tuple of gradients: (∂y/∂x₀, ∂y/∂params)
        calc_psi = jac(step, argnums=(1, 5))  # dy_hat/d[x0, params]

        @compile(kind="MX")
        def scan_fn(carry_flat, input):
            t, u, y = input[0], input[1:nu+1], input[nu+1:]
            x, P, params, V, J, H = unravel_carry(carry_flat)
            params_flat = tree.ravel(params)[0]
            psi_x0, psi_params = calc_psi(t, x, u, y, P, params_flat)
            psi = np.concatenate([psi_x0, psi_params], axis=1)  # shape (ny, nx+na)
            x, P, e = self.predictor.step(t, x, y, P, args=(u, params))
            output = np.concatenate([x, e], axis=0)

            # Accumulate cost function, Jacobian, and Hessian
            V += np.sum(e**2)
            J -= psi.T @ e
            H += psi.T @ psi

            carry, _ = tree.ravel((x, P, params, V, J, H))
            return carry, output

        inputs = np.vstack((ts, us, ys)).T
        carry, scan_output = scan(scan_fn, init_carry, xs=inputs)
        _, _, _, V, J, H = unravel_carry(carry)
        scan_output = scan_output.T
        x_hat, e = scan_output[:nx], scan_output[nx:]

        # Average the function results
        V /= ts.size
        J /= ts.size
        H /= ts.size

        # If not jointly optimizing x0, trim from gradients
        if self.x0 is not None:
            J = J[nx:]
            H = H[nx:, nx:]

        return {
            "x_hat": x_hat,
            "e": e,
            "V": V,
            "J": J,
            "H": H,
        }

    def __call__(self, decision_variables: np.ndarray) -> tuple:
        """Evaluate the residuals

        Args:
            decision_variables: optimization parameters

        Returns:
            tuple of (V, J, H)
            - V: cost function
            - J: Jacobian
            - H: Hessian approximation
        """
        if self.x0 is not None:
            x0 = self.x0
            params = decision_variables
        else:
            x0, params = decision_variables

        results = self.forward(x0, params)
        V = results["V"]
        J = results["J"]
        H = results["H"]
        return V, J, H


def pem(
    predictor: KalmanFilterBase,
    data: Timeseries,
    params_guess: T,
    x0: np.ndarray = None,
    bounds: tuple[T, T] | None = None,
    P0: np.ndarray = None,
    method: str = "lm",
    options: dict | None = None,
) -> LMResult:
    """Solve the prediction error minimization problem

    Args:
        predictor: Kalman filter predictor
        data: Timeseries object containing ts, us, ys
        params_guess: initial guess for parameters
        x0: initial state (optional, defaults to None)
        bounds: tuple of lower and upper bounds for parameters (optional)
            If provided, should be a tuple of two PyTrees with the same structure
            as the parameters
        P0: initial state covariance (optional, defaults to identity)
        method: optimization method (default is "lm")
        options: additional options for the optimizer (optional)

    Returns:
        LMResult containing the optimized parameters and other information
    """
    if method not in {"lm"}:
        raise ValueError(f"Unsupported method: {method}. Only 'lm' is supported.")

    # Set sensible defaults for system identification problems
    if options is None:
        options = {}

    # Apply defaults for any missing options
    default_options = {
        "ftol": 1e-4,
        "xtol": 1e-6, 
        "gtol": 1e-6,
        "maxfev": 200
    }
    options = {**default_options, **options}

    objective = PEMObjective(
        predictor=predictor,
        data=data,
        P0=P0,
        x0=x0,
    )

    return lm_solve(objective, params_guess, bounds=bounds, **options)