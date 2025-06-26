# ruff: noqa: N802, N803, N806, N815, N816

import numpy as np
from scipy.signal import chirp

import archimedes as arc

np.random.seed(0)


def generate_linear_oscillator():
    omega_n = 2.0  # Natural frequency [rad/s]
    zeta = 0.1  # Damping ratio [-]

    # Problem dimensions
    nx = 2  # state dimension (x₁, x₂)
    nu = 1  # input dimension (u)
    ny = 1  # output dimension (y = x₁)

    # Time vector
    t0, tf = 0.0, 20.0
    dt = 0.05
    ts = np.arange(t0, tf, dt)

    us = np.ones((nu, len(ts)))  # Constant input (step)

    def ode_rhs(t, x):
        x1, x2 = x[0], x[1]
        u = np.interp(t, ts, us[0, :])  # Interpolate input at time t

        x1_t = x2
        x2_t = -omega_n**2 * x1 - 2*zeta*omega_n*x2 + omega_n**2 * u

        return np.hstack([x2, x2_t])

    # Step response
    xs = arc.odeint(
        ode_rhs,
        t_span=(t0, tf),
        x0=np.zeros(nx),
        t_eval=ts,
        rtol=1e-8,
        atol=1e-10,
    )

    # Add measurement noise
    noise_std = 0.05
    ys = xs[:1, :] + np.random.normal(0, noise_std, (ny, len(ts)))

    data = np.vstack([ts.reshape(1, -1), us, ys])
    np.savetxt(
        "data/oscillator.csv",
        data.T,
        delimiter="\t",
        header="time\t\tu\t\t\ty",
        comments="",
        fmt="%.6f",
    )

    truth = {
        "omega_n": omega_n,
        "zeta": zeta,
        "noise_std": noise_std,
        "x0": np.zeros(nx),
    }
    np.savez(
        "data/oscillator_truth.npz",
        **truth,
    )


if __name__ == "__main__":
    generate_linear_oscillator()
    print("Data generated")
