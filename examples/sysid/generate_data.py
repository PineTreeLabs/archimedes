# ruff: noqa: N802, N803, N806, N815, N816

import numpy as np
from scipy.signal import chirp

import archimedes as arc

from models import (
    TransferFunction,
    HammersteinWiener,
    TanhNonlinearity,
    CubicNonlinearity,
)

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
        x2_t = -(omega_n**2) * x1 - 2 * zeta * omega_n * x2 + omega_n**2 * u

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


def generate_duffing_oscillator():
    alpha = 1.0  # linear stiffness
    beta = 5.0  # nonlinear stiffness
    delta = 0.02  # damping

    # Forcing parameters
    gamma = 8.0  # forcing amplitude
    omega = 0.5  # forcing frequency

    # Problem dimensions
    nx = 2  # state dimension (x₁, x₂)
    nu = 1  # input dimension (u)
    ny = 1  # output dimension (y = x₁)

    def u(t):
        """Forcing function: γcos(ωt)"""
        return gamma * np.cos(omega * t)

    def ode_rhs(t, x):
        """Duffing oscillator: ẍ + δẋ + αx + βx³ = γcos(ωt)"""
        x1, x2 = x[0], x[1]
        x2_t = -delta * x2 - alpha * x1 - beta * x1**3 + u(t)
        return np.hstack([x2, x2_t])

    t0, tf = 0.0, 40.0  # Longer simulation to capture rich dynamics
    dt = 0.02  # Smaller timestep for nonlinear system
    ts = np.arange(t0, tf, dt)

    x0 = np.array([1.0, 0.5])

    # Generate reference data
    xs = arc.odeint(
        ode_rhs,
        t_span=(t0, tf),
        x0=x0,
        t_eval=ts,
    )

    # Add measurement noise
    noise_std = 0.01
    ys = xs[:1, :] + np.random.normal(0, noise_std, (ny, len(ts)))
    us = u(ts).reshape(1, -1)  # Reshape input to match dimensions

    data = np.vstack([ts.reshape(1, -1), us, ys])
    np.savetxt(
        "data/duffing.csv",
        data.T,
        delimiter="\t",
        header="time\t\tu\t\t\ty",
        comments="",
        fmt="%.6f",
    )

    truth = {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "gamma": gamma,
        "omega": omega,
        "noise_std": noise_std,
        "x0": x0,
    }
    np.savez(
        "data/duffing_truth.npz",
        **truth,
    )


def generate_hammerstein_wiener():
    # Transfer function with zero and mixed pole structure
    # G(s) = K * (τ_z*s + 1) / ((τ_slow*s + 1) * (s² + 2*ζ*ω_n*s + ω_n²))

    K = 20.0  # DC gain
    tau_z = 0.5  # Zero time constant (lead compensation character)
    tau_slow = 2.0  # Slow pole time constant
    omega_n = 6.0  # Natural frequency of oscillatory pair
    zeta = 0.2  # Light damping (ζ = 0.3 gives nice oscillation)

    # Resulting transfer function:
    # G(s) = 2 * (0.5*s + 1) / ((10*s + 1) * (s² + 1.2*s + 4))

    num = np.array([K * tau_z, K])
    den = np.array(
        [
            tau_slow,
            1 + 2 * tau_slow * zeta * omega_n,
            tau_slow * omega_n**2 + 2 * zeta * omega_n,
            omega_n**2,
        ]
    )

    # Problem dimensions
    nx = len(den) - 1  # Number of states (order of the system)
    nu = 1  # Single input
    ny = 1  # Single output

    hw_sys = HammersteinWiener(
        input=TanhNonlinearity(saturation=1.5),
        lin_sys=TransferFunction(num, den),
        output=CubicNonlinearity(coeff=0.05),
    )
    noise_std = 0.01

    def hw_ode(t, x, u, sys):
        return sys.dynamics(t, x, u)

    def obs(t, x, u, sys):
        return sys.observation(t, x, u)

    vmap_obs = arc.vmap(obs, in_axes=(0, 1, 1, None))

    def simulate_system(sys, ts, us, x0, noise_std=0.0):
        """Simulate the Hammerstein-Wiener system."""

        def ode_rhs(t, x, sys):
            u = np.interp(t, ts, us[0]).reshape((nu,))
            return hw_ode(t, x, u, sys)

        xs_true = arc.odeint(
            ode_rhs,
            t_span=(ts[0], ts[-1]),
            x0=x0,
            args=(sys,),
            t_eval=ts,
            rtol=1e-8,
            atol=1e-10,
        )

        ys_ideal = vmap_obs(ts, xs_true, us, sys).T
        ys = ys_ideal + np.random.normal(0, noise_std, ys_ideal.shape)

        return arc.sysid.Timeseries(
            ts=ts,
            us=us,
            ys=ys,
        )

    # 1. Ladder response for steady-state

    # Time vector
    t0, tf = 0.0, 100.0
    dt = 0.1
    ts = np.arange(t0, tf, dt)

    u_ss = np.linspace(-5, 5, 10, endpoint=True)
    us = np.zeros((nu, len(ts)))  # Zero input
    for i, u in enumerate(u_ss):
        us[:, ts > i * 10.0] = u

    ladder_data = simulate_system(hw_sys, ts, us, x0=np.zeros(nx), noise_std=noise_std)

    np.savetxt(
        "data/part3_ladder.csv",
        np.vstack([ladder_data.ts, ladder_data.us, ladder_data.ys]).T,
        delimiter="\t",
        header="time\t\tu\t\t\ty",
        comments="",
        fmt="%.6f",
    )

    # 2. Step response for transient dynamics
    # Time vector
    t0, tf = 0.0, 10.0
    dt = 0.02
    ts = np.arange(t0, tf, dt)

    u0 = 2.0
    us = u0 * np.ones((nu, len(ts)))  # Constant input
    us[:, ts > 10.0] *= -u0  # Step down after 10 seconds
    step_data = simulate_system(hw_sys, ts, us, x0=np.zeros(nx), noise_std=noise_std)

    np.savetxt(
        "data/part3_step.csv",
        np.vstack([step_data.ts, step_data.us, step_data.ys]).T,
        delimiter="\t",
        header="time\t\tu\t\t\ty",
        comments="",
        fmt="%.6f",
    )

    # 3. Chirp input for frequency response
    def dyn_res(x, u):
        return hw_sys.dynamics(0, x, u)

    u0 = 1.0
    x0_chirp = arc.root(dyn_res, x0=np.zeros(nx), args=(u0,))

    us = u0 + 0.2 * chirp(ts, f0=0.01, f1=3.0, t1=tf, method="quadratic")
    chirp_data = simulate_system(
        hw_sys, ts, us.reshape((nu, -1)), x0=x0_chirp, noise_std=noise_std
    )

    np.savetxt(
        "data/part3_chirp.csv",
        np.vstack([chirp_data.ts, chirp_data.us, chirp_data.ys]).T,
        delimiter="\t",
        header="time\t\tu\t\t\ty",
        comments="",
        fmt="%.6f",
    )


if __name__ == "__main__":
    generate_linear_oscillator()
    print("Data generated")

    generate_duffing_oscillator()
    print("Data generated")

    generate_hammerstein_wiener()
    print("Hammerstein-Wiener data generated")
