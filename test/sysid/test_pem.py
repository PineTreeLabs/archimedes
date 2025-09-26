# ruff: noqa: N802, N803, N806, E741

import numpy as np
import pytest

import archimedes as arc
from archimedes import discretize, struct
from archimedes.observers import ExtendedKalmanFilter
from archimedes.sysid import Timeseries, pem

np.random.seed(0)


@struct
class CartPole:
    """CartPole model for testing system identification."""

    m1: float = 1.0  # cart mass
    m2: float = 0.3  # pole mass
    L: float = 0.5  # pole length
    g: float = 9.81  # gravity

    def dynamics(self, t, q, u=None):
        """CartPole dynamics: [x, θ, ẋ, θ̇]"""
        x, θ, ẋ, θ̇ = q
        sθ, cθ = np.sin(θ), np.cos(θ)
        τ = 0.0 if u is None else np.atleast_1d(u)[0]
        den = self.m1 + self.m2 * sθ**2
        ẍ = (self.L * self.m2 * sθ * θ̇**2 + τ + self.m2 * self.g * cθ * sθ) / den
        θ̈ = -(
            self.L * self.m2 * cθ * sθ * θ̇**2
            + τ * cθ
            + (self.m1 + self.m2) * self.g * sθ
        ) / (self.L * den)
        return np.stack([ẋ, θ̇, ẍ, θ̈])


options = {
    "bfgs": None,
    "lm": None,
    "ipopt": None,
}


def second_order_ode(t, x, u, params):
    """Second-order system ODE: ẍ + 2ζωₙẋ + ωₙ²x = ωₙ²u"""
    omega_n = params["omega_n"]
    zeta = params["zeta"]

    x1, x2 = x[0], x[1]
    u_val = u[0]

    x1_t = x2
    x2_t = -(omega_n**2) * x1 - 2 * zeta * omega_n * x2 + omega_n**2 * u_val

    return np.hstack([x1_t, x2_t])


def position_obs(t, x, u, params):
    return x[0]


# Fixture for second-order ODE data
@pytest.fixture(scope="session")
def second_order_data():
    # True system parameters
    omega_n_true = 2.0  # rad/s
    zeta_true = 0.1  # damping ratio
    params_true = {"omega_n": omega_n_true, "zeta": zeta_true}

    # Time vector
    t0, tf = 0.0, 5.0
    dt = 0.05
    ts = np.arange(t0, tf, dt)

    # Problem dimensions
    nu = 1  # input dimension (u)
    ny = 1  # output dimension (y = x₁)

    # Input signal (step input)
    us = np.ones((nu, len(ts)))

    # Initial conditions
    x0_true = np.array([0.0, 0.0])  # start at rest

    # Generate reference data
    xs_true = arc.odeint(
        second_order_ode,
        t_span=(t0, tf),
        x0=x0_true,
        args=(us[:, 0], params_true),
        t_eval=ts,
        rtol=1e-8,
        atol=1e-10,
    )

    # Add small amount of measurement noise
    noise_std = 0.01
    ys = xs_true[:1, :] + np.random.normal(0, noise_std, (ny, len(ts)))

    data = Timeseries(ts=ts, us=us, ys=ys)

    return {
        "data": data,
        "params_true": params_true,
        "x0_true": x0_true,
        "xs_true": xs_true,
        "noise_std": noise_std,
    }


class TestPEMHarmonicOscillator:
    @pytest.mark.parametrize("method", options.keys())
    def test_second_order_system(self, method, second_order_data, plot=False):
        """Test parameter recovery on a second-order damped oscillator.

        System: ẍ + 2ζωₙẋ + ωₙ²x = ωₙ²u
        State space:
            ẋ₁ = x₂
            ẋ₂ = -ωₙ²x₁ - 2ζωₙx₂ + ωₙ²u
        Parameters:
            ωₙ (natural frequency)
            ζ (damping ratio)
        """

        # Problem dimensions
        nx = 2  # state dimension (x₁, x₂)
        ny = 1  # output dimension (y = x₁)

        # Extract data from fixture
        data = second_order_data["data"]
        params_true = second_order_data["params_true"]
        omega_n_true = params_true["omega_n"]
        zeta_true = params_true["zeta"]
        x0_true = second_order_data["x0_true"]
        xs_true = second_order_data["xs_true"]
        noise_std = second_order_data["noise_std"]
        dt = data.ts[1] - data.ts[0]  # time step from data

        # Initial parameter guess (should be different from true values)
        params_guess = {"omega_n": 2.5, "zeta": 0.5}

        R = noise_std**2 * np.eye(ny)  # Measurement noise covariance
        Q = noise_std**2 * np.eye(nx)  # Process noise covariance

        # Initial state covariance (not necessary, just tests proper handling)
        P0 = np.eye(nx)

        # Set up PEM problem
        dyn = discretize(second_order_ode, dt, method="rk4")
        ekf = ExtendedKalmanFilter(dyn, position_obs, Q, R)

        # Set up reasonable bounds (not necessary for convergence, just included
        # for testing purposes)
        bounds = (
            {"omega_n": 0.0, "zeta": 0.0},  # lower bounds
            {"omega_n": 10.0, "zeta": 1.0},  # upper bounds
        )

        result = pem(
            ekf,
            data,
            params_guess,
            x0=x0_true,  # Assume initial conditions are known
            P0=P0,
            bounds=bounds,
            method=method,
            options=options[method],
        )
        params_opt = result.p

        # Validate results
        print("\nSecond-Order System ID Results:")
        print(f"True parameters: ωₙ={omega_n_true:.3f}, ζ={zeta_true:.3f}")
        print(
            f"Estimated parameters: ωₙ={params_opt['omega_n']:.3f}, "
            f"ζ={params_opt['zeta']:.3f}"
        )
        print(f"Success: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Final cost: {0.5 * np.dot(result.fun, result.fun):.2e}")

        # Validate forward simulation accuracy
        xs_pred = arc.odeint(
            second_order_ode,
            t_span=(data.ts[0], data.ts[-1]),
            x0=x0_true,
            args=(data.us[:, 0], params_opt),
            t_eval=data.ts,
            rtol=1e-8,
            atol=1e-10,
        )

        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
            ax[0].plot(data.ts, data.ys[0], label="Measured Output (y₁)")
            ax[0].plot(
                data.ts, xs_true[0], label="True Output (x₁)", c="k", linestyle="--"
            )
            ax[0].plot(data.ts, xs_pred[0], label="Final predictions", linestyle="-.")
            ax[0].legend()
            ax[0].grid()
            ax[0].set_ylabel("State prediction")
            # ax[1].plot(ts, kf_result_init["e"].T, label="Initial residuals")
            # ax[1].plot(ts, kf_result_opt["e"].T, label="Final residuals")
            ax[1].set_ylabel("Kalman residuals")
            ax[1].grid()
            ax[-1].set_xlabel("Time (s)")
            plt.show()

        # Test assertions
        assert result.success, f"Parameter estimation failed: {result.message}"

        # Check parameter recovery accuracy
        omega_n_error = abs(params_opt["omega_n"] - omega_n_true) / omega_n_true
        zeta_error = abs(params_opt["zeta"] - zeta_true) / zeta_true

        assert omega_n_error < 0.01, (
            f"Natural frequency error too large: {100 * omega_n_error:.6f} %"
        )
        assert zeta_error < 0.1, (
            f"Damping ratio error too large: {100 * zeta_error:.6f} %"
        )

        simulation_error = np.sqrt(np.mean((xs_true - xs_pred) ** 2))
        print(f"Forward simulation RMS error: {simulation_error:.2e}")

        assert simulation_error < 0.05, (
            f"Forward simulation error too large: {simulation_error:.6f}"
        )

        # Test convergence performance
        assert result.nit < 50, f"Too many iterations required: {result.nit}"
        final_cost = 0.5 * np.dot(result.fun, result.fun) / len(data)
        assert final_cost / len(data) < 1e-3, f"Final cost too high: {result.fun:.2e}"

    @pytest.mark.parametrize("method", options.keys())
    def test_multi_experiment(self, method, second_order_data):
        # Problem dimensions
        nx = 2  # state dimension (x₁, x₂)
        ny = 1  # output dimension (y = x₁)

        # Extract data from fixture
        data = second_order_data["data"]
        params_true = second_order_data["params_true"]
        omega_n_true = params_true["omega_n"]
        zeta_true = params_true["zeta"]
        x0_true = second_order_data["x0_true"]
        noise_std = second_order_data["noise_std"]
        dt = data.ts[1] - data.ts[0]  # time step from data

        # Initial parameter guess (should be different from true values)
        params_guess = {"omega_n": 2.5, "zeta": 0.5}

        R = noise_std**2 * np.eye(ny)  # Measurement noise covariance
        Q = noise_std**2 * np.eye(nx)  # Process noise covariance

        # Set up PEM problem
        dyn = discretize(second_order_ode, dt, method="rk4")
        ekf = ExtendedKalmanFilter(dyn, position_obs, Q, R)

        result = pem(
            ekf,
            [data, data],
            params_guess,
            x0=[x0_true, x0_true],
            method=method,
            options=options[method],
        )
        params_opt = result.p

        # Test assertions
        assert result.success, f"Parameter estimation failed: {result.message}"

        # Check parameter recovery accuracy
        omega_n_error = abs(params_opt["omega_n"] - omega_n_true) / omega_n_true
        zeta_error = abs(params_opt["zeta"] - zeta_true) / zeta_true

        assert omega_n_error < 0.01, (
            f"Natural frequency error too large: {100 * omega_n_error:.6f} %"
        )
        assert zeta_error < 0.1, (
            f"Damping ratio error too large: {100 * zeta_error:.6f} %"
        )

    @pytest.mark.parametrize("method", options.keys())
    def test_optimize_ics(self, method, second_order_data):
        # Problem dimensions
        nx = 2  # state dimension (x₁, x₂)
        ny = 1  # output dimension (y = x₁)

        # Extract data from fixture
        data = second_order_data["data"]
        params_true = second_order_data["params_true"]
        omega_n_true = params_true["omega_n"]
        zeta_true = params_true["zeta"]
        x0_true = second_order_data["x0_true"]
        noise_std = second_order_data["noise_std"]
        dt = data.ts[1] - data.ts[0]  # time step from data

        # Initial parameter guess (should be different from true values)
        params_guess = {"omega_n": 2.5, "zeta": 0.5}

        # Set up PEM problem
        R = noise_std**2 * np.eye(ny)  # Measurement noise covariance
        Q = (0.01 * noise_std) ** 2 * np.eye(nx)  # Process noise covariance

        dyn = discretize(second_order_ode, dt, method="rk4")
        ekf = ExtendedKalmanFilter(dyn, position_obs, Q, R)

        # Test optimizing initial conditions
        result = pem(
            ekf,
            data,
            params_guess,
            x0=np.array([1.0, 0.0]),
            estimate_x0=True,
            method=method,
            options=options[method],
        )
        x0_est, params_est = result.x0, result.p
        print(f"Estimated initial conditions: {x0_est}")
        print(f"Estimated parameters with x0: {params_est}")

        assert result.success, f"Parameter estimation with x0 failed: {result.message}"

        # Check parameter recovery accuracy (should be good for this clean problem)
        omega_n_error = abs(params_est["omega_n"] - omega_n_true)
        zeta_error = abs(params_est["zeta"] - zeta_true)

        assert omega_n_error < 0.02, (
            f"Natural frequency error too large: {omega_n_error:.6f}"
        )
        assert zeta_error < 0.01, f"Damping ratio error too large: {zeta_error:.6f}"

        assert np.allclose(x0_est[0], x0_true[0], atol=0.1), (
            f"Initial condition error too large: {np.abs(x0_est - x0_true)}"
        )

        # Error handling
        with pytest.raises(ValueError, match=r"Unsupported method.*"):
            pem(
                ekf,
                data,
                params_guess,
                x0=x0_true,
                method="unsupported_method",
            )


class TestPEMVanDerPol:
    def test_van_der_pol(self, plot=False):
        """Test parameter recovery on Van der Pol oscillator (nonlinear system).

        System: ẍ - μ(1-x²)ẋ + x = 0
        State space:
            ẋ₁ = x₂
            ẋ₂ = μ(1-x₁²)x₂ - x₁
        Parameters:
            μ (nonlinearity parameter)
        """
        # True system parameters
        mu_true = 0.5  # Moderate nonlinearity for good convergence
        params_true = {"mu": mu_true}

        # Time vector (shorter simulation for limit cycle development)
        t0, tf = 0.0, 15.0
        dt = 0.05
        ts = np.arange(t0, tf, dt)

        # Problem dimensions
        nx = 2  # state dimension (x₁, x₂)
        ny = 1  # output dimension (y = x₁)

        # No input signal (autonomous system)
        us = np.zeros((1, len(ts)))  # dummy input

        # Initial conditions (start away from equilibrium to excite dynamics)
        x0_true = np.array([2.0, 0.0])  # initial displacement

        # Generate true system response
        def van_der_pol_ode(t, x, u, params):
            """Van der Pol oscillator: ẍ - μ(1-x²)ẋ + x = 0"""
            mu = params["mu"]

            x1, x2 = x[0], x[1]
            # u is unused for autonomous system

            x1_t = x2
            x2_t = mu * (1 - x1**2) * x2 - x1

            return np.hstack([x1_t, x2_t])

        def obs(t, x, u, params):
            return x[0]  # observe position only

        # Generate reference data
        xs_true = arc.odeint(
            van_der_pol_ode,
            t_span=(t0, tf),
            x0=x0_true,
            args=(us[:, 0], params_true),
            t_eval=ts,
            rtol=1e-8,
            atol=1e-10,
        )

        # Add measurement noise
        noise_std = 0.01
        ys = xs_true[:1, :] + np.random.normal(0, noise_std, (ny, len(ts)))

        # Initial parameter guess (should be different from true value)
        params_guess = {"mu": 1.0}  # Start with higher nonlinearity

        R = noise_std**2 * np.eye(ny)  # Measurement noise covariance
        Q = noise_std**2 * np.eye(nx)  # Process noise covariance (scale with R)

        P0 = 1e-4 * np.eye(nx)  # Initial state covariance (small uncertainty)

        # Set up PEM problem
        dyn = discretize(van_der_pol_ode, dt, method="rk4")
        ekf = ExtendedKalmanFilter(dyn, obs, Q, R)
        data = Timeseries(ts=ts, us=us, ys=ys)
        result = pem(
            ekf,
            data,
            params_guess,
            x0=x0_true,  # Assume initial conditions are known
            method="bfgs",
        )
        params_opt = result.p

        # Validate results
        print("\nVan der Pol Oscillator ID Results:")
        print(f"True parameter: μ={mu_true:.3f}")
        print(f"Estimated parameter: μ={params_opt['mu']:.3f}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Final cost: {result.fun:.2e}")

        # Validate forward simulation accuracy
        xs_pred = arc.odeint(
            van_der_pol_ode,
            t_span=(t0, tf),
            x0=x0_true,
            args=(us[:, 0], params_opt),
            t_eval=ts,
            rtol=1e-8,
            atol=1e-10,
        )

        if plot:
            import matplotlib.pyplot as plt

            pem_obj = arc.sysid.PEMObjective(ekf, data, P0, x0=x0_true)

            kf_result_init = pem_obj.forward(x0_true, params_guess)
            kf_result_opt = pem_obj.forward(x0_true, params_opt)

            fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

            # State trajectory
            ax[0].plot(ts, ys[0], label="Measured Output (x₁)", alpha=0.7)
            ax[0].plot(ts, xs_true[0], label="True Output (x₁)", c="k", linestyle="--")
            ax[0].plot(
                ts, kf_result_init["x_hat"][0], label="KF Estimate (x₁)", linestyle=":"
            )
            ax[0].plot(ts, xs_pred[0], label="Final predictions", linestyle="-.")
            ax[0].legend()
            ax[0].grid()
            ax[0].set_ylabel("Position (x₁)")
            ax[0].set_title("Van der Pol Oscillator - State Trajectories")

            # Kalman residuals
            ax[1].plot(ts, kf_result_init["e"].T, label="Initial residuals", alpha=0.7)
            ax[1].plot(ts, kf_result_opt["e"].T, label="Final residuals", alpha=0.7)
            ax[1].set_ylabel("Kalman residuals")
            ax[1].legend()
            ax[1].grid()

            # Phase portrait
            ax[2].plot(
                xs_true[0],
                xs_true[1],
                label="True phase portrait",
                c="k",
                linestyle="--",
            )
            ax[2].plot(
                xs_pred[0], xs_pred[1], label="Predicted phase portrait", linestyle="-."
            )
            ax[2].set_xlabel("Position (x₁)")
            ax[2].set_ylabel("Velocity (x₂)")
            ax[2].legend()
            ax[2].grid()
            ax[2].set_title("Phase Portrait")

            plt.tight_layout()
            plt.show()

        # Test assertions
        assert result.success, f"Parameter estimation failed: {result.message}"

        # Check parameter recovery accuracy
        # Note: Nonlinear systems are typically harder to identify than linear ones
        mu_error = abs(params_opt["mu"] - mu_true)

        assert mu_error < 0.05, (
            f"Nonlinearity parameter error too large: {mu_error:.6f}"
        )

        # Check simulation accuracy (allow larger error for nonlinear system)
        simulation_error = np.sqrt(np.mean((xs_true - xs_pred) ** 2))
        print(f"Forward simulation RMS error: {simulation_error:.2e}")

        assert simulation_error < 0.1, (
            f"Forward simulation error too large: {simulation_error:.6f}"
        )

        # Test convergence performance (allow more iterations for nonlinear system)
        assert result.nit < 100, f"Too many iterations required: {result.nit}"
        assert result.fun < 1e-2, f"Final cost too high: {result.fun:.2e}"


class TestPEMCartPole:
    def test_cartpole(self, plot=False):
        """Test parameter recovery on CartPole system (nonlinear underactuated system).

        System: CartPole with cart mass m1, pole mass m2, and pole length L
        State: [x, θ, ẋ, θ̇] (cart position, pole angle, velocities)
        Input: u (horizontal force on cart)
        Parameters to identify:
            m1 (cart mass)
            m2 (pole mass)
            L (pole length)
        Fixed parameter:
            g (gravity - assumed known)
        """
        # True system parameters (to be identified)
        m1_true = 1.0  # cart mass (kg)
        m2_true = 0.3  # pole mass (kg)
        L_true = 0.5  # pole length (m)
        g_true = 9.81  # gravity (m/s²) - assumed known

        params_true = {"m2": m2_true, "L": L_true}
        system = CartPole(
            **params_true, m1=m1_true, g=g_true
        )  # Create true system instance

        # Time vector (shorter simulation for challenging nonlinear system)
        t0, tf = 0.0, 3.0
        dt = 0.01  # Fine timestep for accurate simulation
        ts = np.arange(t0, tf, dt)

        # Problem dimensions
        nx = 4  # state dimension [x, θ, ẋ, θ̇]
        nu = 1  # input dimension (force on cart)
        ny = 4  # output dimension

        # Input signal
        us = np.zeros((nu, len(ts)))

        # Initial conditions (start near inverted equilibrium)
        x0_true = np.array([0.0, 0.1, 0.0, 0.0])  # Small initial pole angle

        # Generate true system response
        def cartpole_ode(t, x, u, params):
            return system.replace(**params).dynamics(t, x, u)

        def obs(t, x, u, params):
            return x

        def ode_rhs(t, x, params):
            """ODE with interpolated input."""
            u = np.interp(t, ts, us[0]).reshape((nu,))
            return cartpole_ode(t, x, u, params)

        # Generate reference data
        xs_true = arc.odeint(
            ode_rhs,
            t_span=(t0, tf),
            x0=x0_true,
            args=(params_true,),
            t_eval=ts,
            rtol=1e-8,
            atol=1e-10,
        )

        # Add measurement noise
        noise_std = 0.01
        ys = xs_true + np.random.normal(0, noise_std, (ny, len(ts)))

        # Initial parameter guess
        params_guess = {
            "m2": 0.5,
            "L": 1.0,
        }

        R = noise_std**2 * np.eye(ny)  # Measurement noise covariance
        Q = noise_std**2 * np.eye(nx)  # Process noise (smaller than measurement)

        # Set up PEM problem
        dyn = discretize(cartpole_ode, dt, method="rk4")
        ekf = ExtendedKalmanFilter(dyn, obs, Q, R)
        data = Timeseries(ts=ts, us=us, ys=ys)
        result = pem(
            ekf,
            data,
            params_guess,
            x0=x0_true,  # Assume initial conditions are known
            method="bfgs",
        )
        params_opt = result.p

        # Validate results
        print("\nCartPole System ID Results:")
        print(f"True parameters: m1={m1_true:.3f}, m2={m2_true:.3f}, L={L_true:.3f}")
        print(
            f"Estimated parameters: m2={params_opt['m2']:.3f}, L={params_opt['L']:.3f}"
        )
        print(f"Success: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Final cost: {result.fun:.2e}")

        # Validate forward simulation accuracy
        xs_pred = arc.odeint(
            ode_rhs,
            t_span=(t0, tf),
            x0=x0_true,
            args=(params_opt,),
            t_eval=ts,
            rtol=1e-8,
            atol=1e-10,
        )

        if plot:
            import matplotlib.pyplot as plt

            P0 = 1e-4 * np.eye(nx)  # Initial state covariance (small uncertainty)
            pem_obj = arc.sysid.PEMObjective(ekf, data, P0, x0=x0_true)

            kf_result_init = pem_obj.forward(x0_true, params_guess)
            kf_result_opt = pem_obj.forward(x0_true, params_opt)

            fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

            # Cart position
            ax[0].plot(ts, ys[0], label="Measured Cart Position", alpha=0.7)
            ax[0].plot(
                ts, xs_true[0], label="True Cart Position", c="k", linestyle="--"
            )
            ax[0].plot(ts, xs_pred[0], label="Predicted Cart Position", linestyle="-.")
            ax[0].set_ylabel("Cart Position (m)")
            ax[0].legend()
            ax[0].grid()
            ax[0].set_title("CartPole System Identification Results")

            # Pole angle
            ax[1].plot(ts, ys[1], label="Measured Pole Angle", alpha=0.7)
            ax[1].plot(ts, xs_true[1], label="True Pole Angle", c="k", linestyle="--")
            ax[1].plot(ts, xs_pred[1], label="Predicted Pole Angle", linestyle="-.")
            ax[1].set_ylabel("Pole Angle (rad)")
            ax[1].legend()
            ax[1].grid()

            # Control input
            ax[2].plot(ts, us[0], label="Control Force", c="orange")
            ax[2].set_ylabel("Force (N)")
            ax[2].legend()
            ax[2].grid()

            # Kalman residuals
            ax[3].plot(
                ts, kf_result_init["e"][0], label="Initial residuals (x)", alpha=0.7
            )
            ax[3].plot(
                ts, kf_result_opt["e"][0], label="Final residuals (x)", alpha=0.7
            )
            ax[3].plot(
                ts,
                kf_result_init["e"][1],
                label="Initial residuals (θ)",
                alpha=0.7,
                linestyle=":",
            )
            ax[3].plot(
                ts,
                kf_result_opt["e"][1],
                label="Final residuals (θ)",
                alpha=0.7,
                linestyle=":",
            )
            ax[3].set_ylabel("Kalman Residuals")
            ax[3].set_xlabel("Time (s)")
            ax[3].legend()
            ax[3].grid()

            plt.tight_layout()
            plt.show()

        # Test assertions
        assert result.success, f"Parameter estimation failed: {result.message}"

        # Check parameter recovery accuracy
        # CartPole is challenging due to underactuation and nonlinearity
        m2_error = abs(params_opt["m2"] - m2_true) / m2_true
        L_error = abs(params_opt["L"] - L_true) / L_true

        print(f"Relative parameter errors: m2={m2_error:.3f}, L={L_error:.3f}")

        # Allow reasonable tolerances for this challenging system
        assert m2_error < 0.05, f"Pole mass error too large: {m2_error:.6f}"
        assert L_error < 0.05, f"Pole length error too large: {L_error:.6f}"

        # Check simulation accuracy (focus on observed states)
        obs_error = np.sqrt(np.mean((xs_true[:2] - xs_pred[:2]) ** 2))
        print(f"Observed states RMS error: {obs_error:.3e}")

        assert obs_error < 0.05, f"Simulation error too large: {obs_error:.6f}"

        # Test convergence performance
        assert result.nit < 50, f"Too many iterations required: {result.nit}"
        assert result.fun < 1e-1, f"Final cost too high: {result.fun:.2e}"


if __name__ == "__main__":
    # Run individual tests for debugging

    print("=" * 60)
    print("Running System Identification Tests")
    print("=" * 60)

    for i in range(10):
        print(f"Iteration {i + 1}")
        np.random.seed(i)

        print("\n" + "=" * 60)
        print("Running Second Order System Identification Test")
        print("=" * 60)
        TestPEMHarmonicOscillator().test_second_order_system(plot=False)

    # test_suite.test_second_order_system(plot=True)

    # print("\n" + "=" * 60)
    # print("Running Van der Pol Oscillator Identification Test")
    # print("=" * 60)

    # test_suite.test_van_der_pol(plot=True)

    # print("\n" + "=" * 60)
    # print("Running Duffing Oscillator Identification Test")
    # print("=" * 60)

    # test_suite.test_duffing(plot=True)

    # print("\n" + "=" * 60)
    # print("Running CartPole System Identification Test")
    # print("=" * 60)

    # test_suite.test_cartpole(plot=True)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
