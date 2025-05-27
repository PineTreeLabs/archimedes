import numpy as np
import pytest
import archimedes as arc

from archimedes.experimental.sysid import make_pem, lm_solve
from archimedes.experimental.discretize import discretize


np.random.seed(0)


class TestPEMIntegration:
    """Integration tests for system identification using PEM + LM solver."""

    def test_second_order_sysid(self, plot=False):
        """Test parameter recovery on a second-order damped oscillator.
        
        System: ẍ + 2ζωₙẋ + ωₙ²x = ωₙ²u
        State space: ẋ₁ = x₂, ẋ₂ = -ωₙ²x₁ - 2ζωₙx₂ + ωₙ²u
        Parameters: ωₙ (natural frequency), ζ (damping ratio)
        """
        # True system parameters
        omega_n_true = 2.0  # rad/s
        zeta_true = 0.1     # damping ratio
        params_true = {"omega_n": omega_n_true, "zeta": zeta_true}
        
        # Time vector
        t0, tf = 0.0, 10.0
        dt = 0.05
        ts = np.arange(t0, tf, dt)

        # Problem dimensions
        nx = 2  # state dimension (x₁, x₂)
        nu = 1  # input dimension (u)
        ny = 1  # output dimension (y = x₁)
        
        # Input signal (step input)
        us = np.ones((nu, len(ts)))
        
        # Initial conditions
        x0_true = np.array([0.0, 0.0])  # start at rest
    
        # Generate true system response
        def second_order_ode(t, x, u, params):
            """Second-order system ODE: ẍ + 2ζωₙẋ + ωₙ²x = ωₙ²u"""
            omega_n = params["omega_n"]
            zeta = params["zeta"]
            
            x1, x2 = x[0], x[1]
            u_val = u[0]
            
            x1_t = x2
            x2_t = -omega_n**2 * x1 - 2*zeta*omega_n*x2 + omega_n**2 * u_val
            
            return np.hstack([x1_t, x2_t])

        def obs(t, x, u, params):
            return x[0]
        
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
        
        # Initial parameter guess (should be different from true values)
        params_guess = {"omega_n": 2.5, "zeta": 0.5}

        R = noise_std ** 2 * np.eye(ny)  # Measurement noise covariance
        Q = 1e-4 * np.eye(nx)  # Process noise covariance
        
        # Set up PEM problem
        dyn = discretize(second_order_ode, dt, method="rk4")
        pem_obj = make_pem(
            dyn=dyn,
            obs=obs,
            ts=ts,
            us=us,
            ys=ys,
            Q=Q,
            R=R,
            x0=x0_true,  # Assume initial conditions are known
        )
        
        # Solve using LM
        result = lm_solve(
            pem_obj,
            params_guess,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            nprint=1,
        )
        
        # Validate results
        print(f"\nSecond-Order System ID Results:")
        print(f"True parameters: ωₙ={omega_n_true:.3f}, ζ={zeta_true:.3f}")
        print(f"Estimated parameters: ωₙ={result.x['omega_n']:.3f}, ζ={result.x['zeta']:.3f}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Final cost: {result.fun:.2e}")


        # Validate forward simulation accuracy
        xs_pred = arc.odeint(
            second_order_ode,
            t_span=(t0, tf),
            x0=x0_true,
            args=(us[:, 0], result.x),
            t_eval=ts,
            rtol=1e-8,
            atol=1e-10,
        )

        if plot:
            import matplotlib.pyplot as plt
            kf_result = pem_obj.forward(x0_true, params_guess)
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax[0].plot(ts, ys[0], label="Measured Output (y₁)")
            ax[0].plot(ts, xs_true[0], label="True Output (x₁)", linestyle='--')
            ax[0].plot(ts, kf_result["x_hat"][0], label="KF Estimate (x₁)", linestyle=':')
            ax[0].plot(ts, xs_pred[0], label="Predicted Output (x₁)", linestyle='-.')
            ax[0].legend()
            ax[0].grid()
            ax[0].set_ylabel("State prediction")
            ax[1].plot(ts, kf_result["e"].T)
            ax[1].set_ylabel("Estimation Error")
            ax[1].grid()
            ax[-1].set_xlabel("Time (s)")
            plt.show()
        
        # Test assertions
        assert result.success, f"Parameter estimation failed: {result.message}"
        
        # Check parameter recovery accuracy (should be quite good for this clean problem)
        omega_n_error = abs(result.x["omega_n"] - omega_n_true)
        zeta_error = abs(result.x["zeta"] - zeta_true)
        
        assert omega_n_error < 0.01, f"Natural frequency error too large: {omega_n_error:.6f}"
        assert zeta_error < 0.01, f"Damping ratio error too large: {zeta_error:.6f}"
        
        simulation_error = np.sqrt(np.mean((xs_true - xs_pred)**2))
        print(f"Forward simulation RMS error: {simulation_error:.2e}")
        
        assert simulation_error < 0.05, f"Forward simulation error too large: {simulation_error:.6f}"
        
        # Test convergence performance
        assert result.nit < 50, f"Too many iterations required: {result.nit}"
        assert result.fun < 1e-3, f"Final cost too high: {result.fun:.2e}"

        # Compare objective values at different initial guesses
        params_good = {"omega_n": 2.5, "zeta": 0.5}  # Works
        params_bad = {"omega_n": 3.0, "zeta": 0.5}   # Fails

        print(f"Objective at good guess: {pem_obj(params_good)[0]:.2e}")
        print(f"Objective at bad guess:  {pem_obj(params_bad)[0]:.2e}")

        # Also check gradient magnitudes
        print(f"Gradient norm at good guess: {np.linalg.norm(pem_obj(params_good)[1]):.2e}")
        print(f"Gradient norm at bad guess:  {np.linalg.norm(pem_obj(params_bad)[1]):.2e}")


if __name__ == "__main__":
    # Run individual tests for debugging
    test_suite = TestPEMIntegration()
    
    print("=" * 60)
    print("Running Second-Order System Identification Tests")
    print("=" * 60)
    
    test_suite.test_second_order_sysid(plot=True)
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)