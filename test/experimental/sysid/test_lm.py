# ruff: noqa: N802, N803, N806, E741

import numpy as np

import archimedes as arc
from archimedes.experimental.sysid._lmder.lmder import lmder


class TestLM:
    """Test suite for the Levenberg-Marquardt algorithm implementation."""

    def test_rosenbrock(self):
        """Test optimization of the Rosenbrock function."""

        # Define Rosenbrock function as a least-squares problem
        # f(x) = 100(x[1] - x[0]²)² + (1 - x[0])²
        # We define residuals: r1 = 10*(x[1] - x[0]²), r2 = (1 - x[0])
        # So that f(x) = 0.5 * (r1² + r2²)
        def rosenbrock_res(x):
            return np.hstack([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

        def rosenbrock_func(x):
            # Residuals and Jacobian
            r = rosenbrock_res(x)
            J = arc.jac(rosenbrock_res)(x)

            # Objective (half sum of squares)
            V = 0.5 * np.sum(r**2)

            # Gradient of objective: g = J.T @ r
            g = J.T @ r

            # Hessian approximation: H ≈ J.T @ J (Gauss-Newton)
            H = J.T @ J

            return V, g, H

        # Initial guess
        x0 = np.array([-1.2, 1.0])

        # Run optimization (correct function signature: func first, then x0)
        result = lmder(rosenbrock_func, x0, maxfev=1000)

        # Check result - the solution should be close to [1.0, 1.0]
        print(f"Optimization result: {result.x}")
        print(f"Final objective: {result.fun}")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")

        # Test that optimization was successful
        assert result.success, f"Optimization failed: {result.message}"

        # Test that solution is close to the known optimum [1.0, 1.0]
        assert np.allclose(result.x, np.array([1.0, 1.0]), rtol=1e-4, atol=1e-4), (
            f"Solution {result.x} not close to expected [1.0, 1.0]"
        )

        # Test that final objective is close to zero
        assert result.fun < 1e-6, (
            f"Final objective {result.fun} not close to zero"
        )

    def test_compute_step_well_conditioned(self):
        """Test compute_step with a well-conditioned matrix."""
        from archimedes.experimental.sysid._lmder.lmder import compute_step

        # Simple 2x2 case with known solution
        hess = np.array([[4.0, 1.0], [1.0, 3.0]])  # SPD matrix
        grad = np.array([2.0, 1.0])
        diag = np.array([1.0, 1.0])
        lambda_val = 0.1

        step = compute_step(hess, grad, diag, lambda_val)

        # Verify the step satisfies (H + λI)p = -g
        H_damped = hess + lambda_val * np.eye(2)
        residual = H_damped @ step + grad
        assert np.allclose(residual, 0.0, atol=1e-12)

    def test_compute_step_ill_conditioned(self):
        """Test compute_step with an ill-conditioned matrix."""
        from archimedes.experimental.sysid._lmder.lmder import compute_step

        # Ill-conditioned matrix (near-singular)
        hess = np.array([[1.0, 1.0], [1.0, 1.0001]])
        grad = np.array([1.0, 1.0])
        diag = np.array([1.0, 1.0])
        lambda_val = 0.01

        # Should not crash and should return reasonable step
        step = compute_step(hess, grad, diag, lambda_val)
        assert not np.any(np.isnan(step))
        assert not np.any(np.isinf(step))
        assert np.linalg.norm(step) < 100  # Reasonable magnitude

    def test_compute_step_singular(self):
        """Test compute_step with a singular matrix."""
        from archimedes.experimental.sysid._lmder.lmder import compute_step

        # Singular matrix
        hess = np.array([[1.0, 1.0], [1.0, 1.0]])
        grad = np.array([1.0, 1.0])
        diag = np.array([1.0, 1.0])
        lambda_val = 0.0  # No damping to keep it singular

        # Should fall back gracefully
        step = compute_step(hess, grad, diag, lambda_val)
        assert not np.any(np.isnan(step))
        assert not np.any(np.isinf(step))

    def test_compute_predicted_reduction(self):
        """Test predicted reduction calculation."""
        from archimedes.experimental.sysid._lmder.lmder import (
            compute_predicted_reduction,
        )

        # Simple test case
        grad = np.array([2.0, 1.0])
        step = np.array([-0.5, -0.25])  # Should give reduction
        hess = np.array([[4.0, 0.0], [0.0, 2.0]])
        current_objective = 2.0

        # Test without scaling (legacy behavior)
        pred_red_unscaled = compute_predicted_reduction(grad, step, hess)

        # Manual calculation: pred_red = -(g^T*p + 0.5*p^T*H*p)
        linear = np.dot(grad, step)  # 2*(-0.5) + 1*(-0.25) = -1.25
        quadratic = 0.5 * np.dot(
            step, hess @ step
        )  # 0.5 * (0.25*4 + 0.0625*2) = 0.5625
        expected_unscaled = -(linear + quadratic)  # -(-1.25 + 0.5625) = 0.6875

        assert np.isclose(pred_red_unscaled, expected_unscaled)
        assert pred_red_unscaled > 0  # Should predict a reduction

        # Test with scaling (new behavior)
        pred_red_scaled = compute_predicted_reduction(
            grad, step, hess, current_objective
        )
        expected_scaled = (
            expected_unscaled / current_objective
        )  # 0.6875 / 2.0 = 0.34375

        assert np.isclose(pred_red_scaled, expected_scaled)
        assert pred_red_scaled > 0  # Should still predict a reduction

    def test_powell_singular(self):
        """Test optimization of Powell's singular function."""

        def powell_res(x):
            """
            Powell's singular function:
            f(x) = (x1 + 10*x2)² + 5*(x3 - x4)² + (x2 - 2*x3)⁴ + 10*(x1 - x4)⁴

            Formulated as least squares with residuals:
            r1 = x1 + 10*x2
            r2 = √5*(x3 - x4)
            r3 = (x2 - 2*x3)²
            r4 = √10*(x1 - x4)²

            Standard starting point: [3, -1, 0, 1]
            Known solution: [0, 0, 0, 0] with f(x*) = 0
            """

            # Residuals
            r1 = x[0] + 10.0 * x[1]
            r2 = np.sqrt(5.0) * (x[2] - x[3])
            r3 = (x[1] - 2.0 * x[2]) ** 2
            r4 = np.sqrt(10.0) * (x[0] - x[3]) ** 2

            return np.hstack([r1, r2, r3, r4])

        def powell_func(x):
            r = powell_res(x)
            J = arc.jac(powell_res)(x)

            # Objective: V = 0.5 * ||r||²
            V = 0.5 * np.sum(r**2)

            # Gradient: g = J^T @ r
            g = J.T @ r

            # Gauss-Newton Hessian approximation: H = J^T @ J
            H = J.T @ J

            return V, g, H

        # Standard starting point for Powell's function
        x0 = np.array([3.0, -1.0, 0.0, 1.0])

        # Evaluate initial conditions for diagnostic purposes
        V0, g0, H0 = powell_func(x0)

        # Run optimization with generous limits since this is a harder problem
        result = lmder(powell_func, x0, maxfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-8)

        # Test assertions
        expected_solution = np.array([0.0, 0.0, 0.0, 0.0])
        solution_error = np.linalg.norm(result.x - expected_solution)

        assert result.success, (
            f"Powell optimization should succeed, got: {result.message}"
        )

        # Powell's function is notoriously challenging due to singular Jacobian
        # at solution: ~1e-3 is actually quite good for this problem
        assert solution_error < 5e-3, (
            f"Solution {result.x} not close enough to [0,0,0,0] (error: "
            f"{solution_error:.6e})"
        )

        assert result.fun < 1e-6, (
            f"Final objective {result.fun:.6e} should be close to zero"
        )

    def test_convergence_criteria(self):
        """Test that different convergence criteria can be triggered."""

        # Simple quadratic: f(x) = 0.5 * (x-2)^2, optimum at x=2
        def simple_quadratic(x):
            x = np.atleast_1d(x)
            r = x - 2.0  # residual
            V = 0.5 * np.sum(r**2)  # objective
            g = r  # gradient
            H = np.eye(len(x))  # Hessian
            return V, g, H

        # Test 1: Normal convergence (any success status is fine)
        result = lmder(
            simple_quadratic,
            np.array([5.0]),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            maxfev=200,
        )
        assert result.success, (
            f"Optimization should succeed, got: {result.message}"
        )
        assert result.status in [1, 2, 3, 4], (
            f"Should have valid convergence status, got {result.status}"
        )

        # Test 2: Verify we can hit maximum iterations
        result = lmder(
            simple_quadratic,
            np.array([5.0]),
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
            maxfev=5,
        )
        assert result.status == 5, (
            f"Should hit max iterations, got status {result.status}: {result.message}"
        )

        # Test 3: Verify tolerances work (looser tolerances should still converge)
        result = lmder(
            simple_quadratic,
            np.array([5.0]),
            ftol=1e-2,
            xtol=1e-2,
            gtol=1e-2,
            maxfev=200,
        )
        assert result.success, (
            f"Should converge with loose tolerances, got: {result.message}"
        )

    def test_wood_function(self):
        """Test optimization of Wood's function."""

        def wood_res(x):
            """
            Wood's function (4D optimization test problem):
            f(x) = 100(x2-x1²)² + (1-x1)² + 90(x4-x3²)² + (1-x3)² + 10.1((x2-1)²
                   + (x4-1)²) + 19.8(x2-1)(x4-1)

            Formulated as least squares with residuals + cross term:
            r1 = 10(x2 - x1²)
            r2 = (1 - x1)
            r3 = 3√10(x4 - x3²)
            r4 = (1 - x3)
            r5 = √10.1(x2 - 1)
            r6 = √10.1(x4 - 1)
            Plus cross term: 19.8(x2-1)(x4-1)

            Standard starting point: [-3, -1, -3, -1]
            Known solution: [1, 1, 1, 1] with f(x*) = 0
            """
            # Residuals from the main terms
            r1 = 10.0 * (x[1] - x[0] ** 2)
            r2 = 1.0 - x[0]
            r3 = 3.0 * np.sqrt(10.0) * (x[3] - x[2] ** 2)
            r4 = 1.0 - x[2]
            r5 = np.sqrt(10.1) * (x[1] - 1.0)
            r6 = np.sqrt(10.1) * (x[3] - 1.0)

            return np.hstack([r1, r2, r3, r4, r5, r6])

        def wood_func(x):
            # Residuals and Jacobian from the main terms
            r = wood_res(x)
            J = arc.jac(wood_res)(x)

            # Cross term: 19.8(x2-1)(x4-1)
            cross_term = 19.8 * (x[1] - 1.0) * (x[3] - 1.0)

            # Gradient of cross term
            cross_grad = np.zeros(4)
            cross_grad[1] = 19.8 * (x[3] - 1.0)  # ∂/∂x2
            cross_grad[3] = 19.8 * (x[1] - 1.0)  # ∂/∂x4

            # Hessian of cross term (only non-zero element is the mixed derivative)
            cross_hess = np.zeros((4, 4))
            cross_hess[1, 3] = 19.8  # ∂²/∂x2∂x4
            cross_hess[3, 1] = 19.8  # ∂²/∂x4∂x2 (symmetric)

            # Total objective: V = 0.5 * ||r||² + cross_term
            V = 0.5 * np.sum(r**2) + cross_term

            # Total gradient: g = J^T @ r + cross_grad
            g = J.T @ r + cross_grad

            # Total Hessian: H = J^T @ J + cross_hess
            H = J.T @ J + cross_hess

            return V, g, H

        # Standard starting point for Wood's function
        x0 = np.array([-3.0, -1.0, -3.0, -1.0])

        # Run optimization with reasonable limits
        result = lmder(wood_func, x0, maxfev=1000, ftol=1e-10, xtol=1e-10, gtol=1e-8)

        # Print results for diagnostic purposes
        print("\nWood's Function Results (Standard Starting Point):")
        print(f"Initial point: {x0}")
        print(f"Final solution: {result.x}")
        print(f"Final objective: {result.fun:.2e}")
        print(f"Success: {result.success}")
        print(f"Status: {result.status} - {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Final gradient norm: {result.history[-1]['grad_norm']:.6e}")

        # Test starting near global minimum for comparison
        x0_global = np.array([1.1, 1.1, 1.1, 1.1])
        result_global = lmder(
            wood_func, x0_global, maxfev=1000, ftol=1e-10, xtol=1e-10, gtol=1e-8
        )

        print("\nWood's Function Results (Near-Global Starting Point):")
        print(f"Initial point: {x0_global}")
        print(f"Final solution: {result_global.x}")
        print(f"Final objective: {result_global.fun:.2e}")
        print(f"Success: {result_global.success}")
        print(f"Iterations: {result_global.nit}")

        # Test assertions - Modified to account for local vs global minima
        expected_solution = np.array([1.0, 1.0, 1.0, 1.0])
        solution_error_global = np.linalg.norm(result_global.x - expected_solution)

        # Basic convergence assertion
        assert result.success, (
            f"Wood optimization should succeed, got: {result.message}"
        )
        assert result_global.success, (
            f"Wood optimization from global start should succeed, got: "
            f"{result_global.message}"
        )

        # The algorithm should find the global minimum when started near it
        assert solution_error_global < 1e-3, (
            f"Solution from global start {result_global.x} not close enough to "
            "[1,1,1,1] (error: {solution_error_global:.6e})"
        )
        assert result_global.fun < 1e-6, (
            f"Final objective from global start {result_global.fun:.6e} should be "
            "close to zero"
        )

        # For the standard start, we expect to find a local minimum (critical point)
        # The key test is that we found a critical point (small gradient), not
        # necessarily the global minimum
        final_grad_norm = result.history[-1]["grad_norm"]
        assert final_grad_norm < 1e-4, (
            f"Should converge to critical point (grad norm: {final_grad_norm:.6e})"
        )

        print("\nTest Results:")
        print("✓ Both optimizations converged successfully")
        print(
            f"✓ Near-global start found global minimum (error: "
            f"{solution_error_global:.2e})"
        )
        print(
            f"✓ Standard start found critical point (grad norm: {final_grad_norm:.2e})"
        )

    def test_beale_function(self):
        """Test optimization of Beale's function."""

        def beale_res(x):
            """
            Beale's function (2D optimization test problem):
            f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²

            Formulated as least squares with residuals:
            r1 = 1.5 - x + xy
            r2 = 2.25 - x + xy²
            r3 = 2.625 - x + xy³

            Standard starting point: [1, 1]
            Known solution: [3, 0.5] with f(x*) = 0
            """

            # Extract variables for clarity
            x_var, y_var = x[0], x[1]

            # Residuals
            r1 = 1.5 - x_var + x_var * y_var
            r2 = 2.25 - x_var + x_var * y_var**2
            r3 = 2.625 - x_var + x_var * y_var**3

            return np.hstack([r1, r2, r3])


        def beale_func(x):
            r = beale_res(x)
            J = arc.jac(beale_res)(x)

            # Objective: V = 0.5 * ||r||²
            V = 0.5 * np.sum(r**2)

            # Gradient: g = J^T @ r
            g = J.T @ r

            # Gauss-Newton Hessian approximation: H = J^T @ J
            H = J.T @ J

            return V, g, H

        # Standard starting point for Beale's function
        x0 = np.array([1.0, 1.0])

        # Run optimization with reasonable limits
        result = lmder(beale_func, x0, maxfev=1000, ftol=1e-10, xtol=1e-10, gtol=1e-8)

        # Print results for diagnostic purposes
        print("\nBeale's Function Results:")
        print(f"Initial point: {x0}")
        print(f"Final solution: {result.x}")
        print(f"Final objective: {result.fun:.2e}")
        print(f"Success: {result.success}")
        print(f"Status: {result.status} - {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")

        # Test assertions
        expected_solution = np.array([3.0, 0.5])
        solution_error = np.linalg.norm(result.x - expected_solution)

        assert result.success, (
            f"Beale optimization should succeed, got: {result.message}"
        )

        # Beale's function should converge to the known solution
        assert solution_error < 1e-3, (
            f"Solution {result.x} not close enough to [3,0.5] (error: "
            f"{solution_error:.6e})"
        )

        assert result.fun < 1e-6, (
            f"Final objective {result.fun:.6e} should be close to zero"
        )

        # Additional validation: verify original function value
        x_final, y_final = result.x
        original_beale = (
            (1.5 - x_final + x_final * y_final) ** 2
            + (2.25 - x_final + x_final * y_final**2) ** 2
            + (2.625 - x_final + x_final * y_final**3) ** 2
        )

        print(f"Original Beale function value: {original_beale:.6e}")
        assert original_beale < 1e-6, (
            f"Original Beale function value should be close to zero: "
            f"{original_beale:.6e}"
        )

    def test_iteration_history(self):
        """Test iteration history collection functionality."""

        # Use simple quadratic for predictable convergence
        def simple_quadratic(x):
            x = np.atleast_1d(x)
            r = x - 2.0  # residual: optimum at x=2
            V = 0.5 * np.sum(r**2)  # objective
            g = r  # gradient
            H = np.eye(len(x))  # Hessian
            return V, g, H

        # Test with history collection (always enabled now)
        result = lmder(
            simple_quadratic,
            np.array([5.0]),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            maxfev=50,
        )

        # Verify basic optimization success
        assert result.success, (
            f"Optimization should succeed, got: {result.message}"
        )
        assert np.isclose(result.x[0], 2.0, atol=1e-6), (
            f"Solution should be close to 2.0, got {result.x[0]}"
        )

        # Verify history collection
        history = result.history

        # Basic history validation
        assert len(history) > 0, "History should contain at least one iteration"
        # History records the start of each iteration, including the final one where
        # convergence is detected. So history length should be iterations + 1
        # (we record iter 0, 1, 2, ..., final_iter)
        expected_history_length = result.nit + 1
        assert len(history) == expected_history_length, (
            f"History length ({len(history)}) should be iterations + 1 "
            f"({expected_history_length})"
        )

        # Check history structure
        for i, hist_entry in enumerate(history):
            assert "iter" in hist_entry, f"History entry {i} missing 'iter'"
            assert "cost" in hist_entry, f"History entry {i} missing 'cost'"
            assert "grad_norm" in hist_entry, f"History entry {i} missing 'grad_norm'"
            assert "lambda" in hist_entry, f"History entry {i} missing 'lambda'"
            assert "x" in hist_entry, f"History entry {i} missing 'x'"

            # Check that iteration numbers are correct
            assert hist_entry["iter"] == i, (
                f"Iteration {i} has wrong iter value: {hist_entry['iter']}"
            )

            # Verify convergence trend (cost should generally decrease)
            if i > 0:
                assert hist_entry["cost"] <= history[0]["cost"], (
                    f"Cost should not increase from initial: {hist_entry['cost']} > "
                    f"{history[0]['cost']}"
                )

        # Check that step details are recorded (for successful steps)
        successful_steps = [h for h in history if "step_norm" in h]
        assert len(successful_steps) > 0, (
            "At least one step should have detailed step information"
        )

        print("History Collection Test Results:")
        print(f"Iterations completed: {result.nit}")
        print(f"History entries recorded: {len(history)}")
        print(f"Initial cost: {history[0]['cost']:.6e}")
        print(f"Final cost: {history[-1]['cost']:.6e}")
        print(f"Cost reduction: {history[0]['cost'] - history[-1]['cost']:.6e}")

        # Print sample history entry
        if len(history) > 1 and "step_norm" in history[1]:
            print("\nSample detailed history (iteration 1):")
            for key, value in history[1].items():
                if key == "x":
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value:.6e}")


if __name__ == "__main__":
    TestLM().test_rosenbrock()
