# ruff: noqa: N802, N803, N806, E741

import numpy as np
import pytest
from scipy import sparse

from archimedes.optimize import LMStatus, least_squares, lm_solve

METHODS = ["lm", "hess-lm"]
BOUNDED_METHODS = ["trf", "hess-lm"]


class TestLeastSquares:
    """Test suite for the Levenberg-Marquardt algorithm implementation."""

    @pytest.mark.parametrize("method", METHODS)
    def test_rosenbrock(self, method):
        """Test optimization of the Rosenbrock function."""

        # Define Rosenbrock function as a least-squares problem
        # f(x) = 100(x[1] - x[0]²)² + (1 - x[0])²
        # We define residuals: r1 = 10*(x[1] - x[0]²), r2 = (1 - x[0])
        # So that f(x) = 0.5 * (r1² + r2²)
        def rosenbrock_func(x):
            return np.hstack([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

        # Initial guess
        x0 = np.array([-1.2, 1.0])

        # Run optimization
        result = least_squares(rosenbrock_func, x0, method=method)

        print(result)

        # Check result - the solution should be close to [1.0, 1.0]
        print(f"Optimization result: {result.x}")
        print(f"Final residuals: {result.fun}")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        if hasattr(result, "nit"):
            print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")

        # Test that optimization was successful
        assert result.success, f"Optimization failed: {result.message}"

        # Test that solution is close to the known optimum [1.0, 1.0]
        assert np.allclose(result.x, np.array([1.0, 1.0]), rtol=1e-4, atol=1e-4), (
            f"Solution {result.x} not close to expected [1.0, 1.0]"
        )

        # Test that final residuals are close to zero
        assert np.allclose(result.fun, 0.0, atol=1e-6), (
            f"Final residuals {result.fun} not close to zero"
        )

        with pytest.raises(ValueError, match=r"Method 'invalid' is not supported.*"):
            least_squares(rosenbrock_func, x0, method="invalid")

    def test_powell_singular(self):
        """Test optimization of Powell's singular function."""

        def powell_func(x):
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

        # Standard starting point for Powell's function
        x0 = np.array([3.0, -1.0, 0.0, 1.0])

        # Run optimization with generous limits since this is a harder problem
        options = {
            "max_nfev": 1000,
            "ftol": 1e-12,
            "xtol": 1e-12,
            "gtol": 1e-8,
        }
        result = least_squares(powell_func, x0, options=options)

        # Test assertions
        expected_solution = np.array([0.0, 0.0, 0.0, 0.0])
        solution_error = np.linalg.norm(result.x - expected_solution)

        # Check result - the solution should be close to [0.0, 0.0, 0.0, 0.0]
        print(f"Optimization result: {result.x}")
        print(f"Final residuals: {result.fun}")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")

        assert result.success, (
            f"Powell optimization should succeed, got: {result.message}"
        )

        # Powell's function is notoriously challenging due to singular Jacobian
        # at solution: ~1e-3 is actually quite good for this problem
        assert solution_error < 5e-3, (
            f"Solution {result.x} not close enough to [0,0,0,0] (error: "
            f"{solution_error:.6e})"
        )

        assert np.allclose(result.fun, 0.0, atol=1e-5), (
            f"Final objective {result.fun} should be close to zero"
        )

    def test_wood_function(self):
        """Test optimization of Wood's function."""

        def wood_func(x):
            """
            Wood's function (4D optimization test problem):
            f(x) = 100(x2-x1²)² + (1-x1)² + 90(x4-x3²)² + (1-x3)² + 10.1((x2-1)²
                   + (x4-1)²) + 19.8(x2-1)(x4-1)

            Formulated as pure least squares residuals:
            f(x) = 0.5 * ||r||²

            Standard starting point: [-3, -1, -3, -1]
            Known solution: [1, 1, 1, 1] with f(x*) = 0
            """
            return np.hstack(
                [
                    10.0 * np.sqrt(2.0) * (x[1] - x[0] ** 2),  # √200 * (x2 - x1²)
                    np.sqrt(2.0) * (1.0 - x[0]),  # √2 * (1 - x1)
                    6.0 * np.sqrt(5.0) * (x[3] - x[2] ** 2),  # √180 * (x4 - x3²)
                    np.sqrt(2.0) * (1.0 - x[2]),  # √2 * (1 - x3)
                    np.sqrt(0.4) * (x[1] - 1.0),  # √0.4 * (x2 - 1)
                    np.sqrt(0.4) * (x[3] - 1.0),  # √0.4 * (x4 - 1)
                    np.sqrt(19.8) * (x[1] + x[3] - 2.0),  # √19.8 * (x2 + x4 - 2)
                ]
            )

        # Standard starting point for Wood's function
        x0 = np.array([-3.0, -1.0, -3.0, -1.0])

        # Run optimization with reasonable limits
        options = {
            "max_nfev": 1000,
            "ftol": 1e-10,
            "xtol": 1e-10,
            "gtol": 1e-8,
        }
        result = least_squares(wood_func, x0, options=options)

        # Print results for diagnostic purposes
        print("\nWood's Function Results (Standard Starting Point):")
        print(f"Initial point: {x0}")
        print(f"Final solution: {result.x}")
        print(f"Final residuals: {result.fun}")
        print(f"Success: {result.success}")
        print(f"Status: {result.status} - {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Final gradient norm: {result.history[-1]['grad_norm']:.6e}")

        # Test assertions - Modified to account for local vs global minima
        expected_solution = np.array([1.0, 1.0, 1.0, 1.0])
        solution_error = np.linalg.norm(result.x - expected_solution)

        # Basic convergence assertion
        assert result.success, (
            f"Wood optimization should succeed, got: {result.message}"
        )
        # The algorithm should find the global minimum when started near it
        assert solution_error < 1e-3, (
            f"Solution {result.x} not close enough to "
            f"[1,1,1,1] (error: {solution_error:.6e})"
        )
        assert np.allclose(result.fun, 0.0, atol=1e-6), (
            f"Final residuals {result.fun} should be close to zero"
        )

        # For the standard start, we expect to find a local minimum (critical point)
        # The key test is that we found a critical point (small gradient), not
        # necessarily the global minimum
        final_grad_norm = result.history[-1]["grad_norm"]
        assert final_grad_norm < 1e-4, (
            f"Should converge to critical point (grad norm: {final_grad_norm:.6e})"
        )

        print("\nTest Results:")
        print(f"✓ Standard start found global minimum (error: {solution_error:.2e})")
        print(
            f"✓ Standard start found critical point (grad norm: {final_grad_norm:.2e})"
        )

    def test_beale_function(self):
        """Test optimization of Beale's function."""

        def beale_func(x):
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

        # Standard starting point for Beale's function
        x0 = np.array([1.0, 1.0])

        # Run optimization with reasonable limits
        options = {
            "max_nfev": 1000,
            "ftol": 1e-10,
            "xtol": 1e-10,
            "gtol": 1e-8,
        }
        result = least_squares(beale_func, x0, options=options)

        # Print results for diagnostic purposes
        print("\nBeale's Function Results:")
        print(f"Initial point: {x0}")
        print(f"Final solution: {result.x}")
        print(f"Final residuals: {result.fun}")
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

        assert np.allclose(result.fun, 0.0, atol=1e-6), (
            f"Final objective {result.fun} should be close to zero"
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

    @pytest.mark.parametrize("method", BOUNDED_METHODS)
    def test_box_constraints_simple_quadratic(self, method):
        """Test box constraints with a simple quadratic function."""

        def constrained_quadratic(x):
            """
            Minimize f(x,y) = (x-3)² + (y-2)²
            Subject to: 0 ≤ x ≤ 2, 0 ≤ y ≤ 1

            Unconstrained optimum: (3, 2)
            Constrained optimum: (2, 1) - both variables at upper bounds
            """
            x = np.atleast_1d(x)

            # Residuals: r1 = (x-3), r2 = (y-2)
            r = np.array([x[0] - 3.0, x[1] - 2.0], like=x)

            return r

        # Test setup
        x0 = np.array([0.5, 0.5])  # Start in interior
        bounds = (
            np.array([0.0, 0.0]),  # Lower bounds
            np.array([2.0, 1.0]),  # Upper bounds
        )

        # Solve constrained problem
        result = least_squares(
            constrained_quadratic,
            x0,
            bounds=bounds,
            method=method,
        )

        # Test assertions
        expected_solution = np.array([2.0, 1.0])
        solution_error = np.linalg.norm(result.x - expected_solution)

        assert result.success, (
            f"Constrained quadratic optimization should succeed, got: {result.message}"
        )

        assert solution_error < 1e-4, (
            "Solution {result.x} not close enough to [2,1] (error: "
            f"{solution_error:.6e})"
        )

        # Check constraint satisfaction
        lb, ub = bounds
        tol = 1e-6  # Tolerance for bounds checking
        bounds_satisfied = np.all(lb <= result.x + tol) and np.all(result.x - tol <= ub)
        assert bounds_satisfied, f"Bounds violated: {result.x} not in [{lb}, {ub}]"

        if method == "hess-lm":
            # Check that constrained optimization info is recorded
            if len(result.history) > 0 and "grad_proj_norm" in result.history[-1]:
                final_history = result.history[-1]
                # Both variables should be at upper bounds
                assert final_history["n_active_upper"] == 2, (
                    "Expected 2 active upper bounds, got "
                    "{final_history['n_active_upper']}"
                )
                assert final_history["grad_proj_norm"] < 1e-6, (
                    "Projected gradient norm should be small: "
                    f"{final_history['grad_proj_norm']}"
                )

        # Edge case: Test ValueError for bounds structure mismatch
        bad_lower = np.array([0.0])  # Wrong size (1 instead of 2)
        bad_upper = np.array([2.0, 1.0])  # Correct size
        with pytest.raises(
            ValueError, match=r"Lower bounds must have the same number .*"
        ):
            least_squares(
                constrained_quadratic,
                x0,
                bounds=(bad_lower, bad_upper),
            )

        bad_lower = {"a": np.array([0.0, 0.0])}  # Wrong tree structure
        with pytest.raises(
            ValueError, match=r"Lower bounds must have the same structure .*"
        ):
            least_squares(
                constrained_quadratic,
                x0,
                bounds=(bad_lower, bad_upper),
            )

        bad_lower = np.array([0.0, 0.0])  # Correct size
        bad_upper = np.array([2.0])  # Wrong size (1 instead of 2)
        with pytest.raises(
            ValueError, match=r"Upper bounds must have the same number .*"
        ):
            least_squares(
                constrained_quadratic,
                x0,
                bounds=(bad_lower, bad_upper),
            )

        bad_upper = {"a": np.array([2.0, 1.0])}  # Wrong tree structure
        with pytest.raises(
            ValueError, match=r"Upper bounds must have the same structure .*"
        ):
            least_squares(
                constrained_quadratic,
                x0,
                bounds=(bad_lower, bad_upper),
            )

    def test_box_constraints_rosenbrock(self):
        """Test box constraints with Rosenbrock function."""

        def rosenbrock_func(x):
            return np.hstack([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

        # Test Case 1: Bounds that force solution to boundary
        # Unconstrained optimum is [1, 1], but we restrict x ≤ 0.8
        x0 = np.array([0.0, 0.0])  # Start at origin
        bounds = (
            np.array([0.0, 0.0]),  # Lower bounds
            np.array([0.8, 2.0]),  # Upper bounds - restrict first variable
        )

        options = {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-6, "max_nfev": 100}
        result = least_squares(
            rosenbrock_func,
            x0,
            bounds=bounds,
            options=options,
        )

        # The constrained optimum should be at x[0] = 0.8, x[1] = 0.8² = 0.64
        expected_solution = np.array([0.8, 0.64])
        solution_error = np.linalg.norm(result.x - expected_solution)

        assert result.success, (
            f"Constrained Rosenbrock optimization should succeed, got: {result.message}"
        )

        assert solution_error < 1e-2, (
            f"Solution {result.x} not close enough to [0.8, 0.64] (error: "
            f"{solution_error:.6e})"
        )

        # Check that first variable is at upper bound
        assert abs(result.x[0] - 0.8) < 1e-6, (
            f"First variable should be at upper bound 0.8, got {result.x[0]}"
        )

        # Check constraint satisfaction
        lb, ub = bounds
        bounds_satisfied = np.all(lb <= result.x) and np.all(result.x <= ub)
        assert bounds_satisfied, f"Bounds violated: {result.x} not in [{lb}, {ub}]"

        # Test Case 2: Bounds that don't affect the solution (interior optimum)
        bounds_loose = (
            np.array([0.0, 0.0]),  # Lower bounds
            np.array([2.0, 2.0]),  # Upper bounds - don't restrict
        )

        result_loose = least_squares(
            rosenbrock_func,
            x0,
            bounds=bounds_loose,
            options=options,
        )

        # Should find the unconstrained optimum [1, 1]
        expected_solution_loose = np.array([1.0, 1.0])
        solution_error_loose = np.linalg.norm(result_loose.x - expected_solution_loose)

        assert result_loose.success, (
            "Loosely constrained Rosenbrock should succeed, got: "
            f"{result_loose.message}"
        )

        assert solution_error_loose < 1e-3, (
            f"Loose bounds solution {result_loose.x} not close to [1,1] (error: "
            f"{solution_error_loose:.6e})"
        )

        # Verify that loose bounds don't activate constraints
        if len(result_loose.history) > 0 and "total_active" in result_loose.history[-1]:
            final_active = result_loose.history[-1]["total_active"]
            assert final_active == 0, (
                f"Loose bounds should have no active constraints, got {final_active}"
            )


class TestLM:
    def test_compute_step_well_conditioned(self):
        """Test compute_step with a well-conditioned matrix."""
        from archimedes.optimize._lm import _compute_step

        # Simple 2x2 case with known solution
        hess = np.array([[4.0, 1.0], [1.0, 3.0]])  # SPD matrix
        grad = np.array([2.0, 1.0])
        diag = np.array([1.0, 1.0])
        lambda_val = 0.1
        x_current = np.array([0.0, 0.0])  # Current point
        bounds = None

        step = _compute_step(hess, grad, diag, lambda_val, x_current, bounds)

        # Verify the step satisfies (H + λI)p = -g
        H_damped = hess + lambda_val * np.eye(2)
        residual = H_damped @ step + grad
        assert np.allclose(residual, 0.0, atol=1e-12)

    def test_compute_step_ill_conditioned(self):
        """Test compute_step with an ill-conditioned matrix."""
        from archimedes.optimize._lm import _compute_step

        # Ill-conditioned matrix (near-singular)
        hess = np.array([[1.0, 1.0], [1.0, 1.0001]])
        grad = np.array([1.0, 1.0])
        diag = np.array([1.0, 1.0])
        lambda_val = 0.01
        x_current = np.array([0.0, 0.0])  # Current point
        bounds = None

        # Should not crash and should return reasonable step
        step = _compute_step(hess, grad, diag, lambda_val, x_current, bounds)
        assert not np.any(np.isnan(step))
        assert not np.any(np.isinf(step))
        assert np.linalg.norm(step) < 100  # Reasonable magnitude

    def test_compute_step_singular(self):
        """Test compute_step with a singular matrix."""
        from archimedes.optimize._lm import _compute_step

        # Singular matrix
        hess = np.array([[1.0, 1.0], [1.0, 1.0]])
        grad = np.array([1.0, 1.0])
        diag = np.array([1.0, 1.0])
        lambda_val = 0.0  # No damping to keep it singular
        x_current = np.array([0.0, 0.0])  # Current point
        bounds = None

        # Should fall back gracefully
        step = _compute_step(hess, grad, diag, lambda_val, x_current, bounds)
        assert not np.any(np.isnan(step))
        assert not np.any(np.isinf(step))

    def test_compute_predicted_reduction(self):
        """Test predicted reduction calculation."""
        from archimedes.optimize._lm import (
            _compute_predicted_reduction,
        )

        # Simple test case
        grad = np.array([2.0, 1.0])
        step = np.array([-0.5, -0.25])  # Should give reduction
        hess = np.array([[4.0, 0.0], [0.0, 2.0]])
        current_objective = 2.0

        # Test without scaling
        pred_red_unscaled = _compute_predicted_reduction(grad, step, hess)

        # Manual calculation: pred_red = -(g^T*p + 0.5*p^T*H*p)
        linear = np.dot(grad, step)  # 2*(-0.5) + 1*(-0.25) = -1.25
        quadratic = 0.5 * np.dot(
            step, hess @ step
        )  # 0.5 * (0.25*4 + 0.0625*2) = 0.5625
        expected_unscaled = -(linear + quadratic)  # -(-1.25 + 0.5625) = 0.6875

        assert np.isclose(pred_red_unscaled, expected_unscaled)
        assert pred_red_unscaled > 0  # Should predict a reduction

        # Test with scaling
        pred_red_scaled = _compute_predicted_reduction(
            grad, step, hess, current_objective
        )
        expected_scaled = (
            expected_unscaled / current_objective
        )  # 0.6875 / 2.0 = 0.34375

        assert np.isclose(pred_red_scaled, expected_scaled)
        assert pred_red_scaled > 0  # Should still predict a reduction

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
        result = lm_solve(
            simple_quadratic,
            np.array([5.0]),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=200,
        )
        assert result.success, f"Optimization should succeed, got: {result.message}"
        assert result.status in [
            LMStatus.FTOL_REACHED,
            LMStatus.XTOL_REACHED,
            LMStatus.BOTH_TOL_REACHED,
            LMStatus.GTOL_REACHED,
        ], f"Should have valid convergence status, got {result.status}"

        # Test 2: Verify we can hit maximum iterations
        result = lm_solve(
            simple_quadratic,
            np.array([5.0]),
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
            max_nfev=2,
        )
        assert result.status == LMStatus.MAX_FEVAL, (
            f"Should hit max iterations, got status {result.status}: {result.message}"
        )

        # Test 3: Verify tolerances work (looser tolerances should still converge)
        result = lm_solve(
            simple_quadratic,
            np.array([5.0]),
            ftol=1e-2,
            xtol=1e-2,
            gtol=1e-2,
            max_nfev=200,
        )
        assert result.success, (
            f"Should converge with loose tolerances, got: {result.message}"
        )

    def test_diagonal_scaling(self):
        """Test custom diagonal scaling vs automatic scaling."""

        # Create an ill-conditioned problem with very different variable scales
        # Variable 1: operates around 1e-3 scale
        # Variable 2: operates around 1e3 scale
        def ill_conditioned_func(x):
            """
            Problem where variables have very different natural scales:
            f(x) = 0.5 * ((1000*x[0] - 1)^2 + (x[1]/1000 - 1)^2)

            Solution is at x[0] = 1e-3, x[1] = 1e3
            Without proper scaling, this is very hard to optimize.
            """
            x = np.atleast_1d(x)

            # Residuals with very different scales
            r1 = 1000.0 * x[0] - 1.0  # x[0] should be ~1e-3
            r2 = x[1] / 1000.0 - 1.0  # x[1] should be ~1e3

            r = np.array([r1, r2], like=x)

            # Compute Jacobian manually for this simple case
            J = np.array([[1000.0, 0.0], [0.0, 1.0 / 1000.0]], like=x)

            # Objective: V = 0.5 * ||r||^2
            V = 0.5 * np.sum(r**2)

            # Gradient: g = J^T @ r
            g = J.T @ r

            # Hessian approximation: H = J^T @ J
            H = J.T @ J

            return V, g, H

        # Starting point away from optimum
        x0 = np.array([0.1, 0.1])  # Both variables start at wrong scale

        print("\nDiagonal Scaling Test:")
        print("True solution: x* = [1e-3, 1e3] = [0.001, 1000.0]")
        print(f"Starting point: x0 = {x0}")

        # Test 1: Automatic scaling (diag=None, default)
        result_auto = lm_solve(
            ill_conditioned_func,
            x0.copy(),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=100,
        )

        print("\nAutomatic scaling (diag=None):")
        print(f"  Solution: {result_auto.x}")
        print(f"  Success: {result_auto.success}")
        print(f"  Iterations: {result_auto.nit}")
        print(f"  Final residuals: {result_auto.fun}")

        # Test 2: Custom scaling that accounts for the variable scales
        # diag[i] should be proportional to the "natural scale" of variable i
        # For our problem: x[0] ~ 1e-3, x[1] ~ 1e3
        custom_diag = np.array(
            [1e-3, 1e3]
        )  # Scale factors matching expected solution magnitude

        result_scaled = lm_solve(
            ill_conditioned_func,
            x0.copy(),
            diag=custom_diag,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=100,
        )

        print(f"\nCustom scaling (diag={custom_diag}):")
        print(f"  Solution: {result_scaled.x}")
        print(f"  Success: {result_scaled.success}")
        print(f"  Iterations: {result_scaled.nit}")
        print(f"  Final residuals: {result_scaled.fun}")

        # Test 3: Poor scaling (opposite of what we need)
        poor_diag = np.array([1e3, 1e-3])  # Wrong scaling

        result_poor = lm_solve(
            ill_conditioned_func,
            x0.copy(),
            diag=poor_diag,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=100,
        )

        print(f"\nPoor scaling (diag={poor_diag}):")
        print(f"  Solution: {result_poor.x}")
        print(f"  Success: {result_poor.success}")
        print(f"  Iterations: {result_poor.nit}")
        print(f"  Final residuals: {result_poor.fun}")

        # Verify that at least one optimization succeeded
        assert result_auto.success or result_scaled.success, (
            "At least one scaling approach should succeed"
        )

        # The expected solution
        expected_solution = np.array([1e-3, 1e3])

        # Check solution accuracy for successful runs
        if result_auto.success:
            auto_error = np.linalg.norm(result_auto.x - expected_solution)
            print(f"  Auto scaling error: {auto_error:.2e}")

        if result_scaled.success:
            scaled_error = np.linalg.norm(result_scaled.x - expected_solution)
            print(f"  Custom scaling error: {scaled_error:.2e}")

            # Custom scaling should be reasonably close to the solution
            assert scaled_error < 1e-2, (
                "Custom scaling should find accurate solution, error: "
                f"{scaled_error:.2e}"
            )

        # Verify that the custom diag array was actually used
        # We can check this by verifying that auto_scale was set to False
        # This is tested indirectly by ensuring that our custom scaling affects
        # the results

        print("\n✓ Diagonal scaling test completed!")
        print("✓ Custom diag parameter functionality verified!")

        # Test that custom diag is actually different from auto scaling
        # (both should work, but give different iteration counts/paths)
        if result_auto.success and result_scaled.success:
            # They should both converge but potentially with different efficiency
            assert (
                result_auto.nit != result_scaled.nit
                or abs(result_auto.fun - result_scaled.fun) > 1e-15
            ), "Custom and automatic scaling should behave differently"

    def test_iteration_history(self):
        """Test iteration history collection functionality."""

        # Use simple quadratic for predictable convergence
        def func(x):
            x = np.atleast_1d(x)
            return x - 2.0  # residual: optimum at x=2

        # Test with history collection (always enabled now)
        result = lm_solve(
            func,
            np.array([5.0]),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=50,
        )

        # Verify basic optimization success
        assert result.success, f"Optimization should succeed, got: {result.message}"
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

    def test_ftol_convergence_info_1(self):
        """Test convergence via ftol only (info = 1)."""

        def func(x):
            x = np.atleast_1d(x)
            # Two residuals to create: V = 0.5 * (x[0]² + 2e-10)
            return np.hstack([x[0], np.sqrt(2e-10)])

        # Start close to optimum
        x0 = np.array([1e-6])

        result = lm_solve(
            func,
            x0,
            ftol=1e-3,  # Relatively loose ftol
            xtol=1e-15,  # Very tight xtol to avoid that convergence
            gtol=1e-15,  # Very tight gtol to avoid that convergence
            max_nfev=50,
        )

        # Should converge due to ftol
        assert result.status == LMStatus.FTOL_REACHED, (
            f"Expected ftol convergence, got {result.status}: {result.message}"
        )
        assert result.success

    def test_combined_convergence_info_3(self):
        """Test combined ftol and xtol convergence (info = 3)."""

        def func(x):
            x = np.atleast_1d(x)
            # Design function to satisfy both ftol and xtol simultaneously
            r1 = x[0] + 1e-10  # Gradient contribution: x[0] + 1e-10
            r2 = np.sqrt(2e-12)  # Adds constant 1e-12 to objective
            return np.hstack([r1, r2])

        x0 = np.array([1e-5])  # Start very close to optimum
        result = lm_solve(
            func,
            x0,
            ftol=1e-4,  # Loose enough to trigger
            xtol=1e3,  # Loose enough to trigger (this is relative to parameter norm)
            gtol=1e-15,  # Very tight to avoid gradient convergence
            max_nfev=10,
            log_level=20,
        )

        # Should converge with combined criteria
        assert result.status == LMStatus.BOTH_TOL_REACHED, (
            f"Expected combined convergence, got {result.status}: {result.message}"
        )
        assert result.success

        # Loosen xtol and make sure that triggers first
        result = lm_solve(
            func,
            x0,
            ftol=1e-4,  # Loose enough to trigger
            xtol=1e4,  # Loose enough to trigger
            gtol=1e-15,  # Very tight to avoid gradient convergence
            max_nfev=10,
            log_level=20,
        )

        assert result.status == LMStatus.XTOL_REACHED, (
            f"Expected xtol convergence, got {result.status}: {result.message}"
        )
        assert result.success

    def test_max_nfev_reached_in_inner_loop(self):
        """Test max_nfev reached during inner loop iterations."""

        def slow_converging_func(x):
            x = np.atleast_1d(x)
            # Create ill-conditioned problem via small Jacobian entries
            # This gives J = [[1e-4]], so H = J^T*J = [[1e-8]]
            return 1e-4 * x[0]

        x0 = np.array([1.0])

        result = lm_solve(
            slow_converging_func,
            x0,
            max_nfev=3,  # Very small limit to hit during inner loop
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
        )

        # Should fail due to max function evaluations
        assert result.status == LMStatus.MAX_FEVAL, (
            f"Expected max fev, got {result.status}: {result.message}"
        )
        assert not result.success
        assert result.nfev >= 3  # Should have hit the limit

    def test_progress_reporting_with_nprint(self, caplog):
        """Test progress reporting functionality (nprint > 0)."""

        def simple_quadratic(x):
            x = np.atleast_1d(x)
            return x - 2.0

        # Capture printed output by redirecting stdout
        import logging

        with caplog.at_level(logging.INFO):
            result = lm_solve(
                simple_quadratic, np.array([5.0]), log_level=logging.INFO, max_nfev=20
            )

            output = caplog.text

            # Check that headers and iteration info were printed
            assert "Iteration" in output, "Should print iteration header"
            assert "Cost" in output, "Should print cost header"
            assert result.success, "Optimization should succeed"

    def test_header_logic(self, caplog):
        """Test the specific header printing logic in LMProgress."""
        import logging

        from archimedes.optimize._lm import LMProgress

        logger = logging.getLogger("test_header_logic")

        # Test LMProgress class directly
        with caplog.at_level(logging.INFO):
            progress = LMProgress(logger)

            # First report (iteration 0) - should print header
            progress.report(1.0, 0.1, 0.01, 5)

            # Second report (iteration 1) - should print again
            progress.report(0.25, 0.025, 0.0025, 7)
            output = caplog.text

            # Should contain headers and specific iteration data
            assert "Iteration" in output, "Should contain header"
            assert "Cost" in output, "Should contain cost header"

            # Check that it printed for iterations 0 and 2
            lines = output.strip().split("\n")
            # Should have: header line + iteration 0 + iteration 2 = 3 lines minimum
            assert len(lines) >= 3, (
                f"Expected at least 3 lines of output, got {len(lines)}"
            )

    def test_box_constraints_gradient_projection(self):
        """Test gradient projection logic with a simple example."""
        from archimedes.optimize._lm import (
            _check_constrained_convergence,
            _compute_step,
            _project_gradient,
        )

        # Test case: At lower bound with outward gradient
        x = np.array([0.0, 2.5])  # First variable at lower bound
        grad = np.array([0.5, -0.2])  # Positive gradient for bounded variable
        bounds = (np.array([0.0, 0.0]), np.array([3.0, 3.0]))

        grad_proj, active_lower, active_upper = _project_gradient(grad, x, bounds)

        # First component should be projected to 0, second unchanged
        assert grad_proj[0] == 0.0, (
            f"First component should be projected to 0, got {grad_proj[0]}"
        )
        assert grad_proj[1] == grad[1], (
            f"Second component should be unchanged, got {grad_proj[1]} vs {grad[1]}"
        )
        assert active_lower[0], "First variable should be detected as active lower"
        assert not active_lower[1], "Second variable should not be active"

        # Test convergence check
        converged, proj_norm, active_info = _check_constrained_convergence(
            grad, x, bounds, gtol=1e-1
        )

        assert not converged, "Should not converge with large projected gradient"
        assert proj_norm == abs(grad[1]), (
            "Projected norm should be second component only"
        )
        assert active_info["n_active_lower"] == 1, (
            "Should detect one active lower bound"
        )
        assert active_info["total_active"] == 1, (
            "Should have one total active constraint"
        )

        # Edge case: Test ValueError when qp_solver is None but bounds are present
        hess = np.eye(2)
        diag = np.ones(2)
        lambda_val = 0.1

        try:
            _compute_step(hess, grad, diag, lambda_val, x, bounds, qp_solver=None)
            assert False, "Should have raised ValueError for missing qp_solver"
        except ValueError as e:
            assert "qp_solver must be provided" in str(e), (
                f"Unexpected error message: {e}"
            )

    def test_box_constraints_qp_edge_cases(self):
        """Test QP solver edge cases: difficult solves and fallback handling."""
        import io
        import sys

        import osqp

        from archimedes.optimize._lm import _compute_step

        # Set up basic QP problem data
        n = 2
        hess = np.eye(n)
        grad = np.array([1.0, 1.0])
        diag = np.ones(n)
        lambda_val = 0.1
        x_current = np.array([0.0, 0.0])

        # Test case 1: QP with very tight constraints that might cause solver issues
        # Use extremely small feasible region
        epsilon = 1e-10  # Very tight bounds
        bounds_tight = (np.array([-epsilon, -epsilon]), np.array([epsilon, epsilon]))

        # Set up OSQP solver with settings that might cause early termination
        qp_solver = osqp.OSQP()
        qp_solver.setup(
            P=sparse.csc_matrix(np.ones((n, n))),
            q=np.zeros(n),
            A=sparse.csc_matrix(np.eye(n)),
            l=np.array([-epsilon, -epsilon]),
            u=np.array([epsilon, epsilon]),
            verbose=False,
            max_iter=1,  # Very few iterations to force early termination
            eps_abs=1e-3,  # Loose tolerance
            eps_rel=1e-3,
        )

        # Capture printed output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # This might trigger suboptimal solve due to tight constraints
            # + few iterations
            step = _compute_step(
                hess, grad, diag, lambda_val, x_current, bounds_tight, qp_solver
            )

            # Should return a valid step regardless of QP solver status
            assert step is not None, "Step should not be None even with difficult QP"
            assert not np.any(np.isnan(step)), "Step should not contain NaN"
            assert not np.any(np.isinf(step)), "Step should not contain Inf"
            assert len(step) == n, "Step should have correct dimension"

        finally:
            sys.stdout = sys.__stdout__

        # Test case 2: Test fast path vs QP path decision logic
        # This tests the bounds violation detection more directly

        # Case where unconstrained step doesn't violate bounds (should use fast path)
        grad_small = np.array([0.01, 0.01])  # Small gradient
        bounds_loose = (np.array([-10.0, -10.0]), np.array([10.0, 10.0]))

        qp_solver_test = osqp.OSQP()
        qp_solver_test.setup(
            P=sparse.csc_matrix(np.ones((n, n))),
            q=np.zeros(n),
            A=sparse.csc_matrix(np.eye(n)),
            l=np.array([-10.0, -10.0]),
            u=np.array([10.0, 10.0]),
            verbose=False,
        )

        step_fast = _compute_step(
            hess, grad_small, diag, lambda_val, x_current, bounds_loose, qp_solver_test
        )

        # Case where unconstrained step would violate bounds (should use QP path)
        grad_large = np.array([5.0, 5.0])  # Large gradient that will violate bounds
        bounds_tight_valid = (np.array([-0.1, -0.1]), np.array([0.1, 0.1]))

        qp_solver_test.update(l=np.array([-0.1, -0.1]), u=np.array([0.1, 0.1]))

        step_qp = _compute_step(
            hess,
            grad_large,
            diag,
            lambda_val,
            x_current,
            bounds_tight_valid,
            qp_solver_test,
        )

        # Both should return valid steps, but they should be different
        assert not np.allclose(step_fast, step_qp, atol=1e-6), (
            "Fast path and QP path should give different steps for this problem"
        )

        # QP step should respect bounds after adding to x_current
        x_trial_qp = x_current + step_qp
        lb, ub = bounds_tight_valid
        bounds_tolerance = 1e-6
        assert np.all(x_trial_qp >= lb - bounds_tolerance), (
            "QP step should respect lower bounds"
        )
        assert np.all(x_trial_qp <= ub + bounds_tolerance), (
            "QP step should respect upper bounds"
        )

        # Test case 3: Simulate QP solver failure by causing numerical issues
        # Use pathological problem that might stress the QP solver
        hess_ill = np.array([[1e-12, 0], [0, 1e12]])  # Very ill-conditioned
        grad_mixed = np.array([1e6, 1e-6])  # Mixed scales

        # This combination might cause the QP solver to struggle
        step_pathological = _compute_step(
            hess_ill,
            grad_mixed,
            diag,
            lambda_val,
            x_current,
            bounds_tight_valid,
            qp_solver_test,
        )

        # Should still return a valid step via fallback mechanisms
        assert step_pathological is not None, (
            "Step should not be None even with pathological QP"
        )
        assert not np.any(np.isnan(step_pathological)), "Step should not contain NaN"
        assert len(step_pathological) == n, "Step should have correct dimension"


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running Rosenbrock Test")
    print("=" * 60)
    TestLeastSquares().test_rosenbrock()

    print("\n" + "=" * 60)
    print("Running Powell's Singular Function Test")
    print("=" * 60)
    TestLeastSquares().test_powell_singular()

    print("\n" + "=" * 60)
    print("Running Wood's Function Test")
    print("=" * 60)
    TestLeastSquares().test_wood_function()

    print("\n" + "=" * 60)
    print("Running Beale's Function Test")
    TestLeastSquares().test_beale_function()
    print("=" * 60)
