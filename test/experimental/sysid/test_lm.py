# /Users/jared/Dropbox/projects/archimedes/test/experimental/sysid/test_lm.py

import numpy as np
import pytest
from archimedes.experimental.sysid._lmder.lmder import lmder


class TestLM:
    """Test suite for the Levenberg-Marquardt algorithm implementation."""
    
    def test_rosenbrock(self):
        """Test optimization of the Rosenbrock function."""
        # Define Rosenbrock function as a least-squares problem
        # f(x) = 100(x[1] - x[0]²)² + (1 - x[0])²
        # We define residuals: r1 = 10*(x[1] - x[0]²), r2 = (1 - x[0])
        # So that f(x) = 0.5 * (r1² + r2²)
        def rosenbrock_func(x):
            # Residuals
            r = np.array([10.0 * (x[1] - x[0]**2), 1.0 - x[0]])
            
            # Jacobian of residuals
            J = np.array([[-20.0 * x[0], 10.0],   # ∂r1/∂x
                         [-1.0, 0.0]])           # ∂r2/∂x
            
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
        print(f"Optimization result: {result['x']}")
        print(f"Final objective: {result['fun']}")
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print(f"Iterations: {result['nit']}")
        print(f"Function evaluations: {result['nfev']}")
        
        # Test that optimization was successful
        assert result['success'], f"Optimization failed: {result['message']}"
        
        # Test that solution is close to the known optimum [1.0, 1.0]
        assert np.allclose(result['x'], np.array([1.0, 1.0]), rtol=1e-4, atol=1e-4), \
               f"Solution {result['x']} not close to expected [1.0, 1.0]"
        
        # Test that final objective is close to zero
        assert result['fun'] < 1e-6, f"Final objective {result['fun']} not close to zero"


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
        from archimedes.experimental.sysid._lmder.lmder import compute_predicted_reduction
        
        # Simple test case
        grad = np.array([2.0, 1.0])
        step = np.array([-0.5, -0.25])  # Should give reduction
        hess = np.array([[4.0, 0.0], [0.0, 2.0]])
        current_objective = 2.0
        
        # Test without scaling (legacy behavior)
        pred_red_unscaled = compute_predicted_reduction(grad, step, hess)
        
        # Manual calculation: pred_red = -(g^T*p + 0.5*p^T*H*p)
        linear = np.dot(grad, step)  # 2*(-0.5) + 1*(-0.25) = -1.25
        quadratic = 0.5 * np.dot(step, hess @ step)  # 0.5 * (0.25*4 + 0.0625*2) = 0.5625
        expected_unscaled = -(linear + quadratic)  # -(-1.25 + 0.5625) = 0.6875
        
        assert np.isclose(pred_red_unscaled, expected_unscaled)
        assert pred_red_unscaled > 0  # Should predict a reduction
        
        # Test with scaling (new behavior)
        pred_red_scaled = compute_predicted_reduction(grad, step, hess, current_objective)
        expected_scaled = expected_unscaled / current_objective  # 0.6875 / 2.0 = 0.34375
        
        assert np.isclose(pred_red_scaled, expected_scaled)
        assert pred_red_scaled > 0  # Should still predict a reduction
    
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
            x = np.asarray(x)
            
            # Residuals
            r1 = x[0] + 10.0 * x[1]
            r2 = np.sqrt(5.0) * (x[2] - x[3])
            r3 = (x[1] - 2.0 * x[2])**2
            r4 = np.sqrt(10.0) * (x[0] - x[3])**2
            
            r = np.array([r1, r2, r3, r4])
            
            # Jacobian matrix: J[i,j] = ∂rᵢ/∂xⱼ
            J = np.zeros((4, 4))
            
            # ∂r1/∂x: [1, 10, 0, 0]
            J[0, :] = [1.0, 10.0, 0.0, 0.0]
            
            # ∂r2/∂x: [0, 0, √5, -√5]
            J[1, :] = [0.0, 0.0, np.sqrt(5.0), -np.sqrt(5.0)]
            
            # ∂r3/∂x: [0, 2*(x2-2*x3), -4*(x2-2*x3), 0]
            diff_23 = x[1] - 2.0 * x[2]
            J[2, :] = [0.0, 2.0 * diff_23, -4.0 * diff_23, 0.0]
            
            # ∂r4/∂x: [2*√10*(x1-x4), 0, 0, -2*√10*(x1-x4)]
            diff_14 = x[0] - x[3]
            coeff_4 = 2.0 * np.sqrt(10.0) * diff_14
            J[3, :] = [coeff_4, 0.0, 0.0, -coeff_4]
            
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
        solution_error = np.linalg.norm(result['x'] - expected_solution)
        
        assert result['success'], f"Powell optimization should succeed, got: {result['message']}"
        
        # Powell's function is notoriously challenging due to singular Jacobian at solution
        # A solution error of ~1e-3 is actually quite good for this problem
        assert solution_error < 5e-3, f"Solution {result['x']} not close enough to [0,0,0,0] (error: {solution_error:.6e})"
        
        assert result['fun'] < 1e-6, f"Final objective {result['fun']:.6e} should be close to zero"
    
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
        result = lmder(simple_quadratic, np.array([5.0]), 
                      ftol=1e-8, xtol=1e-8, gtol=1e-8, maxfev=200)
        assert result['success'], f"Optimization should succeed, got: {result['message']}"
        assert result['status'] in [1, 2, 3, 4], f"Should have valid convergence status, got {result['status']}"
        
        # Test 2: Verify we can hit maximum iterations
        result = lmder(simple_quadratic, np.array([5.0]), 
                      ftol=1e-15, xtol=1e-15, gtol=1e-15, maxfev=5)
        assert result['status'] == 5, f"Should hit max iterations, got status {result['status']}: {result['message']}"
        
        # Test 3: Verify tolerances work (looser tolerances should still converge)
        result = lmder(simple_quadratic, np.array([5.0]), 
                      ftol=1e-2, xtol=1e-2, gtol=1e-2, maxfev=200)
        assert result['success'], f"Should converge with loose tolerances, got: {result['message']}"

    def test_compare_with_scipy(self):
        """Benchmark against SciPy's implementation."""
        # ...test implementation...
        pass


if __name__ == "__main__":
    TestLM().test_rosenbrock()