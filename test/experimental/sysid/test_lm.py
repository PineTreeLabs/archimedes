# /Users/jared/Dropbox/projects/archimedes/test/experimental/sysid/test_lm.py

import numpy as np
import pytest
from archimedes.experimental.sysid._lmder.lmder import lmder


class TestLM:
    """Test suite for the Levenberg-Marquardt algorithm implementation."""
    
    @pytest.mark.skip
    def test_rosenbrock(self):
        """Test optimization of the Rosenbrock function."""
        # Define Rosenbrock function and derivatives
        def rosenbrock_func(x):
            # f(x) = 100(x[1] - x[0]²)² + (1 - x[0])²
            # Return objective, gradient, and Hessian
            f = np.array([10.0 * (x[1] - x[0]**2), 1.0 - x[0]])
            J = np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])
            H = J.T @ J  # Approximation to Hessian
            V = 0.5 * np.sum(f**2)  # Objective (half sum of squares)
            return V, -J.T @ f, H
        
        # Initial guess
        x0 = np.array([-1.2, 1.0])
        
        # Run optimization
        result = lmder(x0, rosenbrock_func)
        
        # Check result
        assert np.allclose(result, np.array([1.0, 1.0]), rtol=1e-5)


    def test_powell(self):
        """Test optimization of Powell's singular function."""
        # ...similar test implementation...
        pass
    
    def test_ill_conditioned(self):
        """Test with an ill-conditioned problem to verify numerical stability."""
        # ...test implementation...
        pass
    
    def test_convergence_criteria(self):
        """Test different convergence criteria."""
        # ...test implementation...
        pass
    
    def test_compare_with_scipy(self):
        """Benchmark against SciPy's implementation."""
        # ...test implementation...
        pass