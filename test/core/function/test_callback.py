import numpy as np
import pytest

import archimedes as arc


def test_callback():
    def f(x, y):
        print(f"f called: {x=}, {y=}")
        return x * (y + 3)

    @arc.compile
    def call_f(x, y):
        return arc.callback(f, x, y)

    x, y = np.array([1.0, 2.0]), 3.0
    z = call_f(x, y)
    assert np.allclose(z, f(x, y))
    assert z.shape == x.shape


def test_basic_callback():
    """Test basic functionality of callback with simple inputs."""
    def f(x):
        return np.sin(x)
    
    @arc.compile
    def model(x):
        return arc.callback(f, x)
    
    x = np.array([0.0, np.pi/2, np.pi])
    result = model(x)
    expected = f(x)
    
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    assert np.allclose(result, expected)
    print("Basic callback test passed!")
    
    # Test that callback is only evaluated during tracing
    calls = []
    
    def counting_func(x):
        calls.append(x)
        return x * 2
    
    @arc.compile
    def counter_model(x):
        return arc.callback(counting_func, x)
    
    # First call triggers tracing
    counter_model(np.array([1.0]))
    
    # These calls should reuse the traced function
    counter_model(np.array([2.0]))
    counter_model(np.array([3.0]))
    
    # Should only have one call recorded
    assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"
    print("Trace-time evaluation test passed!")


def test_tree_structured_callback():
    """Test with tree-structured data."""
    def tree_func(data):
        return {
            'doubled': data['values'] * 2,
            'squared': data['values'] ** 2
        }
    
    @arc.compile
    def tree_model(data):
        return arc.callback(tree_func, data)
    
    data = {'values': np.array([1.0, 2.0, 3.0])}
    result = tree_model(data)
    expected = tree_func(data)
    
    assert all(np.allclose(result[k], expected[k]) for k in expected)
    print("Tree-structured callback test passed!")


def test_callback_with_side_effects():
    """Test that callbacks with side effects work as expected."""
    results = []
    
    def func_with_side_effect(x):
        # Side effect: append to global list
        results.append(x)
        return x * 2
    
    @arc.compile
    def side_effect_model(x):
        return arc.callback(func_with_side_effect, x)
    
    # First call during tracing
    x1 = np.array([1.0])
    side_effect_model(x1)
    
    # Second call during evaluation (should not trigger side effect again)
    x2 = np.array([2.0])
    side_effect_model(x2)
    
    # Should only have one side effect from trace time
    assert len(results) == 1
    assert np.allclose(results[0], x1)
    print("Side effect test passed!")


def test_multiple_arguments():
    """Test with multiple input arguments."""
    def multi_arg_func(x, y, z):
        return x * y + z
    
    @arc.compile
    def multi_model(x, y, z):
        return arc.callback(multi_arg_func, x, y, z)
    
    x = np.array([1.0, 2.0])
    y = 3.0
    z = np.array([0.5, 1.0])
    
    result = multi_model(x, y, z)
    expected = multi_arg_func(x, y, z)
    
    assert np.allclose(result, expected)
    print("Multiple arguments test passed!")


def test_error_propagation():
    """Test that errors in the callback function are properly propagated."""
    def error_func(x):
        if x[0] < 0:
            raise ValueError("Negative input not allowed")
        return np.sqrt(x)
    
    try:
        @arc.compile
        def error_model(x):
            return arc.callback(error_func, x)
        
        # This should fail at trace time
        error_model(np.array([-1.0, 4.0]))
        
        # If we get here, something went wrong
        assert False, "Should have raised an exception"
    except ValueError as e:
        assert "Negative input not allowed" in str(e)
        print("Error propagation test passed!")


if __name__ == "__main__":
    print("Testing arc.callback functionality")
    test_basic_callback()
    test_tree_structured_callback()
    test_callback_with_side_effects()
    test_multiple_arguments()
    test_error_propagation()
    print("All tests completed!")