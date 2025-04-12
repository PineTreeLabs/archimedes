import io
import sys

import numpy as np
from scipy import signal

import archimedes as arc


def test_callback():
    def f(x, y):
        print(f"f called: {x=}, {y=}")
        return x * (y + 3)

    @arc.compile
    def call_f(x, y):
        result_shape_dtypes = x  # Output has same type as input
        return arc.callback(f, result_shape_dtypes, x, y)

    x, y = np.array([1.0, 2.0]), 3.0
    z = call_f(x, y)
    assert np.allclose(z, f(x, y))
    assert z.shape == x.shape


def test_unsupported_op():
    def calc_sum_psd(x):
        # Computation that is not supported symbolically
        f, Pxx = signal.welch(x)  # noqa: N806
        print(f)
        return sum(Pxx[f > 0.1])

    # Call from within a compiled function
    @arc.compile
    def sum_psd(x):
        result_shape_dtypes = 0.0  # Template data type for
        return arc.callback(calc_sum_psd, result_shape_dtypes, x)

    x = np.arange(1024)

    y = calc_sum_psd(x)
    assert np.allclose(y, sum_psd(x))


def test_tree_structured_callback():
    """Test with tree-structured data."""

    def tree_func(data):
        return {"doubled": data["values"] * 2, "squared": data["values"] ** 2}

    @arc.compile
    def tree_model(data):
        result_shape_dtypes = {"doubled": data["values"], "squared": data["values"]}
        return arc.callback(tree_func, result_shape_dtypes, data)

    data = {"values": np.array([1.0, 2.0, 3.0])}
    result = tree_model(data)
    expected = tree_func(data)

    assert all(np.allclose(result[k], expected[k]) for k in expected)


def test_multiple_arguments():
    """Test with multiple input arguments."""

    def multi_arg_func(x, y, z):
        return x * y + z

    @arc.compile
    def multi_model(x, y, z):
        result_shape_dtypes = x
        return arc.callback(multi_arg_func, result_shape_dtypes, x, y, z)

    x = np.array([1.0, 2.0])
    y = 3.0
    z = np.array([0.5, 1.0])

    result = multi_model(x, y, z)
    expected = multi_arg_func(x, y, z)

    assert np.allclose(result, expected)


def test_error_propagation():
    """Test that errors in the callback function are properly propagated."""

    def error_func(x):
        if x[0] < 0:
            raise ValueError("Negative input not allowed")
        return np.sqrt(x)

    try:

        @arc.compile
        def error_model(x):
            result_shape_dtypes = x  # Output has same type as input
            return arc.callback(error_func, result_shape_dtypes, x)

        # This should fail at runtime
        error_model(np.array([-1.0, 4.0]))

        # If we get here, something went wrong
        assert False, "Should have raised an exception"

    # TODO: Ideally this would return the actual ValueError, but it's not
    # straightforward to extract the exception from CasADi, so the RuntimeError
    # is the best we can do for now.
    except RuntimeError as e:
        assert "Negative input not allowed" in str(e)
        print("Error propagation test passed")


def test_debug_print():
    def print_func(x):
        print("Value: {}".format(x))
        return x * 2

    # Create a function that uses the print functionality
    @arc.compile
    def print_test(x):
        return arc.callback(print_func, x, x)

    # Redirect stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function with test data
    x = np.array([1.0, 2.0, 3.0])
    result = print_test(x)

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Check that the output contains the expected text
    assert "Value: [1. 2. 3.]" in captured_output.getvalue()

    # Also verify that the function returns the correct result
    assert np.allclose(result, x * 2)
