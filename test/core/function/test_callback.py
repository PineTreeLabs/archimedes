import numpy as np

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


def test_tree_structured_callback():
    """Test with tree-structured data."""

    def tree_func(data):
        return {"doubled": data["values"] * 2, "squared": data["values"] ** 2}

    @arc.compile
    def tree_model(data):
        return arc.callback(tree_func, data)

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
        return arc.callback(multi_arg_func, x, y, z)

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
            return arc.callback(error_func, x)

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
