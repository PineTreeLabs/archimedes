import numpy as np
import pytest

from archimedes import discretize


class TestRK4:
    @pytest.mark.parametrize("n_steps", [1, 2])
    def test_rk4(self, n_steps, plot=False):
        def f(t, x, u, p):
            return np.stack([x[1], -x[0]])

        h = 1e-2
        step = discretize(f, h, method="rk4", n_steps=n_steps, name="test_rk4")

        x0 = np.array([1, 0])
        t0 = 0
        t_end = 10.0
        t_eval = np.arange(t0, t_end, h)
        x_ex = np.stack([np.cos(t_eval), -np.sin(t_eval)], axis=1)
        x_arc = np.zeros((len(t_eval), 2))
        x_arc[0] = x0
        for i in range(len(t_eval) - 1):
            x_arc[i + 1] = step(t_eval[i], x_arc[i], 0, 0)

        if plot:
            import matplotlib.pyplot as plt

            plt.plot(t_eval, x_arc, label="arc")
            plt.plot(t_eval, x_ex, "--", label="exact")
            plt.show()

        assert np.allclose(x_arc, x_ex)


class TestRadau5:
    @pytest.mark.parametrize("n_steps", [1, 2])
    def test_radau5(self, n_steps, plot=False):
        def f(t, x, u, p):
            return np.stack([x[1], -x[0]])

        h = 1e-1
        step = discretize(f, h, method="radau5", n_steps=n_steps)

        x0 = np.array([1, 0])
        t0 = 0
        t_end = 10.0
        t_eval = np.arange(t0, t_end, h)
        x_ex = np.stack([np.cos(t_eval), -np.sin(t_eval)], axis=1)
        x_arc = np.zeros((len(t_eval), 2))
        x_arc[0] = x0
        for i in range(len(t_eval) - 1):
            x_arc[i + 1] = step(t_eval[i], x_arc[i], 0, 0)

        if plot:
            import matplotlib.pyplot as plt

            plt.plot(t_eval, x_arc, label="arc")
            plt.plot(t_eval, x_ex, "--", label="exact")
            plt.show()

        assert np.allclose(x_arc, x_ex)

    def test_error_handling(self):
        def f(t, x, u, p):
            return np.stack([x[1], -x[0]])

        # No dt argument
        with pytest.raises(ValueError, match="dt must be specified"):
            discretize(f, method="rk4")

        # Decorator mode without dt
        with pytest.raises(ValueError, match="dt must be specified"):

            @discretize(method="rk4")
            def f(t, x, u, p):
                return np.stack([x[1], -x[0]])

    def test_decorator_usage(self):
        @discretize(dt=0.01, method="rk4")
        def f(t, x, u, p):
            return np.stack([x[1], -x[0]])
