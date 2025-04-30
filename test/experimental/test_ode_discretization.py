import numpy as np

from archimedes.experimental.discretize import discretize


def rk4_step(f, t, x, p, h):
    k1 = f(t, x, p)
    k2 = f(t + h / 2, x + h * k1 / 2, p)
    k3 = f(t + h / 2, x + h * k2 / 2, p)
    k4 = f(t + h, x + h * k3, p)
    return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class TestRK4:
    def test_rk4(self, plot=False):
        def f(t, x, p):
            return np.stack([x[1], -x[0]])

        h = 1e-2
        step = discretize(f, h, method="rk4")

        x0 = np.array([1, 0])
        t0 = 0
        t_end = 10.0
        t_eval = np.arange(t0, t_end, h)
        x_ex = np.stack([np.cos(t_eval), -np.sin(t_eval)], axis=1)
        x_arc = np.zeros((len(t_eval), 2))
        x_arc[0] = x0
        for i in range(len(t_eval) - 1):
            x_arc[i + 1] = step(t_eval[i], x_arc[i], 0)

        if plot:
            import matplotlib.pyplot as plt

            plt.plot(t_eval, x_arc, label="arc")
            plt.plot(t_eval, x_ex, "--", label="exact")
            plt.show()

        assert np.allclose(x_arc, x_ex)


class TestRadau5:
    def test_radau5(self, plot=False):
        def f(t, x, p):
            return np.stack([x[1], -x[0]])

        h = 1e-1
        step = discretize(f, h, method="radau5")

        x0 = np.array([1, 0])
        t0 = 0
        t_end = 10.0
        t_eval = np.arange(t0, t_end, h)
        x_ex = np.stack([np.cos(t_eval), -np.sin(t_eval)], axis=1)
        x_arc = np.zeros((len(t_eval), 2))
        x_arc[0] = x0
        for i in range(len(t_eval) - 1):
            x_arc[i + 1] = step(t_eval[i], x_arc[i], 0)

        if plot:
            import matplotlib.pyplot as plt

            plt.plot(t_eval, x_arc, label="arc")
            plt.plot(t_eval, x_ex, "--", label="exact")
            plt.show()

        assert np.allclose(x_arc, x_ex)
