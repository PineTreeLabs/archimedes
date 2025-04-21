import numpy as np
import pytest

import archimedes as arc


class TestQPSolve:
    def test_casadi_example(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def g(x):
            return x[0] + x[1] - 10

        x0 = np.array([0.0, 0.0])
        sol = arc.qpsol(f, g, x0, lba=0)

        # Check against the values from the CasADi example
        assert np.allclose(sol.x, [5.0, 5.0], atol=1e-2)
        assert np.allclose(sol.lam_a, -10, atol=1e-2)

        # Test with warm starting
        sol2 = arc.qpsol(f, g, x0, lba=0, lam_a0=sol.lam_a)
        assert np.allclose(sol2.x, sol.x, atol=1e-2)
        assert np.allclose(sol2.lam_a, sol.lam_a, atol=1e-2)

    def test_with_args(self):
        def f(x, a):
            return a * x[0] ** 2 + x[1] ** 2

        def g(x, a):
            return x[0] + x[1] - 10

        x0 = np.array([0.0, 0.0])
        sol = arc.qpsol(f, g, x0, lba=0, args=(1.0,))

        # Check against the values from the CasADi example
        assert np.allclose(sol.x, [5.0, 5.0], atol=1e-2)
        assert np.allclose(sol.lam_a, -10, atol=1e-2)

        # Test with no bounds
        sol = arc.qpsol(f, g, x0, args=(1.0,))
        assert np.allclose(sol.x, [0, 0], atol=1e-2)
        assert np.allclose(sol.lam_a, 0, atol=1e-2)
        # print(sol)

        # # Test warm starting
        # sol2 = qpsol(
        #     f,
        #     g,
        #     sol.x,
        #     lba=0,
        #     lam_x0=sol.lam_x,
        #     lam_a0=sol.lam_a,
        #     check_termination=5,
        #     verbose=True,
        # )

    def test_error_handling(self):
        # Test inconsistent signatures
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def g(x, y):
            return x[0] + x[1] - 10

        with pytest.raises(ValueError, match=r".*same number of arguments.*"):
            arc.qpsol(f, g, x0=[0.0, 0.0], lba=0)

        # Test inconsistent static arguments
        def f(a, x):
            return a * (x[0] ** 2 + x[1] ** 2)

        def g(a, x):
            return a * (x[0] + x[1]) - 10

        f = arc.compile(f, static_argnames=["a"])
        g = arc.compile(g)

        with pytest.raises(ValueError, match=r".*same number of static arguments.*"):
            arc.qpsol(f, g, x0=[0.0, 0.0], lba=0)

        # Test inconsistent decision variables type
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def g(x):
            return x[0] + x[1] - 10

        with pytest.raises(ValueError, match=r".*Only scalar and vector.*"):
            arc.qpsol(f, g, x0=np.zeros((2, 2)), lba=0)
