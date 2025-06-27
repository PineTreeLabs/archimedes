"""Tests of adaptive meshing capabilities for the OCP solver"""

import numpy as np
import pytest

from archimedes.experimental import coco as cc


def test_block_push():
    """Minimum-time double integrator problem"""

    def minimum_time(t, x, u, p):
        return 1.0

    def double_integrator(t, x, u, p):
        return np.array([x[1], u[0]], like=x)

    def lagrange_cost(x0, t0, xf, tf, q, p):
        return q

    x0, xf = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    u_min, u_max = -10.0, 10.0
    t0 = 0.0

    boundary_conditions = [
        cc.start_time(t0),
        cc.initial_condition(x0),
        cc.final_condition(xf),
    ]

    # Add bounds to test path constraints
    path_constraints = [
        cc.control_bounds([u_min], [u_max]),
    ]

    # Define the optimal control problem
    ocp = cc.OptimalControlProblem(
        nx=2,
        nu=1,
        ode=double_integrator,
        quad=minimum_time,
        cost=lagrange_cost,
        boundary_constraints=boundary_conditions,
        path_constraints=path_constraints,
    )

    # Linearly interpolate initial guess
    tf_guess = 1.0

    def x_guess(t):
        return x0 + (t - t0) * (xf - x0) / (tf_guess - t0)

    # Initial solution
    domain = cc.RadauFiniteElements(N=[5], knots=[])
    options = {"ipopt": {"print_level": 0}}
    sol = ocp.solve(domain, t_guess=(t0, tf_guess), x_guess=x_guess, options=options)

    # Refine mesh
    max_iter = 10
    eps = 1e-4
    for i in range(max_iter):
        print(f"\n*** Iteration {i + 1} ***")
        residuals = cc.midpoint_residuals(ocp, domain, sol)
        converged, domain = cc.refine_mesh_bisection(
            domain,
            residuals,
            eps=eps,
            incr=3,
            rho=2,
            verbose=True,
        )
        sol = ocp.solve(
            domain,
            t_guess=(sol.t0, sol.tf),
            x_guess=sol.x,
            u_guess=sol.u,
            options=options,
        )

        if converged:
            break

    assert np.allclose(sol.tf, 0.632455549)

    # Check for "bang-bang" control
    t1 = sol.tp[sol.tp < sol.tf / 2]
    t2 = sol.tp[sol.tp >= sol.tf / 2]

    assert np.allclose(sol.u(t1), u_max)
    assert np.allclose(sol.u(t2), u_min)


@pytest.mark.skip(reason="Too slow")
def test_moon_lander():
    """Analytic 'moon lander' example from Darby et al. (2011)"""

    x0 = np.array([10.0, -2.0])
    xf = np.array([0.0, 0.0])
    t0, tf_guess = 0.0, 4.0
    g = 1.5

    def j(t, x, u, p):
        return u

    def f(t, x, u, p):
        return np.array([x[1], -g + u[0]], like=x)

    def cost(x0, t0, xf, tf, q, p):
        return q

    boundary_conditions = [
        cc.start_time(t0),
        cc.initial_condition(x0),
        cc.final_condition(xf),
    ]

    path_constraints = [cc.control_bounds([0.0], [3.0])]

    # Define the optimal control problem
    ocp = cc.OptimalControlProblem(
        nx=2,
        nu=1,
        ode=f,
        quad=j,
        cost=cost,
        boundary_constraints=boundary_conditions,
        path_constraints=path_constraints,
    )

    # Linearly interpolate initial guess
    def x_guess(t):
        return x0 + (t - t0) * (xf - x0) / (tf_guess - t0)

    # Initial solution
    domain = cc.RadauFiniteElements(N=[5], knots=[])
    sol = ocp.solve(
        domain,
        t_guess=(t0, tf_guess),
        x_guess=x_guess,
        ipopt={"print_level": 0},
        print_time=0,
    )

    converged = False
    max_iter = 50

    # Refine
    for i in range(max_iter):
        print(f"\n*** Iteration {i + 1} ***")

        # sol = ocp.solve(domain, t_guess=(t0, tf), x_guess=x_guess)
        sol = ocp.solve(
            domain, t_guess=(t0, tf_guess), x_guess=x_guess, print_level=0, print_time=0
        )
        residuals = cc.midpoint_residuals(ocp, domain, sol)
        converged, domain = cc.refine_mesh_bisection(
            domain,
            residuals,
            eps=1e-6,
            incr=5,
            rho=2,
            verbose=True,
        )
        sol = ocp.solve(
            domain,
            t_guess=(sol.t0, sol.tf),
            x_guess=sol.x,
            u_guess=sol.u,
            ipopt={"print_level": 0},
            print_time=0,
        )

        if converged:
            break

    # Analytic solution
    h0, v0 = x0
    tf_opt = (2 * v0) / 3 + 4 * np.sqrt(0.5 * v0**2 + 1.5 * h0) / 3
    s_opt = 0.5 * tf_opt + v0 / 3  # Optimal switching time

    def h1(t):
        return -0.75 * t**2 + v0 * t + h0

    def v1(t):
        return -1.5 * t + v0

    def h2(t):
        return 0.75 * t**2 + (v0 - 3 * s_opt) * t + 1.5 * s_opt**2 + h0

    def v2(t):
        return 1.5 * t + v0 - 3 * s_opt

    assert np.allclose(sol.tf, tf_opt)

    delta_t = 0.1  # Tolerance for true "bang-bang" control
    t1 = sol.tp[sol.tp < s_opt - delta_t]
    t2 = sol.tp[sol.tp > s_opt + delta_t]

    atol = 1e-4
    assert np.allclose(sol.u(t1), 0.0, atol=atol)
    assert np.allclose(sol.u(t2), 3.0, atol=atol)
    assert np.allclose(h1(t1), sol.x(t1)[:, 0], atol=atol)
    assert np.allclose(v1(t1), sol.x(t1)[:, 1], atol=atol)
    assert np.allclose(h2(t2), sol.x(t2)[:, 0], atol=atol)
    assert np.allclose(v2(t2), sol.x(t2)[:, 1], atol=atol)


if __name__ == "__main__":
    test_block_push()
