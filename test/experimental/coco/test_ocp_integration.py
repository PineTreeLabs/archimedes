"""Integration test of OCPs with analytic solution"""

# ruff: noqa: N802
# ruff: noqa: N803
# ruff: noqa: N806

import numpy as np

from archimedes.experimental import coco as cc


class TestDoubleIntegrator:
    def minimum_control(self, t, x, u, p):
        return u**2

    def minimum_time(self, t, x, u, p):
        return 1.0

    def double_integrator(self, t, x, u, p):
        return np.array([x[1], u[0]], like=x)

    def lagrange_cost(self, x0, t0, xf, tf, q, p):
        return q

    def _make_min_ctrl_problem(self, x0, xf, t0, tf, N=6):
        boundary_conditions = [
            cc.start_time(t0),
            cc.end_time(tf),
            cc.initial_condition(x0),
            cc.final_condition(xf),
        ]

        # Add bounds to test path constraints
        path_constraints = [
            cc.state_bounds([-10.0, -10.0], [10.0, 10.0]),
            cc.control_bounds([-10.0], [10.0]),
        ]

        # Define the optimal control problem
        ocp = cc.OptimalControlProblem(
            nx=2,
            nu=1,
            ode=self.double_integrator,
            quad=self.minimum_control,
            cost=self.lagrange_cost,
            boundary_constraints=boundary_conditions,
            path_constraints=path_constraints,
        )

        # Discretize the domain
        domain = cc.RadauFiniteElements(N=[N], knots=[])
        return ocp, domain

    def test_min_ctrl(self, plot=False):
        x0, xf = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        t0, tf = 0.0, 1.0

        ocp, domain = self._make_min_ctrl_problem(x0, xf, t0, tf, N=6)

        # Linearly interpolate initial guess
        def x_guess(t):
            return x0 + (t - t0) * (xf - x0) / (tf - t0)

        # Solve the optimal control problem with IPOPT
        sol = ocp.solve(domain, t_guess=(t0, tf), x_guess=x_guess)

        #
        # Check against analytic solution
        #
        def x_ex(t):
            return 3 * t**2 - 2 * t**3

        def u_ex(t):
            return 6 - 12 * t

        t0, tf = sol.t0, sol.tf
        t_plt = np.linspace(t0, tf, 10)
        x_plt = sol.x(t_plt)
        u_plt = sol.u(t_plt)

        assert np.allclose(x_plt[:, 0], x_ex(t_plt), atol=1e-3)
        assert np.allclose(u_plt.squeeze(), u_ex(t_plt), atol=1e-3)

        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 1, figsize=(7, 3), sharex=True)
            ax[0].scatter(sol.tp, sol.xp[:, 0], c="tab:red", label="Optimal trajectory")
            ax[0].plot(t_plt, x_plt[:, 0], c="tab:red")
            ax[0].plot(t_plt, x_ex(t_plt), "k--", lw=2, label="Exact solution")
            ax[0].legend()
            ax[0].grid()
            ax[0].set_ylabel(r"$x$")
            ax[1].scatter(sol.tp[:-1], sol.up, c="tab:red")
            ax[1].plot(t_plt, u_plt, c="tab:red")
            ax[1].plot(t_plt, u_ex(t_plt), "k--", lw=2)
            ax[1].grid()
            ax[1].set_ylabel(r"$u$")
            ax[1].set_xlabel(r"$t$")

            plt.show()

    def test_default_initial_guess(self):
        x0, xf = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        t0, tf = 0.0, 2.0

        ocp, domain = self._make_min_ctrl_problem(x0, xf, t0, tf, N=4)

        initial_guess, _lower_bound, _upper_bound = ocp.initialize(domain)
        x, u, t0, tf, p = ocp.unpack_dvs(initial_guess, domain)
        assert np.allclose(x, 0.0)
        assert np.allclose(u, 0.0)
        assert np.allclose(t0, 0.0)
        assert np.allclose(tf, 1.0)

    def test_min_time(self):
        x0, xf = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        u_min, u_max = -10.0, 10.0
        t0 = 0.0
        N = 4

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
            ode=self.double_integrator,
            quad=self.minimum_time,
            cost=self.lagrange_cost,
            boundary_constraints=boundary_conditions,
            path_constraints=path_constraints,
        )

        # Discretize the domain.  We can split into two segments to test "bang-bang"
        # control, knowing that the optimal switching time is halfway through based
        # on the symmetry of the problem.
        domain = cc.RadauFiniteElements(N=[N, N], knots=[0.0])

        # Linearly interpolate initial guess
        tf_guess = 1.0

        def x_guess(t):
            return x0 + (t - t0) * (xf - x0) / (tf_guess - t0)

        # Solve the optimal control problem with IPOPT
        sol = ocp.solve(domain, t_guess=(t0, tf_guess), x_guess=x_guess)

        assert np.allclose(sol.tf, 0.632455549)

        # Check for "bang-bang" control
        t1 = sol.tp[sol.tp < sol.tf / 2]
        t2 = sol.tp[sol.tp >= sol.tf / 2]

        assert np.allclose(sol.u(t1), u_max)
        assert np.allclose(sol.u(t2), u_min)


class TestProjectile:
    def projectile_dynamics(self, t, x, u, p):
        g0 = 9.81

        vx, vy = x[2], x[3]
        ax = np.zeros_like(vx)
        ay = np.zeros_like(vy) - g0
        return np.array([vx, vy, ax, ay], like=x)

    def dummy_quad(self, t, x, u, p):
        return 0.0

    def max_range(self, x0, t0, xf, tf, q, p):
        return -xf[0]

    def test_projectile(self):
        """What angle should you launch a projectile for maximum range?

        Tests a problem with a discrete parameter (launch angle)
        """
        v0 = 1.0

        # Parametric initial condition
        def initial_condition(boundary_data):
            x0 = boundary_data.x0
            theta = boundary_data.p[0]
            return x0 - np.array(
                [0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)], like=x0
            )

        def final_condition(boundary_data):
            return boundary_data.xf[1]

        t0, tf_guess = 0.0, 2.0
        boundary_conditions = [
            cc.start_time(t0),
            cc.Constraint(initial_condition, nc=4),
            cc.Constraint(final_condition, nc=1),
            cc.parameter_bounds([0.0], [np.pi]),
        ]

        # Define the optimal control problem
        ocp = cc.OptimalControlProblem(
            nx=4,
            nu=1,
            np=1,
            ode=self.projectile_dynamics,
            quad=self.dummy_quad,
            cost=self.max_range,
            boundary_constraints=boundary_conditions,
        )

        domain = cc.RadauFiniteElements(N=[3], knots=[])
        sol = ocp.solve(domain, t_guess=(t0, tf_guess))

        # Analytic solution to the projectile problem
        theta_ex = np.pi / 4
        tf_ex = 2 * v0 * np.sin(theta_ex) / 9.81
        range_ex = np.cos(theta_ex) * tf_ex
        x0_ex = np.array([0.0, 0.0, np.cos(theta_ex), np.sin(theta_ex)])
        xf_ex = np.array([range_ex, 0.0, np.cos(theta_ex), -np.sin(theta_ex)])

        assert np.allclose(sol.p[0], theta_ex)
        assert np.allclose(sol.t0, 0.0)
        assert np.allclose(sol.tf, tf_ex, atol=1e-3)
        assert np.allclose(sol.xp[0], x0_ex)
        assert np.allclose(sol.xp[-1], xf_ex, atol=1e-3)


class TestHohmannTransfer:
    def test_single_stage(self, plot=False):
        """Classical ideal Hohmann transfer"""
        mu = 1.0
        r0 = 1.0
        rf = 1.5
        t0 = 0.0

        def j(t, x, phi, p):
            return 0.0

        def f(t, x, phi, p):
            r, θ, u, v = x
            return np.array(
                [
                    u,
                    v / r,
                    v**2 / r - mu / r**2,
                    -u * v / r,
                ],
                like=x,
            )

        def cost(x0, t0, xf, tf, q, p):
            return sum(p**2)  # Total delta-v expenditure

        # Add impulsive (tangential) velocity at the beginning
        def initial_constraint(boundary_data):
            dv1 = boundary_data.p[0]
            x0_transfer = np.array(
                [r0, 0.0, 0.0, np.sqrt(mu / r0) + dv1], like=boundary_data.x0
            )
            return x0_transfer - boundary_data.x0

        # Final state of the transfer orbit should be the target orbit,
        # less the impulsive (tangential) velocity change. The final anomaly
        # is not constrained
        def terminal_constraint(boundary_data):
            dv2 = boundary_data.p[1]
            xf = boundary_data.xf[[0, 2, 3]]
            xf_circular = np.array(
                [rf, 0.0, np.sqrt(mu / rf) - dv2], like=boundary_data.xf
            )
            return xf_circular - xf

        # Discretize the domain
        N = [20]
        knots = []
        domain = cc.RadauFiniteElements(N=N, knots=knots)

        # Boundary conditions
        bcs = [
            cc.start_time(t0),
            cc.Constraint(initial_constraint, 4),
            cc.Constraint(terminal_constraint, 3),
        ]

        # Define the problem in three stages
        ocp = cc.OptimalControlProblem(
            nx=4,
            nu=0,
            np=2,
            ode=f,
            quad=j,
            cost=cost,
            boundary_constraints=bcs,
        )

        # Linearly interpolate initial guess
        x0_guess = np.array([r0, 0.0, 0.0, np.sqrt(mu / r0)])
        xf_guess = np.array([rf, np.pi, 0.0, np.sqrt(mu / rf)])
        tf_guess = 4.5

        def x_guess(t):
            return x0_guess + (t - t0) * (xf_guess - x0_guess) / (tf_guess - t0)

        sol = ocp.solve(domain, t_guess=(t0, tf_guess), x_guess=x_guess)

        # Hohmann solution
        dv1_ex = np.sqrt(mu / r0) * (np.sqrt(2 * rf / (r0 + rf)) - 1)
        dv2_ex = np.sqrt(mu / rf) * (1 - np.sqrt(2 * r0 / (r0 + rf)))

        assert np.allclose(sol.p, np.array([dv1_ex, dv2_ex]), atol=1e-3)
        assert np.allclose(sol.xp[-1, 1], np.pi, atol=1e-3)  # Anomaly of 180 degrees

        if plot:
            import matplotlib.pyplot as plt

            t_plt = np.linspace(t0, sol.tf, 100)
            x_plt = sol.x(t_plt)

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            r, θ = x_plt[:, 0], x_plt[:, 1]
            x, y = r * np.cos(θ), r * np.sin(θ)

            rf = r[-1]
            r0 = r[0]
            rmax = max(r)

            ax.plot(x, y, c="xkcd:brick")

            θ = np.linspace(0, 2 * np.pi, 1000)
            ax.plot(r0 * np.cos(θ), r0 * np.sin(θ), c="grey", ls="--")
            ax.plot(rf * np.cos(θ), rf * np.sin(θ), c="k", ls="--")

            ax.set_xlim([-1.2 * rmax, 1.2 * rmax])
            ax.set_ylim([-1.2 * rmax, 1.2 * rmax])
            ax.grid()
            plt.show()

    def test_multistage(self, plot=False):
        """Test multistage problems by adding a specified final anomaly"""
        mu = 1.0
        r0 = 1.0
        rf = 1.5

        nx, nu = 4, 0
        x0 = np.array([r0, 0.0, 0.0, np.sqrt(mu / r0)])
        xf = np.array([rf, 2 * np.pi, 0.0, np.sqrt(mu / rf)])
        t0 = 0.0

        def j(t, x, phi, p):
            return 0.0

        def f(t, x, phi, p):
            r, θ, u, v = x
            return np.array(
                [
                    u,
                    v / r,
                    v**2 / r - mu / r**2,
                    -u * v / r,
                ],
                like=x,
            )

        def cost_stage1(x0, t0, xf, tf, q, p):
            return sum(p**2)  # Total delta-v expenditure

        def cost_stage2(x0, t0, xf, tf, q, p):
            return sum(p**2)  # Total delta-v expenditure

        # State continuity between stages
        def stage_constraint(boundary_data):
            bd1, bd2 = boundary_data
            bd2_x0 = bd1.xf
            bd2_x0[3] += bd1.p  # Add delta-v to tangential velocity
            return bd2_x0 - bd2.x0

        def terminal_constraint(boundary_data):
            dv2 = boundary_data.p
            stage_xf = boundary_data.xf
            xf_circular = np.array([xf[0], xf[1], xf[2], xf[3] - dv2], like=stage_xf)
            return xf_circular - stage_xf

        # Discretize the domain (same for all three stages)
        N = [20]
        knots = []
        stage_domain = cc.RadauFiniteElements(N=N, knots=knots)

        # Boundary conditions for each stage
        bcs1 = [cc.start_time(t0), cc.initial_condition(x0)]
        bcs2 = [cc.Constraint(terminal_constraint, 4)]

        # Define the problem in three stages
        stage1 = cc.OptimalControlProblem(
            nx=nx,
            nu=nu,
            np=1,
            ode=f,
            quad=j,
            cost=cost_stage1,
            boundary_constraints=bcs1,
        )

        stage2 = cc.OptimalControlProblem(
            nx=nx,
            nu=nu,
            np=1,
            ode=f,
            quad=j,
            cost=cost_stage2,
            boundary_constraints=bcs2,
        )

        # Define the full problem
        ocp = cc.MultiStageOptimalControlProblem(
            stages=[stage1, stage2],
            stage_constraints=[
                cc.Constraint(stage_constraint, 4),
            ],
        )

        domain = [stage_domain, stage_domain]

        # Linearly interpolate initial guess
        tf_guess = 5.0

        def x_guess(t):
            return x0 + (t - t0) * (xf - x0) / (tf_guess - t0)

        sol = ocp.solve(domain, x_guess=x_guess)

        # Hohmann solution
        dv1_ex = np.sqrt(mu / r0) * (np.sqrt(2 * rf / (r0 + rf)) - 1)
        dv2_ex = np.sqrt(mu / rf) * (1 - np.sqrt(2 * r0 / (r0 + rf)))

        assert np.allclose(sol[0].p, dv1_ex, atol=1e-3)
        assert np.allclose(sol[1].p, dv2_ex, atol=1e-3)
        assert np.allclose(sol[0].xp[-1, 1], np.pi)  # Anomaly of 180 degrees
        assert np.allclose(sol[0].xp[-1, :2], sol[1].xp[0, :2])  # State continuity
        assert np.allclose(sol[1].xp[-1, 1], 2 * np.pi)  # Anomaly of 360 degrees

        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            for i in range(len(sol)):
                t_plt = np.linspace(sol[i].t0, sol[i].tf, 100)
                x_plt = sol[i].x(t_plt)
                r, θ = x_plt[:, 0], x_plt[:, 1]
                x, y = r * np.cos(θ), r * np.sin(θ)

                ax.plot(x, y, lw=2, c="xkcd:brick")

            θ = np.linspace(0, 2 * np.pi, 1000)
            ax.plot(r0 * np.cos(θ), r0 * np.sin(θ), c="grey", ls="--")
            ax.plot(rf * np.cos(θ), rf * np.sin(θ), c="k", ls="--")

            ax.set_xlim([-1.2 * rf, 1.2 * rf])
            ax.set_ylim([-1.2 * rf, 1.2 * rf])
            ax.grid()
            plt.show()


if __name__ == "__main__":
    TestDoubleIntegrator().test_min_ctrl(plot=True)
