import numpy as np
import pytest

from archimedes.experimental.coco.ocp import (
    BoundaryData,
    Constraint,
    MultiStageOptimalControlProblem,
    OptimalControlProblem,
    control_bounds,
    end_time,
    final_condition,
    initial_condition,
    parameter_bounds,
    start_time,
    state_bounds,
)


def test_boundary_data():
    t0, x0 = 0.0, np.array([1.0, 2.0])
    tf, xf = 1.0, np.array([3.0, 4.0])
    p = np.array([5.0])
    data = BoundaryData(t0, x0, tf, xf, p)
    assert data.t0 == t0
    assert np.array_equal(data.x0, x0)
    assert data.tf == tf
    assert np.array_equal(data.xf, xf)
    assert np.array_equal(data.p, p)


def test_constraint():
    def func(x):
        return x**2

    nc = 1
    c = Constraint(func, nc, lower_bound=np.array([0.0]), upper_bound=np.array([1.0]))
    assert c.func(2.0) == 4.0
    assert c.nc == nc
    assert np.array_equal(c.lower_bound, [0.0])
    assert np.array_equal(c.upper_bound, [1.0])


def test_initial_condition():
    x0 = np.array([1.0, 2.0])
    c = initial_condition(x0)
    data = BoundaryData(
        0.0, np.array([1.0, 2.0]), 1.0, np.array([3.0, 4.0]), np.array([])
    )
    assert np.array_equal(c(data), np.zeros(2))


def test_final_condition():
    xf = np.array([3.0, 4.0])
    c = final_condition(xf)
    data = BoundaryData(
        0.0, np.array([1.0, 2.0]), 1.0, np.array([3.0, 4.0]), np.array([])
    )
    assert np.array_equal(c(data), np.zeros(2))


def test_start_time():
    t0 = 0.0
    c = start_time(t0)
    data = BoundaryData(
        0.0, np.array([1.0, 2.0]), 1.0, np.array([3.0, 4.0]), np.array([])
    )
    assert c(data) == 0.0


def test_end_time():
    tf = 1.0
    c = end_time(tf)
    data = BoundaryData(
        0.0, np.array([1.0, 2.0]), 1.0, np.array([3.0, 4.0]), np.array([])
    )
    assert c(data) == 0.0


def test_parameter_bounds():
    lb, ub = np.array([-1.0]), np.array([1.0])
    c = parameter_bounds(lb, ub)
    p = np.array([0.5])
    data = BoundaryData(0.0, np.array([1.0, 2.0]), 1.0, np.array([3.0, 4.0]), p)
    assert np.array_equal(c(data), p)
    assert np.array_equal(c.lower_bound, lb)
    assert np.array_equal(c.upper_bound, ub)

    # Error case: bound dimension mismatch
    lb, ub = np.array([-1.0, -2.0]), np.array([1.0])
    with pytest.raises(ValueError):
        parameter_bounds(lb, ub)


def test_state_bounds():
    lb, ub = np.array([-1.0, -2.0]), np.array([1.0, 2.0])
    c = state_bounds(lb, ub)
    t, x, u, p = 0.0, np.array([0.5, 1.5]), np.array([0.0]), np.array([])
    assert np.array_equal(c(t, x, u, p), x)
    assert np.array_equal(c.lower_bound, lb)
    assert np.array_equal(c.upper_bound, ub)

    # Error case: bound dimension mismatch
    lb, ub = np.array([-1.0, -2.0]), np.array([1.0])
    with pytest.raises(ValueError):
        state_bounds(lb, ub)


def test_control_bounds():
    lb, ub = np.array([-1.0]), np.array([1.0])
    c = control_bounds(lb, ub)
    t, x, u, p = 0.0, np.array([0.0, 0.0]), np.array([0.5]), np.array([])
    assert np.array_equal(c(t, x, u, p), u)
    assert np.array_equal(c.lower_bound, lb)
    assert np.array_equal(c.upper_bound, ub)

    # Error case: bound dimension mismatch
    lb, ub = np.array([-1.0, -2.0]), np.array([1.0])
    with pytest.raises(ValueError):
        control_bounds(lb, ub)


def test_optimal_control_problem():
    nx, nu = 2, 1

    def ode(t, x, u, p):
        return np.array([x[1], u[0]], like=x)

    def quad(t, x, u, p):
        return x[0] ** 2 + u[0] ** 2

    def cost(x0, t0, xf, tf, q, p):
        return q

    boundary_constraints = [initial_condition(np.zeros(2)), final_condition(np.ones(2))]
    path_constraints = [state_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))]

    ocp = OptimalControlProblem(
        nx,
        nu,
        ode=ode,
        quad=quad,
        cost=cost,
        boundary_constraints=boundary_constraints,
        path_constraints=path_constraints,
    )

    assert ocp.nx == nx
    assert ocp.nu == nu
    assert ocp.np == 0
    p = np.array([])
    assert np.allclose(
        ocp.ode(0.0, np.array([1.0, 2.0]), np.array([3.0]), p), np.array([2.0, 3.0])
    )
    assert ocp.quad(0.0, np.array([1.0, 2.0]), np.array([3.0]), p) == 10.0
    assert ocp.cost(np.array([1.0, 2.0]), 0.0, np.array([3.0, 4.0]), 1.0, 5.0, p) == 5.0
    assert len(ocp.boundary_constraints) == 2
    assert len(ocp.path_constraints) == 1


def test_multistage():
    # Construct a multistage problem
    nx, nu = 2, 1

    def ode(t, x, u, p):
        return np.array([x[1], u[0]], like=x)

    def quad(t, x, u, p):
        return x[0] ** 2 + u[0] ** 2

    def cost(x0, t0, xf, tf, q, p):
        return q

    stage1 = OptimalControlProblem(
        nx,
        nu,
        ode=ode,
        quad=quad,
        cost=cost,
    )

    stage2 = OptimalControlProblem(
        nx,
        nu,
        ode=ode,
        quad=quad,
        cost=cost,
    )

    def stage_constraint(bd):
        return bd[0].xf - bd[1].x0  # State continuity

    ocp = MultiStageOptimalControlProblem(
        stages=[stage1, stage2],
        stage_constraints=[Constraint(stage_constraint, nx)],
    )

    # TODO: Remove when implemented
    with pytest.raises(NotImplementedError):
        ocp.dynamics_residual(None, None, None, None, None)
