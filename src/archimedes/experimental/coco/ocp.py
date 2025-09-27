import abc
import dataclasses
from typing import Callable

import numpy as np

from archimedes import compile, minimize, struct, field

from .discretization import SplineDiscretization
from .interpolation import LagrangePolynomial


@dataclasses.dataclass
class BoundaryData:
    t0: float
    x0: np.ndarray
    tf: float
    xf: np.ndarray
    p: np.ndarray


@dataclasses.dataclass
class Constraint:
    func: Callable
    nc: int
    lower_bound: np.ndarray = None  # Default to equality constraint
    upper_bound: np.ndarray = None

    def __post_init__(self):
        if self.lower_bound is None:
            self.lower_bound = np.zeros(self.nc)
        if self.upper_bound is None:
            self.upper_bound = np.zeros(self.nc)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


#
# Convenience functions for boundary conditions
#
def initial_condition(x0):
    return Constraint(lambda data: data.x0 - x0, len(x0))


def final_condition(xf):
    return Constraint(lambda data: data.xf - xf, len(xf))


def start_time(t0):
    return Constraint(lambda data: data.t0 - t0, 1)


def end_time(tf):
    return Constraint(lambda data: data.tf - tf, 1)


def parameter_bounds(lb, ub):
    if len(lb) != len(ub):
        raise ValueError("Lower and upper bounds must have the same length")
    return Constraint(lambda data: data.p, len(lb), lower_bound=lb, upper_bound=ub)


#
# Convenience functions for constructing path constraints
#
def state_bounds(lb, ub):
    if len(lb) != len(ub):
        raise ValueError("Lower and upper bounds must have the same length")
    return Constraint(lambda t, x, u, p: x, len(lb), lower_bound=lb, upper_bound=ub)


def control_bounds(lb, ub):
    if len(lb) != len(ub):
        raise ValueError("Lower and upper bounds must have the same length")
    return Constraint(lambda t, x, u, p: u, len(lb), lower_bound=lb, upper_bound=ub)


@struct
class OptimalControlSolution:
    xp: np.ndarray
    up: np.ndarray
    tp: np.ndarray
    x: Callable = field(static=True)
    u: Callable = field(static=True)
    p: np.ndarray
    dvs: np.ndarray = None

    @property
    def t0(self):
        return self.tp[0]

    @property
    def tf(self):
        return self.tp[-1]


class OCPBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build_objective(self, domain):
        """Construct the objective function"""

    @abc.abstractmethod
    def build_constraints(self, domain):
        """Construct the constraint functions"""

    @abc.abstractmethod
    def initialize(self, domain, t0, tf, x_guess=None, u_guess=None):
        """Initialize the decision variables and bounds"""

    @abc.abstractmethod
    def postprocess(self, sol, domain):
        """Postprocess the solution"""

    @abc.abstractmethod
    def dynamics_residual(self, sol, element, x, t0, tf):
        """Compute the residual of the dynamics for a given solution"""

    def solve(self, domain, t_guess=None, x_guess=None, u_guess=None, options=None):
        initial_guess, lower, upper = self.initialize(domain, t_guess, x_guess, u_guess)
        obj = self.build_objective(domain)
        cons = self.build_constraints(domain)
        opt_result = minimize(
            obj,
            initial_guess,
            constr=cons,
            constr_bounds=(lower, upper),
            options=options,
        )
        opt_dvs = opt_result.x
        return self.postprocess(opt_dvs, domain)


@dataclasses.dataclass
class OptimalControlProblem(OCPBase):
    nx: int
    nu: int
    ode: Callable
    quad: Callable
    cost: Callable
    np: int = 0
    boundary_constraints: list[Constraint] = dataclasses.field(default_factory=list)
    path_constraints: list[Constraint] = dataclasses.field(default_factory=list)

    def num_dvs(self, domain):
        N = domain.n_nodes
        nx, nu = self.nx, self.nu
        return (N + 1) * nx + N * nu + 2 + self.np

    def unpack_dvs(self, dvs, domain, order="C"):
        # Note on ordering: CasADi orders the symbolic variables in Fortran style,
        # but the NumPy default is C-style.  So the default behavior is to unpack
        # with F ordering in the objective and constraint functions, but then use
        # C ordering by default in the unpacking function for postprocessing, etc.
        N = domain.n_nodes
        nx, nu = self.nx, self.nu
        i0, i1 = 0, (N + 1) * nx
        x = np.reshape(dvs[i0:i1], (N + 1, nx), order=order)
        i0, i1 = i1, i1 + N * nu
        u = np.reshape(dvs[i0:i1], (N, nu), order=order)
        i0, i1 = i1, i1 + self.np
        p = dvs[i0:i1]
        t0, tf = dvs[-2:]  # Time knots
        return x, u, t0, tf, p

    def boundary_data(self, dvs, domain):
        x, u, t0, tf, p = self.unpack_dvs(dvs, domain, order="C")
        return BoundaryData(t0, x[0, :].T, tf, x[-1, :].T, p)

    def build_objective(self, domain):
        N, w = domain.n_nodes, domain.weights

        # Given a flattened array of decision variables, compute the
        # value of the objective function
        @compile
        def obj(dvs):
            x, u, t0, tf, p = self.unpack_dvs(dvs, domain, order="C")

            tscale = 0.5 * (tf - t0)
            t = domain.time_nodes(t0, tf)

            running_cost = []
            for i in range(N):
                running_cost = np.append(
                    running_cost, self.quad(t[i], x[i, :], u[i, :], p)
                )

            # Evaluate objective
            q = tscale * np.dot(w, running_cost)
            J = self.cost(x[0, :], t0, x[-1, :], tf, q, p)
            return J

        return obj

    def build_constraints(self, domain):
        N, D = domain.n_nodes, domain.diff_matrix

        # Given a flattened array of decision variables, compute the
        # function value of the constraints
        @compile
        def cons(dvs):
            x, u, t0, tf, p = self.unpack_dvs(dvs, domain, order="C")

            t = domain.time_nodes(t0, tf)
            tscale = 0.5 * (tf - t0)

            x_dot = (
                D @ x
            )  # Numerically differentiate with Lagrange differentiation matrix

            # Dynamical constraints
            # TODO: Support mass matrix here for DAEs
            res = []
            # F = tscale * self.ode(t[:-1], x[:-1, :].T, u.T, p)

            # vmap_ode = vmap(self.ode, in_axes=(0, 1, 1, None), out_axes=1)
            # print(t[:-1].shape, x[:-1, :].T.shape, u.T.shape)
            # F = tscale * vmap_ode(t[:-1], x[:-1, :].T, u.T, p)
            for i in range(N):
                # res = np.append(res, x_dot[i, :].T - F[:, i])
                F = tscale * self.ode(t[i], x[i, :], u[i, :], p)
                res = np.append(res, x_dot[i, :].T - F)

            # # Constrain the non-collocated end point
            # w = domain.weights
            # res = np.append(res, x[-1] - (x[0] + F @ w))

            # Boundary condition constraints
            boundary_data = self.boundary_data(dvs, domain)
            for bc in self.boundary_constraints:
                res = np.append(res, bc(boundary_data))

            # Path constraints
            for pc in self.path_constraints:
                for i in range(N):
                    res = np.append(res, pc(t[i], x[i, :], u[i, :], p))

            res = np.append(res, tf - t0)  # Ordering of endpoints

            return res

        return cons

    def initialize(
        self, domain, t_guess=None, x_guess=None, u_guess=None, p_guess=None
    ):
        N = domain.n_nodes
        nx, nu = self.nx, self.nu

        if t_guess is None:
            t_guess = [0.0, 1.0]

        if p_guess is None:
            p_guess = np.zeros(self.np)

        t0, tf = t_guess
        t = domain.time_nodes(t0, tf)

        if x_guess is None:
            x_guess = lambda t: np.zeros(nx)

        if u_guess is None:
            u_guess = lambda t: np.zeros(nu)

        x0 = np.vstack([x_guess(t[i]) for i in range(N + 1)])
        u0 = np.vstack([u_guess(t[i]) for i in range(N)])

        initial_guess = np.hstack(
            [x0.flatten(), u0.flatten(), p_guess.flatten(), t0, tf]
        )

        # Initialize constraint vectors for dynamic constraints
        # lb = np.zeros(nx * (N + 1))
        # ub = np.zeros(nx * (N + 1))
        lb = np.zeros(nx * N)
        ub = np.zeros(nx * N)

        # Boundary condition constraints
        for bc in self.boundary_constraints:
            lb = np.append(lb, bc.lower_bound)
            ub = np.append(ub, bc.upper_bound)

        # Path constraints
        for pc in self.path_constraints:
            lb = np.append(lb, np.tile(pc.lower_bound, N))
            ub = np.append(ub, np.tile(pc.upper_bound, N))

        # Time ordering constraint
        lb = np.append(lb, 0.0)
        ub = np.append(ub, np.inf)

        return initial_guess, lb, ub

    def postprocess(self, sol, domain: SplineDiscretization):
        x, u, t0, tf, p = self.unpack_dvs(sol, domain, order="C")
        t = domain.time_nodes(t0, tf)
        x_fn, u_fn = domain.create_interpolants(x, u, t0, tf)
        return OptimalControlSolution(x, u, t, x_fn, u_fn, p, dvs=sol)

    def dynamics_residual(self, sol, element, x, t0, tf):
        # Differentiate then interpolate
        D = element.diff_matrix
        x_dot_ = D @ x

        # Extrapolate to end time
        τ = element.nodes
        p_radau = LagrangePolynomial(τ[:-1])  # Interpolant for collocated nodes
        x_dot_f = p_radau.interpolate(x_dot_, τ[-1])
        x_dot_ = np.vstack([x_dot_, x_dot_f])
        x_dot_fn = element.create_interpolant(x_dot_, t0, tf)

        def _res(t):
            x = sol.x(t)
            u = sol.u(t)
            F = np.zeros_like(x)
            for i in range(len(x)):
                F[i] = self.ode(t[i], x[i], u[i], sol.p)
            x_dot = x_dot_fn(t)
            return abs(x_dot - 0.5 * (tf - t0) * F)

        return _res


@dataclasses.dataclass
class MultiStageOptimalControlProblem(OCPBase):
    stages: list[OptimalControlProblem]
    stage_constraints: list[Constraint]

    def split_dvs(self, dvs, domain):
        num_dvs = [s.num_dvs(d) for (s, d) in zip(self.stages, domain)]

        # TODO: Implement np.split for symbolic arrays
        # return np.split(dvs, np.cumsum(num_dvs)[:-1])

        split_dvs = []
        idx = 0
        for n in num_dvs:
            split_dvs.append(dvs[idx : idx + n])
            idx += n
        return split_dvs

    def build_objective(self, domain: list[SplineDiscretization]):
        sub_objectives = [s.build_objective(d) for (s, d) in zip(self.stages, domain)]

        @compile
        def obj(dvs):
            # First split dvs by stage
            split_dvs = self.split_dvs(dvs, domain)

            # Sum the objectives of each stage
            return sum(_obj(x) for (_obj, x) in zip(sub_objectives, split_dvs))

        return obj

    def build_constraints(self, domains: list[SplineDiscretization]):
        sub_constraints = [
            s.build_constraints(d) for (s, d) in zip(self.stages, domains)
        ]

        @compile
        def cons(dvs):
            # First split dvs by stage
            split_dvs = self.split_dvs(dvs, domains)

            # Concatenate the constraints of each stage
            res = []
            for _cons, x in zip(sub_constraints, split_dvs):
                res = np.append(res, _cons(x))

            # Then add the multi-stage constraints
            boundary_data = []
            for s, d, x in zip(self.stages, domains, split_dvs):
                boundary_data.append(s.boundary_data(x, d))

            for sc in self.stage_constraints:
                res = np.append(res, sc(boundary_data))

            # Join the end time of each stage to the start time of the next stage
            for i in range(len(self.stages) - 1):
                tf_prev = boundary_data[i].tf
                t0_next = boundary_data[i + 1].t0
                res = np.append(res, tf_prev - t0_next)

            return res

        return cons

    def initialize(
        self,
        domain: list[SplineDiscretization],
        t_guess: list[float] = None,
        x_guess=None,
        u_guess=None,
    ):
        initial_guess = []
        lower_bound = []
        upper_bound = []

        if t_guess is None:
            t_guess = np.linspace(0, 1, len(self.stages) + 1)

        # Initial guess, lower bound, and upper bound for each stage
        for i, (s, d) in enumerate(zip(self.stages, domain)):
            t0, tf = t_guess[i], t_guess[i + 1]
            (ig, lb, ub) = s.initialize(
                d, t_guess=(t0, tf), x_guess=x_guess, u_guess=u_guess
            )
            initial_guess = np.append(initial_guess, ig)
            lower_bound = np.append(lower_bound, lb)
            upper_bound = np.append(upper_bound, ub)

        # Add stage constraints
        for sc in self.stage_constraints:
            lower_bound = np.append(lower_bound, sc.lower_bound)
            upper_bound = np.append(upper_bound, sc.upper_bound)

        # Add time continuity constraints
        for _ in range(len(self.stages) - 1):
            lower_bound = np.append(lower_bound, 0)
            upper_bound = np.append(upper_bound, 0)

        return (initial_guess, lower_bound, upper_bound)

    def postprocess(self, sol, domain):
        split_dvs = self.split_dvs(sol, domain)
        stage_solns = []
        for i in range(len(split_dvs)):
            stage_solns.append(self.stages[i].postprocess(split_dvs[i], domain[i]))

        return stage_solns

    def dynamics_residual(self, sol, element, x, t0, tf):
        raise NotImplementedError("TODO: implement for multi-stage OCPs")
