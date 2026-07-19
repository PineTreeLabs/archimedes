#
# Copyright (c) 2025 Pine Tree Labs, LLC.
#
# This file is part of Archimedes
# (see github.com/pinetreelabs/archimedes).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.#
from __future__ import annotations
import abc
import dataclasses
from typing import Callable

import numpy as np

from archimedes import compile, minimize, vmap

from .ocp import OCPBase, BoundaryData, Constraint, OptimalControlSolution


@dataclasses.dataclass
class UniformTimeGrid:
    N: int

    @property
    def n_nodes(self):
        return self.N

    def time_nodes(self, t0, tf):
        return np.arange(self.n_nodes) / (self.n_nodes - 1) * (tf - t0) + t0

    def create_interpolants(self, xp, up, t0, tf):
        tp = self.time_nodes(t0, tf)

        # TODO: Use cubic spline interpolation for the state
        @compile
        def x_fn(t):
            return np.stack([np.interp(t, tp, xp[:, i]) for i in range(xp.shape[1])]).T

        @compile
        def u_fn(t):
            return np.stack([np.interp(t, tp, up[:, i]) for i in range(up.shape[1])]).T

        return x_fn, u_fn


@dataclasses.dataclass
class HermiteSimpsonCollocation(OCPBase):
    nx: int
    nu: int
    ode: Callable
    quad: Callable
    cost: Callable
    np: int = 0
    nq: int = 1
    boundary_constraints: list[Constraint] = dataclasses.field(default_factory=list)
    lbx: np.ndarray | None = None
    ubx: np.ndarray | None = None
    lbu: np.ndarray | None = None
    ubu: np.ndarray | None = None
    lbp: np.ndarray | None = None
    ubp: np.ndarray | None = None

    def unpack_dvs(self, dvs, domain, order="C"):
        # Note on ordering: CasADi orders the symbolic variables in Fortran style,
        # but the NumPy default is C-style.  So the default behavior is to unpack
        # with F ordering in the objective and constraint functions, but then use
        # C ordering by default in the unpacking function for postprocessing, etc.
        N = domain.n_nodes
        nx, nu = self.nx, self.nu
        i0, i1 = 0, N * nx
        x = np.reshape(dvs[i0:i1], (N, nx), order=order)
        i0, i1 = i1, i1 + N * nu
        u = np.reshape(dvs[i0:i1], (N, nu), order=order)
        i0, i1 = i1, i1 + self.np
        p = dvs[i0:i1]
        t0, tf = dvs[-2:]
        return x, u, t0, tf, p

    def boundary_data(self, dvs, domain):
        x, u, t0, tf, p = self.unpack_dvs(dvs, domain, order="C")
        return BoundaryData(t0, x[0, :].T, tf, x[-1, :].T, p)

    def _integrate(self, wi, wm, wf):
        """Gauss-Lobatto quadrature for the interval (-1, 1)"""
        return (wf + 4 * wm + wi) / 3

    def eval_collocation(self, domain, dvs):
        # Unpack decision variables
        x, u, t0, tf, p = self.unpack_dvs(dvs, domain, order="C")

        N = domain.n_nodes
        t = domain.time_nodes(t0, tf)
        dt = (tf - t0) / (N - 1)

        # Evaluate RHS and quadrature for loss
        f = np.zeros_like(x)
        r = np.zeros((N, self.nq), like=x)  # Running cost
        for k in range(0, N):
            f[k] = self.ode(t[k], x[k], u[k], p)
            r[k] = self.quad(t[k], x[k], u[k], p)

        # Set up collocation constraints
        g = []
        q = np.zeros(self.nq, like=x)
        for k in range(0, N - 1):
            # Midpoint interpolation constraint
            xc = 0.5 * (x[k] + x[k + 1]) + (dt / 8) * (f[k] - f[k + 1])
            uc = 0.5 * (u[k] + u[k + 1])
            tc = 0.5 * (t[k] + t[k + 1])
            fc = self.ode(tc, xc, uc, p)
            rc = self.quad(tc, xc, uc, p)

            # Endpoint interpolation constraint
            dx_quad = (dt / 2) * self._integrate(f[k], fc, f[k + 1])
            dx_bnd = x[k + 1] - x[k]
            g.append(dx_bnd - dx_quad)

            # Quadrature
            q += (dt / 2) * self._integrate(r[k], rc, r[k + 1])

        J = self.cost(x[0], t0, x[-1], tf, q, p)
        g = np.concatenate(g)
        return J, g

    def build_objective(self, domain: UniformTimeGrid):
        # Given a flattened array of decision variables, compute the
        # value of the objective function
        @compile
        def obj(dvs):
            J, _g = self.eval_collocation(domain, dvs)
            return J

        return obj

    def build_constraints(self, domain):
        # Given a flattened array of decision variables, compute the
        # value of the constraints
        @compile
        def constr(dvs):
            _J, g = self.eval_collocation(domain, dvs)

            # Add boundary conditions
            boundary_data = self.boundary_data(dvs, domain)
            for bc in self.boundary_constraints:
                g = np.append(g, bc(boundary_data))

            # Add time ordering constraint
            t0, tf = dvs[-2:]
            g = np.append(g, tf - t0)

            return g

        return constr

    def build_bounds(self, domain, order="C"):
        lbx, ubx = self.lbx, self.ubx
        if lbx is None:
            lbx = -np.inf * np.ones(self.nx)
        if ubx is None:
            ubx = np.inf * np.ones(self.nx)

        lbx = np.tile(lbx, (domain.n_nodes, 1))
        ubx = np.tile(ubx, (domain.n_nodes, 1))

        lbu, ubu = self.lbu, self.ubu
        if lbu is None:
            lbu = -np.inf * np.ones(self.nu)
        if ubu is None:
            ubu = np.inf * np.ones(self.nu)

        lbu = np.tile(lbu, (domain.n_nodes, 1))
        ubu = np.tile(ubu, (domain.n_nodes, 1))

        lbp, ubp = self.lbp, self.ubp
        if lbp is None:
            lbp = -np.inf * np.ones(self.np)
        if ubp is None:
            ubp = np.inf * np.ones(self.np)

        # Bounds on time (handled by time ordering constraint)
        lbt = np.array([-np.inf])
        ubt = np.array([np.inf])

        # Concatenate bounds for decision variables
        lb = np.concatenate([lbx.flatten(order), lbu.flatten(order), lbp, lbt])
        ub = np.concatenate([ubx.flatten(order), ubu.flatten(order), ubp, ubt])

        return lb, ub

    def initialize(
        self, domain, t_guess=None, x_guess=None, u_guess=None, p_guess=None
    ):
        N = domain.n_nodes
        nx, nu = self.nx, self.nu

        if t_guess is None:
            t_guess = [0.0, 1.0]

        t0, tf = t_guess
        t = domain.time_nodes(t0, tf)

        if p_guess is None:
            p_guess = np.zeros(self.np)

        if x_guess is None:
            x_guess = lambda t: np.zeros(nx)

        if u_guess is None:
            u_guess = lambda t: np.zeros(nu)

        x0 = np.vstack([x_guess(t[i]) for i in range(N)])
        u0 = np.vstack([u_guess(t[i]) for i in range(N)])

        initial_guess = np.hstack([x0.flatten(), u0.flatten(), p_guess.flatten(), tf])

        # Initialize constraint vectors for dynamic constraints
        lbg = np.zeros(nx * (N - 1))
        ubg = np.zeros(nx * (N - 1))

        # Boundary condition constraints
        for bc in self.boundary_constraints:
            lbg = np.append(lbg, bc.lower_bound)
            ubg = np.append(ubg, bc.upper_bound)

        # Time ordering constraint
        lbg = np.append(lbg, 0.0)
        ubg = np.append(ubg, np.inf)

        # Bounds on decision variables
        bounds = self.build_bounds(domain)

        constr_bounds = (lbg, ubg)

        return initial_guess, bounds, constr_bounds

    def postprocess(self, sol, domain):
        x, u, t0, tf, p = self.unpack_dvs(sol, domain, order="C")
        t = domain.time_nodes(t0, tf)
        x_fn, u_fn = domain.create_interpolants(x, u, t0, tf)
        return OptimalControlSolution(x, u, t, x_fn, u_fn, p, dvs=sol)

    def dynamics_residual(self, sol, element, x, t0, tf):
        """Compute the residual of the dynamics for a given solution"""
        raise NotImplementedError(
            "Dynamics residual not implemented for Hermite-Simpson collocation"
        )

    def solve(self, domain, t_guess=None, x_guess=None, u_guess=None, **options):
        initial_guess, bounds, constr_bounds = self.initialize(
            domain, t_guess, x_guess, u_guess
        )
        obj = self.build_objective(domain)
        cons = self.build_constraints(domain)
        opt_dvs = minimize(
            obj,
            initial_guess,
            constr=cons,
            constr_bounds=constr_bounds,
            bounds=bounds,
            **options,
        )
        return self.postprocess(opt_dvs, domain)
