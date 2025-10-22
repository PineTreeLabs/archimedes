from __future__ import annotations

import numpy as np

import control

import archimedes as arc
from archimedes import struct
from archimedes.spatial import Rotation, euler_kinematics

from f16 import SubsonicF16, GRAV_FTS2, TrimPoint

@struct
class LongitudinalState:
    vt: float
    alpha: float
    theta: float
    q: float
    pow: float

    @classmethod
    def from_full_state(
        cls,
        x: SubsonicF16.State,
        vt: float = None,
        alpha: float = None
    ) -> LongitudinalState:
        if vt is None:
            vt = np.sqrt(np.dot(x.v_B, x.v_B))
        if alpha is None:
            alpha = np.arctan2(x.v_B[2], x.v_B[0])

        return cls(
            vt=vt,
            alpha=alpha,
            theta=x.att[1],
            q=x.w_B[1],
            pow=x.engine_power,
        )

    def as_full_state(self, rpy_attitude=True) -> SubsonicF16.State:
        # Assume zero sideslip, zero lateral states
        beta = 0.0
        phi = 0.0
        p = 0.0
        r = 0.0

        v_B = np.hstack([
            self.vt * np.cos(self.alpha),
            0.0,
            self.vt * np.sin(self.alpha),
        ])

        rpy = np.hstack([phi, self.theta, 0.0])
        if rpy_attitude:
            att = rpy
        else:
            att = Rotation.from_euler("xyz", [phi, self.theta, 0.0])

        w_B = np.array([p, self.q, r])

        return SubsonicF16.State(
            p_N=np.zeros(3),
            att=att,
            v_B=v_B,
            w_B=w_B,
            engine_power=self.pow,
        )

@struct
class LongitudinalInput:
    throttle: float
    elevator: float

    @classmethod
    def from_full_input(cls, u: SubsonicF16.Input):
        return cls(
            throttle=u.throttle,
            elevator=u.elevator,
        )

    def as_full_input(self) -> SubsonicF16.Input:
        return SubsonicF16.Input(
            throttle=self.throttle,
            elevator=self.elevator,
            aileron=0.0,
            rudder=0.0,
        )

@struct
class LateralState:
    beta: float
    phi: float
    p: float
    r: float

    @classmethod
    def from_full_state(
        cls,
        x: SubsonicF16.State,
        beta: float = None,
    ):
        if beta is None:
            vt = np.sqrt(np.dot(x.v_B, x.v_B))
            beta = np.arcsin(x.v_B[1] / vt)
        return cls(
            beta=beta,
            phi=x.att[0],
            p=x.w_B[0],
            r=x.w_B[2],
        )

    def as_full_state(self, vt: float, rpy_attitude=True) -> SubsonicF16.State:
        # Assume zero longitudinal states
        pow = 0.0

        v_B = np.hstack([
            vt * np.cos(self.beta),
            vt * np.sin(self.beta),
            0.0,
        ])

        rpy = np.hstack([self.phi, 0.0, 0.0])
        if rpy_attitude:
            att = rpy
        else:
            att = Rotation.from_euler("xyz", [self.phi, 0.0, 0.0])

        w_B = np.array([self.p, 0.0, self.r])

        return SubsonicF16.State(
            p_N=np.zeros(3),
            att=att,
            v_B=v_B,
            w_B=w_B,
            engine_power=pow,
        )

@struct
class LateralInput:
    aileron: float
    rudder: float

    @classmethod
    def from_full_input(cls, u: SubsonicF16.Input):
        return cls(
            aileron=u.aileron,
            rudder=u.rudder,
        )

    def as_full_input(self) -> SubsonicF16.Input:
        # Assume zero longitudinal inputs
        elevator = 0.0
        throttle = 0.0

        return SubsonicF16.Input(
            elevator=elevator,
            aileron=self.aileron,
            rudder=self.rudder,
            throttle=throttle,
        )

@struct
class StabilityState:
    long: LongitudinalState
    lat: LateralState

    @classmethod
    def from_full_state(cls, x: SubsonicF16.State) -> StabilityState:
        return cls(
            long=LongitudinalState.from_full_state(x),
            lat=LateralState.from_full_state(x),
        )
    
    @classmethod
    def from_full_derivative(
        cls, x: SubsonicF16.State, x_dot: SubsonicF16.State
    ) -> StabilityState:
        # Compute time derivatives of airspeed, alpha, and beta
        vt = np.sqrt(np.dot(x.v_B, x.v_B))
        vt_dot = np.dot(x.v_B, x_dot.v_B) / vt
        dum = x.v_B[0] ** 2 + x.v_B[2] ** 2
        beta = np.arcsin(x.v_B[1] / vt)
        alpha_dot = (x.v_B[0]*x_dot.v_B[2] - x.v_B[2]*x_dot.v_B[0]) / dum
        beta_dot = (vt * x_dot.v_B[1] - x.v_B[1] * vt_dot) * np.cos(beta) / dum
        return cls(
            long=LongitudinalState.from_full_state(x_dot, vt=vt_dot, alpha=alpha_dot),
            lat=LateralState.from_full_state(x_dot, beta=beta_dot),
        )

    def as_full_state(self, rpy_attitude=True) -> SubsonicF16.State:
        p_N = np.zeros(3)

        v_W = np.hstack([self.long.vt, 0.0, 0.0])
        R_WB = Rotation.from_euler("zy", [-self.lat.beta, self.long.alpha])
        v_B = R_WB.apply(v_W, inverse=True)
        w_B = np.hstack([self.lat.p, self.long.q, self.lat.r])

        rpy = np.hstack([self.lat.phi, self.long.theta, 0.0])
        if rpy_attitude:
            att = rpy
        else:
            att = Rotation.from_euler("xyz", rpy)

        return SubsonicF16.State(
            p_N=p_N,
            att=att,
            v_B=v_B,
            w_B=w_B,
            engine_power=self.long.pow,
        )


@struct
class StabilityInput:
    long: LongitudinalInput
    lat: LateralInput

    @classmethod
    def from_full_input(cls, u: SubsonicF16.Input):
        return cls(
            long=LongitudinalInput.from_full_input(u),
            lat=LateralInput.from_full_input(u),
        )

    def as_full_input(self) -> SubsonicF16.Input:
        return SubsonicF16.Input(
            elevator=self.long.elevator,
            aileron=self.lat.aileron,
            rudder=self.lat.rudder,
            throttle=self.long.throttle,
        )

