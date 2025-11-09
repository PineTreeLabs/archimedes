# ruff: noqa: N806
from __future__ import annotations

import numpy as np
from engine import F16Engine
from f16 import SubsonicF16

from archimedes import struct
from archimedes.spatial import EulerAngles, Quaternion


@struct
class LongitudinalState:
    vt: float
    alpha: float
    theta: float
    q: float
    eng: F16Engine.State

    @classmethod
    def from_full_state(
        cls, x: SubsonicF16.State, vt: float = None, alpha: float = None
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
            eng=x.eng,
        )

    def as_full_state(self) -> SubsonicF16.State:
        # Assume zero sideslip, zero lateral states

        v_B = np.hstack(
            [
                self.vt * np.cos(self.alpha),
                0.0,
                self.vt * np.sin(self.alpha),
            ]
        )

        rpy = Quaternion.from_euler([0.0, self.theta, 0.0])
        w_B = np.hstack([0.0, self.q, 0.0])

        return SubsonicF16.State(
            pos=np.zeros(3),
            att=rpy,
            v_B=v_B,
            w_B=w_B,
            eng=self.eng,
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
    lon: LongitudinalState
    lat: LateralState

    @classmethod
    def from_full_state(cls, x: SubsonicF16.State) -> StabilityState:
        return cls(
            lon=LongitudinalState.from_full_state(x),
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
        alpha_dot = (x.v_B[0] * x_dot.v_B[2] - x.v_B[2] * x_dot.v_B[0]) / dum
        beta_dot = (vt * x_dot.v_B[1] - x.v_B[1] * vt_dot) / (vt**2 * np.cos(beta))
        return cls(
            lon=LongitudinalState.from_full_state(x_dot, vt=vt_dot, alpha=alpha_dot),
            lat=LateralState.from_full_state(x_dot, beta=beta_dot),
        )

    def as_full_state(self) -> SubsonicF16.State:
        p_N = np.zeros(3)

        v_W = np.hstack([self.lon.vt, 0.0, 0.0])
        R_BW = Quaternion.from_euler([-self.lat.beta, self.lon.alpha], "zy").as_matrix()
        v_B = R_BW @ v_W
        w_B = np.hstack([self.lat.p, self.lon.q, self.lat.r])

        rpy = np.hstack([self.lat.phi, self.lon.theta, 0.0])

        return SubsonicF16.State(
            pos=p_N,
            att=EulerAngles(rpy),
            v_B=v_B,
            w_B=w_B,
            eng=self.lon.eng,
        )


@struct
class StabilityInput:
    lon: LongitudinalInput
    lat: LateralInput

    @classmethod
    def from_full_input(cls, u: SubsonicF16.Input):
        return cls(
            lon=LongitudinalInput.from_full_input(u),
            lat=LateralInput.from_full_input(u),
        )

    def as_full_input(self) -> SubsonicF16.Input:
        return SubsonicF16.Input(
            elevator=self.lon.elevator,
            aileron=self.lat.aileron,
            rudder=self.lat.rudder,
            throttle=self.lon.throttle,
        )
