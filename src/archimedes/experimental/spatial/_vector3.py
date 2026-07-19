# ruff: noqa: N806, N803, N815
from __future__ import annotations
import abc
from typing import Protocol

import numpy as np

from archimedes import tree, struct

from archimedes.spatial import Attitude, RigidBody


class Vector3:
    def __init__(self, array: np.ndarray, frame: str):
        self.array = array
        self.frame = frame

    def __getitem__(self, index):
        return self.array[index]

    def __iter__(self):
        return iter(self.array)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.array}, frame='{self.frame}')"


class Velocity(Protocol):
    def kinematics(
        self, x: RigidBody.State, u: RigidBody.Input
    ) -> tuple[np.ndarray, Attitude]:
        """Calculate kinematics (position and attitude derivatives)

        Parameters
        ----------
        x : RigidBody.State
            Current state of the rigid body.
        u : RigidBody.Input
            Current input to the rigid body.

        Returns
        -------
        pos_deriv : np.ndarray
            Time derivative of position in the world frame.
        att_deriv : Attitude
            Time derivative of attitude (e.g. quaternion derivative or Euler rates).

        Notes
        -----
        This function calculates the kinematics (position and attitude derivatives)
        based on the current state (velocity and angular velocity).

        Typically this does not need to be called directly, but is available
        separately for special analysis or testing.
        """

    def derivative(self, x: RigidBody.State, u: RigidBody.Input) -> Velocity:
        """Compute the time derivative of the velocity.

        Note this is returned as the a Velocity object for type consistency
        with ODE solvers.
        """


class BodyVelocity(Vector3):
    def __init__(self, array: np.ndarray):
        super().__init__(array, frame="body")

    def kinematics(
        self, x: RigidBody.State, u: RigidBody.Input
    ) -> tuple[np.ndarray, Attitude]:
        if u.W_E is not None:
            raise ValueError(
                "Body velocity kinematics with rotating earth frame not supported"
            )

        pos_deriv = x.att.rotate(self.array)
        att_deriv = x.att.kinematics(x.w_B)
        return pos_deriv, att_deriv

    def derivative(self, x: RigidBody.State, u: RigidBody.Input) -> Velocity:
        v_B = self.array
        dv_B = (u.F_B / u.m) - np.cross(x.w_B, v_B)
        return BodyVelocity(dv_B)


class EarthVelocity(Vector3):
    def __init__(self, array: np.ndarray):
        super().__init__(array, frame="earth")

    def kinematics(
        self, x: RigidBody.State, u: RigidBody.Input
    ) -> tuple[np.ndarray, Attitude]:
        pos_deriv = self.array
        W_E = u.W_E if u.W_E is not None else np.zeros(3)
        att_deriv = x.att.kinematics(x.w_B - W_E)
        return pos_deriv, att_deriv

    def derivative(self, x: RigidBody.State, u: RigidBody.Input) -> Velocity:
        return EarthVelocity(u.F_B / u.m)
