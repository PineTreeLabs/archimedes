from __future__ import annotations
from typing import ClassVar, NewType

import numpy as np
import scipy.io as sio

import archimedes as arc
from archimedes import struct


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@struct
class SigmoidNonlinearity:
    c: float  # translation
    b: float  # dilation
    s: float  # output coefficient
    y0: float  # offset/bias

    def __call__(self, u):
        return self.y0 + self.s * sigmoid(self.b * u + self.c)


@struct
class StateSpace:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    State: ClassVar[type] = NewType("State", np.ndarray)

    def dynamics(self, t, x, u):
        x = np.atleast_1d(x)
        u = np.atleast_1d(u)
        return self.A @ x + self.B @ u

    def output(self, t, x, u):
        x = np.atleast_1d(x)
        u = np.atleast_1d(u)
        return self.C @ x + self.D @ u

    @property
    def empty_state(self) -> State:
        """Return empty state for this system"""
        return np.zeros(self.A.shape[0])


@struct
class HammersteinWienerSystem:
    sys: StateSpace
    f: SigmoidNonlinearity = None
    g: SigmoidNonlinearity = None

    State: ClassVar[type] = StateSpace.State

    def dynamics(self, t, x, u):
        return self.sys.dynamics(t, x, u)

    def output(self, t, x, u):
        if self.f is not None:
            x = self.f(x)
        y = self.sys.output(t, x, u)
        if self.g is not None:
            y = self.g(y)
        return y

    @property
    def empty_state(self) -> State:
        """Return empty state for this system"""
        return self.sys.empty_state
