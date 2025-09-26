import numpy as np
from archimedes import struct


@struct
class CartPole:
    m1: float = 1.0
    m2: float = 0.3
    L: float = 0.5
    g: float = 9.81

    def dynamics(self, t, q, u=None):
        x, θ, ẋ, θ̇ = q
        sθ, cθ = np.sin(θ), np.cos(θ)
        τ = 0.0 if u is None else np.atleast_1d(u)[0]
        den = self.m1 + self.m2 * sθ**2
        ẍ = (self.L * self.m2 * sθ * θ̇**2 + τ + self.m2 * self.g * cθ * sθ) / den
        θ̈ = -(
            self.L * self.m2 * cθ * sθ * θ̇**2
            + τ * cθ
            + (self.m1 + self.m2) * self.g * sθ
        ) / (self.L * den)
        return np.stack([ẋ, θ̇, ẍ, θ̈])
