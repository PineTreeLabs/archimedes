import numpy as np
from archimedes import struct

__all__ = ["Timeseries"]


@struct.pytree_node
class Timeseries:
    """A class to represent a time series with associated inputs and outputs.

    Attributes:
        ts: Time vector.
        us: Input signals.
        ys: Output measurements.
    """
    ts: np.ndarray
    us: np.ndarray
    ys: np.ndarray

    def __post_init__(self):
        if self.ts.ndim != 1:
            raise ValueError("Time vector must be one-dimensional.")
        if self.ys.ndim != 2:
            raise ValueError(
                "Output measurements must be two-dimensional with shape (ny, nt)."
            )
        if self.us.ndim != 2:
            raise ValueError(
                "Input signals must be two-dimensional with shape (nu, nt)."
            )
        if self.ts.size != self.ys.shape[1]:
            raise ValueError(
                "Time vector size must match the number of time points in ys."
            )
        if self.ts.size != self.us.shape[1]:
            raise ValueError(
                "Time vector size must match the number of time points in us."
            )