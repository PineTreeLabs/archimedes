import numpy as np

from archimedes import struct

__all__ = ["iir_step", "IIRFilter"]


def iir_step(
    u: float, u_prev: np.ndarray, y_prev: np.ndarray, b: np.ndarray, a: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Perform one step of IIR filtering

    Applies an IIR filter using the difference equation:

    .. code-block:: none

        y[n] = (b[0] * u[n] + b[1] * u[n-1] + ... + b[M] * u[n-M]
                - a[1] * y[n-1] - ... - a[N] * y[n-N]) / a[0]

    Parameters
    ----------
    u : float
        Current input sample.
    u_prev : ndarray
        Previous input samples, with the most recent sample at index 0.
    y_prev : ndarray
        Previous output samples, with the most recent sample at index 0.
    b : ndarray
        Coefficients for the numerator, in order of descending degree.
    a : ndarray
        Coefficients for the denominator, in order of descending degree.

    Returns
    -------
    u_prev : ndarray
        Updated input history.
    y_prev : ndarray
        Updated output history.
    y : float
        Current output sample.

    Examples
    --------
    A simple moving average filter with a window size of 3:
    >>> b = np.array([1/3, 1/3, 1/3])  # Numerator coefficients
    >>> a = np.array([1.0, 0.0])       # Denominator coefficients
    >>> u_prev = np.zeros(len(b))       # Initialize input history
    >>> y_prev = np.zeros(len(a) - 1)   # Initialize output history
    >>> inputs = [1, 2, 2, 2, 2]
    >>> outputs = []
    >>> for u in inputs:
    ...     u_prev, y_prev, y = iir_step(u, u_prev, y_prev, b, a)
    ...     outputs.append(y)
    >>> outputs
    [1.0, 1.3333, 1.6666, 2.0, 2.0]
    """

    # Update input history
    u_prev[1:] = u_prev[:-1]
    u_prev[0] = u

    # Apply the difference equation
    y = (np.dot(b, u_prev) - np.dot(a[1:], y_prev[: len(a) - 1])) / a[0]

    # Update output history
    y_prev[1:] = y_prev[:-1]
    y_prev[0] = y

    return u_prev, y_prev, y


@struct
class IIRFilter:
    """Infinite Impulse Response (IIR) filter.

    Implements a discrete-time IIR filter using the difference equation:

    .. code-block:: none

        y[n] = (b[0] * u[n] + b[1] * u[n-1] + ... + b[M] * u[n-M]
                - a[1] * y[n-1] - ... - a[N] * y[n-N]) / a[0]

    where:
    - u[n] is the current input sample
    - y[n] is the current output sample
    - b are the numerator coefficients
    - a are the denominator coefficients

    Parameters
    ----------
    b : ndarray
        Coefficients for the numerator, in order of descending degree.
    a : ndarray
        Coefficients for the denominator, in order of descending degree.
    """

    b: np.ndarray  # Coefficients for the numerator
    a: np.ndarray  # Coefficients for the denominator

    @struct
    class State:
        u_prev: np.ndarray
        y_prev: np.ndarray

    def __call__(self, state: State, u: float) -> tuple[State, float]:
        u_prev, y_prev, y = iir_step(u, state.u_prev, state.y_prev, self.b, self.a)
        state = IIRFilter.State(u_prev=u_prev, y_prev=y_prev)
        return state, y

    @property
    def x0(self) -> State:
        """Create a new state with zeroed input and output history."""
        u_prev = np.zeros(len(self.b))
        y_prev = np.zeros(len(self.a) - 1)
        return IIRFilter.State(u_prev=u_prev, y_prev=y_prev)
