import numpy as np

from archimedes import tree

__all__ = ["Timeseries"]


@tree.struct
class Timeseries:
    """Container for synchronized input-output time series data.

    This class provides a structured way to organize and validate time series
    data for system identification applications. It ensures that input signals,
    output measurements, and time vectors are consistently sized and properly
    formatted for use with Kalman filters and parameter estimation algorithms.

    The class is implemented as a `@struct`, making it compatible with tree
    operations throughout the Archimedes framework.

    Parameters
    ----------
    ts : array_like
        Time vector of shape ``(N,)`` containing monotonic time samples.
        Must be one-dimensional.
    us : array_like
        Input signal matrix of shape ``(nu, N)`` where ``nu`` is the number
        of input channels and ``N`` is the number of time samples. Each row
        represents one input channel over time.
    ys : array_like
        Output measurement matrix of shape ``(ny, N)`` where ``ny`` is the
        number of output channels and ``N`` is the number of time samples.
        Each row represents one output channel over time.

    Attributes
    ----------
    ts : ndarray
        Time vector of shape ``(N,)``.
    us : ndarray
        Input signals of shape ``(nu, N)``.
    ys : ndarray
        Output measurements of shape ``(ny, N)``.

    Notes
    -----
    **Data Organization**:
        The class follows the convention that time varies along the second
        dimension (columns), while different signals vary along the first
        dimension (rows).

    **Validation**:
        The class automatically validates that:

        - Time vector is one-dimensional
        - Input and output matrices are two-dimensional
        - All time dimensions are consistent (same N)
        - Data types are compatible with numerical operations

    **Tree Compatibility**:
        As a `@struct`, ``Timeseries`` objects can be:

        - Flattened, reconstructed, and manipulated using tree utilities
        - Stored and manipulated as structured data
        - Used in optimization algorithms that expect tree parameters

    **Immutability**:
        Instances are frozen (immutable) after creation, preventing accidental
        modification of data during analysis. Use the ``replace`` method to
        create modified copies when needed.

    **Length and Indexing**:
        The class supports indexing and length retrieval, allowing easy
        access to specific time samples or slices of the data.

    Raises
    ------
    ValueError
        If time vector is not one-dimensional.
    ValueError
        If input or output matrices are not two-dimensional.
    ValueError
        If time dimensions are inconsistent between ts, us, and ys.

    See Also
    --------
    pem : Uses Timeseries objects for parameter estimation
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

    def __len__(self):
        """Return the number of time samples."""
        return self.ts.size

    def __getitem__(self, index):
        """Get a slice of the time series data."""
        return Timeseries(
            ts=self.ts[index],
            us=self.us[:, index],
            ys=self.ys[:, index],
        )
