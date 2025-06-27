import numpy as np

__all__ = ["tf2ss"]


# Almost verbatim from scipy.signal.tf2ss
# https://github.com/scipy/scipy/blob/v1.14.1/scipy/signal/_lti_conversion.py#L18-L112
def tf2ss(num, den):
    r"""Transfer function to state-space representation.

    Parameters
    ----------
    num, den : array_like
        Sequences representing the coefficients of the numerator and
        denominator polynomials, in order of descending degree. The
        denominator needs to be at least as long as the numerator.

    Returns
    -------
    A, B, C, D : ndarray
        State space representation of the system, in controller canonical
        form.

    Examples
    --------
    Convert the transfer function:

    .. math:: H(s) = \frac{s^2 + 3s + 3}{s^2 + 2s + 1}

    >>> num = [1, 3, 3]
    >>> den = [1, 2, 1]

    to the state-space representation:

    .. math::

        \dot{\textbf{x}}(t) =
        \begin{bmatrix} -2 & -1 \\ 1 & 0 \end{bmatrix} \textbf{x}(t) +
        \begin{bmatrix} 1 \\ 0 \end{bmatrix} \textbf{u}(t) \\

        \textbf{y}(t) = \begin{bmatrix} 1 & 2 \end{bmatrix} \textbf{x}(t) +
        \begin{bmatrix} 1 \end{bmatrix} \textbf{u}(t)

    >>> from scipy.signal import tf2ss
    >>> A, B, C, D = tf2ss(num, den)
    >>> A
    array([[-2., -1.],
           [ 1.,  0.]])
    >>> B
    array([[ 1.],
           [ 0.]])
    >>> C
    array([[ 1.,  2.]])
    >>> D
    array([[ 1.]])
    """
    # Controller canonical state-space representation.
    #  if M+1 = len(num) and K+1 = len(den) then we must have M <= K
    #  states are found by asserting that X(s) = U(s) / D(s)
    #  then Y(s) = N(s) * X(s)
    #
    #   A, B, C, and D follow quite naturally.
    #
    # num, den = normalize(num, den)   # Strips zeros, checks arrays

    # Normalize transfer function
    # NOTE: The scipy call to `normalize` does other checks and reshaping
    # that are skipped here - this version relies on well-formed transfer
    # functions or else may return NaN/Inf values.
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)
    num, den = num / den[0], den / den[0]

    nn = len(num.shape)
    if nn == 1:
        num = np.array([num], num.dtype, like=num)

    M = num.shape[1]
    K = len(den)
    if M > K:
        msg = "Improper transfer function. `num` is longer than `den`."
        raise ValueError(msg)
    if M == 0 or K == 0:  # Null system
        return (
            np.array([], float),
            np.array([], float),
            np.array([], float),
            np.array([], float),
        )

    # pad numerator to have same number of columns has denominator
    num = np.hstack((np.zeros((num.shape[0], K - M), dtype=num.dtype), num))

    if num.shape[-1] > 0:
        D = np.atleast_2d(num[:, 0])

    else:
        # We don't assign it an empty array because this system
        # is not 'null'. It just doesn't have a non-zero D
        # matrix. Thus, it should have a non-zero shape so that
        # it can be operated on by functions like 'ss2tf'
        D = np.array([[0]], float)

    if K == 1:
        D = D.reshape(num.shape)

        return (
            np.zeros((1, 1)),
            np.zeros((1, D.shape[1])),
            np.zeros((D.shape[0], 1)),
            D,
        )

    frow = -np.array([den[1:]], like=den)
    # A = np.r_[frow, np.eye(K - 2, K - 1)]
    A = np.vstack([frow, np.eye(K - 2, K - 1, like=den)])
    B = np.eye(K - 1, 1, like=den)
    C = num[:, 1:] - np.outer(num[:, 0], den[1:])
    D = D.reshape((C.shape[0], B.shape[1]))

    return A, B, C, D
