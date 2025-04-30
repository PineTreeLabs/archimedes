from scipy.linalg import qr as scipy_qr

def qr(jacobian, pivot=True):
    """
    Compute QR factorization of the Jacobian matrix with pivoting
    
    Parameters
    ----------
    jacobian : ndarray, shape (m, n)
        Jacobian matrix
    pivot : bool, optional
        Whether to use column pivoting
        
    Returns
    -------
    Q : ndarray, shape (m, m)
        Orthogonal matrix Q
    R : ndarray, shape (m, n)
        Upper triangular matrix R
    p : ndarray, shape (n,)
        Permutation indices
    """
    m, n = jacobian.shape
    if pivot:
        q, r, p = scipy_qr(jacobian, pivoting=True, mode='economic')
        return q, r, p
    else:
        q, r = scipy_qr(jacobian, mode='economic')
        return q, r, np.arange(n)