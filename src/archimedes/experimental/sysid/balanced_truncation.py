#
# Copyright (c) 2025 Pine Tree Labs, LLC.
#
# This file is part of Archimedes
# (see github.com/pinetreelabs/archimedes).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.#
import numpy as np
from scipy import linalg


def balanced_truncation(A, B, C, D=None, r=None, tol=1e-5):
    """
    Balanced truncation for linear time-invariant systems using NumPy/SciPy.

    Parameters:
    -----------
    A : ndarray
        System matrix, shape (n, n)
    B : ndarray
        Input matrix, shape (n, m)
    C : ndarray
        Output matrix, shape (p, n)
    D : ndarray, optional
        Feedthrough matrix, shape (p, m). Default is None.
    r : int, optional
        Order of the reduced model. Default is None (automatically determined).
    tol : float, optional
        Tolerance for automatic order selection. Default is 1e-5.
        Only used if r is None.

    Returns:
    --------
    A_r : ndarray
        Reduced system matrix, shape (r, r)
    B_r : ndarray
        Reduced input matrix, shape (r, m)
    C_r : ndarray
        Reduced output matrix, shape (p, r)
    D_r : ndarray
        Reduced feedthrough matrix, shape (p, m)
    hsv : ndarray
        Hankel singular values
    """
    # Check inputs
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)

    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise ValueError("A must be a square matrix")

    n = A.shape[0]

    if not (B.ndim == 2 and B.shape[0] == n):
        raise ValueError("B must have the same number of rows as A")

    if not (C.ndim == 2 and C.shape[1] == n):
        raise ValueError("C must have the same number of columns as A")

    # Set default D matrix if not provided
    if D is None:
        p, m = C.shape[0], B.shape[1]
        D = np.zeros((p, m))
    else:
        D = np.asarray(D)
        if not (D.ndim == 2 and D.shape[0] == C.shape[0] and D.shape[1] == B.shape[1]):
            raise ValueError(
                "D must have shape (p, m) where p is the number of rows in C and m is the number of columns in B"
            )

    # Check system stability before proceeding
    eigs = linalg.eigvals(A)
    if np.max(np.real(eigs)) >= 0:
        raise ValueError(
            "System is not stable. Balanced truncation requires a stable system."
        )

    # Solve Lyapunov equations for Gramians with improved numerical conditioning
    # For controllability Gramian: AWc + WcA^T + BB^T = 0
    Wc = linalg.solve_continuous_lyapunov(A, -B @ B.T)

    # For observability Gramian: A^TWo + WoA + C^TC = 0
    Wo = linalg.solve_continuous_lyapunov(A.T, -C.T @ C)

    # Ensure symmetry (numerical stability)
    Wc = (Wc + Wc.T) / 2
    Wo = (Wo + Wo.T) / 2

    # Check if Gramians are positive definite
    try:
        # This will either apply Cholesky or eigendecomposition based on condition
        sqrt_method = "auto"
        Lc, Lo, U, hsv, Vh = _compute_balancing_transform(Wc, Wo, sqrt_method)
    except Exception as e:
        print(f"Warning: First method failed. Trying with eigendecomposition: {e}")
        sqrt_method = "eig"
        Lc, Lo, U, hsv, Vh = _compute_balancing_transform(Wc, Wo, sqrt_method)

    # Determine reduced order
    if r is None:
        # Automatic order selection based on singular value decay
        if len(hsv) == 1:
            r = 1
        else:
            # Normalize singular values
            hsv_normalized = hsv / hsv[0]
            # Find where normalized HSV falls below tolerance
            r_indices = np.where(hsv_normalized < tol)[0]
            r = r_indices[0] if len(r_indices) > 0 else n
            # Ensure at least one state is kept
            r = max(1, r)

    r = min(r, n)  # Ensure r doesn't exceed system order

    # Truncate to desired order
    hsv_r = hsv[:r]
    U_r = U[:, :r]
    Vh_r = Vh[:r, :]

    # Compute balancing transformation and its inverse
    Sigma_r_inv_sqrt = np.diag(hsv_r ** (-0.5))
    T = Lc @ Vh_r.T @ Sigma_r_inv_sqrt  # Balancing transformation
    Tinv = Sigma_r_inv_sqrt @ U_r.T @ Lo.T  # Inverse transformation

    # Apply balancing transformation and truncate
    A_r = Tinv @ A @ T
    B_r = Tinv @ B
    C_r = C @ T
    D_r = D

    return A_r, B_r, C_r, D_r, hsv


def _compute_balancing_transform(Wc, Wo, method="auto"):
    """
    Helper function to compute the balancing transform with different methods.

    Parameters:
    -----------
    Wc : ndarray
        Controllability Gramian
    Wo : ndarray
        Observability Gramian
    method : str
        Method for computing square root: 'auto', 'chol' or 'eig'

    Returns:
    --------
    Lc : ndarray
        Cholesky/square root factor of Wc
    Lo : ndarray
        Cholesky/square root factor of Wo
    U : ndarray
        Left singular vectors
    hsv : ndarray
        Hankel singular values
    Vh : ndarray
        Right singular vectors
    """
    if method == "auto":
        # Try Cholesky first, fall back to eigendecomposition
        try:
            return _compute_balancing_transform(Wc, Wo, "chol")
        except:
            return _compute_balancing_transform(Wc, Wo, "eig")

    elif method == "chol":
        # Use Cholesky decomposition (more efficient)
        Lc = linalg.cholesky(Wc, lower=True)
        Lo = linalg.cholesky(Wo, lower=True)

        # SVD of the product Lo^T @ Lc
        U, hsv, Vh = linalg.svd(Lo.T @ Lc, full_matrices=False)
        return Lc, Lo, U, hsv, Vh

    elif method == "eig":
        # Eigendecomposition of Gramians
        evals_c, evecs_c = linalg.eigh(Wc)
        evals_o, evecs_o = linalg.eigh(Wo)

        # Filter out negative eigenvalues (numerical issues)
        evals_c = np.maximum(evals_c, 0)
        evals_o = np.maximum(evals_o, 0)

        # Compute square root factors
        Rc = evecs_c @ np.diag(np.sqrt(evals_c))
        Ro = evecs_o @ np.diag(np.sqrt(evals_o))

        # SVD of the product Ro^T @ Rc
        U, hsv, Vh = linalg.svd(Ro.T @ Rc, full_matrices=False)

        return Rc, Ro, U, hsv, Vh

    else:
        raise ValueError(f"Unknown method: {method}")
