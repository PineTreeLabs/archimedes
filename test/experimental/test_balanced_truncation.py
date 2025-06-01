# ruff: noqa: N802, N803, N806, E741
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
from scipy import signal

from archimedes.experimental import balanced_truncation


def test_balanced_truncation(plot=False):
    n = 10  # Full order
    r = 4  # Reduced order

    # Create a stable A matrix
    np.random.seed(0)  # For reproducibility

    A = np.random.randn(n, n)
    A_off = 5 * np.random.randn(n, n)  # Off-diagonal perturbation
    A = A + A_off - np.diag(np.diag(A_off))  # Ensure diagonal dominance
    evals, evecs = np.linalg.eig(A)

    # Ensure the system is stable
    evals = evals - np.max(np.real(evals)) - 0.1  # Shift eigenvalues to left half-plane
    A = evecs @ np.diag(evals) @ np.linalg.inv(evecs)  # Reconstruct A
    A = A.real

    B = np.random.randn(n, 1)  # Single input for simplicity
    C = np.random.randn(1, n)  # Single output for simplicity
    D = np.zeros((1, 1))  # No direct feedthrough

    # Create full-order system
    sys_full = signal.StateSpace(A, B, C, D)

    # Reduce using balanced truncation
    A_r, B_r, C_r, D_r, hsv = balanced_truncation(A, B, C, D, r)
    sys_reduced = signal.StateSpace(A_r, B_r, C_r, D_r)

    if plot:
        import matplotlib.pyplot as plt

        # Compare time responses (more reliable than frequency response)
        t = np.linspace(0, 10, 1000)
        u = np.ones((1, len(t)))  # Step input

        # Get time responses
        _, y_full, _ = signal.lsim(sys_full, u.T, t)
        _, y_reduced, _ = signal.lsim(sys_reduced, u.T, t)

        # Plot Hankel singular values
        plt.figure(figsize=(6, 5))

        plt.subplot(2, 1, 1)
        plt.semilogy(range(1, len(hsv) + 1), hsv, "ko-")
        plt.axvline(x=r, linestyle="--")
        plt.grid(True)
        plt.xlabel("Index")
        plt.ylabel("Hankel Singular Values")
        plt.title("Hankel Singular Values")

        # Plot time responses
        plt.subplot(2, 1, 2)
        plt.plot(t, y_full, "k-", label="Full Order")
        plt.plot(t, y_reduced, "--", label="Reduced Order")
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.title("Step Response Comparison")
        plt.legend()

        plt.tight_layout()
        plt.show()

        w = np.logspace(-2, 3, 500)
        plt.figure(figsize=(6, 3))

        # # Use bode instead of freqresp for better stability
        # mag_full, phase_full, _ = signal.bode(sys_full, w)
        # mag_reduced, phase_reduced, _ = signal.bode(sys_reduced, w)

        # Full system frequency response
        mag_full, phase_full = ss_freqresp(A, B, C, D, w)
        mag_full = mag_full[:, 0, 0]  # Single input/output
        phase_full = phase_full[:, 0, 0]  # Single input/output

        # Reduced system frequency response
        mag_reduced, phase_reduced = ss_freqresp(A_r, B_r, C_r, D_r, w)
        mag_reduced = mag_reduced[:, 0, 0]  # Single input/output
        phase_reduced = phase_reduced[:, 0, 0]  # Single input/output

        plt.subplot(1, 2, 1)
        plt.semilogx(w, mag_full, "k-", label="Full")
        plt.semilogx(w, mag_reduced, "--", label="Reduced")
        plt.grid(True)
        plt.xlabel("Frequency (rad/s)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.semilogx(w, phase_full, "k-", label="Full")
        plt.semilogx(w, phase_reduced, "--", label="Reduced")
        plt.grid(True)
        plt.xlabel("Frequency (rad/s)")
        plt.ylabel("Phase (deg)")

        plt.tight_layout()
        plt.show()

    # Calculate error metrics
    max_hsv_truncated = hsv[r:].max() if len(hsv) > r else 0
    print(f"Largest truncated Hankel singular value: {max_hsv_truncated:.6e}")
    print("This is an upper bound on the H∞ norm of the error system.")

    return sys_full, sys_reduced, hsv


def ss_freqresp(A, B, C, D, w):
    """
    Compute frequency response directly from state-space representation.
    This avoids the numerically sensitive conversion to transfer function.

    Parameters:
    -----------
    A, B, C, D : ndarray
        State-space matrices
    w : ndarray
        Frequency points (rad/s)

    Returns:
    --------
    mag : ndarray
        Magnitude response
    phase : ndarray
        Phase response in degrees
    """
    n = A.shape[0]
    p, m = C.shape[0], B.shape[1]  # Output and input dimensions

    # Initialize outputs
    mag = np.zeros((len(w), p, m))
    phase = np.zeros((len(w), p, m))

    # Identity matrix of appropriate size
    I = np.eye(n)

    # Compute frequency response at each frequency point
    for k, wk in enumerate(w):
        # G(jw) = C * (jwI - A)^(-1) * B + D
        try:
            # Resolvent (jwI - A)^(-1)
            resolvent = np.linalg.solve(1j * wk * I - A, np.eye(n))
            # Transfer function at this frequency
            Gjw = C @ resolvent @ B + D

            # Compute magnitude and phase
            mag[k, :, :] = np.abs(Gjw)
            phase[k, :, :] = np.angle(Gjw, deg=True)
        except np.linalg.LinAlgError:
            # In case of singular matrix
            print(f"Warning: Singular matrix at frequency {wk} rad/s")
            mag[k, :, :] = np.nan
            phase[k, :, :] = np.nan

    # Convert magnitude to dB
    mag_db = 20 * np.log10(mag)

    return mag_db, phase


if __name__ == "__main__":
    test_balanced_truncation(plot=True)
