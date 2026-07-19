import numpy as np

import archimedes as arc

from constants import MU_EARTH


def lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    t: float,
    mu: float = MU_EARTH,
    short_way: bool = True,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Lambert's orbit determination problem.

    Given two position vectors and the time of flight, compute the
    corresponding velocity vectors at the two positions.
    This implementation is based on the algorithm in App C of
    Bate, Mueller, and White.
    """
    dm = np.where(short_way, 1.0, -1.0)

    cos_dnu = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
    var = dm * np.sqrt(np.dot(r1, r2) * ())
