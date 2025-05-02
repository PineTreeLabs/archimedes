import numpy as np

from archimedes import struct

from .constants import EARTH_RADIUS, EARTH_MU
from ..aero.rotations import x_dcm, y_dcm, z_dcm


@struct.pytree_node
class KeplerElements:
    """Class to represent Keplerian orbital elements."""
    a: float
    e: float
    i: float
    omega: float
    RAAN: float
    nu: float


def kepler_to_eci(
    a: float,
    e: float,
    i: float,
    omega: float,
    RAAN: float,
    nu: float,
    mu: float = EARTH_MU,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform the Kepler elements to a state vector in the ECI frame

    Note that for "singular" orbits in the Kepler representation (e.g. circular or
    equatorial orbits), some of the standard elements are undefined.  In these cases,
    we can use the following substitutions:
    - Circular equatorial: use true longitude for ν, and set Ω = ω = 0
    - Circular inclined: use argument of latitude for ν, and set ω = 0
    - Non-circular equatorial: use longitude of periapsis for ω, and set Ω = 0
    These changes are done implicitly, i.e. by assuming that the input values represent
    the above substitutions.  This differs from the MATLAB implementation, for
    instance, where the user must explicitly provide the true longitude for circular
    equatorial orbits, etc. This is done to simplify the interface and reduce the number
    of required parameters, as well as minimizing the need for NaN- or Inf- valued
    parameters.

    Similarly, for parabolic orbits, the periapsis radius is used instead of the
    semi-major axis.  Again, this is done implicitly based on the input parameter
    ``a``, but in a way that is consistent with the inverse transformation function
    ``eci_to_kepler``.
    """
    Ω = RAAN
    ω = omega
    ν = nu

    parabolic = abs(e - 1) < tol
    equatorial = i < tol
    circular = e < tol

    # If orbit is parabolic, treat "a" as periapsis radius
    # (rp = p / 2) instead of the semi-major axis
    p = np.where(parabolic, 2 * a, a * (1 - e**2))

    # Set the longitude of the ascending node to zero if equatorial
    Ω = np.where(equatorial, 0, Ω)

    # Set the argument of periapsis to zero if circular
    ω = np.where(circular, 0, ω)

    # Position and velocity in the perifocal frame
    r_PQR = (p / 1 + e * np.cos(ν)) * np.hstack([
        np.cos(ν),
        np.sin(ν),
        0,
    ])
    v_PQR = np.sqrt(abs(mu / p)) * np.hstack([
        -np.sin(ν),
        e + np.cos(ν),
        0,
    ])

	# Rotation matrix from perifocal to ECI frame.  Note that this is not the most
	# efficient way to do this, but since this will primarily be evaluated
	# symbolically, we're basically relying on CasADi to generate
	# optimized code.

	# R = Rz(Ω) * Rx(i) * Rz(ω)
	# R = Rotations.RotZXZ(Ω, i, ω)
	# R = inv(angle_to_dcm(Ω, i, ω, :ZXZ))
    R = z_dcm(Ω) @ x_dcm(i) @ z_dcm(ω)
    r_ECI = R @ r_PQR
    v_ECI = R @ v_PQR
    return r_ECI, v_ECI