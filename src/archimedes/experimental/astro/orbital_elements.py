from __future__ import annotations
import numpy as np

from archimedes import struct
from archimedes.spatial import Rotation

from .constants import EARTH_RADIUS, EARTH_MU


@struct
class KeplerElements:
    """Class to represent Keplerian orbital elements."""

    a: float
    e: float
    i: float
    omega: float
    RAAN: float
    nu: float


@struct
class CartesianState:
    """Class to represent a Cartesian state vector."""

    r: np.ndarray
    v: np.ndarray

    @classmethod
    def from_kepler(
        cls, kepler: KeplerElements, mu: float = EARTH_MU
    ) -> CartesianState:
        """Convert Kepler elements to Cartesian state vector."""
        return kepler_to_cartesian(
            kepler.a, kepler.e, kepler.i, kepler.omega, kepler.RAAN, kepler.nu, mu=mu
        )


def kepler_to_cartesian(
    oe: KeplerElements,
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
    equatorial orbits, etc. This is done to simplify the interface and reduce the
    number of required parameters, as well as minimizing the need for NaN- or Inf-
    valued parameters.

    Similarly, for parabolic orbits, the periapsis radius is used instead of the
    semi-major axis.  Again, this is done implicitly based on the input parameter
    ``a``, but in a way that is consistent with the inverse transformation function
    ``eci_to_kepler``.
    """
    a = oe.a
    e = oe.e
    i = oe.i
    Ω = oe.RAAN
    ω = oe.omega
    ν = oe.nu

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
    r_PQW = (p / (1 + e * np.cos(ν))) * np.hstack(
        [
            np.cos(ν),
            np.sin(ν),
            0,
        ]
    )
    v_PQW = np.sqrt(abs(mu / p)) * np.hstack(
        [
            -np.sin(ν),
            e + np.cos(ν),
            0,
        ]
    )

    # Rotation matrix from perifocal to ECI frame.
    R = Rotation.from_euler("ZXZ", [Ω, i, ω]).as_matrix()
    r_ECI = R @ r_PQW
    v_ECI = R @ v_PQW
    return CartesianState(r_ECI, v_ECI)


def _e_vec(r, v, mu):
    return (1 / mu) * ((np.dot(v, v) - mu / np.linalg.norm(r)) * r - np.dot(r, v) * v)


def eccentricity(r, v, mu):
    """Calculate the eccentricity from the position and velocity vectors."""
    e = _e_vec(r, v, mu)
    return np.linalg.norm(e)  # Eccentricity vector


def semi_major_axis(r, v, mu):
    """Calculate the semi-major axis from the position and velocity vectors."""
    h = np.cross(r, v)
    p = np.linalg.norm(h) ** 2 / mu
    e = eccentricity(r, v, mu)
    return p / (1 - e**2)  # Semi-major axis


def inclination(r, v):
    """Calculate the inclination from the position and velocity vectors."""
    h = np.cross(r, v)
    return np.arccos(h[2] / np.linalg.norm(h))  # Inclination


def cartesian_to_kepler(
    oe: CartesianState, mu: float = EARTH_MU, tol: float = 1e-8
) -> KeplerElements:
    """Transform a state vector in the Earth-centered inertial frame to Kepler elements

    This function is the inverse of ``kepler_to_cartesian``.  Note that for "singular"
    orbits in the Kepler representation (e.g. circular or equatorial orbits), some of
    the standard elements are undefined.  In these cases, we can use substitute
    variables based on lat/lon as defined in the ``kepler_to_cartesian`` function.
    """
    r, v = oe.r, oe.v
    r_mag = np.linalg.norm(r)

    h = np.cross(r, v)  # Angular momentum
    h_mag = np.linalg.norm(h)  # Angular momentum magnitude
    k_hat = np.array([0, 0, 1])
    n = np.cross(k_hat, h)  # Node vector
    e = _e_vec(r, v, mu)  # Eccentricity vector

    p = h_mag**2 / mu  # Semi-latus rectum
    e_mag = np.linalg.norm(e)  # Eccentricity magnitude
    a = p / (1 - e_mag**2)  # Semi-major axis
    n_mag = np.linalg.norm(n)  # Node vector magnitude

    # Inclination (0 to 180 degrees)
    i = np.arccos(h[2] / h_mag)

    # Right ascension of the ascending node (ranges from 0 to 180 if nJ > 0,
    # otherwise from 180 to 360). Undefined for equatorial orbits.
    arg = n[0] / n_mag
    # Clip to avoid floating point errors out of acos domain
    Ω = np.arccos(np.clip(arg, -1.0, 1.0))
    Ω = np.where(n[1] < 0, 2 * np.pi - Ω, Ω)

    # Argument of periapsis (ranges from 0 to 180 if eK > 0, otherwise from 180
    # to 360). Undefined for both circular and equatorial orbits
    arg = np.dot(n, e) / (n_mag * e_mag)
    # Clip to avoid floating point errors out of acos domain
    ω = np.arccos(np.clip(arg, -1.0, 1.0))
    ω = np.where(e[2] < 0, 2 * np.pi - ω, ω)

    # True anomaly (ranges from 0 to 180 if r⋅v > 0, otherwise from 180 to 360)
    # Undefined for circular orbits (no periapsis)
    arg = np.dot(e, r) / (e_mag * r_mag)
    # Clip to avoid floating point errors out of acos domain
    nu = np.arccos(np.clip(arg, -1.0, 1.0))
    nu = np.where(np.dot(r, v) < 0, 2 * np.pi - nu, nu)

    # Special cases: parabolic, equatorial, circular
    parabolic = abs(e_mag - 1) < tol
    equatorial = i < tol
    circular = e_mag < tol

    # Parabolic: use peripapsis radius instead of semi-major axis
    a = np.where(parabolic, p / 2, a)

    # Circular equatorial: use true longitude for ν
    i_hat = np.array([1, 0, 0])
    truelon = np.arccos(np.dot(i_hat, r) / r_mag)
    nu = np.where(circular * equatorial, truelon, nu)

    # Circular inclined: use argument of latitude for ν
    arglat = np.arccos(np.dot(n, r) / (n_mag * r_mag))
    arglat = np.where(r[2] < 0, 2 * np.pi - arglat, arglat)
    nu = np.where(circular * (1 - equatorial), arglat, nu)

    # Non-circular equatorial: use longitude of periapsis for ω
    lonper = truelon - nu
    ω = np.where((1 - circular) * equatorial, lonper, ω)

    # Set the longitude of the ascending node to zero if equatorial
    Ω = np.where(equatorial, 0, Ω)

    # Set the argument of periapsis to zero if circular
    ω = np.where(circular, 0, ω)

    return KeplerElements(a, e_mag, i, ω, Ω, nu)
