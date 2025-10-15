# ruff: noqa: N802, N803, N806, N815, N816
import abc
from functools import cached_property
from typing import Callable, ClassVar, Protocol

import numpy as np
from scipy.special import roots_legendre

import archimedes as arc
from archimedes.spatial import (
    Rotation,
    RigidBody,
    dcm_from_euler,
)

__all__ = [
    "RigidBody",
    "RotorGeometry",
    "ConstantGravity",
    "PointGravity",
    "QuadraticDragModel",
    "RotorModel",
    "QuadraticRotorModel",
    "MultiRotorVehicle",
    "VehicleDragModel",
    "GravityModel",
]

# (AoA, Cl, Cd, Cm) data for NACA 0012 airfoil
# NOTE: This data was generated using the incompressible flow solver in
# SU2 with a Spalart-Allmaras turbulence model at Re=50k.  However, the
# CFD configuration used a relatively coarse mesh that was likely not
# fully converged.  This data should therefore be considered approximate
# and for representative purposes only.  Higher-fidelity CFD data or wind
# tunnel data should be used for more accurate aerodynamic modeling.
# Also, the Cm data is set to zero based on thin-airfoil assumptions, but
# this may not be accurate for realistic airfoils.
NACA_0012 = np.array(
    [
        [0.0, 0.0, 0.023, 0.0],
        [0.017, 0.134, 0.023, 0.0],
        [0.035, 0.272, 0.025, 0.0],
        [0.052, 0.402, 0.027, 0.0],
        [0.07, 0.532, 0.03, 0.0],
        [0.087, 0.658, 0.033, 0.0],
        [0.105, 0.776, 0.038, 0.0],
        [0.122, 0.892, 0.044, 0.0],
        [0.14, 0.996, 0.05, 0.0],
        [0.157, 1.092, 0.058, 0.0],
        [0.175, 1.172, 0.066, 0.0],
        [0.192, 1.248, 0.076, 0.0],
        [0.209, 1.306, 0.086, 0.0],
        [0.227, 1.354, 0.098, 0.0],
        [0.244, 1.384, 0.111, 0.0],
        [0.262, 1.404, 0.126, 0.0],
        [0.279, 1.402, 0.125, 0.0],
        [0.297, 1.414, 0.158, 0.0],
        [0.314, 1.408, 0.177, 0.0],
        [0.332, 1.406, 0.197, 0.0],
        [0.349, 1.406, 0.218, 0.0],
        [0.436, 1.44, 0.326, 0.0],
        [0.524, 1.524, 0.456, 0.0],
        [0.611, 1.588, 0.58, 0.0],
        [0.698, 1.626, 0.705, 0.0],
        [0.785, 1.626, 0.831, 0.0],
        [0.872, 1.586, 0.952, 0.0],
        [1.047, 1.394, 1.179, 0.0],
        [1.221, 1.068, 1.369, 0.0],
        [1.396, 0.628, 1.482, 0.0],
        [1.571, 0.134, 1.518, 0.0],
        [1.745, -0.36, 1.48, 0.0],
        [1.92, -0.804, 1.363, 0.0],
        [2.094, -1.156, 1.186, 0.0],
        [2.268, -1.39, 0.972, 0.0],
        [2.443, -1.478, 0.734, 0.0],
        [2.618, -1.404, 0.493, 0.0],
        [2.793, -1.3, 0.28, 0.0],
        [2.967, -1.202, 0.11, 0.0],
        [3.142, -0.0, 0.016, 0.0],
        [3.31618531, 1.202, 0.11, 0.0],
        [3.49018531, 1.3, 0.28, 0.0],
        [3.66518531, 1.404, 0.493, 0.0],
        [3.84018531, 1.478, 0.734, 0.0],
        [4.01518531, 1.39, 0.972, 0.0],
        [4.18918531, 1.156, 1.186, 0.0],
        [4.36318531, 0.804, 1.363, 0.0],
        [4.53818531, 0.36, 1.48, 0.0],
        [4.71218531, -0.134, 1.518, 0.0],
        [4.88718531, -0.628, 1.482, 0.0],
        [5.06218531, -1.068, 1.369, 0.0],
        [5.23618531, -1.394, 1.179, 0.0],
        [5.41118531, -1.586, 0.952, 0.0],
        [5.49818531, -1.626, 0.831, 0.0],
        [5.58518531, -1.626, 0.705, 0.0],
        [5.67218531, -1.588, 0.58, 0.0],
        [5.75918531, -1.524, 0.456, 0.0],
        [5.84718531, -1.44, 0.326, 0.0],
        [5.93418531, -1.406, 0.218, 0.0],
        [5.95118531, -1.406, 0.197, 0.0],
        [5.96918531, -1.408, 0.177, 0.0],
        [5.98618531, -1.414, 0.158, 0.0],
        [6.00418531, -1.402, 0.125, 0.0],
        [6.02118531, -1.404, 0.126, 0.0],
        [6.03918531, -1.384, 0.111, 0.0],
        [6.05618531, -1.354, 0.098, 0.0],
        [6.07418531, -1.306, 0.086, 0.0],
        [6.09118531, -1.248, 0.076, 0.0],
        [6.10818531, -1.172, 0.066, 0.0],
        [6.12618531, -1.092, 0.058, 0.0],
        [6.14318531, -0.996, 0.05, 0.0],
        [6.16118531, -0.892, 0.044, 0.0],
        [6.17818531, -0.776, 0.038, 0.0],
        [6.19618531, -0.658, 0.033, 0.0],
        [6.21318531, -0.532, 0.03, 0.0],
        [6.23118531, -0.402, 0.027, 0.0],
        [6.24818531, -0.272, 0.025, 0.0],
        [6.26618531, -0.134, 0.023, 0.0],
        [6.28318531, -0.0, 0.023, 0.0],
    ]
)


@arc.struct
class RotorGeometry:
    offset: np.ndarray = arc.field(
        default_factory=lambda: np.zeros(3)
    )  # Location of the rotor hub in the body frame B [m]
    ccw: bool = True  # True if rotor spins counter-clockwise when viewed from above
    torsional_cant: float = 0.0  # torsional cant angle χ [rad]
    flapwise_cant: float = 0.0  # flapwise cant angle γ [rad]

    @cached_property
    def R_BH(self):
        """Rotation matrix from the hub frame H to the body frame B"""
        x, y, z = self.offset

        # Torsional cant angle rotation (Rχ)
        c, s = np.cos(self.torsional_cant), np.sin(self.torsional_cant)
        R_torsional = np.array(
            [
                [x**2 * (c - 1) + 1, -x * y * (c - 1), -y * s],
                [-x * y * (c - 1), y**2 * (c - 1) + 1, x * s],
                [y * s, -x * s, c],
            ]
        )

        # Flapwise cant angle rotation (Rγ)
        c, s = np.cos(self.flapwise_cant), np.sin(self.flapwise_cant)
        R_flapwise = np.array(
            [
                [x**2 * (c - 1) + 1, x * y * (c - 1), -x * s],
                [x * y * (c - 1), y**2 * (c - 1) + 1, -y * s],
                [x * s, y * s, c],
            ]
        )

        return R_torsional @ R_flapwise

    @property
    def r_B(self):
        """Offset of the rotor hub in the body frame coordinates B"""
        return self.offset

    def __hash__(self):
        return hash(
            (str(self.offset), self.ccw, self.torsional_cant, self.flapwise_cant)
        )


class GravityModel(Protocol):
    def __call__(self, p_N):
        """Gravitational acceleration at the body CM in the inertial frame N

        Args:
            p_N: position vector in inertial frame N

        Returns:
            g_N: gravitational acceleration in inertial frame N
        """


@arc.struct
class ConstantGravity:
    g0: float = 9.81

    def __call__(self, p_N):
        return np.array([0, 0, self.g0], like=p_N)


@arc.struct
class PointGravity:
    G: float = 6.6743e-11  # Gravitational constant [N-m²/kg²]
    R_e: float = 6.371e6  # Radius of earth [m]

    def __call__(self, p_N):
        raise NotImplementedError("Illustrative purposes only")


class VehicleDragModel(Protocol):
    def __call__(self, t, x):
        """Drag forces and moments in body frame B

        Args:
            t: time
            x: state vector

        Returns:
            F_B: drag forces in body frame B
            M_B: drag moments in body frame B
        """


@arc.struct
class QuadraticDragModel:
    """Simple velocity-squared drag model for the main vehicle body"""

    rho: float = 1.225  # air density [kg/m^3]
    Cd: float = 0.0  # drag coefficient
    A: float = 1.0  # reference planform area [m^2]
    r_CoP = np.zeros(3)  # Center of pressure offset from body CG [m]

    def __call__(self, t, x):
        v_B = x.v_B  # Velocity of the center of mass in body frame B

        # Velocity magnitude with guard against near-zero values
        v_mag = np.linalg.norm(v_B)
        v_mag = np.where(v_mag < 1e-6, 1e-6, v_mag)

        # Drag force in body frame (3-element vector)
        D_B = -0.5 * self.rho * self.Cd * self.A * v_mag * v_B

        # Drag moment in body frame (3-element vector)
        M_B = np.cross(self.r_CoP, D_B)

        return D_B, M_B


class RotorModel(metaclass=abc.ABCMeta):
    def __call__(self, t, v_B, w_B, x, Omega, geometry: RotorGeometry):
        """Aerodynamic forces and moments in body frame B

        Args:
            t: time
            v_B: velocity of the center of mass in body frame B
            w_B: angular velocity in body frame B
            x: state vector (may include aerodynamic state variables)
            Omega: rotor speed
            geometry: rotor geometry

        Returns:
            F_B: aerodynamic forces for this rotor in body frame B
            M_B: aerodynamic moments for this rotor in body frame B
            aux_state_derivs: time derivatives of aerodynamic state variables
        """
        # So far this framework assumes no extenal wind; hence the rotor-relative
        # air velocity is just the body velocity transformed to the j-th hub frame.
        v_H = geometry.R_BH.T @ v_B
        w_H = geometry.R_BH.T @ w_B  # Angular velocity in the hub frame

        psi_w = np.arctan2(-v_H[1], -v_H[0])
        # For zero forward velocity, the wind angle is +/- 90 degrees
        # If the side velocity is also zero, the rotations have no
        # effect and the wind angle is undefined (+/- 90 degrees will also work).
        psi_w = np.where(abs(v_H[0]) < 1e-6, np.sign(v_H[1]) * np.pi / 2, psi_w)

        R_WH = Rotation.from_euler("z", psi_w).as_matrix().T

        v_W = R_WH @ v_H  # Rotate velocity to wind frame
        w_W = R_WH @ w_H  # Rotate angular velocity to wind frame

        F_W, M_W, aux_state_derivs = self.wind_frame_loads(
            t, v_W, w_W, x, Omega, geometry
        )

        Fj_H = R_WH.T @ F_W  # Rotate force to hub frame
        Mj_H = R_WH.T @ M_W  # Rotate moment to hub frame

        # Transform the rotor force and moment from the hub frame to the body frame.
        F_B = geometry.R_BH @ Fj_H
        M_B = geometry.R_BH @ Mj_H + np.cross(geometry.r_B, F_B)

        return F_B, M_B, aux_state_derivs

    @property
    def num_aux_states(self) -> int:
        return 0

    @abc.abstractmethod
    def wind_frame_loads(self, t, v_W, w_W, x, Omega, geometry):
        """Aerodynamic forces and moments in wind frame W"""


@arc.struct
class QuadraticRotorModel(RotorModel):
    """Simple velocity-squared model for rotor aerodynamics"""

    kF: float = 1.0  # aerodynamic force constant [kg-m]
    kM: float = 0.1  # aerodynamic torque constant [kg-m^2]

    num_states: ClassVar[int] = 0

    def wind_frame_loads(self, t, v_W, w_W, x, Omega, geometry: RotorGeometry):
        M_sign = 1 if geometry.ccw else -1
        z_W = np.array([0.0, 0.0, 1.0])
        u_sq = Omega**2

        # Note that the wind and hub frame z-axes are coincident
        F_W = -self.kF * u_sq * z_W
        M_W = M_sign * self.kM * u_sq * z_W

        # No internal aerodynamic state variables
        aux_state_derivs = np.array([], like=x)

        return F_W, M_W, aux_state_derivs


@arc.struct
class MultiRotorVehicle:
    rigid_body: RigidBody = arc.field(default_factory=RigidBody)
    rotors: list[RotorGeometry] = arc.field(default_factory=list)
    rotor_model: RotorModel = arc.field(default_factory=QuadraticRotorModel)
    drag_model: VehicleDragModel = arc.field(default_factory=QuadraticDragModel)
    gravity_model: GravityModel = arc.field(default_factory=ConstantGravity)

    m: float = 1.0  # mass [kg]
    J_B: np.ndarray = arc.field(
        default_factory=lambda: np.eye(3)
    )  # inertia matrix [kg·m²]

    @arc.struct
    class State:
        rigid_body: RigidBody.State
        aux: np.ndarray = None  # auxiliary state variables for rotors

        @property
        def p_N(self):
            return self.rigid_body.p_N

        @property
        def att(self):
            return self.rigid_body.att

        @property
        def v_B(self):
            return self.rigid_body.v_B

        @property
        def w_B(self):
            return self.rigid_body.w_B

    def state(self, pos, rpy, vel, w_B, aux=None, inertial_velocity=False) -> State:
        if self.rigid_body.rpy_attitude:
            att = rpy
            R_BN = dcm_from_euler(rpy)
        else:
            att = Rotation.from_euler("xyz", rpy)
            R_BN = att.as_matrix()
        if inertial_velocity:
            vel = R_BN @ vel
        rb_state = self.rigid_body.State(pos, att, vel, w_B)
        return self.State(rb_state, aux)

    def net_forces(self, t, x: State, u):
        p_N = x.p_N  # Position of the center of mass in inertial frame N
        v_B = x.v_B  # Velocity of the center of mass in body frame B
        w_B = x.w_B  # Roll-pitch-yaw rates in body frame (ω_B)

        # Calculate local gravity force on B, expressed in
        # the Newtonian frame N
        Fgravity_N = self.m * self.gravity_model(p_N)

        # Calculate rotor forces and moments in wind frame W_j local to the
        # j-th rotor hub.  These forces and moments must be transformed to
        # the body frame and summed to obtain the net force and moment on B.
        # Frotor_B, Mrotor_B, aux_state_derivs = self.rotor_model(t, x, u)
        Frotor_B = np.zeros_like(v_B)
        Mrotor_B = np.zeros_like(v_B)
        aux_state_derivs = np.array([], like=p_N)
        aux_state_idx = 0
        num_aux_states = self.rotor_model.num_aux_states
        for j, rotor in enumerate(self.rotors):
            if x.aux is not None:
                aux_state = x.aux[aux_state_idx : aux_state_idx + num_aux_states]
                aux_state_idx += num_aux_states
            else:
                aux_state = None
            rotor_speed = u[j]

            Fj_B, Mj_B, aux_state_derivs_j = self.rotor_model(
                t, v_B, w_B, aux_state, rotor_speed, rotor
            )

            # Sum the forces and moments from each rotor
            Frotor_B += Fj_B
            Mrotor_B += Mj_B

            aux_state_derivs = np.hstack([aux_state_derivs, aux_state_derivs_j])

        # Calculate drag forces and moments in body frame B
        Fdrag_B, Mdrag_B = self.drag_model(t, x)

        if self.rigid_body.rpy_attitude:
            R_BN = dcm_from_euler(x.att)
            Fgravity_B = R_BN @ Fgravity_N

        else:
            Fgravity_B = x.att.apply(Fgravity_N, inverse=True)

        F_B = Frotor_B + Fdrag_B + Fgravity_B

        # Net moment in body frame
        M_B = Mrotor_B + Mdrag_B

        return F_B, M_B, aux_state_derivs

    def dynamics(self, t, x: State, u: np.ndarray) -> State:
        """Compute time derivative of the state

        Args:
            t: time
            x: state: (p_N, att, v_B, w_B, rotor aux states)
            u: control inputs

        Returns:
            x_dot: time derivative of the state
        """
        # Compute the net forces
        F_B, M_B, aux_state_derivs = self.net_forces(t, x, u)

        rb_input = RigidBody.Input(
            F_B=F_B,
            M_B=M_B,
            m=self.m,
            J_B=self.J_B,
        )
        rb_derivs = self.rigid_body.dynamics(t, x.rigid_body, rb_input)

        return self.State(
            rigid_body=rb_derivs,
            aux=aux_state_derivs,
        )


class AirfoilModel(metaclass=abc.ABCMeta):
    """Model for aerodynamics of a 2D subsonic airfoil section."""

    compressibility_model: str = None

    @abc.abstractmethod
    def __call__(self, alpha, Re, M_inf, force_only=False):
        """Returns lift, drag, and moment coefficients for the given angle of attack

        Args:
            alpha: angle of attack [rad]
            Re: Reynolds number
            M_inf: freestream Mach number
            force_only: if True, only compute lift and drag coefficients

        Returns:
            Cl: lift coefficient
            Cd: drag coefficient
            Cm: quarter-chord moment coefficient (all zero if force_only=True)
        """

    def initialize(self, nodes_rad):
        """Initialize the airfoil model for the given radial quadrature nodes"""
        pass


@arc.struct
class ThinAirfoil(AirfoilModel):
    """Airfoil model based on thin airfoil theory.

    This highly simplified model assumes constant aerodynamic derivatives
    along the span of the airfoil.
    """

    Cl_0: float = 0.0  # zero-lift coefficient
    Cl_alpha: float = 2 * np.pi  # lift curve slope [1/rad]
    Cd_0: float = 0.0  # zero-drag coefficient
    Cm_0: float = 0.0  # zero-moment coefficient

    def __post_init__(self):
        if self.compressibility_model is not None:
            raise ValueError(
                "Compressibility correction not supported for thin airfoil model"
            )

    def __call__(self, alpha, Re, M_inf, force_only=False):
        Cl = self.Cl_0 + self.Cl_alpha * alpha
        Cd = self.Cd_0
        Cm = self.Cm_0

        return Cl, Cd, Cm


@arc.struct(frozen=False)
class TabulatedAirfoil(AirfoilModel):
    """Airfoil model based on tabulated data"""

    # Airfoil lookup tables: these should be a list of 2D arrays
    # specifying coefficients as a function of angle of attack. Each
    # item in the list corresponds to a different airfoil section;
    # data is interpolated linearly between sections.
    airfoil_data: list[np.ndarray] = arc.field(default_factory=list)
    rad_loc: list[float] = arc.field(
        default_factory=list
    )  # Radial locations of airfoil sections [m]

    def initialize(self, nodes_rad):
        if self.compressibility_model not in (None, "glauert"):
            raise ValueError(
                f"Invalid compressibility model: {self.compressibility_model}"
            )

        airfoil_data = self.airfoil_data
        rad_loc = self.rad_loc

        if len(airfoil_data) != len(rad_loc):
            raise ValueError(
                "airfoil_data, and radial locations must have the same length"
            )

        if len(airfoil_data) == 0:
            raise ValueError("At least one airfoil section must be provided")

        # If there is only one airfoil section, duplicate it for all radial locations
        if len(airfoil_data) == 1:
            airfoil_data = [airfoil_data[0]] * len(rad_loc)

        # Check that all airfoil sections have the same number of data points
        n_data = len(airfoil_data[0])
        if not all(len(data) == n_data for data in airfoil_data):
            raise ValueError(
                "All airfoil sections must have the same number of data points"
            )

        # Interpolate between airfoil sections
        node_airfoil_data = []
        for r in nodes_rad[:, 0]:
            # Find the two airfoil sections that bracket the current radial location
            j = np.searchsorted(rad_loc, r)

            # Extrapolate with the last section beyond the limits
            if j == 0:
                print(f"Warning: Extrapolating Cl and Cd for radial location {r}")
                j = 1
            elif j == len(rad_loc):
                print(f"Warning: Extrapolating Cl and Cd for radial location {r}")
                j = len(rad_loc) - 1

            # Interpolate between the two sections
            x = (r - rad_loc[j - 1]) / (rad_loc[j] - rad_loc[j - 1])
            node_airfoil_data.append(
                airfoil_data[j - 1] * (1 - x) + airfoil_data[j] * x
            )

        # node_airfoil_data will have shape (n_rad * p_rad) x n_alpha x 3
        # with the last dimension corresponding to (AoA, Cl, Cd)
        self.node_airfoil_data = np.array(node_airfoil_data)

    def __call__(self, alpha, Re, M_inf, force_only=False):
        alpha = alpha % (2 * np.pi)  # Wrap to (0, 360)

        # Look up the lift and drag coefficients for each radial node
        Cl = np.zeros_like(alpha)
        Cd = np.zeros_like(alpha)
        Cm = np.zeros_like(alpha)
        for i in range(len(self.node_airfoil_data)):
            alpha_bkpts, Cl_data, Cd_data, Cm_data = self.node_airfoil_data[i].T
            Cl[i] = np.interp(alpha[i], alpha_bkpts, Cl_data)
            Cd[i] = np.interp(alpha[i], alpha_bkpts, Cd_data)
            if not force_only or not np.allclose(Cm_data, 0):
                Cm[i] = np.interp(alpha[i], alpha_bkpts, Cm_data)

        if self.compressibility_model == "glauert":
            glauert = 1 / np.sqrt(1 - M_inf**2)
            Cl = Cl * glauert
            Cm = Cm * glauert

        return Cl, Cd, Cm


@arc.struct(frozen=False)
class BladeElementModel(RotorModel):
    """Blade element model for rotor aerodynamics

    This model is based on blade-element theory with uniform inflow (momentum disk)
    approximation for rotor-induced inflow.

    Uses Gauss-Legendre quadrature for integration over the rotor disk.
    """

    # Aerodynamic model parameters
    Nb: int = 2  # Number of blades
    R: float = 1.0  # Rotor radius [m]
    e: float = 0.1  # Root cut-out ratio [-]
    a: float = 343.0  # Speed of sound [m/s]
    rho: float = 1.225  # Air density [kg/m^3]

    # Airfoil model
    airfoil_model: AirfoilModel = arc.field(default_factory=ThinAirfoil)

    # Iterative inflow solver parameters
    T0: float = 0.0  # Reference thrust per rotor (e.g. weight / number of rotors)
    newton_max_iter: int = 50  # Maximum number of iterations for Newton solve
    newton_abstol: float = 1e-6  # Absolute tolerance for Newton solve

    # Parameters for Gaussian quadrature
    p_rad: int = 2  # Integration order for radial elements
    n_rad: int = 5  # Number of radial elements
    p_az: int = 3  # Integration order for azimuthal elements
    n_az: int = 5  # Number of azimuthal elements

    # Blade pitch as a function of radial position
    blade_pitch: Callable[[float], float] = lambda r: np.zeros_like(r)

    # Chord as a function of radial position
    chord: Callable[[float], float] = lambda r: 0.1 * np.ones_like(r)

    num_states: ClassVar[int] = 0

    def __post_init__(self):
        self._initialize_quadrature(self.p_rad, self.n_rad, self.p_az, self.n_az)

        # Initialize the airfoil model for the given radial quadrature nodes
        self.airfoil_model.initialize(self.nodes_rad)

        # Determine the blade pitch and chord at each radial location
        self.node_pitch = self.blade_pitch(self.nodes_rad)
        self.node_chord = self.chord(self.nodes_rad)

        # Construct an implicit function for the induced inflow ratio as a function
        # of thrust coefficient and dimensionless advance ratios.
        @arc.compile(static_argnames=("geometry",), kind="MX")
        def CT_residual(lambda_, t, v_W, w_W, x, Omega, geometry):
            mu_x = -v_W[0] / (Omega * self.R)
            mu_z = -v_W[2] / (Omega * self.R)
            # Compute thrust coefficient based on momentum disk theory
            # See (2.126) in Leishman
            CT_momentum_disk = 2 * (lambda_ - mu_z) * np.sqrt(mu_x**2 + lambda_**2)
            # Compute thrust coefficient based on blade element theory
            CT_blade_element = self.thrust_coefficient(
                t, v_W, w_W, x, Omega, lambda_, geometry
            )
            # Residual is the difference between the momentum disk and
            # blade element predictions of the thrust coefficient.
            return CT_momentum_disk - CT_blade_element

        self._lambda_solve = arc.implicit(
            CT_residual,
            static_argnames=("geometry",),
            solver="fast_newton",
            max_iter=self.newton_max_iter,
            abstol=self.newton_abstol,
        )

    def _initialize_quadrature(self, p_rad, n_rad, p_az, n_az):
        x_rad, w_rad = roots_legendre(p_rad)
        x_az, w_az = roots_legendre(p_az)

        # Divide the radial span into n_rad elements
        el_rad = np.linspace(self.e * self.R, self.R, n_rad + 1)
        nodes_rad = []
        weights_rad = []
        for i in range(n_rad):
            a, b = el_rad[i], el_rad[i + 1]
            x0 = 0.5 * (a + b)  # Center of the interval
            dx = 0.5 * (b - a)  # Half-width of the interval
            nodes_rad.extend(x0 + dx * x_rad)
            weights_rad.extend(dx * w_rad)

        self.weights_rad = np.array(weights_rad)

        # Construct a (p_rad x n_rad) x (p_az x n_az) grid of the radial nodes
        self.nodes_rad = np.repeat(np.array(nodes_rad)[:, None], (p_az * n_az), axis=1)

        # Divide the azimuthal span into n_az elements
        el_az = np.linspace(0, 2 * np.pi, n_az + 1)
        nodes_az = []
        weights_az = []
        for i in range(n_az):
            a, b = el_az[i], el_az[i + 1]
            x0 = 0.5 * (a + b)
            dx = 0.5 * (b - a)
            nodes_az.extend(x0 + dx * x_az)
            weights_az.extend(dx * w_az)

        self.weights_az = np.array(weights_az)

        # Construct a (p_rad x n_rad) x (p_az x n_az) grid of the azimuthal nodes
        self.nodes_az = np.repeat(np.array(nodes_az)[None, :], (p_rad * n_rad), axis=0)

    def inflow(self, r, psi, v_W, w_W, Omega, lambda_):
        """Calculate the rotor inflow at a given radius and azimuthal position

        Args:
            r: radial position from rotor hub to blade element [m]
            psi: azimuthal position of the rotor blade [rad]
            v_W: velocity of the center of mass in wind frame W
            w_W: angular velocity in wind frame W
            Omega: rotor speed [rad/s]
            lambda_: rotor-induced inflow ratio
        """
        # The blade frame has the x-axis pointing towards the trailing edge and
        # the z-axis pointing upwards (inverted from the wind frame). Hence, the
        # relative velocity is the negative of v_W in the x and z directions.
        # However, the way U_tan and U_perp are defined is along the _positive_
        # blade-frame x-axis and _negative_ blade-frame z-axis, respectively.
        Vx, Vz = v_W[0], -v_W[2]

        # Flow components relative to the blade element, including rigid body
        # rotations of the entire vehicle and rotor-induced inflow
        ux_b = Omega * r + Vx * np.sin(psi) - r * w_W[2]
        uz_b = (
            -Vz
            - lambda_ * Omega * self.R
            + r * (w_W[0] * np.sin(psi) + w_W[1] * np.cos(psi))
        )

        # induced inflow angle γ (angle of attack α = pitch - γ)
        gamma = np.arctan2(-uz_b, ux_b)

        # Total inflow velocity squared
        U_sq = ux_b**2 + uz_b**2

        return U_sq, gamma

    def induced_inflow_ratio(self, t, v_W, w_W, x, Omega, geometry: RotorGeometry):
        # Reference thrust coefficient with guard against zero rotor speed
        Omega = np.where(Omega < 0.0, min(Omega, -1.0), max(Omega, 1.0))
        CT0 = self.T0 / (0.5 * self.rho * np.pi * self.R**2 * Omega**2)
        lambda_guess = np.sqrt(0.5 * CT0)  # Initial guess for the inflow ratio
        return self._lambda_solve(lambda_guess, t, v_W, w_W, x, Omega, geometry)

    def thrust_coefficient(
        self, t, v_W, w_W, x, Omega, lambda_, geometry: RotorGeometry
    ):
        F_W = self._compute_forces_moments(
            t, v_W, w_W, x, Omega, lambda_, geometry, force_only=True
        )
        # Different definitions are sometimes used: see (2.31) in Leishman
        CT = abs(F_W[2]) / (self.rho * np.pi * self.R**4 * Omega**2)
        return CT

    def wind_frame_loads(self, t, v_W, w_W, x, u, geometry: RotorGeometry):
        # Determine a uniform inflow ratio that is consistent between momentum
        # theory and the blade element model in terms of thrust.
        lambda_ = self.induced_inflow_ratio(t, v_W, w_W, x, u, geometry)
        return self._compute_forces_moments(t, v_W, w_W, x, u, lambda_, geometry)

    def _differential_shear(self, v_W, w_W, u, lambda_, force_only=False):
        # Compute infinitesimal shear forces and moments dS(r, ψ), dM(r, ψ) at each node

        # Determine the relative velocity U for the blade element at each node
        U_sq, gamma = self.inflow(self.nodes_rad, self.nodes_az, v_W, w_W, u, lambda_)

        # Determine the angle of attack for each node
        alpha = self.node_pitch - gamma

        if self.airfoil_model.compressibility_model is not None:
            M_inf = np.sqrt(U_sq) / self.a
        else:
            M_inf = 0.0

        Cl, Cd, Cm = self.airfoil_model(
            alpha, Re=None, M_inf=M_inf, force_only=force_only
        )

        # Integrate in radial direction to get root forces and moments as a function
        # of azimuthal position
        c = self.node_chord

        # Blade root drag shear dSx(r, ψ)
        dFx = 0.5 * self.rho * U_sq * c * (Cl * np.sin(gamma) + Cd * np.cos(gamma))

        # Note that blade root radial shear Sy(ψ) is identically zero

        # Blade root vertical shear dSz(r, ψ)
        dFz = 0.5 * self.rho * U_sq * c * (Cl * np.cos(gamma) - Cd * np.sin(gamma))

        if force_only:
            return dFx, dFz, None, None, None

        # Root flapwise bending moment Mβ(ψ)
        dMx = dFz * self.nodes_rad

        # Root torsional moment Mϕ(ψ)
        dMy = 0.5 * self.rho * c**2 * U_sq * Cm

        # Root lagwise bending moment Mζ(ψ)
        dMz = -dFx * self.nodes_rad

        return dFx, dFz, dMx, dMy, dMz

    def _compute_forces_moments(
        self, t, v_W, w_W, x, u, lambda_, geometry: RotorGeometry, force_only=False
    ):
        # Given a specified induced inflow ratio, compute the wind-frame forces
        # and moments on the rotor blade elements using blade element theory.
        # This is called both within the iterative inflow solver to produce the thrust
        # coefficient and afterwards for the full computation of forces and moments.

        sign = 1 if geometry.ccw else -1

        # Compute the differential shear forces and moments at each node
        dFx, dFz, dMx, dMy, dMz = self._differential_shear(
            v_W, w_W, u, lambda_, force_only
        )

        Fx_b = dFx.T @ self.weights_rad  # Blade root drag shear Fx_b(ψ)
        Fy_b = np.zeros_like(Fx_b)  # Blade root radial shear Fy_b(ψ)
        Fz_b = dFz.T @ self.weights_rad  # Blade root vertical shear Sz(ψ)

        # Integrate in azimuthal direction to get mean forces in wind frame
        Nb_fac = self.Nb / (2 * np.pi)
        psi = self.nodes_az[0]  # Only need the array of unique azimuthal nodes
        Fx_W = Nb_fac * np.dot(
            -Fx_b * np.sin(psi) - Fy_b * np.cos(psi), self.weights_az
        )
        Fy_W = (
            sign
            * Nb_fac
            * np.dot(-Fx_b * np.cos(psi) + Fy_b * np.sin(psi), self.weights_az)
        )
        Fz_W = Nb_fac * np.dot(-Fz_b, self.weights_az)

        F_W = np.array([Fx_W, Fy_W, Fz_W], like=v_W)

        if force_only:
            return F_W

        Mx_b = dMx.T @ self.weights_rad  # Root flapwise bending moment Mβ(ψ)
        My_b = dMy.T @ self.weights_rad  # Root torsional moment Mϕ(ψ)
        Mz_b = dMz.T @ self.weights_rad  # Root lagwise bending moment Mζ(ψ)

        # Integrate in azimuthal direction to get mean moments in wind frame
        Mx_W = (
            sign
            * Nb_fac
            * np.dot(-Mx_b * np.sin(psi) - My_b * np.cos(psi), self.weights_az)
        )
        My_W = Nb_fac * np.dot(
            -Mx_b * np.cos(psi) + My_b * np.sin(psi), self.weights_az
        )
        Mz_W = sign * Nb_fac * np.dot(-Mz_b, self.weights_az)

        # No aerodynamic states
        aux_state_derivs = np.array([], like=x)
        M_W = np.array([Mx_W, My_W, Mz_W], like=v_W)

        return F_W, M_W, aux_state_derivs
