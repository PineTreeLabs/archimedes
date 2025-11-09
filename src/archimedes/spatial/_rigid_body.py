# ruff: noqa: N806, N803, N815
from __future__ import annotations

import numpy as np

from archimedes import struct

from ._attitude import Attitude

__all__ = [
    "RigidBody",
]


@struct
class RigidBody:
    """6-dof rigid body dynamics model

    This class implements 6-dof rigid body dynamics based on reference equations
    from Lewis, Johnson, and Stevens, "Aircraft Control and Simulation" [1]_.

    This implementation is general and does not make any assumptions about the
    forces, moments, or mass properties.  These must be provided as inputs to the
    dynamics function.

    The model does assume a non-inertial body-fixed reference frame B and a Newtonian
    inertial reference frame N.  The body frame is assumed to be located at the
    vehicle's center of mass.

    With these conventions, the state vector is defined as
        ``x = [pos, att, v_B, w_B]``

    where

    - ``pos`` = position of the center of mass in the inertial frame
    - ``att`` = attitude (orientation) of the vehicle
    - ``v_B`` = velocity of the center of mass in body frame B
    - ``w_B`` = angular velocity in body frame (ω_B)

    Note that the attitude can be any object implementing the :py:class:`Attitude`
    protocol, commonly :py:class:`Quaternion` or :py:class:`EulerAngles`.
    The transformation implemented by ``as_matrix`` with this attitude represents
    ``R_BN``, the rotation from the inertial frame N to the body frame B.

    The equations of motion for a quaternion attitude are given by

    .. math::
        \\dot{\\mathbf{p}}^N &= \\mathbf{R}_{BN}^T(\\mathbf{q}) \\mathbf{v}^B \\\\
        \\dot{\\mathbf{q}} &= \\frac{1}{2} \\mathbf{q} \\otimes \\boldsymbol{\\omega}^B
            \\\\
        \\dot{\\mathbf{v}}^B &= \\frac{1}{m}\\mathbf{F}^B
            - \\boldsymbol{\\omega}^B \\times \\mathbf{v}^B \\\\
        \\dot{\\boldsymbol{\\omega}}^B &= \\mathbf{J}_B^{-1}(\\mathbf{M}^B
            - \\boldsymbol{\\omega}^B \\times (\\mathbf{J}^B \\boldsymbol{\\omega}^B))

    where

    - :math:`\\mathbf{R}_{BN}(\\mathbf{q})` = direction cosine matrix (DCM)
    - :math:`m` = mass of the vehicle
    - :math:`\\mathbf{J}^B` = inertia matrix of the vehicle in body axes
    - :math:`\\mathbf{F}^B` = net forces acting on the vehicle in body frame B
    - :math:`\\mathbf{M}^B` = net moments acting on the vehicle in body frame B

    The inputs to the dynamics function are a ``RigidBody.Input`` struct
    containing the forces, moments, mass, and inertia properties.  By default
    the time derivatives of the mass and inertia are zero unless specified
    in the input struct.

    The attitude representation can be anything that implements the
    :py:class:`Attitude` protocol, which requires methods for rotating vectors
    between frames and calculating the attitude kinematics.  Common choices
    are :py:class:`Quaternion` or :py:class:`EulerAngles`.
    The time derivative of the attitude will be calculated with the classes'
    own kinematics method, so for example Euler angle rates will be subject to
    the usual gimbal lock singularity.

    Examples
    --------
    
    This class is implemented as a "singleton", meaning it does not need to
    actually be instantiated and all methods are class methods.

    >>> import archimedes as arc
    >>> from archimedes.spatial import RigidBody, Quaternion
    >>> import numpy as np
    >>> t = 0
    >>> v_B = np.array([1, 0, 0])  # Constant velocity in x-direction
    >>> att = Quaternion([1, 0, 0, 0])  # No rotation
    >>> x = RigidBody.State(
    ...     pos=np.zeros(3),
    ...     att=att,
    ...     v_B=v_B,
    ...     w_B=np.zeros(3),
    ... )
    >>> u = RigidBody.Input(
    ...     F_B=np.array([0, 0, -9.81]),  # Gravity
    ...     M_B=np.zeros(3),
    ...     m=2.0,
    ...     J_B=np.diag([1.0, 1.0, 1.0]),
    ... )
    >>> RigidBody.dynamics(t, x, u)
    State(pos=array([1., 0., 0.]),
      att=Quaternion([0., 0., 0., 0.]),
      v_B=array([ 0.   ,  0.   , -4.905]),
      w_B=array([0., 0., 0.]))

    A common pattern is to have a vehicle model inherit from ``RigidBody.State``,
    even while the model class itself does not inherit from ``RigidBody``.  This
    allows the vehicle model to have its own state representation while still being
    compatible with the rigid body dynamics:

    >>> @arc.struct
    ... class AircraftState(RigidBody.State):
    ...     eng: np.ndarray  # engine state
    ...
    >>> state = AircraftState(
    ...     pos=np.zeros(3),
    ...     att=Quaternion([1, 0, 0, 0]),
    ...     v_B=np.zeros(3),
    ...     w_B=np.zeros(3),
    ...     eng=np.array([0.0]),
    ... )
    >>> x_dot_rb = RigidBody.dynamics(t, state, u)
    >>> x_dot_rb
    State(pos=array([0., 0., 0.]),
      att=Quaternion([0., 0., 0., 0.]),
      v_B=array([ 0.   ,  0.   , -4.905]),
      w_B=array([0., 0., 0.]))

    The ``dynamics`` method returns a ``RigidBody.State`` struct containing only the
    derivatives of rigid body states.  To use this in a vehicle model with additional
    states, you can manually combine the rigid body state derivatives with the
    derivatives of the additional states.

    >>> x_dot = AircraftState(
    ...     pos=x_dot_rb.pos,
    ...     att=x_dot_rb.att,
    ...     v_B=x_dot_rb.v_B,
    ...     w_B=x_dot_rb.w_B,
    ...     eng=np.array([0.1]),  # engine state derivative
    ... )
    >>> x_dot
    AircraftState(pos=array([0., 0., 0.]),
      att=Quaternion([0., 0., 0., 0.]),
      v_B=array([ 0.   ,  0.   , -4.905]),
      w_B=array([0., 0., 0.]),
      eng=array([0.1]))

    Notes
    -----

    The equations of motion implemented here are technically correct only for the
    case of a rigid body with constant mass, inertia, and center of gravity moving
    in an inertial reference frame and without "internal" angular velocity (gyroscopic
    effects).  However, the model can be extended to account for these effects if
    needed by passing pseudo-forces and moments.

    In all the following cases, the effects can be treated as constant, quasi-steady
    (time-varying but with negligible rates), or fully dynamic (time-varying with
    non-negligible rates).  In both cases, the current value and time derivatve should
    be tracked and computed outside of the rigid body model, and the appropriate
    values passed in the input struct.

    - **Variable mass**: Quasi-steady mass may be handled by passing the current mass
        in the input struct.  The mass rate of change :math:`\\dot{m}` enters the
        equations of motion via the time derivative of linear momentum:

        .. math::
            \\frac{d}{dt}(m \\mathbf{v}^B) = \\mathbf{F}^B
            \\implies m \\dot{\\mathbf{v}}^B + \\dot{m} \\mathbf{v}^B = \\mathbf{F}^B

        Hence, mass flow rates can be accounted for by including the pseudo-force
        :math:`-\\dot{m} \\mathbf{v}^B` in the net forces passed as input.

    - **Variable inertia**: In the same way, quasi-steady inertia may be handled
        by passing the current inertia matrix in the input struct.  The inertia rate
        of change :math:`\\dot{\\mathbf{J}}^B` enters the equations of motion via the
        time derivative of angular momentum:

        .. math::
            \\frac{d}{dt}(\\mathbf{J}^B \\boldsymbol{\\omega}^B) = \\mathbf{M}^B
            \\implies \\mathbf{J}^B \\dot{\\boldsymbol{\\omega}}^B
            + \\dot{\\mathbf{J}}^B \\boldsymbol{\\omega}^B = \\mathbf{M}^B

        Non-negligible inertia rates can be accounted for by including the
        pseudo-moment :math:`-\\dot{\\mathbf{J}}^B \\boldsymbol{\\omega}^B` in the
        net moment passed as input.

    - **Variable center of mass**: The equations of motion are derived about the
        center of mass (CM).  However, typically the body-fixed reference frame B is
        defined at some convenient reference point that may not coincide with the
        instantaneous center of mass.  Properties like aerodynamics and propulsion
        behaviors are also often defined with respect to the reference CM.

        If the reference CM is at the origin of the body frame B and the actual CM
        is at a point :math:`\\mathbf{r}_{CM}^B` in body frame B moving with velocity
        :math:`\\dot{\\mathbf{r}}_{CM}^B` with respect to the reference point, then
        the relationship between the state velocity :math:`\\mathbf{v}^B` (that is, the
        inertial velocity of the CM expressed in body frame B) and the velocity of
        the reference point :math:`\\mathbf{v}_{ref}^B` is

        .. math::
            \\mathbf{v}^B = \\mathbf{v}_{ref}^B +
            \\dot{\\mathbf{r}}_{CM}^B + \\boldsymbol{\\omega}^B \\times
            \\mathbf{r}_{CM}^B

        Often this correction is negligible, but if needed then the state velocity
        should be converted to the reference point velocity before computing
        aerodynamics or other quantities referenced to the body frame origin.
        In the common case that the CM is moving due to fuel consumption or payload
        release, the relative velocity :math:`\\dot{\\mathbf{r}}_{CM}^B` is usually
        negligible.

        A more important effect is the moment transfer from the offset of the forces
        acting at the reference point to the actual CM.  If the net force acting on
        the vehicle at the reference point is :math:`\\mathbf{F}_{ref}^B`, then the
        moment about the CM is given by

        .. math::
            \\mathbf{M}^B = \\mathbf{M}_{ref}^B -
            \\mathbf{r}_{CM}^B \\times \\mathbf{F}_{ref}^B

        The same transformation applies to forces computed about an arbitrary reference
        point, but the moment arm will then be the vector from that reference point to
        the instantaneous CM.

    - **Gyroscopic effects**: The full Euler equation for rotational dynamics in a
        non-inertial body-fixed frame is

        .. math::
            \\mathbf{M}^B = \\frac{d\\mathbf{h}^B}{dt} + \\boldsymbol{\\omega}^B
            \\times \\mathbf{h}^B,

        where :math:`\\mathbf{h}^B` is the net angular momentum of the vehicle in the
        body frame B.  If the vehicle does not have any "internal" angular momentum,
        then :math:`\\mathbf{h}^B = \\mathbf{J}^B \\boldsymbol{\\omega}^B` and the
        equations reduce to those implemented here.
        
        However, if there are significant additional contributions to angular momentum,
        these affect the dynamics via gyroscopic pseudo-moments.  If a system has
        internal angular momentum :math:`\\mathbf{h}_{int}^B =
        \\sum_{i} \\mathbf{J}_{int,i}^B \\boldsymbol{\\omega}_{int,i}^B`, these
        contributions must be included:

        .. math::
            \\mathbf{M}^B = \\frac{d}{dt}(\\mathbf{J}^B \\boldsymbol{\\omega}^B)
            + \\frac{d\\mathbf{h}_{int}^B}{dt}
            + \\boldsymbol{\\omega}^B \\times \\mathbf{J}^B \\boldsymbol{\\omega}^B
            + \\boldsymbol{\\omega}^B \\times \\mathbf{h}_{int}^B

        The additional terms involving :math:`\\mathbf{h}_{int}^B` can be treated as
        pseudo-moments and included in the net moment passed as input.  The usual
        logical flow would be to compute both the internal angular momentum and its
        time derivative outside of the rigid body model (e.g. as a subsystem
        calculation), and then pass the net effective moment

        .. math::
            \\mathbf{M}_\\mathrm{eff}^B = \\mathbf{M}^B
            - \\frac{d}{dt}(\\mathbf{h}_{int}^B)
            - \\boldsymbol{\\omega}^B \\times \\mathbf{h}_{int}^B

        as the input to the rigid body dynamics.

        For example, a calculation of the gyroscopic effects of a spinning rotor
        with inertia :math:`\\mathbf{J}_\\mathrm{rot}^B`, angular velocity
        :math:`\\boldsymbol{\\omega}_\\mathrm{rot}^B`, and negligible angular
        acceleration might look like:

        .. code-block:: python

            h_int_B = J_rot_B @ w_rot_B  # Rotor angular momentum

            # Compute effective moment including gyroscopic effects
            M_eff_B = M_B - np.cross(w_B, h_int_B)

    - **Non-inertial reference frame**: These equations of motion are valid only
        when referenced to a Newtonian inertial frame N.  This is of course an
        idealization in all cases, but it is always possible to find _some_ frame
        that is nearly enough inertial for modeling purposes.

        However, a common situation in aerospace applications is to model a body
        moving relative to a rotating planetary frame E (e.g. the Earth-centered,
        Earth-fixed frame ECEF) that is assumed to be in non-accelerating but
        rotating with some angular velocity :math:`\\boldsymbol{\\Omega}_{E}` with
        respect to the inertial frame N.  In this case an alternative formulation
        uses a state vector composed of:

        - :math:`\\mathbf{p}^E` = position of the center of mass in the frame E
        - :math:`\\mathbf{q}` = attitude (orientation) of the vehicle with respect to E
        - :math:`\\mathbf{v}^E` = velocity of the center of mass in rotating frame E
        - :math:`\\boldsymbol{\\omega}^B` = angular velocity in body frame (ω_B) with
            respect to the inertial frame N

        Then the equations of motion are [1]_:

        .. math::
            \\dot{\\mathbf{p}}^E &= \\mathbf{v}^E \\\\
            \\dot{\\mathbf{q}} &= \\frac{1}{2} \\mathbf{q} \\otimes \\left(
                \\boldsymbol{\\omega}^B - \\boldsymbol{\\Omega}_{E}^B \\right) \\\\
            \\dot{\\mathbf{v}}^E &= \\frac{1}{m}\\mathbf{F}^E
                - 2 \\boldsymbol{\\Omega}_{E}^E \\times \\mathbf{v}^E
                - \\boldsymbol{\\Omega}_{E}^E \\times
                (\\boldsymbol{\\Omega}_{E}^E \\times \\mathbf{p}^E)
            \\dot{\\boldsymbol{\\omega}}^B &= \\mathbf{J}_B^{-1}(\\mathbf{M}^B
                - \\boldsymbol{\\omega}^B \\times (\\mathbf{J}^B
                \\boldsymbol{\\omega}^B))

        Unfortunately, this cannot be straightforwardly reconciled with the
        implementation here, even with the addition of the Coriolis and centrifugal
        pseudo-forces.  This is because of the definition of the attitude and angular
        velocity with respect to different reference frames (E and N, respectively).
        Using the angular velocity relative to frame N allows the use of the Euler
        dynamics equation without complex pseudo-moments, but means that the angular
        velocity must be modified by :math:`-\\boldsymbol{\\Omega}_{E}^B` in the
        attitude kinematics.

        The equations above could be implemented in an ECEF frame with a simple custom
        rigid body class:

        .. code-block python

            @struct
            class EarthReferencedBody:
                rot_earth: float = 7.292e-5

                @struct
                class State:
                    pos: np.ndarray  # ECEF position
                    att: Attitude  # Body attitude relative to ECEF
                    v_E: np.ndarray  # ECEF velocity
                    w_B: np.ndarray

                @struct
                class Input:
                    F_E: np.ndarray  # Forces in Earth frame
                    M_B: np.ndarray  # Moments in body frame
                    m: float  # Mass
                    J_B: np.ndarray  # Inertia matrix in body frame

                def dynamics(self, t: float, x: State, u: Input) -> State:
                    Ω_E = np.hstack([0.0, 0.0, self.rot_earth])
                    R_BE = x.att.as_matrix()
                    v_E, p_E = x.v_E, x.pos  # ECEF position, velocity

                    # Position and attitude kinematics
                    pos_deriv = v_E
                    att_deriv = x.att.kinematics(x.w_B - R_BE @ Ω_E)

                    # Force equation with Coriolis and centrifugal effects
                    dv_E = (u.F_E / u.m) - np.cross(Ω_E, 2 * v_E - np.cross(Ω_E, p_E))
                    
                    # Moment equation (same as body-referenced formulation)
                    dw_B = np.linalg.solve(
                        u.J_B, u.M_B - np.cross(x.w_B, u.J_B @ x.w_B)
                    )

                    # Output time derivatives of substates
                    return RigidBody.State(
                        pos=pos_deriv, att=att_deriv, v_E=dv_E, w_B=dw_B
                    )

        While this formulation does cover a substantial number of orbtial mechanics
        applications, it is not one-size-fits all.  Are centrifugal effects accounted
        for in the gravity model?  Are precession and nutation important?  Is the
        angular velocity time-varying?  The present design prioritizes _customization_
        over _comprehensiveness_.

        In short, handling of non-inertial frames in Archimedes still needs some
        design work and is not robustly supported.  The recommendation is to
        implement custom rigid body dynamics based on the above equations.  If you
        would like to see support for non-inertial frames be a higher priority,
        please feel free to raise the issue in the `Discussions
        <https://github.com/PineTreeLabs/archimedes/discussions>`__ page on GitHub.

    As a combined example of pseudo-force handling, if you have a flight dynamics
    model with constant inertia and quasi-steady mass and CM location, a typical
    calculation of the rigid body inputs might look like the following:

    .. code-block:: python

        @struct
        class FlightVehicle:
            dry_mass: float
            wet_mass: float
            J_B: np.ndarray  # constant inertia matrix
            dCM_dm: np.ndarray  # CM location rate w.r.t. mass [m/kg]
    
            @struct
            class State(RigidBody.State):
                fuel_mass: float
                ...

            @struct
            class Input:
                throttle: float
                ...

            def dynamics(t, x: State, u: Input) -> State:
                # Compute current mass and CM location
                m = self.dry_mass + x.fuel_mass
                r_CM_B = self.dCM_dm * (self.wet_mass - m)  # ref point -> CM

                # Velocity at reference point
                v_ref_B = x.v_B - np.cross(x.w_B, r_CM_B)

                # Aerodynamic forces and moments at body frame origin
                F_aero_B, M_aero_B = self.aerodynamics(v_ref_B, ...)

                # Propulsion forces and moments at body frame origin
                F_prop_B, M_prop_B = self.propulsion(v_ref_B, ...)

                # Net forces
                F_B = F_aero_B + F_prop_B

                # Net moments, including moment transfer of forces to CM
                M_B = M_aero_B + M_prop_B - np.cross(r_CM_B, F_B)

                # Compute rigid body dynamics
                rb_input = RigidBody.Input(F_B=F_B, M_B=M_B, m=m, J_B=self.J_B)
                x_dot_rb = RigidBody.dynamics(t, x, rb_input)

                # Combine with other state derivatives (e.g. fuel mass)
                return Vehicle.State(
                    pos=x_dot_rb.pos,
                    att=x_dot_rb.att,
                    v_B=x_dot_rb.v_B,
                    w_B=x_dot_rb.w_B,
                    fuel_mass=-self.propulsion.burn_rate(u),
                    ...
                )


    References
    ----------
    .. [1] Lewis, F. L., Johnson, E. N., & Stevens, B. L. (2015).
            Aircraft Control and Simulation. Wiley.
    """

    @struct
    class State:
        pos: np.ndarray  # Position of the center of mass in the inertial frame
        att: Attitude  # Attitude (orientation) of the vehicle
        v_B: np.ndarray  # Velocity of the center of mass in body frame B
        w_B: np.ndarray  # Angular velocity in body frame (ω_B)

    @struct
    class Input:
        F_B: np.ndarray  # Forces in body frame
        M_B: np.ndarray  # Moments in body frame
        m: float  # Mass
        J_B: np.ndarray  # Inertia matrix in body frame

    @classmethod
    def dynamics(cls, t: float, x: State, u: Input) -> State:
        """Calculate 6-dof dynamics

        Args:
            t: time
            x: dynamic state
            u: input struct with net forces and moments

        Returns:
            x_t: time derivative of the state vector
        """
        R_BN = x.att.as_matrix()
        pos_deriv = R_BN.T @ x.v_B
        att_deriv = x.att.kinematics(x.w_B)
        dv_B = (u.F_B / u.m) - np.cross(x.w_B, x.v_B)
        dw_B = np.linalg.solve(u.J_B, u.M_B - np.cross(x.w_B, u.J_B @ x.w_B))
        return RigidBody.State(pos=pos_deriv, att=att_deriv, v_B=dv_B, w_B=dw_B)
