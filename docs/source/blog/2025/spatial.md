---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: archimedes
---

# [Spatial Mechanics]{.hidden-title}

```{image} _static/spatial_graduates_light.png
:class: only-light
```

```{image} _static/spatial_graduates_dark.png
:class: only-dark
```

**_Inside the new `spatial` module_**

Jared Callaham • 16 Oct 2025

<!-- Graphic: 6dof gimbal with graduation cap -->

Release v0.3.1 marks the graduation of the [`spatial`](#archimedes.spatial) module out of experimental status and into production.
This module includes core functionality for 3D vehicle dynamics modeling in a range of domains.

In this post we'll introduce the most important members of this module: the [`Attitude`](#archimedes.spatial.Attitude) protocol and [`RigidBody`](#archimedes.spatial.RigidBody) class.
These let you represent 3D rotations and 6dof rigid body dynamics in a way that's extensible, customizable, and compatible with the rest of Archimedes, including C code generation, autodiff, and tree operations.

We'll cover:

- Why you might want to use the `spatial` module
- Attitude representations and the `Attitude` protocol
- 6dof dynamics and the `RigidBody` class
- Building your own vehicle models
- What's next for `spatial`

This post serves as an announcement of these new features, but will also be updated to provide a basic reference for the relevant conventions and equations.

## What `spatial` Does

The [`spatial`](#archimedes.spatial) module is designed for _cross-domain spatial mechanics_.
This means that there's no specific physics modeling like aerodynamics or gravity, but there are reusable components that come in handy across a wide range of application areas; satellites, airplanes, rockets, drones, watercraft, and cars can all use the same basic spatial dynamics primitives.

The module is built on two main capabilities: **3D rotation representations**, and **6dof rigid body dynamics**.
We'll cover these next.

### What's not in `spatial`

This module does not (yet) handle any multibody dynamics - RNA, CRB, contact mechanics, etc. - although this is on the longer-term [roadmap](../../roadmap.md).

Much sooner on the roadmap, `spatial` will eventually add functionality for _spatial transforms_ combining translation and rotation, plus _kinematic tree_ data structures to handle complex and moving reference frame situations (common in robotics and orbital mechanics, for instance).

In short, the current functionality is useful for 3D orientations and when simulating isolated rigid bodies in 3D (especially vehicle dynamics), but not for full-fledged multibody systems including joints and collisions.

## 3D Rotations

Direction cosine matrices, Euler angles, and quaternions are all representations of 3D rotations - how one frame or body is oriented relative to another in space.
Archimedes represents these rotations using the [`Attitude`](#archimedes.spatial.Attitude) protocol and classes that implement this abstract interface, most importantly [`EulerAngles`](#archimedes.spatial.EulerAngles) and [`Quaternion`](#archimedes.spatial.Quaternion).

This class is modeled directly on [SciPy's `Rotation` class](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) and is unit tested directly against the SciPy behavior.
However, by re-implementing it in Archimedes we can ensure that it is compatible with all of the symbolic-numeric capabilities like autodiff and codegen.

```{code-cell} python
:tags: [remove-cell]
# ruff: noqa: N803, N816
```

```{code-cell} python
:tags: [hide-cell]
import numpy as np

import archimedes as arc
from archimedes.spatial import RigidBody, Rotation, Quaternion
```

```{code-cell} python
# Rotate a vector from the body frame B to an inertial reference frame N if the body
# attitude is given by (roll, pitch, yaw) Euler angles rpy
def to_inertial(rpy, v_B):
    att = Rotation.from_euler("xyz", rpy)
    return att.apply(v_B)


rpy = np.array([0.1, 0.2, 0.3])
v_N = np.array([10.0, 0.0, 0.0])
print(arc.jac(to_inertial)(rpy, v_N))  # dv_B/drpy
```

```{code-cell} python
:tags: [remove-cell]
from scipy.spatial.transform import Rotation as ScipyRotation

v_B_arc = to_inertial(rpy, v_N)
v_B_sp = ScipyRotation.from_euler("xyz", rpy).apply(v_N)
assert np.allclose(v_B_sp, v_B_arc)
```

Internally, `Rotation` uses a quaternion representation, as in the SciPy implementation.
However, it can be instantiated from a quaternion, a rotation matrix (DCM), or any combination of Euler angles, giving you a lot of flexibility in how you think about representing your attitude while still providing a robust, minimal, singularity-free representation of 3D rotations.

:::{note}
One difference from the SciPy version is that by default Archimedes uses a scalar-first component ordering, more common in engineering applications compared to, for instance, computer graphics.
:::

Archimedes also diverges from the SciPy implementation by providing a `derivative` method that calculates attitude kinematics, assuming the rotation represents the orientation of a moving body with respect to some reference frame.
Given the angular velocity of the body in its own frame, $\omega_B$, this function calculates the time derivative of the rotation using quaternion kinematics:

```{math}
\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \mathbf{\omega}_B
```

The actual implementation of quaternion kinematics deviates slightly from the ideal form by adding a "Baumgarte stabilization" to preserve the unit-norm requirement.
With a stabilization factor of $\lambda$, the full kinematics model is:

```{math}
\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \mathbf{\omega}_B - \lambda * (||\mathbf{q}||² - 1) \mathbf{q}.
```

A factor of $\lambda = 1$ is a good default (and is the default in `RigidBody` as well).

```{code-cell} python
att = Rotation.from_euler("xyz", rpy)
w_B = np.array([0.0, 0.1, 0.0])  # 0.1 rad/sec pitch-up
att.derivative(w_B, baumgarte=1.0)
```

:::{caution}
The `kinematics` method returns the time derivative of the `Rotation` as a new `Rotation` instance.
This is convenient for working with ODE solvers and other algorithms that expect the output to have the same structure as the input state.
However, keep in mind that the time derivative $\dot{\mathbf{q}}$ is _not_ itself a valid rotation.
Hence, you CANNOT use `att.kinematics(w_B).as_euler("xyz")` to get the Euler angle rates (instead use [`euler_kinematics`](#archimedes.spatial.euler_kinematics) if you need this).
:::

This attitude kinematics calculation comes in particularly handy for the second major functionality released with `spatial`: 6dof rigid body dynamics modeling.

## 6dof Dynamics

A "6dof" rigid body has three translational and three rotational "degrees of freedom" from the point of view of Lagrangian mechanics.
From a state-space modeling perspective, this system has either 12 or 13 dynamical states (depending on whether you use Euler or quaternion kinematics).
The rigid body dynamics model implements the equations of motion of such a body given specified forces, torques, and mass/inertia characteristics.

The Archimedes [`RigidBody`](#archimedes.spatial.RigidBody) implementation follows the conventions of the classic GNC textbook ["Aircraft Control and Simulation"](https://doi.org/10.1002/9781119174882) by Stevens, Lewis, and Johnson.
Hence, the terminology and implementation is heavily based on flight dynamics applications, though this can be adapted straightforwardly to other domains.
For an in-depth description of the conventions, notation, and derivation of the equations of motion, refer to the textbook.

Our rigid body model assumes two reference frames: a body-fixed frame "B" with the origin at the center of mass, and a Newtonian inertial frame "N" (for instance the world or ground frame in flight dynamics).

```{image} _static/spatial_frames_light.png
:class: only-light
```

```{image} _static/spatial_frames_dark.png
:class: only-dark
```

Vectors are suffixed with `*_B` or `*_N` to indicate their coordinate systems.
In this convention, the dynamical states for a rigid body are four vectors:

- `p_N`: the position of the body in the inertial frame ($\mathbf{p}_N$)
- `att`: the attitude of the body with respect to the inertial frame (by default a `Rotation` $\mathbf{q}$, but can optionally be roll-pitch-yaw sequence)
- `v_B`: the translational velocity of the body in its own coordinate system B ($\mathbf{v}_B$)
- `w_B`: the body-relative angular velocity vector ($\mathbf{\omega}_B$)

The governing equations for these four state components depend on the applied forces and moments in the body frame ($\mathbf{F}_B$ and $\mathbf{M}_B$, respectively), as well as on the mass $m$ and inertia matrix $J_B$ of the vehicle.
If the mass and/or inertia matrix are changing significantly in time, their time derivatives can also be provided (we'll ignore this here, since this is uncommon).

Then the equations of motion are:

```{math}
\begin{align*}
\dot{\mathbf{p}}_N &= \mathbf{R}_{BN}^T(\mathbf{q}) \mathbf{v}_B \\
\dot{\mathbf{q}} &= \frac{1}{2} \mathbf{q} \otimes \mathbf{\omega}_B
    - \lambda * (||\mathbf{q}||^2 - 1) \mathbf{q} \\
\dot{\mathbf{v}}_B &= \frac{1}{m}\mathbf{F}_B - \mathbf{\omega}_B \times \mathbf{v}_B \\
\dot{\mathbf{\omega}}_B &= \mathbf{J}_B^{-1}(\mathbf{M}_B
    - \mathbf{\omega}_B \times (\mathbf{J}_B \mathbf{\omega}_B))
\end{align*}
```

The `RigidBody` class exists to calculate these equations for a generic body - you just have to provide forces, moments, mass, and inertia characteristics. 
The idea is that you can use this as a building block and construct your own vehicle models (or models of whatever it is you're building) by implementing the domain-specific physics models and letting Archimedes handle the generic parts.

### Implementation

The `RigidBody` class structure looks roughly like the following:

```python
@struct
class RigidBody:
    rpy_attitude: bool = False  # If True, use roll-pitch-yaw for attitude
    baumgarte: float = 1.0  # Baumgarte stabilization factor for quaternion kinematics

    @struct
    class State:
        p_N: np.ndarray  # Position of the center of mass in the Newtonian frame N
        att: Rotation | np.ndarray  # Attitude (orientation) of the vehicle
        v_B: np.ndarray  # Velocity of the center of mass in body frame B
        w_B: np.ndarray  # Angular velocity in body frame (ω_B)

    @struct
    class Input:
        F_B: np.ndarray  # Net forces in body frame B
        M_B: np.ndarray  # Net moments in body frame B
        m: float  # mass [kg]
        J_B: np.ndarray  # inertia matrix [kg·m²]
        dm_dt: float = 0.0  # mass rate of change [kg/s]
        # inertia rate of change [kg·m²/s]
        dJ_dt: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # type: ignore


    def dynamics(self, t: float, x: State, u: Input) -> State:
        ...
```

See the [source code](https://github.com/PineTreeLabs/archimedes/blob/main/src/archimedes/spatial/_rigid_body.py) for the actual implementation.

The inner classes `State` and `Input` help to organize the data and states, and the `dynamics` method does the work of actually calculating the equations of motion as given above.

:::{note}
**On time-varying mass/inertia**: it is relatively common to have vehicles that change mass and inertia properties over time (e.g. a rocket burning fuel).
`RigidBody` takes these as inputs so that you can manage their characteristics however you want.
Technically, when the mass and inertia are time-varying this adds terms like $\dot{m} v_B$ to the dynamics equations, though under most circumstances these contributions are negligible even if $\dot{m} \neq 0$.
The compromise model in this case is the "quasi-steady" approximation: provide time-varying mass $m(t)$ as inputs to the `dynamics` method, but leave $\dot{m} = 0$.
This is why the `Input` struct by default has `dm_dt = dJ_dt = 0`.
But it's up to you to ignore/include these effects in whatever way makes the most sense for your application.
:::

While [`RigidBody`](#archimedes.spatial.RigidBody) uses quaternion kinematics by default for stability and robustness - critical for vehicles like satellites, quadrotors, and fighter jets - there is also the option to use roll-pitch-yaw Euler kinematics for bodies like cars and ships that (nominally) won't reach 90-degrees pitch-up and hit the gimbal lock singularity.
In these cases you can set `rpy_attitude = True` and use a roll-pitch-yaw sequence instead of the `Rotation` for the attitude representation.

<!--
:::{note}
Why are there _six_ degrees of freedom? For historical reasons (I assume related to Lagrangian mechanics) the number of _degrees of freedom_ of a mechanical system are the number of variables required to say where something _is_, but not how it is moving.
If there are three components of position ($x$, $y$, $z$) and three components of attitude (roll, pitch, yaw), we have a total of six degrees of freedom.
The quaternion attitude representation has four components but it must be unit norm, so technically this still only contributes three degrees of freedom.

So, a point mass in 2D has two degrees of freedom ($x$ and $y$), while a 3D point mass and longitudinal vehicle dynamics both have three degrees of freedom (3D location or 2D location plus pitch).
Full 3D rigid body dynamics have six degrees of freedom, and certain aftermarket Deloreans have seven (all the usual ones plus time).

This is a bit with the prevalence of modern state-space modeling, since there are roughly twice as many dynamical states as there are "degrees of freedom".
As implemented here, the "6dof" model has thirteen dynamical states (three position, three velocity, four quaternion, and three angular velocity).
Flux capacitors not included.
:::
-->

Here's the `RigidBody` class in action:

```{code-cell} python

x = RigidBody.State(
    pos=np.array([0.0, 0.0, 10.0]),
    att=Quaternion.identity(),
    vel=np.zeros(3),
    w_B=np.zeros(3),
)

u = RigidBody.Input(
    F_B=np.array([0.0, 0.0, 9.8]),
    M_B=np.array([0.0, 0.1, 0.0]),
    m=10.0,
    J_B=np.eye(3),
)

RigidBody.dynamics(0.0, x, u)
```

In this simple case, since the body and world axes are aligned (`Rotation.identity()`) and we start out with zero angular velocity, most of the complexity from non-inertial frames in the equations of motion disappears. and we just get $m \dot{\mathbf{v}}_B = \mathbf{F}_B$ and $\mathbf{J}_B \dot{\mathbf{\omega}}_B = \mathbf{M}_B$.

## Custom Vehicle Models

The power of `RigidBody` comes from being able to use this as a component inside more complex vehicle models.

Two basic patterns you might use for this are **inheritance** and **composition**.

### Inheritance

With this pattern, the vehicle model simply inherits from `RigidBody` directly.
This is convenient when there are no additional state variables in the model, for instance with a flight dynamics model that uses lookup tables for aerodynamics and propulsion models:

```python
@struct
class Aircraft(RigidBody):
    m: float
    J_B: np.ndarray

    @struct
    class Input:
        throttle: float
        rudder: float
        aileron: float
        elevator: float

    def calc_aero(
        self, x: RigidBody.State, u: Aircraft.Input
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate aerodynamic forces and moments"""

    def calc_eng(
        self, x: RigidBody.State, u: Aircraft.Input
    ) -> np.ndarray:
        """Calculate engine thrust"""

    def dynamics(
        self, t: float, x: RigidBody.State, u: Aircraft.Input
    ) -> RigidBody.State:
        # Aerodynamics and propulsion models
        F_aero_B, M_aero_B = self.calc_aero(x, u)
        F_eng_B = self.calc_eng(x, u)

        # Use the state attitude to calculate gravity in body axes
        F_grav_N = self.m * np.hstack([0, 0, 9.81])
        F_grav_B = x.att.rotate(F_grav_N, inverse=True)

        # Net forces/moments
        F_B = F_aero_B + F_eng_B + F_grav_B
        M_B = M_aero_B

        # Use RigidBody.dynamics to evaluate the equations of motion
        u_rb = RigidBody.Input(F_B=F_B, M_B=M_B, m=self.m, J_B=self.J_B)
        return super().dynamics(t, x, u_rb)
```

### Composition

For more complex models it is usually more convenient to instead treat the `RigidBody` as one component of several.
This can be a more natural way to organize hierarchical state variables:

```python
@struct
class Aircraft:
    gravity: GravityModel
    atmosphere: AtmosphereModel

    aero: AeroModel
    engine: EngineModel

    m: float
    J_B: np.ndarray

    @struct
    class State(RigidBody.State):
        aero: AeroModel.State
        engine: Engine.State

    @struct
    class Input:
        throttle: float
        rudder: float
        aileron: float
        elevator: float

    def dynamics(
        self, t: float, x: Aircraft.State, u: Aircraft.Input
    ) -> Aircraft.State:
        # Aerodynamics and propulsion models
        F_aero_B, M_aero_B = self.aero.output(x, u)
        F_eng_B = self.engine.output(x, u)

        # Time derivatives of aerodynamic and engine states
        x_aero_dot = self.aero.dynamics(x, u)
        x_eng_dot = self.engine.dynamics(x, u)

        # Use the state attitude to calculate gravity in body axes
        F_grav_N = self.gravity(x.rigid_body.pos)
        F_grav_B = x.rigid_body.att.rotate(F_grav_N, inverse=True)

        # Net forces/moments
        F_B = F_aero_B + F_prop_B + F_grav_B
        M_B = M_aero_B

        # Evaluate the equations of motion
        u_rb = RigidBody.Input(F_B=F_B, M_B=M_B, m=self.m, J_B=self.J_B)
        x_rb_dot = RigidBody.dynamics(x, u_rb)

        return self.State(
            pos=x_rb_dot.pos,
            att=x_rb_dot.att,
            vel=x_rb_dot.vel,
            w_B=x_rb_dot.w_B,
            aero=x_aero_dot,
            engine=x_eng_dot,
        )
```

Now the aerodynamic state can handle lag effects or other unsteady aerodynamic behavior, and the engine can have its own internal dynamics as well.
This can be a much more flexible and powerful approach - since the `Aircraft` implementation doesn't handle the details of any of these subsystems, it's easy to create and test a range of different component models.
For instance, the engine model here could be anything from a simple linear thrust approximation to a detailed physics-based propulsion system model including turbomachinery and combustion calculations.

:::{note}
For a deeper dive on hierarchical modeling in Archimedes, check out the [tutorial series](../../tutorials/hierarchical/hierarchical00.md), which goes into detail on the [`@struct`](#archimedes.struct) decorator, recommended design patterns, and configuration management for complicated hierarchical models.
:::

We'll be releasing more in-depth examples of different vehicle dynamics models soon, so be sure to [sign up for the mailing list](https://jaredcallaham.substack.com/embed) to stay in the loop.

## What's Next

The new `spatial` module is the first core physics modeling functionality in Archimedes, but this is just the beginning.
For `spatial` itself, the next priorities are _spatial transformations_ (transformation + rotation), _kinematic trees_ for handling multiple reference frames, and _interpolations_ (slerp) for trajectory generation and optimization.

Beyond `spatial`, we'll be adding some common functionality for different classes of vehicle models, such as reference gravitational and atmospheric models like WGS84 and USSA1976.
Tools for detailed propulsion systems modeling are more niche and thus farther out on the roadmap, but there are proof of concept demos already, so feel free to reach out if you're interested in that.

Finally, to see this module in action check out the [Subsonic F-16 series](../../tutorials/f16/f16_00.md), where we implement the NASA F-16 benchmark from scratch, relying heavily on `spatial` for attitude representations and 6dof dynamics.
More detailed application examples will be released soon to provide full reference implementations of different classes of vehicle dynamics models (particularly aerospace-related, but also reach out if there's something else you'd like to see).

Speaking of getting in touch, if you're interested in this topic or Archimedes more generally, be sure to:

- **⭐ Star the Repository**: This shows support and interest and helps others discover the project
- **📢 Spread the Word**: Think anyone you know might be interested?
- **🗞️ Stay in the Loop**: [Subscribe](https://jaredcallaham.substack.com/embed) to the newsletter for updates and announcements

The [GitHub Discussions](https://github.com/pinetreelabs/archimedes/discussions) page is also a great place to give feedback, ask questions, or share any projects you want to use Archimedes for.
Bug reports, feature requests, and complaints about documentation quality are invaluable for open-source projects like Archimedes; these are also welcome on the [Issues](https://github.com/pinetreelabs/archimedes/issues) tab.

---

:::{admonition} About the Author
:class: blog-author-bio

**Jared Callaham** is the creator of Archimedes and principal at Pine Tree Labs.
He is a consulting engineer on modeling, simulation, optimization, and control systems with a particular focus on applications in aerospace engineering.

*Have questions or feedback? [Open a discussion on GitHub](https://github.com/jcallaham/archimedes/discussions)*
:::