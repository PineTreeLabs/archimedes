# Development Roadmap

This page outlines the development priorities and future direction of Archimedes. As an open-source project, we welcome community input and contributions across any of these areas. In particular, feel free to create or contribute to a thread with the "RFC" (request for comments) tag on the [GitHub Discussions](https://github.com/PineTreeLabs/archimedes/discussions) page with your thoughts on the roadmap.

This document was last updated **29 Sep 2025** for **v0.2.0**.
It will be updated regularly, but with ongoing development it's possible that work has proceeded beyond the status given here.

## Vision & Design Philosophy

Archimedes is designed to accelerate the development of complex hardware systems through three integrated pillars: **physics modeling**, **algorithm development**, and **hardware deployment**. Our goal is to provide a cohesive, Python-native framework that bridges the gap between rapid prototyping and production deployment.

### Core Design Principles

Here's what we're aiming for:

- **Progressive Disclosure**: Simple things stay simple, while complex capabilities remain accessible. Beginners can start without understanding the full system, while experts can access advanced features.
- **Python-Native Experience**: Interface design that leverages Python's strengths rather than simply wrapping existing C++ libraries.
- **One-Stop Shop**: A cohesive ecosystem that avoids the fragmentation and integration overhead of assembling disparate tools.
- **Made for Hackers**: Extensible, composable, and transparent — users should understand and be able to modify the tools they depend on.

The goal is to go slow, test thoroughly, and provide a complete solution rather than rushing to release incomplete capabilities.
From a codebase perspective, this also means that we enforce standards like 100% code coverage (every line of code is covered by at least one unit test) and static type checking with [MyPy](https://mypy-lang.org/).

## Current State: Beta Release

The reliability of Archimedes gets a huge boost from being built on CasADi, which has an excellent track record and is widely used and tested in a variety of applications.
That said, it is still an early-stage project and some things are still likely to evolve.

### What's Ready

- **Stable Core**: Symbolic compilation, automatic differentiation, ODE solving, and optimization are stable with well-tested APIs
- **Semantic Versioning**: We guarantee API stability for all public interfaces going forward

### What's Experimental

We're committed to smooth upgrade paths and clear deprecation notices. The `experimental` module provides a sandbox for new features without destabilizing the core API.

- **Physics Library**: Expanding rapidly but still in the `experimental` module
- **Code Generation**: Functional proof-of-concept with ongoing API refinements
- **Hybrid Systems**: Core concepts implemented, advanced features in development

For instance, some features currently living in `experimental` include:

- Support for 3D spatial rotations via quaternions, DCM, and Euler angles
- 6 degree-of-freedom flight dynamics with modules for atmosphere, gravity, and sensors
- A trajectory optimization code using pseudo-spectral collocation with adaptive mesh refinement
- Basic neural network functionality including feedforward nets and an ADAM optimizer

## Development Focus Areas

The Archimedes roadmap is organized into four primary development areas, each addressing critical needs in engineering simulation and deployment:

### 1. Hybrid Simulations

Modern engineering systems combine continuous dynamics with discrete events and distributed control logic. Archimedes is building comprehensive support for these hybrid systems.

**Current Capabilities:**

Currently, Archimedes provides ODE solving capabilities with a SciPy-like interface for solving pure continuous-time dynamics
Under the hood, this uses the powerful and highly stable [CVODES](https://computing.llnl.gov/projects/sundials/cvodes) solver, suitable for both stiff and non-stiff systems.
It also leverages sparse automatic differentiation to calculate Jacobians as needed.
The solver supports automatic differentiation, so you can create fully differentiable simulations.

**Near-term:**
- **Zero-crossing Event Detection**: SUNDIALS root-finding interface for triggered events (collisions, switches, limits)
- **Event-triggered Actions**: Discontinuous state changes upon event detection
- **DAE Solving**: Continuous-time dynamical systems with algebraic constraints
- **Basic Task Framework**: Periodic tasks and simple scheduling

**Long-term:**
- **Multirate Control Framework**: Task groups, dependency resolution, priority-based scheduling
- **Distributed Control Systems**: Inter-group communication with port-based interfaces
- **Communication Protocols**: Shared memory, UDP, CAN
- **Co-simulation**: FMI/FMU integration for external simulator coupling

### 2. Path to Hardware

Seamless transition from simulation to hardware deployment is a core Archimedes capability, targeting embedded systems and real-time applications.

**Current Capabilities:**

Currently, Archimedes supports C code generation via CasADi's codegen engine.
CasADi generates self-contained, deterministic low-level C implementations, and Archimedes layers on a predictable API that maps Python data structures to their C equivalents.
The result is portable, standalone C code generated easily from Python.

For more details, see the [codegen tutorial series](tutorials/codegen/codegen00.md).

We also have a [proof-of-concept deployment workflow](tutorials/deployment/deployment00.md) incorporating a DIY hardware-in-the-loop (HIL) testing setup, with expanded capabilities outlined below.

**Near-term:**
- **Production HIL Testing**: Complete framework for automated hardware validation
- **Performance Analysis**: Memory usage and execution time profiling tools

**Long-term:**
- **Enhanced Code Generation**: Type inference, fixed-point arithmetic support
- **Advanced Deployments**: RTOS, distributed systems, hybrid or multiprocessor MCU architectures
- **Certification Support:** Documentation/traceability for automotive/aerospace safety-critical systems

### 3. Physics Modeling

A comprehensive library of reusable, well-tested physics models accelerates development across engineering domains.

**Current Capabilities:**

At the moment there are a few modules with the `experimental` tag, including 6-DOF rigid body dynamics, gravity and atmospheric effects, spatial rotations, and orbital elements.
We plan to expand on these features and migrate them to the main codebase when they're ready, but in the meantime you can "invent it here" with fully customized code.


**Near-term:**
- **Spatial Rotations**: Quaternions, DCM, Euler angles
- **Reference Frames**: Kinematic trees to manage complex and time-varying transformations
- **Linear Systems**: LTI utilities, transfer functions, conversions
- **Environment Models**: Atmosphere, gravity, wind (Dryden/von Karman)

**Long-term:**
- **Sensors/Actuators**: IMU, GNSS, motors, encoders with realistic models
- **Multibody Systems**: Joint constraints, Featherstone algorithms, contact/collision
- **Orbital Mechanics**: Elements, propagation, coordinate frames
- **Advanced Physics**: 1D FEA/FVM primitives for structural/thermal/fluid applications
- **Propulsion Systems**: Turbomachinery, combustion, two-phase flow modeling

### 4. Algorithm Development

Archimedes will continue to expand support for control systems building blocks, helping you design and evaluate cutting-edge architectures and algorithms.

**Current Capabilities**

Currently Archimedes has "production" support for some basic control systems tools:

- **Core Filters**: Support for Kalman variants (EKF, UKF), including autodiff for Jacobians
- **Parameter Estimation**: Implementation of the nonlinear prediction error method (PEM)
- **Ecosystem Tools**: Easily leverage SciPy, python-control, and other libraries for design and analysis

The `experimental` module also includes preliminary support for codegen-compatible LQR controllers, IIR filters and discrete transfer functions, trajectory optimization, and model reduction.

**Near-term**:
- **Signal Processing**: IIR/FIR filters, FFT, windowing
- **Finite State Machines**: Hierarchical states, and triggered transitions
- **Trajectory Optimization/MPC**: Solve optimal control problems offline or in real-time 
- **Sensor Fusion**: Expanded support for advanced Kalman filtering, including error-state methods
- **System Identification**: Implementation of the N4SID algorithm for linear system ID
- **Gray-box Modeling**: Hybrid models with first-principles physics and data/driven approximations

**Long-term**:
- **Uncertainty Quantification**: Polynomial chaos, Monte Carlo, covariance analysis
- **PyTorch Integration**: Train and deploy machine learning-based control algorithms

## The Road to 1.0

The 1.0 release will represent our commitment that users can build production systems on Archimedes without worrying about foundational changes. It signals API stability and comprehensive validation across core use cases.
We're targeting this 1.0 release in mid-2026, following 6-9 months of beta stabilization.
This timeline allows for:

- **Real-world Validation**: Complex applications built on the current API to stress-test design decisions
- **Community Feedback**: Integration of user feedback from beta adoption
- **API Maturation**: Ensuring core interfaces remain stable without breaking changes
- **Performance Characterization**: Well-understood behavior across different application domains

### 1.0 Criteria

- **Stability**: 6+ months without breaking API changes in core functionality
- **Validation**: Successful deployment in multiple complex, real-world applications
- **Documentation**: Complete onboarding path from tutorials to production use
- **Performance**: Documented and predictable performance characteristics
- **Community**: Active user base with established best practices

The beta period is intentionally designed to let the API mature through actual usage before making long-term stability commitments.

## Documentation and User Experience

Excellent developer experience is essential for adoption and productivity.

**Short-term:**
- Enhanced examples covering key use cases
- Complete API documentation with comprehensive docstrings
- Improved error messages and debugging capabilities

**Medium-term:**
- Interactive Jupyter notebook tutorials
- Visual debugging tools for computational graphs
- Performance profiling and optimization guidance

## Community Engagement

As an open-source project, Archimedes thrives on community contributions and feedback. We welcome input on roadmap prioritization and are open to partnerships for sponsored development or proprietary applications.

### Ways to Contribute

- **Feature Development**: Implement capabilities aligned with this roadmap
- **Domain Expertise**: Partner with us on physics modeling in your area of expertise
- **Bug Reports and Fixes**: Help improve stability and reliability
- **Documentation**: Enhance examples, tutorials, and API documentation
- **Use Cases**: Share novel applications and real-world validation projects

### What We're NOT Building

- **Acausal/Equation-based Modeling**: We focus on causal state-space dynamics models rather than the acausal equation-based approach of Modelica or ModelingToolkit.jl
- **Full Simulink Replacement**: We're focused on programmatic workflows, not graphical modeling
- **General-purpose PDE Solver**: We plan to integrate with FEniCS/Firedrake for FEA capabilities
- **Deep Learning Framework**: We plan to integrate with PyTorch for neural network components

### Integration Philosophy

Rather than reimplementing everything, we prioritize integrations with best-in-class tools, for instance:

- **PyTorch**: Deep learning and neural network components
- **Cantera**: Combustion and chemical kinetics modeling
- **SPICE**: Trajectory and ephemeris calculations
- **FEniCS/Firedrake**: Advanced PDE solving capabilities

These integrations allow you to embed specialized functionality into your Archimedes computational graphs while maintaining differentiability and code generation support.

### Partnership Opportunities

- **Industry Validation**: Real-world applications in aerospace, robotics, and process control
- **Educational Collaboration**: Curriculum development and classroom integration
- **Open Source Integration**: Connecting Archimedes with complementary tools and frameworks

---

*This roadmap reflects our current priorities and technical vision. We embrace "building in public" and welcome community feedback to help shape Archimedes' development. For questions, suggestions, or collaboration opportunities, please reach out through our GitHub repository.*