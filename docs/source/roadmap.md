# Development Roadmap

This page outlines the development priorities and future direction of Archimedes. As an open-source project, we welcome community input and contributions across any of these areas. In particular, feel free to create or contribute to a thread with the "RFC" (request for comments) tag on the [GitHub Discussions](https://github.com/PineTreeLabs/archimedes/discussions) page with your thoughts on the roadmap.

## ⚠️ WARNING: PRE-RELEASE! ⚠️

If you are seeing this, it means that this project has not "officially" been released yet and this documentation page is incomplete.
Check back soon for more details!

<!-- 
The Archimedes roadmap is organized into three primary development buckets:

1. **Hybrid Simulations**: Enhancing capabilities for mixed continuous/discrete systems (zero-crossing events and multirate control)
2. **Path to Hardware**: Streamlining the transition from simulation to hardware deployment (code generation, HIL testing)
3. **Physics Modeling**: Expanding pre-built components for physical systems modeling (esp. aerospace and robotics)

We will be adding more comprehensive design documents to each of these, but for now here are some basic thoughts about where we're going.

## 1. Hybrid Simulations

Hybrid systems that combine continuous dynamics with discrete events are essential for realistic modeling of many engineering systems.

The two biggest practical challenges in modeling hybrid systems are dealing with unscheduled "triggered" or "zero-crossing" events (e.g. a rigid body collision or emergency shutoff switch) and correctly handling multirate/multitasking control systems.

### Next steps

- **Zero-crossing Event Detection**: Implement support for event handling in ODE/DAE solves using SUNDIALS root-finding interface
- **Event-triggered Actions**: Support for discontinuous changes to system state upon event detection
- **Basic Task Framework**: Initial implementation of "tasks" (periodic events) and simple scheduling


### Medium-term

- **Multirate Control Framework**: 
  - Task group abstractions for organizing computations
  - Dependency resolution for intra-group communication
  - Priority-based scheduling within rate groups
  - Deterministic timing control

### Long-term

- **Distributed Control Systems**: 
  - Inter-group communication with explicit port-based interfaces
  - Support for multiple communication protocols (shared memory, UDP, CAN)
  - Realistic communication models with appropriate delays and constraints
- **Co-simulation Capabilities**:
  - Integration with external simulators via FMI/FMU standard
  - Time synchronization between continuous and discrete components

## 2. Path to Hardware

A key goal of Archimedes is enabling a seamless transition from simulation to real hardware deployment, particularly for embedded systems.
We currently support basic C code generation from symbolically compiled functions (thanks to CasADi) and have a simple templating system for generating platform-specific "driver" code with a few simple example templates.  This lets you customize the templates for your own hardware and applications.

**Please note**: if you have difficulties getting any of these templates to work, or you'd like to see a different platform supported, please bring it up in a Discussion!

The immediate roadmap for hardware support will be focused on creating a HIL testing framework that can be easily managed from Python (but using C/C++ behind the scenes for precise timing).
This isn't intended to be a replacement for highly optimized, "hard real-time" platforms like SpeedGoat and dSPACE, but as a low-cost solution for rapid development and early-stage testing.

Longer-term goals include:

- Integration with static analysis tools for memory analysis and execution time
- Expanded support for 


- **Expanded Target Support**:
  - Arduino platform integration with driver code generation
  - STM32 microcontroller platform support
  - Platform-specific optimizations and constraints

### Long-term Goals

- **Hardware-in-the-Loop (HIL) Testing**:
  - Framework for connecting simulations to physical hardware
  - Automated test generation for validating deployed code
  - Real-time performance profiling and validation
- **Advanced Deployments**:
  - Multi-processor system support
  - Real-time operating system (RTOS) integration
  - Automotive-grade and aerospace-grade certification support

## 3. Physics Modeling

Building a library of reusable, high-performance physics models is essential for accelerating development of complex engineering applications.

### Short-term Goals

- **Foundational Mechanics**:
  - Rigid body dynamics with 6DOF
  - Collision detection and response
  - Spring-mass-damper systems
- **Basic Aerodynamics**:
  - Lift and drag models
  - Simple stability derivatives
  - Atmospheric models

### Medium-term Goals

- **Multibody Systems**:
  - Joint constraints (revolute, prismatic, spherical)
  - Contact dynamics
  - Featherstone algorithm implementation for O(n) dynamics
- **Advanced Fluids and Thermal**:
  - Heat transfer models
  - Pipe network flow
  - Basic CFD integration

### Long-term Goals

- **Domain-specific Libraries**:
  - Comprehensive aerospace vehicle modeling
  - Robotics/manipulator dynamics and kinematics
  - Power systems and electronics
- **Uncertainty Quantification**:
  - Sensitivity analysis tools
  - Monte Carlo simulation capabilities
  - Polynomial chaos expansions

## Documentation and User Experience

Alongside these core development areas, we're committed to improving documentation and user experience.

### Short-term Goals

- **Enhanced Examples**: Expand the collection of worked examples covering key use cases
- **Comprehensive API Documentation**: Complete docstrings for all public functions
- **Debugging Tools**: Improved error messages and diagnostic capabilities

### Medium-term Goals

- **Interactive Tutorials**: Jupyter notebooks with step-by-step guides
- **Visual Debugging**: Tools for visualizing computational graphs and model structure
- **Performance Profiling**: Tooling to help identify and resolve performance bottlenecks

## Community Engagement

As an open-source project, Archimedes thrives on community contributions and feedback.

### Ways to Contribute

- **Feature Development**: Implement new capabilities aligned with the roadmap
- **Bug Reports and Fixes**: Help improve stability and reliability
- **Documentation**: Enhance examples, tutorials, and API documentation
- **Use Cases**: Share novel applications and use cases of Archimedes

We welcome input on prioritization of these roadmap items. If you're interested in contributing or have suggestions, please reach out through our GitHub repository.
 -->
