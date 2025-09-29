# Multirotor Dynamics

Multirotor flight vehicles are a rapidly growing domain of aerospace engineering, boasting applications ranging from photography and package delivery using smaller drones to passenger transport with larger eVTOL concepts.
At the same time, the dynamics of these aircraft are notoriously complex, creating challenges in modeling, design, and control.

The tightly coupled, strongly nonlinear dynamics, unsteady and often extreme aerodynamic conditions, and sensitive dependence on geometry create opportunities for advanced model-based engineering approaches. These include parameter estimation, trajectory optimization, and integrated design optimization.

## Tutorial overview

In this tutorial, we will walk through the process of modeling a multirotor vehicle using Archimedes. We'll start with a simplified approximation of the aerodynamics and progressively build towards a higher-fidelity blade-element model. This approach will showcase how Archimedes allows us to:

- **Implement models using familiar NumPy code**: We'll write our numerical implementation in standard NumPy, then execute it using the powerful CasADi symbolic/numeric C++ framework.
- **Achieve significant performance gains**: Archimedes can reach 5-10x speedups in simulation compared to vanilla NumPy implementations.
- **Leverage automatic differentiation**: This enables efficient optimization and control design.
- **Generate C code**: For deployment to embedded controllers or other performance-critical applications.
- **Develop modular, extensible models**: We'll use modern software design practices to create a flexible modeling framework.

<!-- Goals of the tutorial -->
This tutorial will cover the following topics:

1. [**Fundamentals of multirotor dynamics**](multirotor01): 
   - 6-DOF rigid body dynamics
   - Rotorcraft aerodynamics
   - Blade-element momentum theory

2. [**Implementation in Archimedes**](multirotor02): 
   - Designing a modular multirotor modeling framework
   - Adapting NumPy code to work with Archimedes

3. [**Running the model**](multirotor03):
   - Simulating with a standard SciPy solver
   - Accelerating simulation with Archimedes

4. [**Linear systems analysis**](multirotor04): 
   - Efficient trim point identification
   - Linear stability analysis

5. [**Blade-element momentum theory**](multirotor05): 
   - Extending the initial model with a more accurate rotor dynamics model

## Prerequisites

While this tutorial aims to be largely self-contained, it will reference established texts for detailed presentations of modeling methods, focusing instead on implementation and applications with Archimedes. A basic familiarity with rigid body dynamics and aerodynamics will be helpful. We'll closely follow the notation and methods from these references:

* "Principles of Helicopter Aerodynamics" by Leishman
* "Aircraft Control and Simulation" by Stevens, Lewis, and Johnson
* "Flight Dynamics" by Stengel
* ["Development and application of a medium-fidelity analysis code for multicopter aerodynamics and flight dynamics"](https://dspace.rpi.edu/items/14f1cb03-4c62-4365-a389-c70de7afb442) by Niemiec

Some basic Python knowledge will also be beneficial. Our code makes extensive use of:
- [Dataclasses](https://realpython.com/python-data-classes/)
- [Composition and inheritance](https://realpython.com/inheritance-composition-python/)
- [Callable objects](https://realpython.com/python-callable-instances/)

To run the code, you'll need a Python environment configured with Archimedes and its dependencies (NumPy, SciPy, Matplotlib, and CasADi).
You'll also need the [multirotor.py](https://github.com/jcallaham/archimedes/tree/main/docs/source/notebooks/multirotor/multirotor.py) source code for the models.

## Demo contents

```{toctree}
:maxdepth: 1
multirotor01
multirotor02
multirotor03
multirotor04
multirotor05
multirotor06
   
```
