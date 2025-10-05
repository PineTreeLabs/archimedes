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

# Hierarchical Design Patterns

Part 1 of this tutorial covers best practices and design patterns for creating composable dynamical systems models, control algorithms, and other modular functionality using Archimedes.
By leveraging the [`struct`](#archimedes.tree.struct) decorator, you can create modular components that can be combined into complex hierarchical models while maintaining clean, organized code.
But while this guide provides some tips and best practices, these are strictly suggestions; you can design your models and workflows however you wish.

## Core Concepts

The basic concepts of structured data types and tree operations are covered in the ["Structured Data Types"](../../trees.md) documentation page.
Here we'll build on these concepts to see how they can be used for intuitive and scalable design patterns.

Dynamical systems often have natural subsystems and state variables that benefit from logical grouping.
Using tree-structured representations allows you to:

1. Group related state variables together
2. Create nested hierarchies that mirror the physical structure of your system
3. Maintain clean interfaces between subsystems
4. Flatten and unflatten states automatically for simulation, optimization, stability analysis, etc.

### Design Patterns

The first step in creating a hierarchical dynamics model is to identify the subsystems, components, etc. that naturally decompose a complex system into its logical building blocks.
For each of these modular dynamics components, you can then implement state-space systems of the form

$$
\begin{align}
\dot{x} &= f(t, x, u, p) \\
y &= h(t, x, u, p),
\end{align}
$$

where $t$ is time, $x$ is the state, $u$ are external (control) inputs, and $p$ are parameters.
Some recommended patterns to implement these components include:

1. **Structured Data Classes**: Decorate each class as a [`struct`](#archimedes.tree.struct)
3. **Nested Struct Definitions**: Define a `State` class inside each model component (and `Input` and `Output` as necessary)
2. **Hierarchical Parameters**: Add model parameters as fields in the struct or as a nested `Parameters` class
4. **Standardized Methods**: Implement `dynamics(self, t, state, ...)` methods that return state derivatives, and/or `output(self, t, state, ...)` methods that return the outputs of a state-space model

A template for a generic dynamics model following these patterns would look like this:

```python
@struct
class Component:
    # Define model parameters here
    ...

    # OR create an inner struct to organize them
    @struct
    class Parameters:
        ...  # Model parameters (p)

    @struct
    class State:
        ...  # Dynamic state (x)

    @struct
    class Input:
        ...  # Inputs to the system (u)

    @struct
    class Output:
        ...  # Measured outputs (y)

    def dynamics(self, t: float, x: State, u: Input, p: Parameters) -> State:
        ...  # Implement ẋ = f(t, x, u, p)

    def output(self, t: float, x: State, u: Input, p: Parameters) -> Output:
        ...  # Implement y = h(t, x, u, p)
```

One important recommendation is to use `abc` metaclasses or `Protocol` classes to clearly define interfaces for components - then you can define multiple variants and easily switch between them without touching the rest of your system model.
This template code itself could be a `Protocol`, for instance.

Then you can create composite models by nesting these components together to organize and abstract the details of complex system models.
The `abc`/`Protocol` approach also lets you define an interface for a component and then implement "multi-fidelity modeling" by creating implementations of varying speed and accuracy.

For example, you might create a sensor component and then have three variants that implement a (1) simplified version with no dynamics, (2) a simple linear system model (e.g. second-order transfer function), and (3) a high-fidelity physics-based model.
Then you can separately calibrate each using [parameter estimation](../sysid/parameter-estimation.md) and easily switch between them depending on the context (e.g. low-fidelity for model-based control or high-fidelity for simulated evaluation).

We'll see more on implementing the `Protocol` concept in [Part 2](hierarchical02.md), covering configuration management for hierarchical models.

## Basic Component Pattern

Here's a basic example of using these patterns to creating a modular dynamical system component:


```{code-cell} python
:tags: [hide-cell]
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import archimedes as arc
from archimedes import struct
```

```{code-cell} python
:tags: [remove-cell]
from pathlib import Path

plot_dir = Path.cwd() / "_plots"
plot_dir.mkdir(exist_ok=True)
```


```{code-cell} python
@struct
class Oscillator:
    """A basic mass-spring-damper component."""

    # Define model parameters as fields in the struct
    m: float  # Mass
    k: float  # Spring constant
    b: float  # Damping constant

    # Define a nested State class as another struct
    @struct
    class State:
        """State variables for the mass-spring-damper system."""

        x: float
        v: float

    def dynamics(self, t, state: State, f_ext: float = 0.0) -> State:
        """Compute the time derivatives of the state variables."""

        # Compute derivatives
        f_net = f_ext - self.k * state.x - self.b * state.v

        # Return state derivatives in the same structure
        return self.State(
            x=state.v,
            v=f_net / self.m,
        )


system = Oscillator(m=1.0, k=1.0, b=0.1)
x0 = system.State(x=1.0, v=0.0)
x0
```

Note that since this particular system is very simple we didn't implement the full machinery shown in the `Component` protocol above - that would be overkill here.
In fact, for such a simple system, the advantages to following this design pattern are relatively limited overall (it would have been easier to just implement a simple ODE right-hand-side function).
But because these nodes can be nested within each other, it can be a useful way to organize states, parameters, and functions associated with more complex models.

## Working with tree-structured data

Many functions like ODE solvers expect to work with flat vectors.
Tree operations in Archimedes make conversion to and from flat vectors easy.
For example, we can "ravel" a tree-structured state to a vector and "unravel" back to the original state:


```{code-cell} python
x0_flat, unravel = arc.tree.ravel(x0)
print(x0_flat)
print(unravel(x0_flat))
```

The `unravel` function created by [`tree.ravel`](#archimedes.tree.ravel) is specific to the original argument data type, so it can be used within ODE functions, for example:


```{code-cell} python
@arc.compile
def ode_rhs(t, state_flat, system):
    # Unflatten the state vector to our structured state
    state = unravel(state_flat)

    # Compute state derivatives using model dynamics
    state_deriv = system.dynamics(t, state)

    # Flatten derivatives back to a vector
    state_deriv_flat, _ = arc.tree.ravel(state_deriv)

    return state_deriv_flat


# Solve the ODE
t_span = (0.0, 10.0)
t_eval = np.linspace(*t_span, 100)
solution_flat = arc.odeint(
    ode_rhs,
    t_span=t_span,
    x0=x0_flat,
    t_eval=t_eval,
    args=(system,),
)
```

Since the model itself is also a [`struct`](#archimedes.tree.struct), we can also apply [`ravel`](#archimedes.tree.ravel) directly to it, giving us a flat vector of the parameters defined as fields:


```{code-cell} python
p_flat, unravel_system = arc.tree.ravel(system)
print(p_flat)  # [1.  1.  0.1]
```

This is useful for applications in optimization and parameter estimation.

Another common pattern is to define yet another `struct` for the parameters, rather than having them as fields in the system model.
This comes down to individual preference and whatever works best for the specific application.

## Composite System: Coupled Oscillators

Larger systems can be built by composing multiple components together:


```{code-cell} python
@struct
class CoupledOscillators:
    """A system of two coupled oscillators."""

    osc1: Oscillator
    osc2: Oscillator
    coupling: float

    @struct
    class State:
        """Combined state of both oscillators."""

        osc1: Oscillator.State
        osc2: Oscillator.State

    def dynamics(self, t, state):
        """Compute dynamics of the coupled system."""
        # Extract states
        x1 = state.osc1.x
        x2 = state.osc2.x

        # Compute equal and opposite coupling force
        f_ext = self.coupling * (x2 - x1)

        return self.State(
            osc1=self.osc1.dynamics(t, state.osc1, f_ext),
            osc2=self.osc2.dynamics(t, state.osc2, -f_ext),
        )


# Create a coupled oscillator system
system = CoupledOscillators(
    osc1=Oscillator(m=1.0, k=4.0, b=0.1),
    osc2=Oscillator(m=1.5, k=2.0, b=0.2),
    coupling=0.5,
)

# Create initial state
x0 = system.State(
    osc1=Oscillator.State(x=1.0, v=0.0),
    osc2=Oscillator.State(x=-0.5, v=0.0),
)

# Flatten the state for ODE solver
x0_flat, state_unravel = arc.tree.ravel(x0)


# ODE function that works with flat arrays
@arc.compile
def ode_rhs(t, state_flat, system):
    state = state_unravel(state_flat)
    state_deriv = system.dynamics(t, state)
    state_deriv_flat, _ = arc.tree.ravel(state_deriv)
    return state_deriv_flat


# Solve the system
t_span = (0.0, 20.0)
t_eval = np.linspace(*t_span, 200)
sol_flat = arc.odeint(
    ode_rhs,
    t_span=t_span,
    x0=x0_flat,
    t_eval=t_eval,
    args=(system,),
)

# Postprocessing: create a "vectorized map" of the unravel
# function to map back to the original tree-structured state
sol = arc.vmap(state_unravel, in_axes=1)(sol_flat)
```

```{code-cell} python
:tags: [hide-cell, remove-output]
# Plot the results

plt.figure(figsize=(7, 2))
plt.plot(t_eval, sol.osc1.x, label="Oscillator 1")
plt.plot(t_eval, sol.osc2.x, label="Oscillator 2")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Coupled Oscillators")
plt.legend()
plt.grid(True)
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.theme.set_theme(theme)
    plt.figure(figsize=(7, 2))
    plt.plot(t_eval, sol.osc1.x, label="Oscillator 1")
    plt.plot(t_eval, sol.osc2.x, label="Oscillator 2")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Coupled Oscillators")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / f"hierarchical0_0_{theme}.png")
    plt.close()
```

```{image} _plots/hierarchical0_0_light.png
:class: only-light
```

```{image} _plots/hierarchical0_0_dark.png
:class: only-dark
```


## Summary

The recommended approach to building hierarchical and modular dynamical systems in Archimedes follows these key patterns:

1. Use [`@struct`](#archimedes.tree.struct) to define structured component classes
2. Create nested `State` classes to organize state variables (and maybe `Input`, `Output`, `Parameters`)
3. Implement `dynamics` methods that compute state derivatives (and maybe `output`)
4. Compose larger systems from smaller components
5. Add helper methods to simplify simulation and analysis

Other best practices include:

1. **Immutable States**: Always return new state objects instead of modifying existing ones
2. **Physical Units**: Document physical units in comments or docstrings
3. **Meaningful Names**: Use descriptive names that reflect physical components, or consistent pseudo-mathematical notation like the [monogram](https://drake.mit.edu/doxygen_cxx/group__multibody__notation__basics.html) convention
4. **Domain Decomposition**: Decompose complex systems into logical components (mechanical, electrical, etc.)
5. **Physical Parameters**: Define physical parameters as fields in the structs or as inner classes
6. **Configuration Parameters**: Define as fields in the struct and use the [`field(static=True)`](#archimedes.tree.field) annotation.

These patterns enable clean, organized, and reusable model components by leveraging Archimedes' tree operations to handle the conversion between structured and flat representations needed by ODE solvers.
