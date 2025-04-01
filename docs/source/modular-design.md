# Hierarchical Design Patterns

This page covers best practices and design patterns for creating composable dynamic systems using Archimedes.
By leveraging the [`pytree_node`](#archimedes.tree.struct.pytree_node) decorator, you can create modular components that can be combined into complex hierarchical models while maintaining clean, organized code.
However, the recommendations in this guide are strictly suggestions; you can design your models and workflows however you wish.

## Core Concepts

### PyTree Nodes for Structured States

Dynamical systems often have state variables that benefit from logical grouping.
Using tree-structured representations allows you to:

1. Group related state variables together
2. Create nested hierarchies that mirror the physical structure of your system
3. Maintain clean interfaces between subsystems
4. Flatten and unflatten states automatically for ODE solvers

### Design Patterns

Some recommended patterns for building modular dynamics components in Archimedes are:

1. **Modular Components**: Create a [`pytree_node`](#archimedes.tree.struct.pytree_node) for each logical system component
2. **Hierarchical Parameters**: Add model parameters as fields in the PyTree nodes
3. **Nested State Classes**: Define a `State` class inside each model component
4. **Dynamics Methods**: Implement `dynamics(self, t, state)` methods that return state derivatives
5. **Compositional Models**: Build larger models by combining smaller components

## Basic Component Pattern

Here's a basic example of using these patterns to creating a modular dynamical system component:

```python
import numpy as np
import matplotlib.pyplot as plt
import archimedes as arc
from archimedes.tree import struct

@struct.pytree_node
class Oscillator:
    """A basic mass-spring-damper component."""
    
    # Define model parameters as fields in the PyTree node
    m: float  # Mass
    k: float  # Spring constant
    b: float  # Damping constant

    # Define a nested State class as another PyTree node
    @struct.pytree_node
    class State:
        """State variables for the mass-spring-damper system."""
        x: np.ndarray
        v: np.ndarray
    
    def dynamics(self, t, state: State, f_ext=0.0):
        """Compute the time derivatives of the state variables."""

        # Compute derivatives
        f_net = f_ext - self.k * state.x - self.b * state.v

        # Return state derivatives in the same structure
        return self.State(
            x=state.v,
            v=f_net / self.m,
        )

system = Oscillator(m=1.0, k=1.0, b=0.1)
x0 = system.State(x=1.0, v=0.0)  # Oscillator.State(x=1.0, v=0.0)
```

For such a simple system, the advantages to this design are relatively limited, but because these nodes can be nested within each other, it can be a useful way to organize states, parameters, and functions associated with complex models.

## Working with PyTree models

Many functions like ODE solvers expect to work with flat vectors.
PyTree utilities in Archimedes make conversion to and from flat vectors easy.
For example, we can "ravel" a PyTree-structured state to a vector and "unravel" back to the original state:

```python
x0_flat, unravel = arc.tree.ravel(x0)
print(x0_flat)  # [1. 0.]
print(unravel(x0_flat))  # MassSpringDamper.State(x=array(1.), v=array(0.))
```

The `unravel` function created by `tree.ravel` is specific to the original PyTree argument, so it can be used within ODE functions, for example:

```python
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

Since the model itself is also a PyTree, we can also apply `ravel` directly to it, giving us a flat vector of the parameters defined as fields:

```python
p_flat, unravel_system = arc.tree.ravel(system)
print(p_flat)  # [1.  1.  0.1]
```

This is useful for applications in optimization and parameter estimation.

## Complete Example: Coupled Oscillators

Larger systems can be built by composing multiple components together.
Let's build a system of coupled oscillators to demonstrate these patterns.

```python
@struct.pytree_node
class CoupledOscillators:
    """A system of two coupled oscillators."""

    osc1: Oscillator
    osc2: Oscillator
    coupling_constant: float

    @struct.pytree_node
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
        f_ext = self.coupling_constant * (x2 - x1)
    
        return self.State(
            osc1=self.osc1.dynamics(t, state.osc1, f_ext),
            osc2=self.osc2.dynamics(t, state.osc2, -f_ext)
        )

# Create a coupled oscillator system
system = CoupledOscillators(
    osc1=Oscillator(m=1.0, k=4.0, b=0.1),
    osc2=Oscillator(m=1.5, k=2.0, b=0.2),
    coupling_constant=0.5
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
solution_flat = arc.odeint(
    ode_rhs,
    t_span=t_span,
    x0=x0_flat, 
    t_eval=t_eval,
    args=(system,),
)

# Postprocessing
x1 = np.zeros(len(t_eval))
x2 = np.zeros(len(t_eval))
for i in range(len(t_eval)):
    state = state_unravel(solution_flat[:, i])
    x1[i] = state.osc1.x
    x2[i] = state.osc2.x

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_eval, x1, label='Oscillator 1')
plt.plot(t_eval, x2, label='Oscillator 2')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Coupled Oscillators')
plt.legend()
plt.grid(True)
plt.show()
```

:::{note}
Planned development may simplify some of this workflow.
For example, pervasive support of PyTrees in functions like `odeint` will mean that the `ravel` and `unravel` operations will happen automatically and there will be no need to write wrapper functions like `ode_rhs` above or do the "postprocessing" step of manually unraveling the flat ODE solution.
:::


## Summary

The recommended approach to building hierarchical and modular dynamical systems in Archimedes follows these key patterns:

1. Use `@struct.pytree_node` to define structured component classes
2. Create nested `State` classes to organize state variables
3. Implement `dynamics` methods that compute state derivatives
4. Compose larger systems from smaller components
5. Add helper methods to simplify simulation and analysis

Other best practices include:

1. **Consistent Interfaces**: Keep the `dynamics(self, t, state, *args)` method signature consistent across all components
2. **Immutable States**: Always return new state objects instead of modifying existing ones
3. **Physical Units**: Document physical units in comments or docstrings
4. **Input Validation**: Add validation in constructors to catch errors early
5. **Meaningful Names**: Use descriptive names that reflect physical components, or consistent pseudo-mathematical notation like the [monogram](https://drake.mit.edu/doxygen_cxx/group__multibody__notation__basics.html) convention
6. **Domain Decomposition**: Decompose complex systems into logical components (mechanical, electrical, etc.)
7. **Structured Parameters**: Define physical parameters as fields in the PyTree nodes, and use the `struct.field(static=True)` annotation to mark configuration variables.


These patterns enable clean, organized, and reusable model components while leveraging Archimedes' PyTree functionality to handle the conversion between structured and flat representations needed by ODE solvers.