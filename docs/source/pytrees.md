# Working with PyTrees

PyTrees in Archimedes provide a way to work with structured data in numerical algorithms that typically expect flat vectors. This page explains what PyTrees are, how to use them in your code, and how to create custom structures that work seamlessly with Archimedes functions.

The Archimedes concept of a PyTree is borrowed from [JAX](https://docs.jax.dev/en/latest/pytrees.html), while the `pytree_node` decorator produces composable classes that work similarly to [PyTorch Modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
You may also want to scan these documentation pages to get ideas about what these structures are and how they are used.

## What is a PyTree?

A PyTree is any nested structure of "containers" (dictionaries, lists, tuples, NamedTuples) and "leaves" (arrays or scalars). Archimedes automatically recognizes built-in Python containers as PyTree nodes, while arrays and scalars are treated as leaves.

Examples of valid PyTrees:

```python
# A list of scalars
[1.0, 2.0, 3.0]  

# A dictionary of arrays
{"position": np.array([0.0, 1.0, 2.0]), "velocity": np.array([3.0, 4.0, 5.0])}  

# A nested structure with multiple container types
(np.array([1.0, 2.0]), [3.0, 4.0], {"value": 5.0})
```

## Flattening and Unflattening

The primary operations with PyTrees are:
- **Flattening**: Converting a structured PyTree into a flat vector
- **Unflattening**: Restoring a flat vector to its original structure

```python
import archimedes as arc
import numpy as np

# Create a structured state
state = {"pos": np.array([0.0, 1.0, 2.0]), "vel": np.array([3.0, 4.0, 5.0])}

# Flatten to a vector
flat_state, unravel = arc.tree.ravel(state)
print(flat_state)  # array([0., 1., 2., 3., 4., 5.])

# Restore the original structure
restored_state = unravel(flat_state)
print(restored_state)  # {'pos': array([0., 1., 2.]), 'vel': array([3., 4., 5.])}
```

This pattern is essential when working with:
- ODE solvers that require state vectors
- Optimization algorithms operating on parameter vectors
- Root-finding methods that expect flat systems

## Custom PyTree Types

For more complex models, create custom classes that work as PyTrees using the `struct.pytree_node` decorator:

```python
from archimedes.tree import struct

@struct.pytree_node
class VehicleState:
    position: np.ndarray      # [x, y, z]
    velocity: np.ndarray      # [vx, vy, vz]
    attitude: np.ndarray      # quaternion [qx, qy, qz, qw]
    angular_velocity: np.ndarray  # [wx, wy, wz]

# Create an instance
state = VehicleState(
    position=np.zeros(3),
    velocity=np.zeros(3),
    attitude=np.array([0, 0, 0, 1]),
    angular_velocity=np.zeros(3)
)

# Flatten to a vector
flat_state, unravel = arc.tree.ravel(state)

# Use in symbolic functions
@arc.compile
def dynamics(state, control, dt=0.1):
    # Access fields naturally
    new_position = state.position + dt * state.velocity
    # ...other calculations...
    return VehicleState(
        position=new_position,
        # ...other updated fields...
    )
```

The `struct.pytree_node` decorator automatically registers your class with Archimedes' PyTree system, combining the benefits of Python [dataclasses](https://docs.python.org/3/library/dataclasses.html) with PyTree functionality.

### Custom nodes: advanced usage

Since the `pytree_node` decorator converts your class into a standard (frozen) Python dataclass, many typical dataclass considerations carry over directly to custom nodes.
For instance, you should typically avoid implementing `__init__` yourself (as this is constructed automatically by the dataclass), but you can implement `__post_init__` for custom initializations.

In addition, you can apply the usual `field` to any field defined for the node.
The `struct` module provides its own wrapper of `field`, which extends it with the ability to label a field as `static`.
Among other things, this means that it should not be included when translating to/from a flat vector.

These custom nodes are otherwise normal Python classes, so you can define methods on them as usual.

Here is an expanded example with some advanced features:

```python
import numpy as np
import archimedes as arc
from archimedes import struct

@struct.pytree_node
class Rocket:
    # Dynamic variables (included in flattening)
    h: float  # height in meters
    v: float  # velocity in m/s
    m: float  # Current mass in kg
    
    # Static parameters (excluded from flattening)
    thrust: float = struct.field(static=True, default=10000.0)  # Thrust in Newtons
    isp: float = struct.field(static=True, default=300.0)       # Specific impulse in seconds
    
    def __post_init__(self):
        # Validate inputs
        if self.m <= 0:
            raise ValueError("Mass must be positive")

# Create a rocket state
rocket = Rocket(
    h=0.0,
    v=0.0,
    m=1000.0,
    thrust=15000.0,  # Override the default
)

print(rocket)  # Rocket(h=0.0, v=0.0, m=1000.0, thrust=15000.0, isp=300.0)

# Flatten to vector - note that static fields are excluded
flat_state, unravel = arc.tree.ravel(rocket)
print(f"Flat state shape: {flat_state.shape}")  # (3,) for height + velocity + mass

# Modify the flat state
flat_state[0] += 10  # Increase height by 10 meters

# Unravel back to object - static fields are restored
new_rocket = unravel(flat_state)
print(new_rocket)  # Rocket(h=10.0, v=0.0, m=1000.0, thrust=15000.0, isp=300.0)
```

You can also nest custom PyTree nodes within each other and define special methods like `__call__`, giving you the ability to create modular and reusable model components.
For example, if we wanted to simulate a rendezvous between our rocket and the ISS, we could create another PyTree node `Satellite` and the combined state of our system could be defined by a `NamedTuple`, making the entire composite state a valid PyTree:

```python
from typing import NamedTuple

class Satellite(NamedTuple):
    pos: np.ndarray
    vel: np.ndarray

class RendezvousState(NamedTuple):
    rocket: Rocket
    satellite: Satellite

satellite = Satellite(pos=np.zeros(3), vel=np.ones(3))
state = RendezvousState(rocket, satellite)

flat_state, unravel = arc.tree.ravel(state)
print(flat_state.shape)  # (9,): three from rocket and six from satellite
```

## Example: Pendulum Simulation Using PyTrees

Here's how PyTrees simplify ODE solving with structured data:

```python
import numpy as np
import archimedes as arc

# Define a custom PyTree node with dynamics method
@arc.struct.pytree_node
class PendulumState:
    theta: float      # angle
    omega: float      # angular velocity

    @classmethod
    def dynamics(cls, t, state):
        g, L = 9.81, 1.0

        # Calculate derivatives
        theta_t = state.omega
        omega_t = -(g/L) * np.sin(state.theta)

        # Return in the same structure
        return cls(theta=theta_t, omega=omega_t)

# Initial state and simulation parameters
initial_state = PendulumState(theta=np.pi/4, omega=0.0)
t_span = (0.0, 10.0)
t_eval = np.linspace(*t_span, 100)

# Convert to flat vector for solver
x0, unravel = arc.tree.ravel(initial_state)

# Create flat dynamics wrapper
@arc.compile
def flat_dynamics(t, x):
    state = unravel(x)  # Unflatten to 
    derivatives = state.dynamics(t, state)
    dx, _ = arc.tree.ravel(derivatives)
    return dx  # Return the flattened state

# Solve the ODE
solution = arc.odeint(flat_dynamics, t_span=t_span, x0=x0, t_eval=t_eval)

# Convert results back to structured form and unpack
states = [unravel(x) for x in solution.T]
theta = np.array([state.theta for state in states])
omega = np.array([state.omega for state in states])
```

## Current Limitations

The following limitations will be resolved with further development:

- **Postprocessing**: The `unravel` functions assume that their arguments have the same shape as the original PyTree, leading to somewhat complicated unpacking operations, particularly for ODE solutions like the previous example.
- **No automatic conversion**: Functions like `odeint` and `minimize` don't automatically convert all their arguments to PyTrees, meaning you have to manually construct wrapper functions like `flat_dynamics` above.

**If you encounter what looks like a bug with PyTrees or another limitation, please file an issue!**

## Best Practices and Tips

- **Structure Consistency**: When unraveling, the flat array length must match the original PyTree structure
- **Immutability**: Treat PyTrees as immutable, creating new instances rather than modifying in place
- **Method Support**: Custom PyTree classes can include methods for operations on your data
- **Performance**: PyTree flattening/unflattening does reshaping at the tracing stage, meaning that it has minimal runtime overhead compared to typical numerical operations
- **Type Support**: PyTrees work with both NumPy arrays and Archimedes symbolic arrays

For more advanced PyTree operations, explore the `archimedes.tree` module.