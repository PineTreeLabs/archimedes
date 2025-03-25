# What is Archimedes?

Archimedes is an open-source Python framework designed to simplify complex modeling and simulation tasks, with the ultimate goal of enabling practical hardware engineering with Python.

The core functionality of Archimedes is the ability to transform plain NumPy functions into high-performance C++ _computational graphs_. By leveraging CasADi, a symbolic framework for automatic differentiation and numerical optimization designed specifically for optimal control applications, Archimedes allows users to implement complex models using familiar NumPy syntax while gaining significant performance improvements and advanced capabilities.

Archimedes is based on three core concepts:

1. **Symbolic-numeric Computation**
2. **Function transformations**
3. **Tree-Structured Data**

## Symbolic-Numeric Computation

Archimedes provides a seamless interface between symbolic and numeric computation by wrapping CasADi's symbolic engine in a NumPy-compatible array API. This approach, combined with JAX-style composable function transformations, enables a powerful workflow:

1. Write functions using standard NumPy operations
2. Convert these functions into "symbolic functions" with a simple decorator
3. When called, these symbolic functions create an efficient computational graph in compiled C++
4. This computational graph can be used for fast execution, automatic differentiation, and C code generation

Consider this simple example:

```python
import numpy as np
import archimedes as arc

# Write your function using standard NumPy
def f(x, y):
    return x + np.sin(y)

# Convert it into a "symbolic" function
f_sym = arc.sym_function(f)

# Call the symbolic function with standard arrays
z = f_sym(1.0, np.array([2.0, 3.0]))
```

### The Computational Graph

From an abstract point of view, the function `f` defines a "computational graph" encapsulating the operations required to produce the output.
A _computational graph_ is a structured representation of a mathematical function as a directed graph, where nodes represent operations and edges represent data flow between them. For the function `f`, this graph encapsulates all operations required to produce the output from the inputs.

For example, the function `f(x, y) = x + sin(y)` could be represented by a computational graph with:
- Input nodes for `x` and `y`
- An operation node for `sin(y)`
- An addition node connecting `x` and the result of `sin(y)`
- An output node representing the final result

<!-- TODO: Add a figure for the computational graph -->

Typically Python won't actually construct such a graph; it will just march through the statements and evaluate them with no knowledge of what happened before or after.
This is the "interpreted" execution model.

In a compiled language, on the other hand, the compiler constructs internal representations of the code (such as abstract syntax trees and control flow graphs) and performs optimizations on these structures to improve performance and memory usage.
Just-in-time (JIT) compilation-based frameworks like Numba and JAX do something similar on the fly in order to achieve their impressive speedups over vanilla Python.
These compilers do process the code, but they also do not necessarily create an explicit representation of the computational graph.

Archimedes works a little differently.
Instead of using a low-level compiler to construct and process the code, it uses _symbolic arrays_ to explicitly construct the high-level computational graph.
This representation of the code actually lives in CasADi's efficient C++ data structures, meaning that Archimedes doesn't need to compile anything.
Instead we "trace" the function symbolically and can then work with the computational graph, which can be much more efficient than standard interpreted code, and enables advanced functionality like automatic differentiation and code generation.

Obviously, each approach has its advantages.
Interpreted code is easy to write and highly flexible, while compiled code offers hard-to-beat performance.
Archimedes aims to strike a balance between these two, targeting both ease of use and high performance for the types of computations often used in numerical modeling, simulation, and optimization.

### How Symbolic Functions Work

<!-- TODO: Add a figure to represent this graphically -->

When a symbolic function is called with specific arguments, Archimedes performs three key steps:

1. **Symbolic Replacement**: Arguments are replaced with equivalent symbolic variables having the same shape and dtype.  Here, `1.0` is replaced by a symbolic scalar, and `[2.0, 3.0]` is replaced by an array with shape `(2,)`.

2. **Symbolic Evaluation**: NumPy operations are intercepted via the array API, creating symbolic representations of each operation. For example, `np.sin(y)` returns a symbolic representation rather than a concrete value, building up a complete computational graph. This is the "tracing" step described earlier.

3. **Numerical Evaluation**: The symbolic inputs are replaced with their original numerical values and passed to the CasADi computational graph, which executes efficiently in compiled C++ to return the numerical result.

## Function Transformations

The true power of symbolic computation emerges when applying function transformations or using symbolic functions in complex algorithms. Two key advantages are:

1. **Performance optimization** for complex functions that need to be evaluated repeatedly
2. **Function transformation** capabilities that would otherwise require difficult manual implementations

### Automatic Differentiation

Computing gradients for optimization becomes straightforward:

```python
import archimedes as arc

@arc.sym_function
def f(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

df = arc.grad(f)  # Transform the computational graph
df(np.array([1.0, 1.0]))  # Evaluate numerically
```

The `arc.grad(f)` transformation constructs a new symbolic function that computes the exact gradient of the original function. This provides numerically exact derivatives computed in C++, avoiding the slow and potentially unstable finite differencing methods used by default in MATLAB and Python.

### Implicit Functions

Engineering problems frequently involve implicit functions (where a relationship is defined but not directly solvable). Consider an implicit function $0 = f(x, y)$:

```python
import numpy as np
import archimedes as arc

@arc.sym_function
def f(x, y):
    return x ** 2 + x * np.sin(y) - y
```

Rather than manually implementing an iterative solver, Archimedes can transform this into an explicit function $x = F(y)$:

```python
F = arc.implicit(f)
x = F(y=1.0, x0=0.0)  # x0 provides the initial guess
```

Archimedes constructs a computational graph that automatically applies a Newton solver or equivalent method to efficiently solve the equation $f(x, y) = 0$ for any given value of $y$.

### C Code Generation

For embedded applications, the C++ computational graphs can be exported as standalone C code:

```python
def f(x, y):
    return x + np.sin(y)

# Create templates with appropriate shapes and dtypes
x_type = np.empty((), dtype=float)
y_type = np.empty((2,), dtype=float)

arc.codegen(f, "func.c", (x_type, y_type), header=True)
```

While the current code generation relies on CasADi's conventions and formatting, enhanced code generation for embedded systems is on the project roadmap.

**If you are interested in using Archimedes for embedded applications, please let us know!**

## Tree-Structured Data

Modern engineering models often involve complex, nested data structures that go beyond simple arrays.
Archimedes adopts the ["PyTree" concept from JAX](https://docs.jax.dev/en/latest/pytrees.html) to seamlessly work with tree-structured data and blends it with the [composable "Module" design from PyTorch](https://pytorch.org/docs/stable/notes/modules.html) for constructing hierarchical, modular functionality.

### What are PyTrees?

PyTrees are nested structures of containers (lists, tuples, dictionaries) and leaves (arrays or scalars) that can be flattened to a vector and "unflattened" back to their original structure.
They provide a systematic way to:

1. **Organize complex data** - Maintain logical structure in your models
2. **Simplify function interfaces** - Pass structured arguments instead of numerous separate parameters
3. **Enable operations on nested structures** - Apply transformations that work naturally with trees

For example, ODE solvers are typically written to work with functions that accept and return a vector state.
However, for complex systems the "state" may contain many sub-components and it can be time-consuming and error-prone to manually index into a monolithic vector.

```python
# Example of a PyTree representing a robot state
robot_state = {
    'joints': {
        'positions': np.array([0.1, 0.2, 0.3]),
        'velocities': np.array([0.01, 0.02, 0.03])
    },
    'end_effector': {
        'position': np.array([1.0, 2.0, 3.0]),
        'orientation': np.array([0.0, 0.0, 0.0, 1.0])
    }
}
```

As in the example of this "robot state", a PyTree can be made up of built-in Python data structures like dictionaries, lists, and tuples.
Archimedes already knows how to work with these containers, but as we'll see below, you can also easily construct custom PyTrees using a simple dataclass-derived class decorator.

### The PyTree Interface

Archimedes provides utilities in the `tree` module to work efficiently with tree-structured data.
The most common operation is to "ravel", or flatten, a tree to a single vector, and to "unravel" a vector of the same length back to the original tree.

```python
# Flatten a PyTree into a single vector for optimizers, ODE solvers, etc.
flat_state, unravel = arc.tree.ravel(robot_state)
print(flat_state)  # array([0.  , 0.  , 0.  , 1.  , 1.  , 2.  , ...

# Reconstruct the original tree structure
print(unravel(flat_state))  # {'end_effector': {'orientation': array([0., 0., 0., 1.]), ...
```

### Structured Data Classes

If you want to go beyond built-in Python containers, Archimedes provides a `struct` module that combines Python's `dataclass` with automatic PyTree registration:

```python
from archimedes.tree import struct

@struct.pytree_node
class VehicleState:
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    attitude: np.ndarray  # quaternion [qx, qy, qz, qw]
    angular_velocity: np.ndarray  # [wx, wy, wz]

# Create an instance
state = VehicleState(
    position=np.zeros(3),
    velocity=np.zeros(3),
    attitude=np.array([0, 0, 0, 1]),
    angular_velocity=np.zeros(3)
)

# Flatten to a single vector
flat_state, unravel = arc.tree.ravel(state)

# Pass to/from symbolic functions
@arc.sym_function
def euler_dynamics(state, control):
    # Access structured fields naturally
    new_position = state.velocity + dt * state.position
    # ...
    return VehicleState(
        position=new_position,
        # ...
    )
```

In addition, these custom PyTree nodes are regular classes, so you can define methods as usual.
For example, you can create a callable class with some parameters using the special `__call__` method:

```python

@struct.pytree_node
class PredatorPrey:
    a: float
    b: float
    c: float
    d: float

    def __call__(self, t, x):
        return np.array([
            self.a * x[0] - self.b * x[0] * x[1],
            -self.c * x[1] + self.d * x[0] * x[1]
        ], like=x)

model = PredatorPrey(a=1.5, b=1.0, c=1.0, d=3.0)
t0, tf = 0, 100
x0 = np.array([1.0, 1.0])

print(model(t0, x0))  # Call like a function
```

### Why PyTrees Matter

PyTrees are particularly valuable when:

1. **Working with complex dynamical systems** - Maintain logical separation of system components while allowing operations on the whole system
2. **Building hierarchical models** - Compose sub-models into larger systems
3. **Implementing optimization problems** - Package decision variables in meaningful structures
4. **Designing control systems** - Keep controller states organized

## Comparison with Deep Learning Frameworks

While frameworks like JAX and PyTorch offer similar capabilities, they are fundamentally designed for deep learning applications. Archimedes addresses several limitations that make these frameworks less suitable for engineering tasks:

1. **CPU Optimization**: While deep learning frameworks prioritize GPU acceleration, most engineering computation occurs on CPUs. Archimedes is optimized for CPU performance.

2. **C Code Generation**: Deep learning frameworks cannot typically generate standalone C code, which is essential for embedded systems and safety-critical applications.

3. **Compilation Efficiency**: Deep learning frameworks use JIT compilation, which can be prohibitively slow for complex engineering models. Archimedes leverages pre-compiled C++ through CasADi, avoiding lengthy compilation times.

4. **Sparse Differentiation**: Engineering applications often require sparse Jacobians and Hessians for constrained or second-order optimization, whereas deep learning typically focuses on dense first-order gradients. Archimedes provides efficient sparse automatic differentiation through CasADi.

By addressing these engineering-specific requirements, Archimedes bridges the gap between scientific computing in Python and the performance demands of practical hardware engineering.