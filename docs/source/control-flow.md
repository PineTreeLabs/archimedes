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

# Structured Control Flow

Control flow is a fundamental aspect of programming, allowing us to make decisions (if/else) and repeat operations (loops). However, in symbolic computation frameworks like Archimedes, standard Python control flow constructs don't work as expected. This page explains why this happens and introduces structured control flow mechanisms that enable these patterns in a symbolic-computation-friendly way.

Note that much of the design of these control flow functions is based on JAX; it may be worth taking a look at [the JAX documentation on control flow](https://docs.jax.dev/en/latest/control-flow.html) for further reading.



```{code-cell} python
:tags: [hide-cell]
import numpy as np

import archimedes as arc
from archimedes import struct
```


## The Problem with Standard Control Flow

When working with symbolic computation, Python's standard control flow constructs often fail because they require evaluating conditions on symbolic values, which cannot be directly converted to boolean values. For example:


```{code-cell} python
@arc.compile
def f(x):
    if x > 0:  # This doesn't work! x is symbolic here
        return np.sin(x)
    else:
        return np.cos(x)


x = 1.0
f(x), np.sin(x)  # Incorrect!
```

This fails because `x > 0` produces a symbolic expression representing the condition, not a concrete `True` or `False` value that Python's `if` statement requires.

Similarly, loops with symbolic bounds or termination conditions don't work:

```{code-cell} python
@arc.compile
def f(x):
    y = 0
    for i in range(x):  # Error: x is symbolic, can't be converted to int
        y += i
    return y


try:
    f(5)
except TypeError as e:
    print("Error:", e)
```

## When You Can Use Standard Python Loops

Despite these limitations, standard Python loops can be used provided

1. Loop bounds are static (known at compile time)
2. There's no early termination based on symbolic conditions

For example, this will work:

```{code-cell} python
@arc.compile
def f(x):
    y = 0
    for i in range(len(x)):  # Fixed, static bound
        y += x[i]
    return y


x = np.array([1, 2, 3, 4, 5])
f(x)
```

However, even when loops are structurally valid, using standard Python loops inside compiled functions can lead to large computational graphs, which may impact performance and memory usage.

## Structured Control Flow Mechanisms

Archimedes provides three primary mechanisms for structured control flow:

1. [`np.where`](#numpy.where) - For element-wise conditional operations
2. [`scan`](#archimedes.scan) - For iterative computations (similar to functional fold/reduce)
3. [`switch`](#archimedes.switch) - For selecting between multiple computational branches
4. [`vmap`](#archimedes.vmap) - For vectorizing operations across batch dimensions

Let's explore each of these in detail.


### Iterative Computation with [`scan`](#archimedes.scan)

For loops and iterative algorithms, [`scan`](#archimedes.scan) provides a functional way to express loops that are compatible with symbolic computation:


```{code-cell} python
# Define a function for a single iteration
@arc.compile
def iteration_step(carry, x):
    new_carry = carry + x
    return new_carry, new_carry  # Return both state and output


# Apply this function repeatedly
xs = np.array([1, 2, 3, 4, 5])
final_state, ys = arc.scan(iteration_step, 0, xs)

print(final_state)  # 15 (sum of all values)
print(ys)  # [1, 3, 6, 10, 15] (running sum)
```

[`scan`](#archimedes.scan) takes a function with the signature `f(carry, x) -> (new_carry, y)`, applies it to each element of `xs` (or for a specified number of iterations), and returns the final state and all intermediate outputs `ys`.

This is useful for constructing efficient computational graphs when there is a loop with many iterations; the [`scan`](#archimedes.scan) operation condenses all of these to a single node in the computational graph, compared to one node per loop iteration.


### Conditional Logic with `np.where`

The simplest way to implement conditional logic is using NumPy's [`where`](#numpy.where) function, which works with symbolic values:


```{code-cell} python
@arc.compile
def f(x):
    return np.where(x > 0, np.sin(x), np.cos(x))


print(f(1.0), np.sin(1.0))
print(f(-1.0), np.cos(-1.0))
```

This approach works for simple conditionals but becomes unwieldy for complex branching logic or when the branches involve substantial computation.

### Branch Selection with [`switch`](#archimedes.switch)

For more complex conditional branching, where different functions need to be applied based on an index value, [`switch`](#archimedes.switch) provides a clean solution:

```{code-cell} python
@arc.compile
def apply_operation(x, op_index):
    return arc.switch(
        op_index,
        (
            lambda x: x**2,  # Branch 0
            lambda x: np.sin(x),  # Branch 1
            lambda x: -x,
        ),  # Branch 2
        x,
    )


# Call with different branch indices
result0 = apply_operation(2.0, 0)  # 4.0 (square)
result1 = apply_operation(2.0, 1)  # ~0.91 (sine)
result2 = apply_operation(2.0, 2)  # -2.0 (negate)

print(result0, result1, result2)
```

[`switch`](#archimedes.switch) evaluates all branches during compilation to ensure they return compatible outputs, but at runtime, only the selected branch executes (i.e. evaluation is "short-circuiting").

### Vectorization with [`vmap`](#archimedes.vmap)

For applying the same operation to multiple inputs in parallel, [`vmap`](#archimedes.vmap) transforms a function that works on single elements into one that works on batches:

```{code-cell} python
def dot(a, b):
    return np.dot(a, b)


# Vectorize to compute multiple dot products at once
batched_dot = arc.vmap(dot)

# Input: batch of vectors (3 vectors of length 2)
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([[7, 8], [9, 10], [11, 12]])

# Output: batch of scalars (3 dot products)
print(batched_dot(x, y))  # [23, 67, 127]
```

## Common Control Flow Patterns

Now let's look at how to implement common control flow patterns using these mechanisms.

### Implementing if/else logic

As mentioned above, for simple conditionals it is easiest to use [`np.where`](#numpy.where).  However, when the branches are more complex, an `if_else` function can be constructed with [`switch`](#archimedes.switch):

```{code-cell} python
def true_branch(x, y):
    return x + y


def false_branch(x, y):
    return x - y


@arc.compile
def f(condition, x, y):
    # Convert boolean condition to 0/1 index
    return arc.switch(condition, (false_branch, true_branch), x, y)


print(f(True, 2, 3))  # Returns 5 (true branch)
print(f(False, 2, 3))  # Returns -1 (false branch)
```

### Creating a Bounded While Loop

While [`scan`](#archimedes.scan) typically iterates for a fixed number of steps, you can implement a bounded while loop by carrying a condition flag and using early-return values:

```{code-cell} python
@arc.compile(static_argnames=("loop_func", "max_iterations"))
def bounded_while(loop_func, init_state, max_iterations=100):
    def body(state, i):
        # Unpack state: (value, done)
        x, done = state

        # Compute new value if not done
        new_x = np.where(done, x, loop_func(x))

        # Check termination condition (with a maximum iteration bound)
        done = np.where(done + (np.abs(new_x - x) < 1e-6), 1, 0)

        # Return updated state and the current value
        return (new_x, done), new_x

    # Initialize with not-done flag
    init_full_state = (init_state, False)

    # Run the scan for the maximum number of iterations
    final_state, values = arc.scan(body, init_full_state, length=max_iterations)

    # Return the final converged value
    return final_state[0]
```

### Vectorizing Tree Operations

When working with structured data types, [`vmap`](#archimedes.vmap) is a particularly useful transformation:


```{code-cell} python
@struct
class Particle:
    pos: np.ndarray
    vel: np.ndarray


def update(particle, dt):
    new_pos = particle.pos + dt * particle.vel
    return Particle(pos=new_pos, vel=particle.vel)


# Create a batch of particles
positions = np.random.randn(100, 3)  # 100 particles in 3D space
velocities = np.random.randn(100, 3)
particles = Particle(pos=positions, vel=velocities)

# Update all particles at once
batch_update = arc.vmap(update, in_axes=(0, None))
new_particles = batch_update(particles, 0.1)
```

[`vmap`](#archimedes.vmap) can also be used to batch-unravel flat arrays, for example the results of a simulation:


```{code-cell} python
x0 = Particle(pos=np.array([0.0, 0.0]), vel=np.array([1.0, 1.0]))

# Flatten the single particle
x0_flat, unravel = arc.tree.ravel(x0)

# Dummy data in the shape of a simulation result
xs_flat = np.random.randn(4, 100)  # 4 timesteps, 100 particles

# We cannot directly unravel the data
try:
    x0_unraveled = unravel(xs_flat)
except ValueError as e:
    print("Error:", e)

# Instead use vmap:
xs = arc.vmap(unravel, in_axes=1, out_axes=1)(xs_flat)
print(xs.pos.shape)  # (2, 100)
```
