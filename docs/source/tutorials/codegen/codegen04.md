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

```{code-cell} python
:tags: [hide-cell]
# ruff: noqa: N802, N803, N806, N815, N816
import pprint

import numpy as np
from scipy import signal

import archimedes as arc
from archimedes import struct
```

```{code-cell} python
:tags: [remove-cell]
from utils import cleanup

from archimedes.docs.utils import display_text

cleanup()  # Clean up any previous generated code
```

```{code-cell} python
:tags: [remove-cell]
from pathlib import Path

plot_dir = Path.cwd() / "_plots"
plot_dir.mkdir(exist_ok=True)
```

# Structured data types

So far we have been working with a function that operates with regular arrays:


```{code-cell} python
# Design a simple IIR filter with SciPy
dt = 0.01  # Sampling time [seconds]
Wn = 10  # Cutoff frequency [Hz]
order = 4
b, a = signal.butter(order, Wn, "low", analog=False, fs=1 / dt)


@arc.compile(return_names=("u_hist", "y_hist"))
def iir_filter_flat(u, b, a, u_prev, y_prev):
    # Update input history
    u_prev[1:] = u_prev[:-1]
    u_prev[0] = u

    # Compute output using the direct II transposed structure
    y = (np.dot(b, u_prev) - np.dot(a[1:], y_prev[: len(a) - 1])) / a[0]

    # Update output history
    y_prev[1:] = y_prev[:-1]
    y_prev[0] = y

    return u_prev, y_prev
```

However, this approach of passing each individual input, output, or state component as an arg doesn't scale well to more complex algorithms.
Additionally, for stateful functions it requires manual state management:

```c
// Copy output arrays back to inputs
for (int j = 0; j < n; j++) {
    arg.u_prev[j] = res.u_hist[j];
    arg.y_prev[j] = res.y_hist[j];
}
arg.u_prev[n] = res.u_hist[n];
```

This single filter uses five arguments; even if we just added two extra IIR filters, we might need as many as 15 arguments to keep track of the filter histories, coefficients, etc.
This quickly becomes difficult to understand and maintain.

One solution is to use hierarchical data structures with well-defined fields and sizes to organize the arguments and returns.
We call these "structured data types" or "trees", using terminology borrowed from [JAX](https://docs.jax.dev/en/latest/pytrees.html).
These let you organize state and parameters in dataclass-like containers which can be mapped directly to a C `struct`.
They can even be nested inside of one another to arbitrary depth.
See the overview of [structured data types](../../trees.md) or the documentation for the [`@struct`](#archimedes.tree.struct) decorator for more details.

## Multiple filters

For example, let's say we have an algorithm that is composed of three discrete transfer functions, implemented as IIR filters:

```
                  |--> G --> y_g
u --> F --> y_f --|
                  |--> H --> y_h
```

Instead of maintaining six arrays (one each for `u` and `y` histories, across three filters), plus inputs and outputs, we can organize these into a logical hierarchical structure.

First, let's rewrite the basic IIR filter to work with a structured state:


```{code-cell} python
@struct
class FilterState:
    u_prev: np.ndarray
    y_prev: np.ndarray


def iir_filter(
    x: FilterState, u: float, b: np.ndarray, a: np.ndarray
) -> tuple[FilterState, float]:
    u_prev, y_prev = x.u_prev, x.y_prev

    # Update input history
    u_prev[1:] = u_prev[:-1]
    u_prev[0] = u

    # Compute output using the direct II transposed structure
    y = (np.dot(b, u_prev) - np.dot(a[1:], y_prev[: len(a) - 1])) / a[0]

    # Update output history
    y_prev[1:] = y_prev[:-1]
    y_prev[0] = y

    return FilterState(u_prev, y_prev), y
```

Next we can combine three such filters into our "compound" filter with transfer functions `F`, `G`, and `H`:


```{code-cell} python
@struct
class CompoundState:
    x_f: FilterState
    x_g: FilterState
    x_h: FilterState


@struct
class CompoundOutput:
    y_f: float
    y_g: float
    y_h: float


@arc.compile(return_names=("state_new", "y"))
def compound_filter(
    state: CompoundState, u: float
) -> tuple[CompoundState, CompoundOutput]:
    # For simplicity, just re-use the coefficients from the low-pass filter
    # we've already designed
    x_f, y_f = iir_filter(state.x_f, u, b, a)
    x_g, y_g = iir_filter(state.x_g, y_f, b, a)
    x_h, y_h = iir_filter(state.x_h, y_f, b, a)

    return CompoundState(x_f, x_g, x_h), CompoundOutput(y_f, y_g, y_h)


x0 = CompoundState(
    x_f=FilterState(u_prev=np.zeros(len(b)), y_prev=np.zeros(len(a) - 1)),
    x_g=FilterState(u_prev=np.zeros(len(b)), y_prev=np.zeros(len(a) - 1)),
    x_h=FilterState(u_prev=np.zeros(len(b)), y_prev=np.zeros(len(a) - 1)),
)
u0 = 1.0

args = (x0, u0)
x, y = compound_filter(*args)

pprint.pprint(y)
```

If you've read the documentation on [structured data types](../../trees.md) and/or worked with them a bit, you can probably already see where this is going - a class decorated with `@struct` behaves a lot like a C `struct`!
This makes it straightforward to generate code that maps directly back to your Python data structures.


```{code-cell} python
arc.codegen(compound_filter, args)
```

```{code-cell} python
:tags: [remove-input]
with open("compound_filter.h", "r") as f:
    c_code = f.read()

display_text(c_code)
```

We see that the `FilterState` struct is automatically translated to `filter_state_t`, and the `CompoundState` and `CompoundOutput` structs are likewise translated to `compound_state_t` and `compound_output_t`.

These can be nested in one another just as in the Python code, so for example a `compound_state_t` is composed of three `filter_state_t`s

## State management

As before, the `_step` function will **not** copy the output state in `res` back into `arg`; it is still implemented as a "pure" function.

As a result, you do still need to manually manage the state by copying the data from `res` back the `arg`, but now this is much easier by virtue of C's "direct assignment" semantics for structs.
Since all of the data in our structs is either a scalar, a statically-allocated array, or another struct (or statically-allocated array of structs) with the same property, copying all stateful data is now a one-liner, regardless of how complex the state is:

```c
filter_arg.state = filter_res.state_new;
```

## Using the generated code

Code using the generated API will now look like the following:

```c
// Initialize as before
compound_filter_arg_t filter_arg;
compound_filter_res_t filter_res;
compound_filter_work_t filter_work;
compound_filter_init(&filter_arg, &filter_res, &filter_work);

// Set up the inputs to the function
filter_arg.u = read_sensor();

// Access nested states if necessary
check_state(&filter_arg.state.x_f);

// Evaluate the function numerically
compound_filter_step(&filter_arg, &filter_res, &filter_work);

// Copy state back to the argument
filter_arg.state = filter_res.state_new;

// Do something with the outputs
handle_outputs(filter_res.y.y_g, filter_res.y.y_h);
```

When you call the `_step` function, all that happens is that the pointers to the underlying data are "marshalled" into a pointer array and passed to CasADi, amounting to minimal overhead compared to the gain in flexibility and readability for more complex functions.

With this approach you can create maintainable and scalable data types that match how you think about parameters, plant models, and control algorithms.

## Supported data types

We've already seen that codegen is supported for scalars (floats), NumPy arrays (up to 2D), and `@struct` classes.

Similar to `@struct` classes, dicts and NamedTuples have named fields containing heterogeneous data; these can also be mapped to C `struct` types.

For instance,

```python
class Point(NamedTuple):
    x: float
    y: float

state = {"pos": np.zeros(3), "vel": np.zeros(3)}
```

would generate the following C code:

```c
typedef struct {
    float x;
    float y;
} point_t;

typedef struct {
    float pos[3];
    float vel[3];
} state_t;
```

Lists and tuples map to arrays, provided their data is "homogeneous", meaning that all items have the same type.

That is, a list of three NamedTuple `Point` objects in Python will generate a C array like `point_t points[3]`, but a list containing one `Point` and a dict will raise an error (since this cannot be mapped to a C array).
For containers with heterogeneous data types, use a NamedTuple, dict, or (recommended) [`@struct`](#archimedes.tree.struct).


The full set of supported conversions is summarized in this table:

| Python type | C type |
| :------: | :------: |
| `float`  | `float`  |
| `np.ndarray`  | `float[]`  |
| `@struct`  | `{name}_t`  |
| `dict \| NamedTuple`  | `{name}_t`  |
| `list[T] \| tuple[T]`  | `T[]`  |

These can compose with one another; you can have a dict whose entries are lists of NamedTuples: this will generate a struct containing arrays of structs.

This flexibility lets you create composite data structures in Python to represent complex input, output, and state and get a predictable C API that maps logically to the Python.

## Summary

In this final part of the hardware deployment tutorial, we've finally put the pieces together to illustrate a scalable workflow for converting compatible Python functions to embedded-friendly C code.

This code generation provides a convenient workflow for writing and evaluating high-level logic in Python and then rapidly deploying to a range of C environments.

### Where to Go From Here

For an example of a more comprehensive development workflow, a great next step is the [end-to-end hardware deployment tutorial](../deployment/deployment00.md).
In that series we walk through the development of a simple DC motor controller, including system identification, controller design, HIL testing, and hardware deployment.

This code generation and hardware deployment workflow is an early proof of concept, but it is central to the Archimedes roadmap.
**If you're using Archimedes for hardware development and have questions, comments, or requests, please don't hesitate to reach out!**


```{code-cell} python
:tags: [remove-cell]
cleanup()
```
