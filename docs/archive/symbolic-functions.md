<!--
Copyright (c) 2025 Pine Tree Labs, LLC.

This file is part of Archimedes 
(see github.com/pinetreelabs/archimedes).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.-->
<!--
Copyright (c) 2025 Pine Tree Labs, LLC.

This file is part of Archimedes 
(see github.com/pinetreelabs/archimedes).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.-->
# Symbolic Functions in Archimedes

Symbolic functions are a powerful feature that allows you to write regular Python code and automatically get additional capabilities like automatic differentiation, optimization, and C code generation. This guide will help you understand how to use them effectively in your projects.

## What are Symbolic Functions?

When you decorate a Python function with `@compile`, Archimedes transforms your regular function into one that can work with both normal numeric arrays and symbolic expressions. This transformation enables:

- Fast ODE/DAE solves
- Integration with optimization tools
- Efficient numerical computation
- Automatic differentiation (sensitivity analysis)

Here's a simple example:

```python
import archimedes as arc
import numpy as np

@arc.compile
def spring_force(x, k=1.0):
    return -k * x  # Hooke's law: F = -kx
```

This function can now be:
- Called normally: `spring_force(0.1)`
- Differentiated automatically: `arc.grad(spring_force)(0.1)`
- Used in optimization problems
- Compiled to C++ code

## How Does It Work?

When you call a symbolic function, Archimedes:
1. Replaces your numeric inputs with symbolic variables
2. Traces how these variables flow through your calculations
3. Creates an efficient computational graph
4. Caches this graph for future use with similar inputs
5. Evaluates the function with your actual input values

Under the hood, this works because the Archimedes `SymbolicArray` type uses the NumPy [dispatch mechanism](https://numpy.org/doc/stable/user/basics.dispatch.html).  This means that when a function like `np.cos` is called on a symbolic array,
NumPy knows to send the arguments back to Archimedes for symbolic evaluation.  This is why you can use plain NumPy functions like `np.sin` and `np.cos` in your symbolic functions and switch seamlessly between symbolic and numeric calculations.

## Decorator Options

### Basic Usage: `@compile`
```python
@arc.compile
def my_function(x, y):
    return x * np.sin(y)
```

### Static Arguments: `static_argnums` and `static_argnames`
Use these when some arguments should be treated as fixed constants:

```python
@arc.compile(static_argnames="frequency")
def oscillator(t, amplitude, frequency=1.0):
    return amplitude * np.sin(2 * np.pi * frequency * t)
```

The `frequency` parameter will be treated as a constant in the symbolic calculations, which simplifies the symbolic calculations.

### Symbolic Variable Type: `kind`
- `"SX"` (default): Best for basic mathematical operations
- `"MX"`: Required for advanced features like ODE solving or optimization

```python
@arc.compile(kind="MX")
def solve_dynamics(x0, t_span):
    # Function involving ODE solving
    return solution
```

## Caching and Reuse

The symbolic function operates by "tracing" the Python function.  Specifically,
when the function is called, all non-static arguments are replaced with symbolic
equivalents and the function is evaluated symbolically.  The symbolic output is
then used to construct a computational graph in CasADi.

If the function is evaluated with different argument types or different static
arguments, the symbolic "tracing" is repeated to create a new computational graph.
Each new graph is cached according to the shapes, dtypes, and static arguments.
Hence, the static arguments must be hashable.

For example, if we create a function as follows:

```python
@compile
def f(x, y):
    return x + np.cos(y)
```

and evaluate it with `x=1.0` and `y=2.0`, the symbolic function will first replace
the numeric arguments with scalar symbolic variables and then pass the symbolic
variables to the original function `f`.  The NumPy function `cos` dispatches
to the symbolic equivalent based on the type of the input, so the return value
from the function is another symbolic variable.  Once this "computational graph"
is constructed, it is stored under the key `(((), np.float64), ((), np.float64))`
in the cache and evaluated using the original input values `x=1.0` and `y=2.0`.
If the function is called again with `x=1.0` and `y=np.array([2.0, 3.0])`, the
symbolic graph will be fully recomputed using the new argument data types, and
will be stored under the key `(((), np.float64), ((2,), np.float64))`.  If the
function is executed a third time with argument types that match either of these
cases, the cached symbolic graph will be used.

In short, it is _relatively fast_ to call a symbolic function with the same shape,
dtype, and static arguments as a previous call, and _relatively slow_ to call it
with different argument types or static arguments (though the actual speed will depend
on the complexity of the function).  Keep this in mind when designing and structuring
your code.

## Common Gotchas and Tips

### 1. Array Creation
When creating arrays inside symbolic functions, use:
```python
# Good - works with both numeric and symbolic inputs
y = arc.array([x[0], x[1]])
# or
y = np.array([x[0], x[1]], like=x)
# or
y = np.hstack([x[0], x[1]])

# Bad - may fail with symbolic inputs
y = np.array([x[0], x[1]])  
```

### 2. Conditionals
Avoid python conditionals with symbolic variables:
```python
# Good - uses numpy where
@arc.compile
def clip(x, limit):
    return np.where(x > limit, limit, x)

# Good - conditional depends on a known static variable
@arc.compile(static_argnames="flag")
def clip(x, flag):
    if flag:
        return np.cos(x)
    else:
        return np.sin(x)

# Bad
@arc.compile
def clip(x, limit):
    if x > limit:  # Can't evaluate symbolic expression in if statement
        return limit
    return x
```

### 3. Loops
Avoid loops with symbolic variables:

__TODO__


## Next Steps

- Try the examples in the [examples folder](examples/)
- Read about [automatic differentiation](link-to-ad-docs)
- Explore [optimization capabilities](link-to-optimization-docs)

## Getting Help

If you encounter issues or have questions:
- Check the [Common Gotchas](#common-gotchas-and-tips) section
- Search existing GitHub issues
- Create a new issue with a minimal example demonstrating your problem


# OLD

The symbolic function operates by "tracing" the Python function.  Specifically,
when the function is called, all non-static arguments are replaced with symbolic
equivalents and the function is evaluated symbolically.  The symbolic output is
then used to construct a computational graph in CasADi that can be evaluated
numerically or symbolically, differentiated, used for codegen, etc.

If the function is evaluated with different argument types or different static
arguments, the symbolic "tracing" is repeated to create a new computational graph.
Each new graph is cached according to the shapes, dtypes, and static arguments.
Hence, the static arguments must be hashable.


For example, if we create a function as follows:

```python
@compile
def f(x, y):
    return x + np.cos(y)
```

and evaluate it with `x=1.0` and `y=2.0`, the symbolic function will first replace
the numeric arguments with scalar symbolic variables and then pass the symbolic
variables to the original function `f`.  The NumPy function `cos` dispatches
to the symbolic equivalent based on the type of the input, so the return value
from the function is another symbolic variable.  Once this "computational graph"
is constructed, it is stored under the key `(((), np.float64), ((), np.float64))`
in the cache and evaluated using the original input values `x=1.0` and `y=2.0`.
If the function is called again with `x=1.0` and `y=np.array([2.0, 3.0])`, the
symbolic graph will be fully recomputed using the new argument data types, and
will be stored under the key `(((), np.float64), ((2,), np.float64))`.  If the
function is executed a third time with argument types that match either of these
cases, the cached symbolic graph will be used.