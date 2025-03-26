# Under the Hood

Once you are up and running with Archimedes, it may be useful to understand some of the internal details of how some of the main functionality works.
This is helpful for debugging and writing more effective and performant code with Archimedes.
This page will revisit the core concepts from ["What is Archimedes?"](about.md) but will get more into the weeds about symbolic arrays, function tracing, and transformations.

(symbolic-arrays)=
## Symbolic arrays

Archimedes is built around _symbolic arrays_, which are a wrapper around [CasADi](https://web.casadi.org/docs/) symbolic objects that implement the NumPy array dispatch mechanism.

It's easiest to see what this means by example. Typical Python classes don't work with NumPy functions out of the box:

```python
import numpy as np

class MyArray:
    def __init__(self, data):
        self.data = data

x = MyArray([1.0, 2.0])
np.sin(x)  # AttributeError: 'MyArray' object has no attribute 'sin'
```

What's happening here is that the NumPy function `sin` will check to see if (a) the argument has a `sin` method it can call, (b) the argument has an `__array__` method that returns a NumPy array, or (c) the argument is a "custom array container" implementing the [NumPy dispatch mechanism](https://numpy.org/doc/stable/user/basics.dispatch.html).
The dispatch mechanism is essentially a way to tell NumPy functions how to work with non-NumPy-native data types like [dask](http://dask.pydata.org/) and [cupy](https://docs-cupy.chainer.org/en/stable/) arrays.
Since we haven't done any of these things for our class yet, NumPy throws an error.

CasADi is a powerful symbolic framework that is ideal in many ways for constructing the computational graphs discussed in the [introduction to Archimedes](about).
However, while its symbolic arrays have some NumPy compatibility because they have methods like `sin`, their overall compatibility with the NumPy API is limited:

```python
import casadi as cs

x = cs.MX.sym("x", 2, 2)  # Create a 2x2 symbolic array
np.sin(x)  # Calls x.sin to return the symbolic expression MX(sin(x))
np.linalg.inv(x)  # numpy.linalg.LinAlgError
```

Archimedes defines a class called `SymbolicArray` that wraps CasADi symbolics in an container that implements the NumPy dispatch mechanism, making it much more compatible with the NumPy API:

```python
import archimedes as arc

x = arc.sym("x", shape=(2, 2))  # Create a SymbolicArray
np.sin(x)  # SymbolicArray containing sin(x)
np.linalg.inv(x)  # SymbolicArray containing the expression @1=((x_0*x_3)-(x_1*x_2)), [[(x_3/@1), (-(x_1/@1))],  [(-(x_2/@1)), (x_0/@1)]]
```

Currently, Archimedes does not implement any core functionality that is not available in the CasADi symbolic backend.
That is to say, anything you can do with Archimedes you could in principle also do directly with CasADi.
Archimedes aims to create an intuitive interface familiar to NumPy and SciPy users that adds as much functionality as possible while having to learn as little as possible.

:::{note}
The NumPy dispatch interface hasn't been fully implemented yet, although many common functions have been implemented.  If something is missing, feel free to open a feature request (or bump an existing feature request).
:::

On its own, the NumPy API compatibility is not especially useful; the value comes with the ability to define and manipulate symbolic functions, as we'll see next.

(symbolic-functions)=
## Symbolic functions

The next building block in Archimedes is _symbolic functions_, which convert plain NumPy code into symbolic computational graphs.
This is the key abstraction that makes it so you don't have to think about symbolic arrays.

### The mechanics of symbolic functions

The NumPy API compatiblity means that you can write [(almost)](gotchas.md) standard NumPy functions that can be evaluated either symbolically or numerically:

```python
def f(A, b):
    b[[0, -1]] = 0.0  # Set boundary conditions
    return np.linalg.solve(A, b)

n = 4

# Evaluate numerically
A = np.random.randn(n, n)
b = np.random.randn(n)
x = f(A, b)  # returns np.ndarray

# Evaluate symbolically
A = arc.sym("A", (n, n))
b = arc.sym("b", (n,))
x = f(A, b)  # returns SymbolicArray
```

However, ultimately we're often not interested in doing symbolic calculations (this is what tools like Mathematica are great at).
The symbolic arrays are a means to an end: constructing the computational graph, which is what enables fast execution, automatic differentiation, etc.

In CasADi this is done by converting a symbolic expression into a `Function` object that essentially acts like a new primitive function that can be evaluated numerically or embedded in more symbolic expressions.
The drawback to this is that the shape of the function arguments must be specified ahead of time and it requires working directly with the symbolic arrays before converting them to a `Function`.

To get around this, Archimedes introduces the `@sym_function` decorator, which converts a standard function into a `SymbolicFunction`.

```python
@arc.sym_function
def f(A, b):
    b[[0, -1]] = 0.0  # Set boundary conditions
    return np.linalg.solve(A, b)
```

At first the `SymbolicFunction` object doesn't do anything except keep a reference to the original code `f`.
However, when it's called as a function, the `SymbolicFunction` will "trace" the original code as follows:

1. Replace all arguments with `SymbolicArray`s that match the shape and dtype
2. Call the original function `f` with the symbolic arguments and gather the symbolic outputs
3. Convert the symbolic arguments and returns into a CasADi `Function` and cache for future use
4. Evaluate the cached `Function` with the original arguments.

If the `SymbolicFunction` is called again with arguments that have the same shape and dtype, the tracing isn't repeated; we can just look up the cached `Function` and evaluate it directly.

What this means is that you can write a single generic function like the one above using pure NumPy and without specifying anything about the arguments, and use it with any valid array sizes.
The `SymbolicFunction` will automatically take care of all of the symbolic processing for you so you never have to actually create or manipulate symbolic variables yourself.

Now you can forget everything you read until `@sym_function`.

### Static arguments and caching

The caching mechanism happens by storing the traced computational graph (as a CasADi `Function`) in a `dict` indexed by the shape and dtype of the arguments.
This means that the tracing is a one-time cost for any combination of shape and dtype, after which evaluation is much faster.

:::{tip}
There is some overhead involved in transferring data back and forth between NumPy and CasADi, so for very small functions you may not see much performance improvement, if at all.
With more complex codes you should be able to see significant performance improvements over pure Python, since the traced computational graph gets executed in compiled C++.
This also makes it beneficial to embed as much of your program as possible into a single `sym_function`, for example by using the built-in ODE solvers instead of the SciPy solvers.
:::

Sometimes functions will take arguments that act as configuration parameters instead of "data".  We call these "static" arguments, since they have a fixed value no matter what data the function is called with.
If we tell the `sym_function` decorator about these, they don't get traced with symbolic arrays.
Instead, whatever value the `SymbolicFunction` gets called with is passed literally to the original function.
Since this can lead to different computational graphs for any value of the static argument, the static arguments are also used as part of the cache key.
This means that the function is also re-traced whenever it is called with a static argument that it hasn't seen before.

So let's say our example function `f` makes it optional to apply the "boundary conditions" using a boolean-valued argument `apply_bcs`.

```python

@arc.sym_function(static_argnames=("apply_bcs",))
def f(A, b, apply_bcs=True):
    if apply_bcs:
        b[[0, -1]] = 0.0  # Set boundary conditions
    return np.linalg.solve(A, b)

f(A, b, apply_bcs=True)  # Compile on first call
f(A, 5 * b, apply_bcs=True)  # Use cached Function
f(A, b, apply_bcs=False)  # Different static argument, need to recompile
```

One caveat to this is that you must return the same number of variables no matter what the static arguments are (this could be remedied, but it's not the best programming practice anyway).

So this is okay:

```python
@arc.sym_function(static_argnames=("flag",))
def f(x, y, flag=True):
    if flag:
        return x, np.sin(y)
    return np.cos(x), y

f(1.0, 2.0, True)
f(1.0, 2.0, False)
```

but this will raise an error:

```python
@arc.sym_function(static_argnames=("flag",))
def f(x, y, flag):
    if flag:
        return x, y
    return x * np.cos(y)

f(1.0, 2.0, True)  # OK
f(1.0, 2.0, False)  # ValueError: Expected 1 return values, got 2 in call to f
```

:::{caution}
Note that we are using standard Python `if`/`else` statements in these examples.
This is fine when applied to static arguments, but should **strictly be avoided** for non-static arguments, since it can lead to unpredictable results as shown below:
:::

```python
@arc.sym_function
def f(x):
    if x > 3:
        return np.cos(x)
    return np.sin(x)

f(5.0)  # Returns sin(5.0)
```

During tracing, Python will check whether the symbolic array `x` is greater than 3.
This is not a well-defined operation; in this case Python decides that this is `False` and we end up with the `sin` branch no matter what the actual value of `x` is.
You should always use "structured" control flow like `np.where` for this:

```python
@arc.sym_function
def f(x):
    return np.where(x > 3, np.cos(x), np.sin(x))

f(5.0)  # Returns cos(5.0)
```

For more details, see documentation on [gotchas](gotchas.md) and [control flow](control-flow.md)

### Symbolic types: MX and SX

(function-transformations)=
## Function transformations

### Automatic differentiation

### Implicit functions and root-finding

### Integrators

