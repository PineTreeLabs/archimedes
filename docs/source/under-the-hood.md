# Under the Hood

Once you are up and running with Archimedes, it may be useful to understand some of the internal details of how some of the main functionality works.
This is helpful for debugging and writing more effective and performant code with Archimedes.
This page will get into the weeds about symbolic arrays, function tracing, and transformations.

The core functionality of Archimedes is the ability to transform plain NumPy functions into high-performance C++ _computational graphs_. By leveraging [CasADi](https://web.casadi.org/docs/), a symbolic framework for automatic differentiation and numerical optimization designed specifically for optimal control applications, Archimedes allows users to implement complex models using familiar NumPy syntax while gaining significant performance improvements and advanced capabilities.

Archimedes is based on three core concepts:

1. **Symbolic-numeric computation**
2. **Function transformations**
3. **Tree-structured data**

## 1. Symbolic-numeric computation

Archimedes provides a seamless interface between symbolic and numeric computation by wrapping CasADi's symbolic engine in a NumPy-compatible array API. This approach, combined with JAX-style composable function transformations, enables a powerful workflow:

1. Write functions using standard NumPy operations
2. Convert these functions into "compiled" functions with a simple decorator
3. When called, these functions create an efficient computational graph in compiled C++
4. This computational graph can be used for fast execution, automatic differentiation, and C code generation

Consider this simple example:

```python
import numpy as np
import archimedes as arc

# Write your function using standard NumPy
def f(x, y):
    return x + np.sin(y)

# Convert it into a "compiled" function
f_sym = arc.compile(f)

# Call the compiled function with standard arrays
z = f_sym(1.0, np.array([2.0, 3.0]))
```

### The computational graph

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

(symbolic-arrays)=
### Symbolic arrays

This functionality relies on _symbolic arrays_, which are a wrapper around [CasADi](https://web.casadi.org/docs/) symbolic objects that implement the NumPy array dispatch mechanism.

It's easiest to see what this means by example. Typical Python classes don't work with NumPy functions out of the box:

```python
import numpy as np

class MyArray:
    def __init__(self, data):
        self.data = data

x = MyArray([1.0, 2.0])
np.sin(x)  # AttributeError: 'MyArray' object has no attribute 'sin'
```

What's happening here is that the NumPy function `sin` will check to see if (1) the argument has a `sin` method it can call, (2) the argument has an `__array__` method that returns a NumPy array, or (3) the argument is a "custom array container" implementing the [NumPy dispatch mechanism](https://numpy.org/doc/stable/user/basics.dispatch.html).
The dispatch mechanism is essentially a way to tell NumPy functions how to work with non-NumPy-native data types like [dask](http://dask.pydata.org/) and [cupy](https://docs-cupy.chainer.org/en/stable/) arrays.
Since we haven't done any of these things for our class yet, NumPy throws an error.

CasADi is a powerful symbolic framework that is ideal in many ways for constructing the computational graphs discussed in the [introduction to Archimedes](blog/2025/introduction.md).
However, while its symbolic arrays have some NumPy compatibility because they have methods like `sin`, their overall compatibility with the NumPy API is limited:

```python
import casadi as cs

x = cs.SX.sym("x", 2, 2)  # Create a 2x2 symbolic array
np.sin(x)  # Calls x.sin to return the symbolic expression MX(sin(x))
np.linalg.inv(x)  # numpy.linalg.LinAlgError
```

Archimedes defines a class called `SymbolicArray` that wraps CasADi symbolics in an container that implements the NumPy dispatch mechanism, making it much more compatible with the NumPy API:

```python
import archimedes as arc

x = arc.sym("x", shape=(2, 2), kind="SX")  # Create a SymbolicArray
np.sin(x)  # SymbolicArray containing sin(x)
np.linalg.inv(x)  # SymbolicArray containing the expression @1=((x_0*x_3)-(x_1*x_2)), [[(x_3/@1), (-(x_1/@1))],  [(-(x_2/@1)), (x_0/@1)]]
```

Currently, Archimedes does not implement any core functionality that is not available in the CasADi symbolic backend.
That is to say, anything you can do with Archimedes you could in principle also do directly with CasADi.
Archimedes aims to create an intuitive interface familiar to NumPy and SciPy users that adds as much functionality as possible while having to learn as little as possible.

:::{note}
The NumPy dispatch interface hasn't been fully implemented yet, although many common functions have been implemented.  If something is missing, feel free to open a feature request (or bump an existing feature request).
:::

On its own, the NumPy API compatibility is not especially useful; the value comes with the ability to define and manipulate compiled functions, as we'll see next.

(symbolic-compilation)=
### Symbolic compilation

The next building block in Archimedes is _compiled functions_, which convert plain NumPy code into symbolic computational graphs.
This is the key abstraction that makes it so you don't have to think about symbolic arrays.

#### The mechanics of compilation

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

To get around this, Archimedes introduces the `@compile` decorator, which converts a standard function into a `FunctionCache`.

```python
@arc.compile
def f(A, b):
    b[[0, -1]] = 0.0  # Set boundary conditions
    return np.linalg.solve(A, b)
```

At first the `FunctionCache` object doesn't do anything except keep a reference to the original code `f`.
However, when it's called as a function, the `FunctionCache` will "trace" the original code as follows:

1. Replace all arguments with `SymbolicArray`s that match the shape and dtype
2. Call the original function `f` with the symbolic arguments and gather the symbolic outputs
3. Convert the symbolic arguments and returns into a CasADi `Function` and cache for future use
4. Evaluate the cached `Function` with the original arguments.

If the `FunctionCache` is called again with arguments that have the same shape and dtype, the tracing isn't repeated; we can just look up the cached `Function` and evaluate it directly.

What this means is that you can write a single generic function like the one above using pure NumPy and without specifying anything about the arguments, and use it with any valid array sizes.
The `FunctionCache` will automatically take care of all of the symbolic processing for you so you never have to actually create or manipulate symbolic variables yourself.

Now you can forget everything you read until `@compile`.

#### Static arguments and caching

The caching mechanism happens by storing the traced computational graph (as a CasADi `Function`) in a `dict` indexed by the shape and dtype of the arguments.
This means that the tracing is a one-time cost for any combination of shape and dtype, after which evaluation is much faster.

:::{tip}
There is some overhead involved in transferring data back and forth between NumPy and CasADi, so for very small functions you may not see much performance improvement, if at all.
With more complex codes you should be able to see significant performance improvements over pure Python, since the traced computational graph gets executed in compiled C++.
This also makes it beneficial to embed as much of your program as possible into a single `compile`, for example by using the built-in ODE solvers instead of the SciPy solvers.
:::

Sometimes functions will take arguments that act as configuration parameters instead of "data".  We call these "static" arguments, since they have a fixed value no matter what data the function is called with.
If we tell the `compile` decorator about these, they don't get traced with symbolic arrays.
Instead, whatever value the `FunctionCache` gets called with is passed literally to the original function.
Since this can lead to different computational graphs for any value of the static argument, the static arguments are also used as part of the cache key.
This means that the function is also re-traced whenever it is called with a static argument that it hasn't seen before.

So let's say our example function `f` makes it optional to apply the "boundary conditions" using a boolean-valued argument `apply_bcs`.

```python

@arc.compile(static_argnames=("apply_bcs",))
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
@arc.compile(static_argnames=("flag",))
def f(x, y, flag=True):
    if flag:
        return x, np.sin(y)
    return np.cos(x), y

f(1.0, 2.0, True)
f(1.0, 2.0, False)
```

but this will raise an error:

```python
@arc.compile(static_argnames=("flag",))
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
@arc.compile
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
@arc.compile
def f(x):
    return np.where(x > 3, np.cos(x), np.sin(x))

f(5.0)  # Returns cos(5.0)
```

For more details, see documentation on [gotchas](gotchas.md) and [control flow](control-flow)

(symbolic-types)=
#### Symbolic types: MX and SX

There are two basic symbolic types inherited from CasADi: `SX` and `MX`.
The kind of symbolic array or function that is created is specified by the `kind=` keyword argument, for example:

```python
@arc.compile(kind="SX")
def norm(x):
    return np.dot(x, x)
```

`SX` produces scalar symbolic arrays, meaning that every entry in the array has its own scalar symbol.
This can produce highly efficient code, but is limited to a subset of possible operations.
For example, `SX` symbolics don't support interpolation with lookup tables.

`MX` symbolics are array-valued, meaning that the entire array is represented by a single symbol.
This allows for embedding much more general operations like interpolation, ODE solves, and optimization solves into the computational graph, but may not be as fast as `SX` for functions that are dominated by scalar operations.

Using both `SX` and `MX` can be done to a limited extent, but can be error-prone and should be done with caution.
The current default is `MX` and the current recommendation is to use `MX` symbolics unless you want to do targeted performance optimizations and feel comfortable with the symbolic array concepts.
In this case the most reliable approach is to create compiled `SX` functions that use only supported mathematical operations and then call them from inside any more complex functions using general operations.
For example, you may be able to define an ODE function using `SX` and then do parameter estimation in a more general function traced with `MX` symbolics.

(function-transformations)=
## 2. Function transformations

The second key concept in Archimedes is _function transformations_, which take an existing function and produce a new function with modified behavior.

### How Function Transformations Work

Internally, a function transformation in Archimedes follows these steps:

1. Compile the input function to a computational graph (if not already done)
2. Apply transformation-specific operations to this computational graph
3. Wrap the transformed graph in a new function with an appropriate signature

For example, when you call `arc.grad(f)`, Archimedes:
1. Traces `f` with symbolic inputs to construct its computational graph
2. Applies automatic differentiation to the graph to obtain the gradient
3. Returns a new function that, when called, evaluates this gradient graph

This approach allows transformations to be composable - you can apply multiple transformations in sequence:

```python
def f(x, y):
    return x**2 + np.sin(y)

# Compute the gradient with respect to x, then the Jacobian of that gradient with respect to y
ddf_dxdy = arc.jac(arc.grad(f, argnums=0), argnums=1)
```

### Core Transformation Categories

#### Automatic Differentiation

Automatic differentiation transformations leverage CasADi's powerful AD capabilities:

```python
@arc.compile
def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2 

# First-order derivatives
df = arc.grad(f)  # Returns gradient (vector of partial derivatives)
J = arc.jac(f)    # Returns Jacobian matrix

# Higher-order derivatives
H = arc.hess(f)   # Returns Hessian matrix
```

CasADi implements AD using a hybrid approach of forward and reverse mode differentiation, optimizing for computational efficiency based on the number of inputs and outputs.
It also supports _sparse_ automatic differentiation, which is crucial for performance in many practical large-scale constrained optimization problems.

#### Solver Embeddings

Another category of transformations embeds iterative solvers directly into the computational graph:

```python
# Create an ODE solver function
def dynamics(t, x):
    return np.array([x[1], -np.sin(x[0])], like=x)

solver = arc.integrator(dynamics, method="cvodes")
```

When you call `arc.integrator`, Archimedes:
1. Traces the dynamics function to construct its computational graph
2. Creates an ODE solver with the specified method
3. Returns a function that, when called, applies this integrator to solve the ODE

Similarly, `nlp_solver` embeds an optimization solver, and `implicit` embeds a root-finding solver. These embedded solvers:

1. Execute in compiled C++ for high performance
2. Can (sometimes) be differentiated through themselves (implicit differentiation)
3. Can be composed with other functions and transformations


### Implementation Details

Function transformations operate on CasADi's graph representation rather than directly on Python code. This means:

1. The transformation sees the already-traced computational graph, not the original Python logic
2. Transformations are only aware of operations captured in the graph during tracing
3. The resulting function is a new C++ computational graph wrapped in a Python callable

For example, in gradient computation:
```python
# Original function
@arc.compile
def f(x):
    return np.sin(x**2)

# Gradient transformation
df = arc.grad(f)
```

Archimedes traces `f` with a symbolic input, producing a symbolic output `sin(x^2)`. It then applies autodiff to the original computational graph to get `df/dx = 2x * cos(x^2)`. The resulting function `df` is a Python wrapper around this new CasADi Function.

For advanced use cases, transformations accept additional parameters for customization:

```python
# Gradient with respect to second argument only
df_dx1 = arc.grad(f, argnums=(1,))  

# Customize ODE solver behavior
solver = arc.integrator(dynamics, method="cvodes", rtol=1e-8, atol=1e-8)
```

These function transformations enable complex operations like sensitivity analysis, optimal control, and parameter estimation with minimal code.

(tree-structure)=
## 3. Tree-structured data

Modern engineering models often involve complex, nested data structures that go beyond simple arrays.
Archimedes adopts the ["PyTree" concept from JAX](https://docs.jax.dev/en/latest/pytrees.html) to seamlessly work with tree-structured data and blends it with the [composable "Module" design from PyTorch](https://pytorch.org/docs/stable/notes/modules.html) for constructing hierarchical, modular functionality.

Here we'll give a brief overview of this at a conceptual level; for more detail see the documentation page on ["Structured Data Types"](trees.md)

### What is tree-structured data?

In this context, "trees" are nested structures of containers (lists, tuples, dictionaries) and leaves (arrays or scalars) that can be flattened to a vector and "unflattened" back to their original structure.
They provide a systematic way to:

1. **Organize complex data** - Maintain logical structure in your models
2. **Simplify function interfaces** - Pass structured arguments instead of numerous separate parameters
3. **Enable operations on nested structures** - Apply transformations that work naturally with trees

For example, ODE solvers are typically written to work with functions that accept and return a vector state.
However, for complex systems the "state" may contain many sub-components and it can be time-consuming and error-prone to manually index into a monolithic vector.

```python
# Example of a tree representing a simple robot state
state = {
    'joints': {
        'pos': np.array([0.1, 0.2, 0.3]),
        'vel': np.array([0.01, 0.02, 0.03])
    },
    'end_effector': {
        'pos': np.array([1.0, 2.0, 3.0]),
        'q': np.array([1.0, 0.0, 0.0, 0.0])
    }
}
```

As in the example of this "robot state", a tree can be made up of built-in Python data structures like dictionaries, lists, and tuples.
Archimedes already knows how to work with these containers, but as we'll see below, you can also easily construct custom structured data types that are compatible with tree operations using a simple dataclass-derived class decorator.

### Tree operations

Archimedes provides utilities in the `tree` module to work efficiently with tree-structured data.
The most common operation is to "ravel", or flatten, a tree to a single vector, and to "unravel" a vector of the same length back to the original tree.

```python
# Flatten a tree into a single vector for optimizers, ODE solvers, etc.
flat_state, unravel = arc.tree.ravel(state)
print(flat_state)  # array([1.  , 0.  , 0.  , 0.  , 1.  , 2.  , ...

# Reconstruct the original tree structure
print(unravel(flat_state))  # {'end_effector': {'q': array([1., 0., 0., 0.]), ...
```

You can also perform other functional operations like mapping a function over the leaves of a tree:

```python
arc.tree.map(lambda x: -x, state)  # {'end_effector': {'q': array([-1., 0., 0., 0.]), ...
```

### Structured data types

While you can build up tree-structured data using built-in Python types like dicts, tuples, etc., it is often cleaner to write your own classes.

Archimedes offers a [`@struct`] decorator for classes that turns the class into a Python [dataclass](https://docs.python.org/3/library/dataclasses.html) that is compatible with tree operations.

For example, the `state` dictionary above could be rewritten as:

```python
from archimedes import struct

@struct
class JointState:
    pos: float
    vel: float

@struct
class EndEffectorState:
    pos: np.ndarray
    q: np.ndarray

@struct
class ArmState:
    joints: tuple[JointState]
    end_effector: EndEffectorState
```

These `struct` types are also standard Python classes, so you can define methods on them as usual.
These gives you the ability to construct hierarchical data types in the same vein as the PyTorch `Module` system - but with the ability to flatten/unflatten and apply other tree operations.
For more on recommended design patterns for `struct`-based classes, including configuration management for complex systems, see the [Hierarchical Modeling](tutorials/hierarchical/hierarchical00.md) tutorial series.

These `struct` types also work naturally with C code generation.
For example, if an `ArmState` is used as the argument to a codegen function, the following C code will be produced:

```c
typedef struct {
    float pos;
    float vel;
} joint_state_t;

typedef struct {
    float pos[3];
    float q[4];
} end_effector_state_t;

typedef struct {
    joint_state_t joints[4];
    end_effector_state_t end_effector;
} arm_state_t;
```

This gives you a predictable and intuitive way to switch back and forth between Python and auto-generated C.

For details on structured data types and codegen, see the [C Code Generation](tutorials/codegen/codegen00.md) tutorial.

## Conclusion

Understanding the "under the hood" mechanisms of Archimedes—symbolic arrays, symbolic compilation, and function transformations—provides valuable insight into how to use the framework effectively. While you don't need to fully understand these concepts to use Archimedes productively, this knowledge can help you:

1. **Debug difficult problems** - When encountering unexpected behavior, knowing how symbolic tracing works can help identify if the issue relates to control flow, array creation, or other common pitfalls.

2. **Optimize performance** - Understanding how function caching works and how computational graphs are constructed allows you to structure your code for maximum efficiency.

3. **Leverage advanced features** - With knowledge of how transformations operate, you can compose them in powerful ways to solve complex problems with minimal code.

Archimedes provides the performance benefits of symbolic computation without requiring you to work directly with symbolic expressions.
By using familiar NumPy syntax and adding just a few decorators and transformations, you can create high-performance numerical code that automatically supports features like automatic differentiation, efficient ODE solving, and optimization.

For more detailed information on specific aspects of using Archimedes effectively, refer to the [Quirks and Gotchas](gotchas.md) page and the various examples in the documentation.