# 🔪 Quirks and Gotchas 

Archimedes is a Python framework for numerical modeling and simulation that bridges NumPy's ease of use with the performance benefits of symbolic computation through CasADi.
While it makes complex modeling tasks more approachable, there are several "sharp bits" you should be aware of to avoid unexpected behavior.
This document outlines common pitfalls and quirks you might encounter when using Archimedes.

This page is modeled after ["🔪 JAX - The Sharp Bits 🔪"](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html), since JAX and Archimedes share many design features.

## 🔪 Pure Functions

Archimedes transformations and symbolic functions are designed to work only with functions that are *functionally pure*: all input data is passed through the function parameters, and all results are returned through the function return values.

```python
# NOT recommended: impure function that uses a global variable
g = 9.81
def impure_acceleration(v, t):
    return np.array([v[1], -g * np.sin(v[0])], like=v)  # Uses global g

# Recommended: pure function with all inputs as parameters
def pure_acceleration(v, t, g=9.81):
    return np.array([v[1], -g * np.sin(v[0])], like=v)  # g passed as parameter

# BAD: impure function that modifies a global variable
i = 0
def impure_accumulator(x):
    global i
    i += 1  # Modifies global i
    return x * i
```

While Archimedes works with impure functions in simple cases, this can lead to surprising behavior when functions are transformed or reused. For example, if the global variable `g` changes value, the function might still use the original value in some contexts.

Less obviously, print statements are a "side effect" that makes functions impure.
For tips on printing and debugging see the [debug](#debugging) section below.


(numpy-compatibility-limitations)=
## 🔪 NumPy Compatibility Limitations

Archimedes currently supports many common NumPy operations, but not all.
To see the exhaustive list and current implementation status, see:

* `archimedes._core._array_ops._array_ufunc.SUPPORTED_UFUNCS`
* `archimedes._core._array_ops._array_function.SUPPORTED_FUNCTIONS`

Any entries with `NotImplemented` values are currently not supported.
**If you'd like to see one of these implemented, feel free to file a feature request (or bump an existing one)!**

## 🔪 Matrix- vs. scalar-valued symbolics

When creating symbolic functions, note that there are two basic symbolic types inherited from CasADi: `SX` and `MX`.
`SX` symbolics create symbolic arrays as a collection of symbolic scalars, while `MX` directly create a single symbol to represent an array.
There are performance and flexibility tradeoffs between the two, and trying to mix them can be error prone.

Specifically, if you get `NotImplemented` errors related to functions that you know are supported (see [above](#numpy-compatibility-limitations)), a likely reason is that you have somehow mixed `SX` and `MX` symbolics.
For more information see the [section on symbolic types](symbolic-types) in Under the Hood.

## 🔪 Array Dimensions

In contrast to standard NumPy N-dimensional arrays, CasADi symbolic arrays are always two-dimensional (an artifact of its MATLAB-oriented heritage).
Archimedes allows for 0-D, 1-D, and 2-D arrays to mimic NumPy behavior, but higher-dimensional arrays are currently not supported.

## 🔪 Array Creation

One of the most common pitfalls in Archimedes involves creating arrays within symbolic functions. Standard NumPy array creation doesn't automatically work with symbolic arrays:

```python
# This will NOT work with symbolic inputs:
def bad_function(x):
    return np.array([x[1], x[0]])  # NumPy doesn't know to make SymbolicArrays

# This will NOT work with symbolic inputs:
def also_bad(x):
    y = np.zeros(x.shape[0])  # Creates a NumPy array, not a symbolic array
    for i in range(x.shape[0]):
        y[i] = x[i]  # Fails - cannot assign symbolic value to numeric array
    return y
```

Instead, you must explicitly use `like=` parameter, dispatched functions, or Archimedes array constructors:

```python
# Good: use like= to create compatible arrays
def good_function1(x):
    return np.array([x[1], x[0]], like=x)

# Good: use dispatched functions like hstack, vstack, etc.
def good_function2(x):
    return np.hstack([x[1], x[0]])

# Good: use Archimedes array constructor
import archimedes as arc
def good_function3(x):
    return arc.array([x[1], x[0]])

# Good: use zeros_like and other dispatched functions
def good_function4(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i] = x[i]
    return y

# Good: use Archimedes zeros and similar functions
def good_function4(x):
    y = arc.zeros(len(x))
    for i in range(x.shape[0]):
        y[i] = x[i]
    return y
```

:::{note}
If you get the error `ValueError: setting an array element with a sequence. The requested array would exceed the maximum number of dimension of 64.` this is most likely due to incorrect array creation.
:::

## 🔪 Control Flow

Python's standard control flow constructs like `if/else` and loops with dynamic bounds don't work well with symbolic evaluation. For example:

```python
# This will NOT work correctly when x is symbolic
@arc.compile
def bad_conditional(x):
    if x > 0:  # Can't evaluate symbolic expression in boolean context
        return np.sin(x)
    else:
        return np.cos(x)
```

Instead, use `np.where` for conditionals:

```python
@arc.compile
def good_conditional(x):
    return np.where(x > 0, np.sin(x), np.cos(x))
```

For loops:
1. Ensure loop bounds are static (not symbolic expressions)
2. Consider using `arc.scan` for more complex loops

```python
# This will NOT work with symbolic input
@arc.compile
def bad_loop(x):
    y = 0
    for i in range(sum(x)):  # sum(x) is symbolic, can't be used as loop bound
        y += i
    return y

# This will work
@arc.compile
def good_loop(x):
    y = 0
    for i in range(x.shape[0]):  # Fixed shape is fine
        y += x[i]
    return y
```

See the [control flow](control-flow)  section of Getting Started for a simple example with `scan`, or see the `scan` API documentation for details.

## 🔪 Static Arguments

Some function arguments shouldn't be traced symbolically, but instead treated as fixed configuration parameters. Archimedes provides a way to specify these:

```python
@arc.compile(static_argnames=("apply_bc",))
def solve_with_optional_bc(A, b, apply_bc=True):
    if apply_bc:  # This works because apply_bc is static
        b[[0, -1]] = 0.0  # Apply boundary conditions
    return np.linalg.solve(A, b)
```

When a function with static arguments is called, it will be retraced whenever the static argument value changes. This allows conditional evaluation based on configuration, but can lead to performance issues if done excessively.

## 🔪 Function Caching

Symbolic functions in Archimedes are cached based on the shapes and dtypes of their arguments, as well as on the values of static parameters. This means the first call to a function with a specific argument shape might be slower (due to tracing), but subsequent calls with the same shape will be faster.

```python
@arc.compile
def f(x):
    return np.sin(x**2)

f(np.array([1.0, 2.0]))  # First trace with shape (2,)
f(np.array([3.0, 4.0]))  # Reuse cached function for shape (2,)
f(np.array([1.0, 2.0, 3.0]))  # New trace with shape (3,)
```

Be aware that excessive retracing with different argument shapes can lead to performance degradation and memory usage growth.

(in-place-operations)=
## 🔪 In-place Operations

Unlike NumPy, which allows in-place modification of arrays, symbolic arrays in Archimedes should be treated as immutable. Operations that appear to modify arrays in-place in Python actually create new symbolic expressions behind the scenes.

```python
# In NumPy - modifies array in place
numpy_array = np.zeros(3)
numpy_array[0] = 1.0  # Original array is modified

# In Archimedes - creates new symbolic expression
@arc.compile
def f(x):
    x[0] = 1.0  # This doesn't actually modify x in-place when symbolic
    return x    # Instead, it creates a new expression

# When using arc.compile, this becomes pure even if the Python code isn't!
x = np.zeros(3)
y = f(x)
print(x)  # [0., 0., 0.] - original is unchanged
print(y)  # [1., 0., 0.] - new array is returned
```

This behavior differs from both NumPy (which modifies arrays in-place) and JAX (which disallows in-place operations entirely).

The current recommendation is that it is okay to do in-place operations, but do it with caution.
Specifically, it is a good idea when implementing a function like this to always decorate it so that there is not a "NumPy version" that may inadvertently behave differently than the "Archimedes version".

(debugging)=
## 🔪 Debugging Symbolic Code

Debugging symbolic code can be challenging because:
1. Error messages may refer to CasADi internals rather than your code
2. Standard Python debuggers can't step through symbolically evaluated code
3. `print` statements in symbolic functions won't execute at runtime

```python
@arc.compile
def f(x):
    print("This won't print during symbolic tracing")  # Won't execute during tracing
    y = np.sin(x)
    return y * 2

# Instead, debug the inputs and outputs:
x = np.array([1.0, 2.0])
print("Input:", x)
y = f(x)
print("Output:", y)
```

Some tips for debugging:
- Print array shapes and dtypes instead of values at trace time
- Add trace/debug prints before and after symbolic function calls
- Split complex operations into smaller, testable pieces
- Test with concrete numeric values before using symbolic evaluation

A reliable workflow is to write small functions in pure NumPy first, validate the NumPy code, and then add the `compile` decorator once you're confident in the results.

:::{note}
Aside from specific cases like [in-place operations](#in-place-operations), Archimedes and NumPy results should be consistent.
If you are seeing unexpected divergences, **please file a bug report.**
:::

<!-- TODO: Add arc.debug.print and arc.debug.callback when available -->

## 🔪 Symbolic vs. Numeric Interface Differences

While Archimedes tries to make symbolic and numeric functions behave identically, there are some unavoidable differences:

1. Symbolic functions can't use arbitrary Python objects as inputs - stick to numeric values and arrays, or PyTree data structures that Archimedes knows how to work with
2. Some NumPy functions may have subtly different behavior in symbolic vs. numeric contexts (e.g. [in-place operations](#in-place-operations))
3. Not all NumPy functionality is available symbolically.  **If you find a missing function you need, feel free to file a feature request (or bump an existing one)**.

```python
# Works in both contexts
def simple_function(x):
    return np.sin(x) + np.cos(x)

# Works only in numeric context
def complex_function(x, callback):
    result = np.sin(x)
    callback(result)  # Can't symbolically trace an arbitrary callback
    return result

# Works IF the callback is hashable
traced_function = arc.compile(complex_function, static_argnames=("callback",))
```

## 🔪 PyTree Handling

While PyTrees provide a powerful way to handle nested data structures, they come with their own quirks:

```python
# Be careful when flattening and unflattening complex structures
nested_data = {"a": np.array([1.0, 2.0]), "b": {"c": np.array([3.0])}}
flat, unflatten = arc.tree.ravel(nested_data)

# The flat array contains all numeric values concatenated
print(flat)  # [1. 2. 3.]

# When modifying flat array, make sure the shape is preserved
modified = flat * 2
restored = unflatten(modified)  # This works because shape is preserved

# This would fail because the shape changes
#larger = np.concatenate([flat, np.array([4.0])])
#broken = unflatten(larger)  # Error: incompatible flat array shape
```

Custom PyTree nodes should be well-tested with symbolic functions to ensure proper behavior.

## 🔪 ODE Solving Intricacies

When using ODE solvers, be aware of:

1. The dynamics function signature must match the expected format (t, x, *args)
2. Integrator state must be a flat vector (use PyTree operations to flatten/unflatten if not)
3. In order to maintain static array sizes, ODE solvers can only output data at fixed sample times

For best performance, create an integrator once and reuse it with different inputs:

```python
# Less efficient: recreates the integrator for every call
def simulate_multiple(initial_conditions, t_eval):
    results = []
    t_span = (t_eval[0], t_eval[-1])
    for x0 in initial_conditions:
        xs = arc.odeint(dynamics, t_span=t_span, x0=x0, t_eval=t_eval)  # New integrator each time
        results.append(xs)
    return results

# More efficient: create integrator once
def simulate_multiple_efficient(initial_conditions, t_eval):
    solver = arc.integrator(dynamics, t_eval=t_eval)  # Create once
    results = []
    for x0 in initial_conditions:
        xs = solver(x0)  # Reuse solver with different initial conditions
        results.append(xs)
    return results
```

If the dynamics function has additional arguments, the `solver` function will have the signature `solver(x0, *args)`.
See the API documentation for `integrator` for details.

## 🔪 Conclusion

Archimedes can be helpful for improving the performance and functionality of scientific computing work, but it does have its share of sharp edges.
Some of these may be resolved with further development, but some are due to the nature of the symbolic-numeric computing paradigm.
By understanding these quirks and adopting the recommended patterns, you can avoid common pitfalls and build robust, high-performance numerical applications.

If you find yourself struggling with an issue not covered here, the best approach is to:
1. Simplify your code to isolate the problem
2. Check if your functions are pure
3. Ensure array creation is handled correctly
4. Verify control flow is compatible with symbolic evaluation

If you've done all of this and can't resolve your problem, consider filing an issue on GitHub, particularly if:
- The NumPy and Archimedes versions of the same function don't produce the same output
- Archimedes is producing a hard-to-understand error message
- You suspect a bug in the Archimedes code

Remember that Archimedes is under active development, and the API may evolve to address these and other challenges.