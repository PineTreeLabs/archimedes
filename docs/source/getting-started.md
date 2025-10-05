# Getting Started

Archimedes lets you develop control algorithms in Python and automatically generate optimized C code for embedded deployment.
It combines NumPy-based development with automatic differentiation, efficient ODE solving, and hardware deployment - giving you the productivity of Python with the deployability of C.
New to Archimedes? Read the [introduction](blog/2025/introduction.md) to understand the motivation and design philosophy.
You may also want to see the [quickstart](quickstart) for installation and a simple "Hello, World!" example.

(compiling-functions)=
## Compiling functions

The most basic operation in Archimedes is creating a "compiled" function, which is done as follows:

1. Write out your computation in NumPy
2. Encapsulate it into a function
3. Decorate the function with `@compile`

For example:

```python
import numpy as np
import archimedes as arc

@arc.compile
def rotate(x, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ], like=x)
    return R @ x

rotate(np.array([1.0, 0.0]), 0.1)
```

That's it! Archimedes automatically converted your NumPy-based function into an efficient C++ implementation that can be evaluated symbolically or numerically, or converted into other useful forms via function transformations.

(computing-derivatives)=
## Computing derivatives

Many applications across scientific computing rely on derivatives: optimization, implicit ODE solvers, Newton solvers, linearization, and various forms of sensitivity analysis.

Archimedes makes it easy to compute these derivatives efficiently and accurately by making use of CasADi's powerful _automatic differentiation_ system:

```python
# Simple example: derivative of a scalar function
def f(x):
    return np.sin(x**2)

df = arc.grad(f)
print(np.allclose(df(1.0), 2.0 * np.cos(1.0)))

# Jacobian with respect to a particular argument
def dynamics(t, x):
    a, b, c, d = 1.5, 1.0, 1.0, 3.0
    return np.array([
        a * x[0] - b * x[0] * x[1],  # prey growth rate
        c * x[0] * x[1] - d * x[1]   # predator growth rate
    ], like=x)

J = arc.jac(dynamics, argnums=1)
print(J(0.0, [1.0, 1.0]))

# Hessian of an objective function
def obj(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

H = arc.hess(obj)
x0 = np.array([1.0, 1.0])
print(H(x0))
```

Archimedes also supports more specialized derivative functionality like Jacobian-vector products (`jvp`) and vector-Jacobian products (`vjp`).
The automatic differentiation feature also composes with functionality like ODE solving, so for example you can compute the sensitivity of an ODE solution or embed a derivative in the dynamics function of an ODE.

Finally, the CasADi autodiff backend supports _sparse_ automatic differentiation for the common case that the derivatives with respect to many variables are zero.
For example, in an optimization problem with 1000 variables, the Hessian is a $1000 \times 1000$ matrix where in practice many entries are zero.
Working with sparse autodiff can have huge performance implications in large-scale constrained optimization.
This is a major benefit of Archimedes/CasADi compared to deep learning frameworks that typically work with dense gradient vectors for unconstrained first-order optimization.

(ode-solving)=
## ODE solving

Let's solve a simple differential equation (the Lotka-Volterra predator-prey model):

```python
import numpy as np
import archimedes as arc
import matplotlib.pyplot as plt

# Define and compile the system dynamics
@arc.compile
def dynamics(t, x):
    a, b, c, d = 1.5, 1.0, 1.0, 3.0
    return np.array([
        a * x[0] - b * x[0] * x[1],  # prey growth rate
        c * x[0] * x[1] - d * x[1]   # predator growth rate
    ], like=x)

# Initial conditions and timespan
x0 = np.array([1.0, 1.0])
t_span = (0.0, 10.0)
ts = np.linspace(*t_span, 100)

# Solve the ODE system
xs = arc.odeint(dynamics, t_span=t_span, x0=x0, t_eval=ts)

# Plot the results
plt.plot(ts, xs.T)
plt.legend(['Prey', 'Predator'])
plt.show()
```

The `odeint` function is designed to closely mimic `scipy.integrate.solve_ivp` for a familiar interface.
By default Archimedes uses the powerful [CVODES](https://computing.llnl.gov/projects/sundials/cvodes) solver that can handle stiff or non-stiff ODE systems.  
When needed, Archimedes will use automatic differentiation to efficiently compute the Jacobian of the dynamics.

The one thing you should note here is that when we create the array in the `dynamics` function we need to tell NumPy to create an array `like=x` in order to dispatch to the Archimedes `array` function.
Otherwise NumPy will attempt to create its own `ndarray` full of symbolic objects that it doesn't know how to manipulate, and will ultimately throw an error.
Alternatively, you can use functions like `vstack`, which will work with either symbolic or numeric arguments.

If the same ODE solve will be applied repeatedly (for instance varying initial conditions or parameters), it can be convenient to use a _function transformation_ to create an integrator function instead of calling `odeint` to do a one-time solve:

```python
solver = arc.integrator(f, method="cvodes", rtol=1e-8, atol=1e-6)
xs = solver(x0, t_span)
```

(optimization)=
## Optimization

Archimedes also provides an interface designed after `scipy.optimize.minimize` that supports nonlinear constrained optimization problems.
This uses sparse automatic differentiation to construct all required gradients, Jacobians, and Hessians and calls out to powerful large-scale solvers like IPOPT,

For example, here we solve the classic Rosenbrock problem:

```python
def f(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

result = arc.minimize(f, x0=[-1.0, 1.0])
```

and a [constrained variation](https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization):

```python

def g(x):
    g1 = (x[0] - 1) ** 3 - x[1] + 1
    g2 = x[0] + x[1] - 2
    return np.array([g1, g2], like=x)

x_opt = arc.minimize(f, constr=g, x0=[2.0, 0.0], constr_bounds=(-np.inf, 0))
```

As with the `integrator`, we can also define an `nlp_solver` function for efficient repeated optimization solves (in model-predictive control applications, for instance):

```python
solver = arc.nlp_solver(f, constr=g)
x_opt = solver([2.0, 0.0], -np.inf, 0.0)  # x0, lower_bound, upper_bound
```

(implicit-functions)=
## Root-finding and implicit functions

A root-finding problem is defined as the solution to a system of equations $f(x, p) = 0$, where $p$ is a parameter vector and $x$ is a vector of optimization variables.
Archimedes provides root-finding functionality with an interface similar to `scipy.optimize.root`, but which uses automatic differentiation to compute the Jacobian and can itself be differentiated.

Here's an example borrowed from the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html), without the need for manually constructing a Jacobian:

```python
def f(x):
    return np.array([
        x[0] + 0.5 * (x[0] - x[1])**3 - 1.0,
        0.5 * (x[1] - x[0])**3 + x[1]
    ], like=x)

x = arc.root(f, x0=np.array([0.0, 0.0]))
print(x)  # array([0.8411639, 0.1588361])
```

The more general form $f(x, p) = 0$ can also be thought of as defining $x$ as an implicit function of $p$.
That is, $x = F(p)$, where it might be difficult or impossible to explicitly write out how to compute $F(p)$.
In Archimedes, we can directly construct a function that is equivalent to `F` using the `implicit` _function transformation_:

```python
def f(x, p):
    return x - p * np.cos(x)

F = arc.implicit(f)  # Transform f(x, p) = 0 to x = F(p)
p = 2.0
x0 = 0.0  # Initial guess
x = F(x0, p)
print(f(x, p) == 0)
```

(trees)=
## Tree-structured data

Many scientific computing functions assume that data is arranged into flat vectors.
This includes ODE integrators, optimizers, root-finders, and other solvers.
This is because is often easier and more computationally efficient to mathematically represent the state of a system or a set of parameters as a vector.

However, this is not necessarily the most natural way to implement a physical model, which may have distinct components or hierarchical structure that makes transforming to/from a flat vector tedious and error-prone.

Archimedes addresses this with the concept of tree-structured data (borrowed from "PyTrees" in JAX and "Modules" in PyTorch).
Here, a "tree" is a data structure consisting of nested containers that can be flattened, unflattened, or manipulated directly as a tree.

```python
state = {"pos": np.array([0.0, 1.0, 2.0]), "vel": np.array([3.0, 4.0, 5.0])}
flat, unravel = arc.tree.ravel(state)
print(flat)  # [0. 1. 2. 3. 4. 5.]
print(unravel(flat)) # {'pos': array([0., 1., 2.]), 'vel': array([3., 4., 5.])}
```

A similar approach can work for containers nested to arbitrary depth, i.e. tuples of dictionaries of lists.

Because the tree representations are flattened before use with optimizers, ODE solvers, and similar functions, this allows you to easily switch between the improved code experience of structured data with the numerical and mathematical convenience of a flat vector.

### Structured data classes

If you want to go beyond built-in Python containers, Archimedes provides a `struct` decorator that combines Python's `dataclass` with automatic compatibility with tree operations:

```python
from archimedes import struct

@struct
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

# Pass to/from compiled functions
@arc.compile
def euler_dynamics(state, control):
    # Access structured fields naturally
    new_position = state.velocity + dt * state.position
    # ...
    return VehicleState(
        position=new_position,
        # ...
    )
```

In addition, these custom structs are regular classes, so you can define methods as usual.
For example, you can create a callable class with some parameters using the special `__call__` method:

```python

@struct
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

For more information, see ["Structured data types"](trees.md).

(interpolation)=
## Interpolation

Basic one-dimensional interpolation can be done with the standard NumPy interface:

```python
xp = np.linspace(1, 6, 6)
fp = np.array([-1, -1, -2, -3, 0, 2])

@compile(kind="MX")
def f(x):
    return np.interp(x, xp, fp)

x = 2.5
print(np.isclose(f(x), np.interp(x, xp, fp)))
```

N-dimensional lookup tables can be constructed as `interpolant` functions, for example:

```python
from scipy.interpolate import RectBivariateSpline

xgrid = np.linspace(-5, 5, 11)
ygrid = np.linspace(-4, 4, 9)
X, Y = np.meshgrid(xgrid, ygrid, indexing='ij')
R = np.sqrt(5 * X ** 2 + Y ** 2)+ 1
Z = np.sin(R) / R
f = arc.interpolant(
    [xgrid, ygrid],
    Z.ravel(order='F'),
    arg_names=("x", "y"),
    ret_name="z",
    name="f",
    method="bspline",  # one of ("linear", "bspline")
)

x, y = 0.5, 1
interp = RectBivariateSpline(xgrid, ygrid, Z)
print(np.allclose(f(x, y), interp.ev(x, y)))
```

The Archimedes interpolant matches the SciPy one, but can be embedded in other functions, differentiated through, etc.

## Control flow

One limitation of the symbolic-numeric computations in Archimedes is that it is not compatible with standard Python control flow constructs like `if`/`else` and `while` loops.
This is because Python does not have a dispatch mechanism for control flow similar to NumPy's for array functions.
In other words, Python doesn't know how to handle symbolic conditional execution.
When applied to symbolic arrays, these will either throw errors or result in unpredictable behavior.

To implement a conditional like `if`/`else`, a simple workaround is to use `np.where`, which can use NumPy's dispatch mechanism to handle symbolic arguments:

```python
@arc.compile
def f(x):
    # # Will not work when x is symbolic
    # if x > 0.5:
    #     return np.cos(x)
    # else:
    #     return np.sin(x)

    # Works with symbolic or numeric x
    return np.where(x > 0.5, np.cos(x), np.sin(x))
```

To implement `for` loops, if the length of the loop is known ahead of time it is possible to write a plain Python loop.
However, it is recommended that you put the contents of the loop in a separately compiled function to help keep the size of the computational graph small by minimizing "call sites":

```python
dt = 0.01
g0 = 9.8

@arc.compile
def euler_step(x):
    v = np.array([x[1], -g0 * np.cos(x[0])], like=x)
    return x + v * dt

@arc.compile(static_argnames=("tf",))
def solve_pendulum(x0, tf):
    n = int(tf // dt)
    x = arc.zeros((n, len(x0)))
    x[0] = x0
    for i in range(n-1):
        x[i+1] = euler_step(x[i])
    return x
```

When repeatedly applying a function across an axis of inputs, a more efficient approach is to create a "vectorized map" of the function using `vmap`, which provides similar functionality to [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) or [`numpy.vectorize`](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html). This allows you to write a function as if it acted on a single example of input and then map across a set of inputs.

For example, here we write a function that takes the dot product of two vectors, and then apply it across a "batch" of vectors:

```python
def dot(a, b):
    return np.dot(a, b)

# Vectorize to compute multiple dot products at once
batched_dot = arc.vmap(dot)

# Input: batch of vectors (3 vectors of length 2)
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([[7, 8], [9, 10], [11, 12]])

# Output: batch of scalars (3 dot products)
batched_dot(x, y)
```

The axes to map the function along can be specified by the `in_axes` keyword argument:

```python
batched_dot = arc.vmap(dot, in_axes=(0, 1))
batched_dot(x, y.T)  # Map along the second axis of y.T
```

For more complex loops or improved compilation times, Archimedes also provides a `scan` function similar to [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html), which creates a single call site for the loop body function and does something similar to the following pure Python code:

```python
def scan(func, init_carry, xs, length):
    if xs is None:
        xs = range(length)
    carry = init_carry
    ys = []
    for x in xs:
        carry, y = func(carry, x)
        ys.append(y)
    return carry, np.stack(ys)
```

This loop maintains a state, and also returns an output for each element of the input.
Here's a simple example:

```python
def func(carry, x):
    # carry is the accumulated state, x is the current element
    new_carry = carry + x
    y = new_carry * 2  # some computation with the updated state
    return new_carry, y

init_carry = 0
xs = np.array([1, 2, 3, 4, 5])
final_carry, ys = arc.scan(func, init_carry, xs)

print("Final state:", final_carry)  # 15 (sum of all elements)
print("Outputs:", ys)  # [ 2  6 12 20 30]
```

This function also supports tree-structured arguments for `carry`, but the return type of `func` must match the initial carry type.
There is a learning curve associated with structured control flow functions like `scan`, but it is a powerful and flexible function for constructing symbolically traceable control flow like bounded while loops and map-accumulate operations.

For more details, see the [full documentation page on control flow](control-flow)

(functional-programming)=
## Functional programming

There is a common thread between all of these examples that is worth keeping in mind; because of its design around function transformations, Archimedes maintains a primarily "functional" programming paradigm.

Specifically, Archimedes works by symbolically tracing your functions to create a computational graph, and then applying transformations to that graph or embedding it in more complex functions.
As a result, you should structure all code to be "functionally pure", meaning that the outputs are a direct function of the inputs, and the function does not attempt to modify inputs or any global variables.
Doing so can lead to unpredictable behavior or errors.

```python
# Pure function
def f(x, y):
    return x * np.sin(y)

# Impure: modifies a global variable
i = 0
def f(x):
    global i
    i += 1
    return i * x
```

One subtle exception is that it is allowed for functions to modify symbolic inputs.  This is because the actual input is replaced by a new symbolic variable before tracing, so the modification of the input is simply traced as part of the computational graph and the original numeric value is not modified.
That is, the resulting function will be "pure" even if the original Python code wasn't.

This can be a convenient shorthand, but should be used with caution due to the divergence in behavior between Archimedes and standard Python.

```python
def f(x):
    x[0] = 1
    return x * 2

# The priginal Python function mutates the input in place
x = np.zeros(3)
y = f(x)
print(y)  # [2. 0. 0.]
print(x)  # [1. 0. 0.]

# The compile is forced to be pure
f_sym = arc.compile(f)
x = np.zeros(3)
y = f_sym(x)
print(y)  # [2. 0. 0.]
print(x)  # [0. 0. 0.]
```

In general, learning to work with pure functions can be an adjustment.
Fortunately, domains like math and physics are built on pure functions, and impure functions usually arise as an implementation detail (e.g. allocating work arrays in legacy Fortran code).
If you keep your code as close to the underlying mathematical model as possible, it will tend to be functionally pure.

(custom-callbacks)=
## Custom callbacks

"Callbacks" let you implement pure-Python code and embed it in an otherwise symbolically-evaluated graph.
In other words, this allows you to use functionality that is available in Python but not supported by CasADi's symbolic computation engine.

```python
import math

def unsupported_code(x):
    print("Evaluating unsupported code")
    # The "math" library is not supported symbolically
    return math.tanh(x[0]) * math.exp(-0.1 * x[1]**2)

# Use in a compiled function
@arc.compile
def model(x):
    result_shape_dtypes = 0.0  # Output is a scalar
    y = arc.callback(unsupported_code, result_shape_dtypes, x)
    return y * 2

model(np.array([0.5, 1.5]))  # array(0.73801609)
```

## Using SciPy Functionality

Archimedes supports some common functionality like [ODE solving](#ode-solving), [constrained nonlinear optimization](#optimization), and [nonlinear root-finding](#implicit-functions), but in some cases you may wish to use particular solvers or functionality from larger libraries like SciPy.

In this case you have three basic options:

1. Use the SciPy function as an "offline" step to generate static arrays.
2. Pass Archimedes functions (and possibly derivatives) to SciPy functions
3. Write a [callback function](#custom-callbacks) to embed the external function in the computational graph

As an example of the first case, you can construct an IIR filter outside of an Archimedes function and then use the filter coefficients as NumPy arrays.  This doesn't require symbolically tracing the IIR filter construction.

```python
# Construct an IIR filter and write a `compile` stepping it forward
import numpy as np
import scipy.signal as signal
import archimedes as arc
import matplotlib.pyplot as plt


# 1. Design a Butterworth lowpass filter (offline step with SciPy)
fs = 1000  # Sample frequency in Hz
cutoff = 100  # Cutoff frequency in Hz
order = 4  # Filter order
b, a = signal.butter(order, cutoff / (fs/2), 'low')

# 2. Create a compiled function that implements the IIR filter
@arc.compile
def iir_filter_step(x, state, b_coef, a_coef):
    # Calculate output: y = b[0]*x + state[0]
    y = b_coef[0] * x + state[0]
    
    # Update state using direct form II transposed structure
    state_new = np.zeros_like(state)
    
    for i in range(len(state) - 1):
        state_new[i] = state[i+1] + b_coef[i+1] * x - a_coef[i+1] * y
    
    # Handle the last state element
    if len(state) > 0:
        i = len(state) - 1
        if i+1 < len(b_coef):
            state_new[i] = b_coef[i+1] * x - a_coef[i+1] * y
        else:
            state_new[i] = -a_coef[i+1] * y
    
    return y, state_new

# Function to filter an entire signal using our IIR filter
def filter_signal(signal_data, b_coef, a_coef):
    # Initialize state array (size depends on filter order)
    n = max(len(a_coef), len(b_coef)) - 1
    state = np.zeros(n)
    output = np.zeros_like(signal_data)
    
    # Apply the filter step by step
    for i in range(len(signal_data)):
        output[i], state = iir_filter_step(signal_data[i], state, b_coef, a_coef)
    
    return output

# Create a test signal with mixed frequencies
t = np.linspace(0, 1, fs)
signal_data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)

# Apply our filter
filtered = filter_signal(signal_data, b, a)

# Compare with SciPy's lfilter (should be identical)
filtered_scipy = signal.lfilter(b, a, signal_data)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, signal_data, 'b-', alpha=0.5, label='Original Signal')
plt.plot(t, filtered, 'r-', label='Archimedes Filtered')
plt.plot(t, filtered_scipy, 'g--', label='SciPy Filtered')
plt.legend()
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('IIR Filter Comparison')
plt.tight_layout()
plt.show()
```

For applications like data processing, it is clearly preferable to just use the SciPy implementation.
However, combining the "offline" filter design with a symbolic filtering function allows you to use the filter as part of a control algorithm within a larger computational graph, and also opens up the possibility of automated C code generation for embedded applications. 

In the second case, you may want to use a SciPy optimizer like BFGS for an unconstrained optimization problem.
In this case you can still take advantage of the efficient C++ computations and automatic differentiation in Archimedes, for example:

```python
import numpy as np
import archimedes as arc
from scipy.optimize import minimize

# Solve Rosenbrock problem using BFGS
@arc.compile
def func(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

x0 = np.array([-1.2, 1.0])  # Initial guess
result = minimize(
    func, 
    x0,
    method='BFGS',
    jac=arc.grad(func),
)
```

Note that there is some overhead in translating between the NumPy arrays and CasADi's internal array structures, so this is not guaranteed to be faster than vanilla NumPy; the performance will be dictated by the relative complexity of an individual function evaluation compared to the number of function evaluations.
For instance, in the Rosenbrock example above the fastest solution is actually plain NumPy with finite differencing, even though this uses about 3x as many function evaluations!
This will often not be the case for more complex practical applications.

<!-- TODO: Write an example callback when this is implemented -->

All of these strategies will also work for libraries beyond SciPy.
Integration with other libraries like Matplotlib is largely straightforward, since Archimedes works with NumPy arrays and visualizations can be created "offline" as a postprocessing step.

(basic-troubleshooting)=
## Basic Troubleshooting

For more details, see the [Quirks and Gotchas](gotchas.md) page, but here are some common early pitfalls:

- **Error when creating arrays**: Remember to use `like=x` or functions like `vstack` with symbolic arguments
- **Unexpected results with conditional logic**: Use `np.where` instead of Python `if/else` with symbolic arguments
- **Poor performance**: Ensure you're reusing solver functions rather than recreating them for repeated calls