# Getting Started with Archimedes

This page will walk through some basic functionality and examples to get you started with some of the patterns and conventions of Archimedes.
Before working through this page, you may want to make sure you have a working [installation](quickstart) and read through ["What is Archimedes?"](about) to understand the basic conceptual framework and terminology.

(ode-solving)=
## ODE solving

Let's solve a simple differential equation (the Lotka-Volterra predator-prey model):

```python
import numpy as np
import archimedes as arc
import matplotlib.pyplot as plt

# Define the system dynamics and convert to a symbolic function
@arc.sym_function
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

That's it! Archimedes automatically converted your NumPy-based function into an efficient C++ implementation and solved the ODE.
The `odeint` function is designed to closely mimic SciPy's `solve_ivp` for a familiar interface.

Specifically, by default Archimedes uses the powerful [CVODES](https://computing.llnl.gov/projects/sundials/cvodes) solver that can handle stiff or non-stiff ODE systems.  
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

Use IPOPT to solve the Rosenbrock problem 

```python
def f(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

x_opt = arc.minimize(f, x0=[-1.0, 1.0])
```

and constrained Rosenbrock problem

```python

def g(x):
    g1 = (x[0] - 1) ** 3 - x[1] + 1
    g2 = x[0] + x[1] - 2
    return np.array([g1, g2], like=x)

x_opt = arc.minimize(f, constr=g, x0=[2.0, 0.0], constr_bounds=(-np.inf, 0))
```

As with the `integrator`, we can also define an `nlp_solver` function for efficient repeated optimization solves:

```python
solver = arc.nlp_solver(f, constr=g)
x_opt = solver([2.0, 0.0], -np.inf, 0.0)  # x0, lower_bound, upper_bound
```

(automatic-differentiation)=
## Automatic differentiation

Many applications across scientific computing rely on derivatives: optimization, implicit ODE solvers, Newton solvers, linearization, and various forms of sensitivity analysis.

Archimedes makes it easy to compute these derivatives efficiently and accurately by making use of CasADi's powerful automatic differentiation system:

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

# Hessian
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

(pytrees)=
## PyTrees

Many scientific computing functions assume that data is arranged into flat vectors.
This includes ODE integrators, optimizers, root-finders, and other solvers.
It is usually easier to mathematically represent the state of a system or a set of parameters as a vector.

However, this is not necessarily the most natural way to implement a physical model, which may have distinct components or hierarchical structure that makes transforming to/from a flat vector tedious and error-prone.

Archimedes addresses this with the concept of a PyTree; a data structure consisting of nested containers that can easily be flattened and unflattened, for instance:

```python
state = {"pos": np.array([0.0, 1.0, 2.0]), "vel": np.array([3.0, 4.0, 5.0])}
flat, unravel = arc.tree.ravel(state)
print(flat)  # [0. 1. 2. 3. 4. 5.]
print(unravel(flat)) # {'pos': array([0., 1., 2.]), 'vel': array([3., 4., 5.])}
```

A similar approach can work for containers nested to arbitrary depth, i.e. tuples of dictionaries of lists.
`NamedTuple` containers are also supported, and it is easy to define custom containers:

```python
from archimedes import struct

@struct.pytree_node
class Point:
    x: float
    y: float
    @property
    def dist(self):
        return np.sqrt(self.x **2 + self.y ** 2)

p = Point(1.0, 2.0)
flat, unravel = arc.tree.ravel(p)
print(flat)  # [1., 2.]
print(unravel(flat).dist)  # 2.23606797749979
```

For more information, see the section on tree-structured data in ["What is Archimedes?"](about.md#tree-structured-data) and ["Working with PyTrees"](pytrees.md).

(interpolation)=
## Interpolation

Basic one-dimensional interpolation can be done with the standard NumPy interface:

```python
xp = np.linspace(1, 6, 6)
fp = np.array([-1, -1, -2, -3, 0, 2])

@sym_function(kind="MX")
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

The Archimedes interpolant matches the SciPy one, but can be embedded in other symbolic functions, differentiated through, etc.

(control-flow)=
## Control flow (scan, where)

One limitation of the symbolic-numeric computations in Archimedes is that it is not compatible with standard Python control flow constructs like `if`/`else` and `while` loops.

<!-- TODO: finish this -->


(functional-programming)=
## Functional programming

Archimedes works by symbolically tracing your functions to create a computational graph, and then applying transformations to that graph or embedding it in more complex functions.
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

# The sym_function is forced to be pure
f_sym = arc.sym_function(f)
x = np.zeros(3)
y = f_sym(x)
print(y)  # [2. 0. 0.]
print(x)  # [0. 0. 0.]
```

In general, learning to work with pure functions can be an adjustment.
Fortunately, domains like math and physics are built on pure functions, and impure functions usually arise as an implementation detail (e.g. allocating work arrays in legacy Fortran code).
If you keep your code as close to the underlying mathematical model as possible, it will tend to be functionally pure.

(custom-callbacks)=
# Custom callbacks

<!-- TODO: Write this once the functionality is implemented. -->

(basic-troubleshooting)=
## Basic Troubleshooting

- **Error when creating arrays**: Remember to use `like=x` or functions like `vstack` with symbolic arguments
- **Unexpected results with conditional logic**: Use `np.where` instead of Python `if/else` with symbolic arguments
- **Poor performance**: Ensure you're reusing solver functions rather than recreating them for repeated calls