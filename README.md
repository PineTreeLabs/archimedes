# Archimedes

Archimedes is an open-source Python framework designed to simplify complex modeling and simulation tasks. By leveraging the power of CasADi, a symbolic framework for automatic differentiation and numerical optimization, Archimedes allows engineers and researchers to implement complex models using familiar NumPy syntax while gaining significant performance improvements and advanced capabilities.

This module hides the CasADi symbolics behind an array API that dispatches from NumPy functions, along with JAX-style composable function transformations. This means that you can write (almost) normal NumPy functions, add a decorator, and get a "compiled" function that executes in pure C++, supports automatic differentiation, and can be evaluated either numerically or symbolically.

### Key features

By building on CasADi, Archimedes provides a number of key features that make it a powerful tool for modeling and simulation:

- Efficient execution of computational graphs in compiled C++
- Automatic differentiation: support for forward- and reverse-mode sparse autodiff
- Interface to "plugin" solvers for ODE/DAEs, root-finding problems, and nonlinear programming.
- Automated C code generation

The key new element of Archimedes is tying the CasADi symbolic framework to the NumPy array API:

- Symbolically evaluate NumPy code using automated dispatch
- Compile and differentiate pure NumPy functions with JAX-style transformations (e.g. `grad`, `scan`, etc.)
- Work with nested tree-like data structures ("PyTrees" in JAX lingo)

# Installation

### Basic setup

Archimedes is currently not available on PyPI (pending resolution of an existing defunct project on PyPI also named Archimedes), but can be installed from source:

```bash
git clone https://github.com/jcallaham/archimedes.git
cd archimedes
pip install .
```

### Recommended development setup

For development, we recommend using [UV](https://docs.astral.sh/uv/) for faster dependency resolution and virtual environment management:

```bash
# Create and activate a virtual environment 
uv venv
source .venv/bin/activate

git clone https://github.com/jcallaham/archimedes.git
cd archimedes

# Install the package with development dependencies
uv pip install -e ".[all]"
```

# Examples

### Automatic differentiation

```python
def f(x):
    return np.sin(x**2)

df = arc.grad(f)
np.allclose(df(1.0), 2.0 * np.cos(1.0))
```

### ODE solving with SUNDIALS

```python
import numpy as np
import archimedes as arc


# Lotka-Volterra model
def f(t, x):
    a, b, c, d = 1.5, 1.0, 1.0, 3.0
    return np.hstack([
        a * x[0] - b * x[0] * x[1],
        c * x[0] * x[1] - d * x[1],
    ])


x0 = np.array([1.0, 1.0])
t_span = (0.0, 10.0)
t_eval = np.linspace(*t_span, 100)

xs = arc.odeint(f, t_span=t_span, x0=x0, t_eval=t_eval)
```

### Constrained optimization

The [constrained Rosenbrock problem](https://en.wikipedia.org/wiki/Test_functions_for_optimization) has a local minimum at (0, 0) and a global minimum at (1, 1)

```python
import numpy as np
import archimedes as arc

def f(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

def g(x):
    g1 = (x[0] - 1) ** 3 - x[1] + 1
    g2 = x[0] + x[1] - 2
    return np.hstack([g1, g2])

x_opt = arc.minimize(f, constr=g, x0=[2.0, 0.0], constr_bounds=(-np.inf, 0))
print(np.allclose(x_opt, [1.0, 1.0], atol=1e-3))
```

### Extended examples

- [Trajectory optimization with feedforward stabilization](examples/cartpole/finite-horizon.ipynb) (Work in progress)
- [System identification with nonlinear Kalman filters](examples/cartpole/sysid.ipynb) (Work in progress)
- [Multirotor vehicle dynamics](examples/multirotor/quadrotor.ipynb)
- [Pressure-fed rocket engine](examples/draco/draco-model.ipynb)
- [Subsonic F-16 benchmark](examples/f16/f16_plant.py)
- [Adaptive optimal control with pseudospectral collocation](examples/coco/)


# Current status

Archimedes is still in active development, and the API is subject to change.  e are actively seeking feedback from users to help shape the direction of the project.

Key planned directions include hardware support (e.g. HIL testing) and extending physics modeling capabilities.  Please reach out to us if you have any questions or feedback.

# Gotchas
### Array creation

Most NumPy functions can be used out of the box and will dispatch to the corresponding symbolic operation.
However, this won't work:

```python
def f(x):
    return np.array([x[1], x[0]])
```

and neither will this:

```python
def f(x):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = x[i]
    return y
```

In both cases NumPy dispatch doesn't know it should be creating `SymbolicArray` objects. In the first case, NumPy will create an `ndarray` of type `object` full of the two indexed `SymbolicArray`s.  In the second case, it will throw an error since the `SymbolicArray` can't be converted to a float.

The solution to this is to call `archimedes.array` and similar functions, which will create either a symbolic or numeric array depending on the type of the inputs.  `np.array` can only be called if the array will be "static" - that is, will never contain any symbolic components.

All of the following are valid and will work with either symbolic or numeric inputs, however:

```python
import archimedes as arc

# OK: like=x ensures that the array creation dispatches to the symbolic version when x is symbolic
def f1(x):
    return np.array([x[1], x[0]], like=x)

# OK: this is a dispatched function
def f2(x):
    return np.hstack([x[1], x[0]])

# OK: the `array` function can handle either NumPy or symbolic inputs
def f3(x):
    return arc.array([x[1], x[0]])

# OK: zeros_like is a dispatched function
def f4(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i] = x[i]
    return y

# OK: like=x ensures that the array creation dispatches to the symbolic version when x is symbolic
def f5(x):
    y = np.zeros(4, like=x)
    for i in range(y.size):
        y[i] = i * sum(x)
    return y
```

### Control flow

Some familiar control flow constructs in Python are not supported in symbolic evaluations.  For example, the following will fail:

```python
@arc.sym_function
def f(x):
    if x > 0:
        return x
    else:
        return 0
```

The reason for this is that `x > 0` is a symbolic expression, and symbolic expressions can't be evaluated in a boolean context.  An easy workaround for this is to use `np.where` to implement a conditional:

```python
@arc.sym_function
def f(x):
    return np.where(x > 0, x, 0)
```

Similarly, loop bounds must be "static" - that is, they can't be symbolic expressions.  For example, the following will fail:

```python
@arc.sym_function
def f(x):
    y = 0
    for i in range(sum(x)):
        y += i
    return y
```

As with the conditional, `sum(x)` is a symbolic expression that can't be evaluated in a loop bound.
However, array sizes and other constants can be used in loop bounds.  For example, the following will work:

```python
@arc.sym_function
def f(x):
    y = 0
    for i in range(x.shape[0]):
        y += i
    return y
```

For repeated evaluations of a function, it is recommended to use `arc.scan` to efficiently construct the computational graph (see for example the [implementation of the prediction error method](archimedes/experimental/sysid/pem.py)).


### Backend conversion
Conversion back and forth between `casadi.DM` and `np.ndarray` is expensive (see ODE example).  For cases where the call will happen many times, conversion should be avoided via CasADi symbolic primitives like `integrator`.


### Symbolic function evaluation

Keyword arguments are supported, but all function arguments must be allowed to be either positional or keyword.  In addition, all arguments must be defined explicitly.  That is, the following signatures are valid:

```
def f1(x):
    return np.sin(x)

sym_function(f1)  # OK, positional args are supported

def f2(x, a=2.0):
    return np.sin(a * x)

sym_function(f2)  # OK, kwargs are supported
```

but positional-only, keyword-only, varargs, etc. are not allowed.  The following signatures are therefore invalid:

```
def f3(x, /):
    return np.sin(x)

sym_function(f3)  # Positional-only not allowed

def f4(x, *, a=2.0):
    return np.sin(a * x)

sym_function(f4)  # Keyword-only not allowed

def f6(x, *params):
    return sum(params) * x

sym_function(f5)  # Variadic args not allowed

def f6(x, **kwargs):
    return kwargs["A"] * np.sin(kwargs["f"] * x)

sym_function(f6)  # Variadic kwargs not allowed
```

Note that this requirement only applies to the top-level function that is evaluated symbolically.  So for example, the `**kwargs` example `f6` above could be called from another function:

```
def g(x, A=2.0, f=np.pi):
    return f6(x, A=A, f=f)

sym_function(g)   # OK, all arguments are defined explicitly at top level
```

<!-- 

# Target demos

### Minimal

- [x] Accelerated ODE solves and optimization vs SciPy
- [x] C code generation for Arduino (IIR filter design in SciPy)
- [ ] Simulink-style block diagram?

### Basic

- [ ] Hybrid dynamics modeling (collisions)
- [ ] Modelica-style physical modeling (mass-spring-damper or RC circuit)
- [ ] Optimal control via direct collocation (CartPole swing-up)

### Advanced

- [ ] Indirect (Pontryagin) optimal control (orbit transfer)
- [x] Optimal control via pseudospectral collocation
- [x] Neural network training
- [ ] Embedded MPC/LQG deployment
- [ ] Rigid body mechanics and multibody systems (Featherstone algorithms?)
- [ ] UQ with polynomial chaos
- [ ] Parameter estimation
- [ ] 1D FEA models (structural, thermal, fluid, beam equations)
- [ ] Model reduction

### Future

- [ ] PDE systems with UFL + DOLFINx
- [ ] Reinforcement learning
- [ ] Requirements tracking

### Applications

- [ ] Rocket engine
- [ ] Battery model
- [ ] Quadrotor (multifidelity aero models)
- [ ] Walking robot (MuJoCo "humanoid")
- [ ] Manipulator robot (iiwa?)

-->


# Testing

For a test coverage report, run `pytest --cov=archimedes --cov-report=html`

To print the coverage report to the terminal, run `pytest --cov=archimedes --cov-report=term-missing`