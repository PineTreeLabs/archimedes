# Archimedes

![Build Status](https://github.com/jcallaham/archimedes/actions/workflows/ci.yaml/badge.svg)
[![codecov](https://codecov.io/gh/jcallaham/archimedes/graph/badge.svg?token=37QNTHS42R)](https://codecov.io/gh/jcallaham/archimedes)

Archimedes is an open-source Python framework designed to simplify complex modeling and simulation tasks, with the ultimate goal of making it possible to do practical hardware engineering with Python.

### Key features

By building on CasADi, Archimedes provides a number of key features that make it a powerful tool for modeling and simulation:

* NumPy-compatible array API with automatic dispatch
* Efficient execution of computational graphs in compiled C++
* Automatic differentiation with forward- and reverse-mode sparse autodiff
* Interface to "plugin" solvers for ODE/DAEs, root-finding, and nonlinear programming
* Automated C code generation for embedded applications
* JAX-style function transformations
* PyTorch-style hierarchical data structures for parameters and dynamics modeling


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

To install the Jupyter notebook kernel, if you have installed `[all]` dependencies you can run

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=archimedes
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


# Testing and development

You can run a version of the CI test workflow locally as follows.

First, set up an environment using UV and installing development dependencies as described above.
Then you can run the basic unit tests with:

```bash
uv run pytest
```

We require 100% code coverage of the tests to help ensure reliability.  To print a test test coverage report to the terminal run

```bash
uv run pytest --cov=archimedes --cov-report=term-missing
```

Alternatively, generate a detailed report with

```bash
uv run pytest --cov=archimedes --cov-report=html
```

Check that the notebooks run with

```bash
uv run pytest --nbmake docs/source/notebooks/*.ipynb
```

Linting and formatting is done with [ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff check src test docs
uv run ruff format --check src test docs
```

Finally, to build the documentation locally, run

```bash
cd docs
make clean && make nbconvert && make html
```

This will scrape API documentation from the docstrings, convert Jupyter notebooks to Markdown files, and then create the HTML website from the outputs.

## Security scanning

First, scan the project requirements for known vulnerabilities:

```bash
uv export --no-emit-project --format requirements-txt > requirements.txt
uv run pip-audit -r requirements.txt --disable-pip
rm requirements.txt
```

Then run [Bandit](https://bandit.readthedocs.io/) to do a static analysis of the Archimides code itself:

```bash
uv run bandit -r src
```


# Current status

Archimedes is still in active development, and the API is subject to change.  We are actively seeking feedback from users to help shape the direction of the project.  Key planned directions include hardware support (e.g. HIL testing) and extending physics modeling capabilities.  Please don't hesitate to reach out to us if you have any questions or feedback.
