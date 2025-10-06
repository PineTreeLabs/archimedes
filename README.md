# Archimedes

![Build Status](https://github.com/pinetreelabs/archimedes/actions/workflows/ci.yaml/badge.svg)
![Security Scan](https://github.com/pinetreelabs/archimedes/actions/workflows/security.yaml/badge.svg)
[![codecov](https://codecov.io/gh/pinetreelabs/archimedes/graph/badge.svg?token=37QNTHS42R)](https://codecov.io/gh/pinetreelabs/archimedes)
[![REUSE status](https://api.reuse.software/badge/github.com/PineTreeLabs/archimedes)](https://api.reuse.software/info/github.com/PineTreeLabs/archimedes)

**Archimedes** is an open-source Python framework designed to simplify the development of complex engineering systems by providing tools for **modeling**, **simulation**, **optimization**, **controls**, and **hardware deployment**. 
The ultimate goal is to make it possible to do practical
hardware engineering with Python.

For more details, see the [documentation site](https://pinetreelabs.github.io/archimedes/)

### Key features

By combining the powerful symbolic capabilities of [CasADi](https://web.casadi.org/docs/) with the intuitive interface designs of NumPy, PyTorch, and JAX, Archimedes provides a number of key features:

* NumPy-compatible array API with automatic dispatch
* Efficient execution of computational graphs in compiled C++
* Automatic differentiation with forward- and reverse-mode sparse autodiff
* Interface to "plugin" solvers for ODE/DAEs, root-finding, and nonlinear programming
* Automated C code generation for embedded applications
* JAX-style function transformations
* PyTorch-style hierarchical data structures for parameters and dynamics modeling

**⚠️ API Stability Notice ⚠️**: Archimedes is currently pre-1.0 software. The API is still evolving and may change between minor versions. We'll document breaking changes in the changelog and will follow semantic versioning for 0.X releases, but expect to see some instability until version 1.0 is released.

# Examples

### Automatic differentiation

```python
import numpy as np
import archimedes as arc

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

### C code generation

Archimedes can convert plain NumPy functions to standalone C code for use in embedded applications:

```python
import numpy as np
import archimedes as arc

def f(x, y):
    return x + np.sin(y)

# Create templates with appropriate shapes and dtypes
x_type = np.zeros((), dtype=float)
y_type = np.zeros((2,), dtype=float)

arc.codegen(f, (x_type, y_type), return_names=("z", ))
```

For more details, see the tutorial on [C code generation](https://pinetreelabs.github.io/archimedes/tutorials/codegen/codegen00.html)

### Tutorials

- [Hierarchical systems modeling](https://pinetreelabs.github.io/archimedes/tutorials/hierarchical/hierarchical00.html)
- [C code generation](https://pinetreelabs.github.io/archimedes/tutorials/codegen/codegen00.html)
- [Nonlinear system identification](https://pinetreelabs.github.io/archimedes/tutorials/sysid/parameter-estimation.html)
- [Hardware development workflow](https://pinetreelabs.github.io/archimedes/tutorials/deployment/deployment00.html)
<!-- - [Multirotor vehicle dynamics](https://pinetreelabs.github.io/archimedes/tutorials/multirotor/multirotor00.html) -->
<!-- - [Pressure-fed rocket engine](examples/draco/draco-model.ipynb) -->
<!-- - [Adaptive optimal control with pseudospectral collocation](examples/coco/) -->
<!-- - [Subsonic F-16 benchmark](examples/f16/f16_plant.py) (Work in progress) -->
<!-- - [CartPole control](examples/cartpole/finite-horizon.ipynb) (Work in progress) -->

### Examples

The [examples folder](examples) includes some examples that are not as well-documented as those on the website, but showcase some additional functionality in different application domains.
These include:

- [Multirotor vehicle dynamics](examples/multirotor)
- [CartPole system ID + control](examples/cartpole)
- [Subsonic F-16 benchmark](examples/f16)
- [Trajectory optimization](examples/trajopt/)

# Installation

### Basic setup

The easiest way to install is from PyPI:

```bash
pip install archimedes
```

Test with any of the examples shown above.

### Recommended setup

For development (or just a more robust environment configuration), we recommend using [UV](https://docs.astral.sh/uv/) for faster dependency resolution and virtual environment management:

```bash
# Create and activate a virtual environment 
uv venv
source .venv/bin/activate

# Install with minimal dependencies
uv pip install archimedes

# OR install with extras (control, jupyter, matplotlib, etc.)
uv pip install archimedes[all]
```

To install a Jupyter notebook kernel, if you have installed the additional dependencies with `[all]` you can run:

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=archimedes
```

This will create a kernel named `archimedes` - you can change the name to whatever you'd like.

### Source installation

To install from source locally (e.g. for development or building the docs), the recommended procedure is to create a UV virtual environment as described above and then run:

```bash
git clone https://github.com/pinetreelabs/archimedes.git
cd archimedes

# Install the package with development dependencies
uv pip install -e ".[all]"
uv sync --all-extras
```

# Testing and development

Development standards outlined in the [testing guide](dev/TESTING.md) include:

- 100% code coverage
- Ruff formatting
- MyPy static type checking
- Vulnerability scanning with `pip-audit`
- Static analysis for security issues using [Bandit](https://bandit.readthedocs.io/)
- Licensing compliance with [REUSE](https://reuse.software/)

# Licensing

Archimedes is available under a dual-licensing model:

The open-source code is licensed under the [GNU General Public License v3.0](LICENSE).
For organizations that would like more flexible licensing contact [info@archimedes.sh](mailto:info@archimedes.sh) for details.

## Third-Party Components
Archimedes incorporates code from several open source projects, including JAX (Apache 2.0), Flax (Apache 2.0), SciPy (BSD-3), and NumPy (NumPy license). See [NOTICE.md](NOTICE.md) for a complete list of attributions, including licenses for key dependencies (CasADi and NumPy).

# Citing Archimedes

At this time Archimedes does not have a DOI-linked publication, though a draft is in progress.
Feel free to link to the repository in the meantime.

If you use Archimedes in published work, please also consider citing [CasADi](https://web.casadi.org/), the symbolic-numeric backend for Archimedes:

```raw
@Article{Andersson2018,
  Author = {Joel A E Andersson and Joris Gillis and Greg Horn
            and James B Rawlings and Moritz Diehl},
  Title = {{CasADi} -- {A} software framework for nonlinear optimization
           and optimal control},
  Journal = {Mathematical Programming Computation},
  Year = {2018},
}
```

# Getting involved

We're excited to build a community around Archimedes - here's how you can get involved at this stage:

- **⭐ Star the Repository**: The simplest way to show support and help others discover the project
- **🐛 Report Issues**: Detailed bug reports, documentation gaps, and feature requests are invaluable
- **💬 Join Discussions**: Share your use cases, ask questions, or provide feedback in our [GitHub Discussions](github.com/pinetreelabs/archimedes/discussions)
- **🗞️ Stay In the Loop**: [Subscribe](https://jaredcallaham.substack.com/embed) to the newsletter for updates and announcements
- **📢 Spread the Word**: Tell colleagues, mention us in relevant forums, or share on social media
- **📝 Document Use Cases**: Share how you're using (or planning to use) Archimedes

## Contributing

At this early stage of development:

- **👍 We welcome issue reports** with specific bugs, documentation improvements, and feature requests
- **⏳ We are not currently accepting pull requests** as we establish the project's foundation and architecture
- **❓ We encourage discussions** about potential applications, implementation questions, and project direction

If you've built something with Archimedes or are planning to, we definitely want to hear about it! Your real-world use cases directly inform our development priorities.

We appreciate your interest and support!