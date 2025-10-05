# Quickstart

This page will walk you through getting up and running with Archimedes, from installation and environment setup through a minimal example and some recommendations for navigating the rest of the documentation.

## Installation

You can install however you wish, but we recommend using a virtual environment like Conda or virtualenv to help keep track of package versions and dependencies.
We use [UV](https://github.com/astral-sh/uv) for development and recommend it as an environment as well:

```bash
# Create and activate a virtual environment 
uv venv
source .venv/bin/activate
```

Package installation can be done by installing from source (pending listing on PyPI):

```bash
# Download the source code
git clone https://github.com/jcallaham/archimedes.git
cd archimedes

# Install the package with development dependencies
uv pip install -e ".[all]"
```

If you are not using UV, simply omit `uv` in the final `pip install` command.

This setup will have you ready to use Archimedes in your own projects, or to modify or contribute to the source code.
If you want a more minimal installation, you can do `(uv) pip install .`, which will not install optional development-related dependencies.


## Simple example

```python
import numpy as np
import archimedes as arc

# Automatically convert NumPy code to efficient C++ with symbolic tracing
@arc.compile
def rotate(x, theta):
    # Rotate the vector x through the angle theta
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ], like=x)
    y = R @ x
    return y

# Evaluate as usual
x = np.array([1.0, 0.0])
theta = 0.1
print(rotate(x, theta))

# Transform to the Jacobian dy/dθ
J = arc.jac(rotate, argnums=1)
print(J(x, theta))
```

## Next steps

If you have not read about the fundamental concepts in Archimedes on [What is Archimedes?](about.md), you may want to start there.
Alternatively, you can skip to [Getting Started](getting-started.md) if you just want to jump right into coding.
Optionally, to get a deeper understanding of how Archimedes works (and therefore how to work with it), see [Under the Hood](under-the-hood.md).

Once you're comfortable with the basics, there are deep dives on [working with tree-structured data](trees.md), recommended [design patterns](tutorials/hierarchical/hierarchical00.md), and [quirks and gotchas](gotchas.md).

<!-- TODO: Add pointers to examples and API documentation once available -->