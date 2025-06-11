
==========================
Archimedes
==========================

**Archimedes** is an open-source Python framework designed to simplify complex modeling
and simulation tasks, with the ultimate goal of making it possible to do practical
hardware engineering with Python. It builds on the powerful symbolic computation
capabilities of `CasADi <https://web.casadi.org/docs/>`_ with an interface designed to
be familiar to NumPy users.

Key features:

* NumPy-compatible array API with automatic dispatch
* Efficient execution of computational graphs in compiled C++
* Automatic differentiation with forward- and reverse-mode sparse autodiff
* Interface to "plugin" solvers for ODE/DAEs, root-finding, and nonlinear programming
* Automated C code generation for embedded applications
* JAX-style function transformations
* PyTorch-style hierarchical data structures for parameters and dynamics modeling


Quick Example
-------------

.. code-block:: python

   import numpy as np
   import archimedes as arc

   def f(x):
       return np.sin(x**2)

   # Use automatic differentiation to compute df/dx
   df = arc.grad(f)
   np.allclose(df(1.0), 2.0 * np.cos(1.0))  # True

   # Automatically generate C code
   template_args = (0.0,)  # Data type for C function
   arc.codegen(df, "grad_f.c", template_args)

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   quickstart
   about
   getting-started

.. toctree::
   :maxdepth: 1
   :caption: Core Concepts

   under-the-hood
   pytrees
   generated/notebooks/control-flow
   generated/notebooks/modular-design

.. toctree::
   :maxdepth: 1
   :caption: Examples

   notebooks/multirotor/multirotor00
   notebooks/deployment/deployment00
   notebooks/sysid/sysid00

.. toctree::
   :maxdepth: 1
   :caption: Reference

   gotchas
   reference/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   roadmap
   community
   licensing-model


Indices and Tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`