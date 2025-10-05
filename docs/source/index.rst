
.. ==========================
.. Archimedes
.. ==========================

.. image:: _static/rocket.png
   :alt: Archimedes Rocket Logo
   :align: center

-----------------

**Archimedes** is an open-source Python framework designed to simplify complex modeling
and simulation tasks, with the ultimate goal of making it possible to do practical
hardware engineering with Python. It builds on the powerful symbolic computation
capabilities of `CasADi <https://web.casadi.org/docs/>`_ with an interface designed to
be familiar to NumPy users.

.. grid:: 2

    .. grid-item-card:: 📚 Quickstart
        :link: quickstart
        :link-type: doc
        
        Learn how to use Archimedes
    
    .. grid-item-card:: 📝 Blog
        :link: blog/index
        :link-type: doc
        
        Announcements, deep dives, and case studies

**Key features**:

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
   arc.codegen(df, "grad_f.c", template_args, return_names=("z",))

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   quickstart
   about
   getting-started


.. toctree::
   :maxdepth: 1
   :caption: Resources

   blog/index

.. toctree::
   :maxdepth: 1
   :caption: Core Concepts

   under-the-hood
   trees
   control-flow

.. toctree::
   :maxdepth: 1
   :caption: Applications

   tutorials/hierarchical/hierarchical00
   tutorials/sysid/parameter-estimation
   tutorials/codegen/codegen00
   tutorials/deployment/deployment00
   tutorials/multirotor/multirotor00

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