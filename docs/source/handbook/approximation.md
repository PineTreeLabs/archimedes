---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: archimedes
---

```{code-cell} python
:tags: [hide-cell]
# ruff: noqa: N802, N803, N806, N815, N816

import matplotlib.pyplot as plt
import numpy as np

import archimedes as arc
```

```{code-cell} python
:tags: [remove-cell]
from pathlib import Path

plot_dir = Path.cwd() / "_plots"
plot_dir.mkdir(exist_ok=True)
```

# Function Approximation

This page gives an overview of the infrastructure that Archimedes provides for _function approximation with linear basis expansions_.
All of the methods we will consider approximate methods of the form:

```{math}
f(x) \approx \hat{f}(x) \equiv \sum_{i=1}^n a_i \phi_i(x)
```

With different definitions of the basis, this can encompass a fairly wide variety of numerical methods, from simple lookup tables and power series expansions to complex representations like discontinuous-Galerkin finite element methods and pseudospectral optimal control.

<!-- TODO: Callout on other function approximation: RBF, GPR, DNN -->

The design of the function approximation system in Archimedes is heavily influenced by other standout packages, most notably [chebfun](TODO) and FEniCS/Firedrake's [Unified Form Language](TODO) but also [shenfun](TODO) and [ApproxFun.jl](TODO).
What these have in common is that the "function" is a first-class citizen, freeing you to think about what you are modeling instead of bookkeeping nodes, weights, local-to-global maps, and assembly.

**So why write a new function approximation module?** Fair question, and Archimedes doesn't claim to beat any of these on their home turf - chebfun and ApproxFun.jl for adaptive, machine-precision work, shenfun for large-scale pseudospectral CFD, and FEniCS/Firedrake for serious FEM work.

What's new here is _composition_ with the other Archimedes capabilities:
* Write a pseudospectral trajectory optimization code using a [hierarchical dynamics model](../tutorials/hierarchical/hierarchical00.md) and solve the NLP using sparse autodiff
* Fit a custom black-box component of a physics-constrained dynamics model using [system identification](../tutorials/sysid/parameter-estimation.md) and then deploy a real-time feedforward compensation controller to hardware using [C code generation](../tutorials/codegen/codegen00.md)
* Approximate continuum mechanics degrees of freedom with a basis expansion, then form the Lagrangian and derive equations of motion using autodiff
* Optimize under uncertainty using a differentiable polynomial chaos expansion to model random variables without Monte Carlo

None of these requires working with inflexible pre-rolled APIs that don't *quite* do what you want; the system is designed to be composable building blocks in the vein of NumPy/SciPy, giving you the freedom to do whatever you want in the spirit of Archimedes' "Made for Hackers" philosophy.

## Function Approximation in Brief

This is a deep topic, and we will barely scratch the surface here, but the abstract math is not overly complicated and helps to understand the software design.
Please forgive any imprecision in the math; this is made by and for engineers.

### Function Spaces

The core abstraction is a _function space_, $\mathcal{V}$ with an inner product $\langle \cdot, \cdot \rangle_\mathcal{V}$.
Specifically, every example we'll look at uses a weighted inner product on a domain $\mathcal{D}$, that is:

```{math}
\langle u, v \rangle_\mathcal{V} = \int_\mathcal{D} u(x) ~ v(x) ~ w(x) ~ dx
```

for a weight function $w(x)$ that depends on the particular function space.

<!-- TODO: Callout for explaining working with scalars -->

Note that this combination of a domain and weight function is exactly the [`Measure`](#archimedes.measure.Measure) that underpins [orthogonal polynomials and Gaussian quadrature](quadrature.md).
In fact, the integrals are implemented the quadrature rules supplied by the [`quadrature`](#archimedes.quadrature) module.

That inner product is the key concept; basically everything else is either a consequence of different choices of inner product or of basis functions.

For example, if we work with a basis of functions $\phi_i(x)$ that are orthogonal with respect to $w(x)$, then by definition

```{math}
\langle \phi_i, \phi_j \rangle_\mathcal{V} = ||phi_i(x)||^2_\mathcal{V} \delta_{ij},
```

where $\delta_{ij}$ is the Kroenecker delta, and

```{math}
||phi_i(x)||^2 \equiv \int_\mathcal{D} \phi_i^2(x) ~ w(x) ~ dx.
```

In this case we can easily extract the expansion coefficients $a_i$ for any function $f(x)$:

```{math}
a_i = \frac{\langle \phi_i, f \rangle_\mathcal{V}}{||phi_i(x)||^2}.
```

To make this example more concrete, if $\mathcal{D}$ is periodic and $\phi_i(x)$ are Fourier (sine/cosine) functions with $w(x) \equiv 1$, then this is the familiar spectral projection method taught in undergrad math methods classes.

### Galerkin Projection

In general the basis functions may not be orthogonal, in which case the coefficient vector $\mathbf{a}$ for a function can be determined by _Galerkin projection_, requiring that the approximation residual not lie in the span of the basis vectors.
This residual-orthogonality condition is that for any "test" function $\phi_j(x)$,

```{math}
\langle \phi_j, \hat{f} - f \rangle_\mathcal{V} = 0,
```

where $\hat{f}(x)$ is again the basis expansion approximation $\sum_i a_i \phi_i(x)$.

This can be equivalently written as a least-squares problem:

```{math}
\min_\mathbf{a} \bigg| \bigg| \hat{f} - f \bigg| \bigg|_\mathcal{V}^2.
```

Later we will see how this can be implemented with basic matrix multiplications that scale with the number of basis functions.

### Petrov-Galerkin Projection

As an aside (not critical for what follows), note that it is also possible to draw the test functions from a different basis $\mathcal{W}$ than the expansion basis (although the weight functions should be the same).
This is known as _Petrov-Galerkin projection_, and the condition is that for any $\psi \in \mathcal{W}$:

```{math}
\langle \psi_j, \hat{f} - f \rangle_\mathcal{W} = 0.
```

### From Continuous to Discrete

<!-- TODO -->

## The `approximation` Module

The design of the [`approximation`](#archimedes.experimental.approximation) module follows directly from the theory above.

There are four key abstractions:

- [`Basis`](#archimedes.experimental.approximation.Basis): the definition of the $\phi(x)$ functions
- [`FunctionSpace`](#archimedes.experimental.approximation.FunctionSpace): combination of a basis with 
- [`BasisMatrix`](#archimedes.experimental.approximation.BasisMatrix)
- [`Function`](#archimedes.experimental.approximation.Function)