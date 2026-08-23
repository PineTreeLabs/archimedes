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
from archimedes.experimental.approximation import FunctionSpace
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
f(x) \approx \hat{f}(x) \equiv \sum_{i=1}^n c_i \phi_i(x)
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
\langle \phi_i, \phi_j \rangle_\mathcal{V} = \|\phi_i(x)\|^2_\mathcal{V} \delta_{ij},
```

where $\delta_{ij}$ is the Kronecker delta, and

```{math}
\|\phi_i(x)\|^2 \equiv \int_\mathcal{D} \phi_i^2(x) ~ w(x) ~ dx.
```

In this case we can easily extract the expansion coefficients $a_i$ for any function $f(x)$:

```{math}
c_i = \frac{\langle \phi_i, f \rangle_\mathcal{V}}{\|\phi_i(x)\|^2}.
```

To make this example more concrete, if $\mathcal{D}$ is periodic and $\phi_i(x)$ are Fourier (sine/cosine) functions with $w(x) \equiv 1$, then this is the familiar spectral projection method taught in undergrad math methods classes.

### Galerkin Projection

In general the basis functions may not be orthogonal, in which case the coefficient vector $\mathbf{c}$ for a function can be determined by _Galerkin projection_, requiring that the approximation residual not lie in the span of the basis vectors.
This residual-orthogonality condition is that for any "test" function $\phi_j(x)$,

```{math}
\langle \phi_j, \hat{f} - f \rangle_\mathcal{V} = 0,
```

where $\hat{f}(x)$ is again the basis expansion approximation $\sum_i c_i \phi_i(x)$.

This can be equivalently written as a least-squares problem:

```{math}
\min_\mathbf{c} \bigg| \bigg| \hat{f} - f \bigg| \bigg|_\mathcal{V}^2.
```

Later we will see how this can be implemented with basic matrix multiplications that scale with the number of basis functions.

### Petrov-Galerkin Projection

As an aside (not critical for what follows), note that it is also possible to draw the test functions from a different basis $\mathcal{W}$ than the expansion basis (although the weight functions should be the same).
This is known as _Petrov-Galerkin projection_, and the condition is that for any $\psi \in \mathcal{W}$:

```{math}
\langle \psi_j, \hat{f} - f \rangle_\mathcal{W} = 0.
```

### From Continuous to Discrete

The inner product is thus a key calculation when working with basis expansions because of its role in finding an L2 projection of an arbitrary function.
Of course, the inner product as a (possibly weighted) integral on the domain typically has no useful closed-form solution, so in practice we must resort to [numerical quadrature](quadrature.md).

Gaussian quadrature approximates the integral with a weighted sum over a fixed set of $m$ nodes $\{x_i\}_{i=1}^m$ and weights $\{w_i\}_{i=1}^m$, chosen so the approximation is exact for the polynomial degrees that appear in the basis:

```{math}
\langle u, v \rangle_\mathcal{V} = \int_\mathcal{D} u(x) \, v(x) \, w(x) \, dx \approx \sum_{i=1}^m w_i \, u(x_i) \, v(x_i).
```

If we construct vectors $\mathbf{u}$ by sampling the continuous function $u(x)$ at the quadrature nodes $\mathbf{x}$, then Gaussian quadrature turns the inner product into a diagonally-weighted dot product of the two vectors $\mathbf{u}$ and $\mathbf{v}$.
In the event that the basis is polynomial and quadrature rule is chosen to exactly integrate any polynomials that appear in the basis, the quadrature is actually a _numerically exact_ evaluation of the inner product.

This lets us introduce a key construction: the basis matrix (also known as a "generalized Vandermonde matrix") $\boldsymbol{\Phi}$, constructed so that the $j$-th column is $\phi_j$ evaluated at the Gaussian quadrature nodes:

```{math}
\boldsymbol{\Phi}_{ij} = \phi_j(x_i), \qquad i = 1, \dots, m, \quad j = 1, \dots, n.
```

:::{note}
The ordinary [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix) is the basis matrix for the special case of a monomial basis $\phi_j(x) = x^{j-1}$
:::

The basis matrix uses the _continuous_ basis functions sampled at the _discrete_ quadrature nodes, so it is the bridge between the whiteboard theory and the numerical implementation.
The basis matrix and the diagonal weight matrix $\mathbf{W} = \operatorname{diag}(w_1, \dots, w_m)$ allow us to conveniently evaluate a range of operations on basis expansion representations of continuous functions.

For example, the Galerkin projection condition $\langle \phi_j, \hat{f} - f \rangle_\mathcal{V} = 0$ for every $j$ can be compactly expressed as the square linear system:

```{math}
\boldsymbol{\Phi}^\top \mathbf{W} \boldsymbol{\Phi} \mathbf{c} = \boldsymbol{\Phi}^\top \mathbf{W} \mathbf{f},
```

where $f_i \equiv f(x_i)$ is again the continuous function $f$ sampled at the quadrature points and $\mathbf{c}$ is the coefficient vector.

Here $\boldsymbol{\Phi}^\top \mathbf{W} \boldsymbol{\Phi}$ is the Gram (or "mass") matrix $\mathbf{M}$ consisting of inner products between two basis functions:

```{math}
M_{ij} = \langle \phi_i, \phi_j \rangle_\mathcal{V},
```

The right-hand side of the projection equation, $\boldsymbol{\Phi}^\top \mathbf{W} \mathbf{f}$, is the inner product between $f$ and each element of the basis, which (because of its appearance in linear finite element methods) is sometimes called the _load vector_ $\mathbf{b}$:

```{math}
b_j = \langle \phi_j, f \rangle_\mathcal{V}.
```

Galerkin projection is then simply the solution to the linear system

```{math}
\mathbf{M} \mathbf{c} = \mathbf{b},
```

which are exactly the normal equations for a weighted least-squares projection.

## The `approximation` Module

The design of the [`approximation`](#archimedes.experimental.approximation) module follows directly from the theory above.

There are four key abstractions:

- [`Basis`](#archimedes.experimental.approximation.Basis): the definition of the $\phi(x)$ functions
- [`FunctionSpace`](#archimedes.experimental.approximation.FunctionSpace): combination of a basis with a domain and associated quadrature rule, together implying an inner product
- [`BasisMatrix`](#archimedes.experimental.approximation.BasisMatrix): the generalized Vandermonde matrix $\boldsymbol{\Phi}$ associated with the basis and quadrature rule
- [`Function`](#archimedes.experimental.approximation.Function): A coefficient vector for a particular element of a function space, defining a (piecewise) continuous function in terms of a basis expansion.

These are summarized in the following table

| Math concept    | Math notation | Code equivalent | Code convention |
| --------------- | --------------| --------------- | --------------- |
| Basis functions | $\{ \phi_i(x) \}_{i=1}^n$ | `Basis` | `phi` |
| Function space  | $\operatorname{span}\{\phi_1, \dots, \phi_n\}$ | `FunctionSpace` | `V` |
| Generalized Vandermonde matrix | $\Phi_{ij} = \phi_j(x_i)$ | `BasisMatrix` | `Phi` |
| Function | $f(x) = \sum_{i=1}^n c_i \phi_i(x)$ | `Function` | `f` |

### Math to Code

The operations in [From Continuous to Discrete](#from-continuous-to-discrete) map onto this module in two layers: matrix-level assembly primitives that directly use `BasisMatrix`, and higher-level `Function` operations that use them internally.

#### Assembly Primitives

`BasisMatrix.T` is defined as the adjoint under the *weighted* inner product, not a plain transpose.
This avoids the need to manually track quadrature weights $W$ so that for instance `M = Phi.T @ Phi`, computed as $\Phi^\top W \Phi$, and `b = Phi.T @ f(x)`.

| Math concept | Math notation | Code |
| --- | --- | --- |
| Basis matrix | $[\Phi]_{ij} = \phi_j(x_i)$ | `Phi = space.basis_matrix()` |
| Derivative basis matrix | $[\Phi']_{ij} = \phi_j'(x_i)$ | `dPhi = space.basis_matrix(deriv=1)` |
| Mass (Gram) matrix | $M_{jk} = \langle \phi_j, \phi_k \rangle_\mathcal{V}$ | `Phi.T @ Phi` |
| Stiffness matrix | $K_{jk} = \langle \phi_j', \phi_k' \rangle_\mathcal{V}$ | `dPhi.T @ dPhi` |
| Load vector | $b_j = \langle \phi_j, f \rangle_\mathcal{V}$ | `Phi.T @ f(x)` |

These are commonly used in PDE discretizations (e.g. spectral or finite element methods), but many function approximation applications don't require directly using these matrix-level operations at all:

#### `Function` Operations

| Math concept | Math notation | Code |
| --- | --- | --- |
| Inner product | $\langle u, v \rangle_\mathcal{V} = \int_\mathcal{D} u \, v \, w \, dx$ | `f.dot(g)` |
| Galerkin projection | $\min_\mathbf{c} \lVert \hat{f} - f \rVert_\mathcal{V}^2$ | `space.project(f)` |
| Petrov-Galerkin projection | $\langle \psi_j, \hat{f} - f \rangle_\mathcal{W} = 0, ~ \psi_j \in \mathcal{W}$ | `space.project(f, test_space=Psi)` |
| Differentiation | $f'(x) = \sum_i c_i \phi_i'(x)$ | `f.derivative()` |
| Antiderivative | $F(x) = \int_a^x f(t) \, dt$ | `f.integral()` |
| Definite integral | $\int_a^b f(x) \, dx$ | `f.integrate()` |

### Basis Families

All of the math and code abstractions above work for any finite-dimensional basis $\{\phi_i\}_{i=1}^n$.
The main choice in function approximation is choosing an appropriate basis for the problem.

The linear-basis-expansion representation is surprisingly broad, including for example:

- Monomials ($1, x, x^2, \dots, x^{n-1}$)
- Orthogonal polynomials (Legendre, Hermite, Laguerre, etc.)
- Lagrange polynomials (nodal representation)
- Fourier series expansions
- B-splines
- Lookup tables (piecewise linear basis)
- Cubic Hermite polynomials
- Piecewise finite/spectral element bases
- Multivariate tensor product bases

`FunctionSpace` provides constructors for these common basis families, or you can write your own custom basis and pass it to the general `FunctionSpace(basis, domain, quad_rule=...)` constructor.

| Family | Constructor | Domain | Weight $w(x)$ | Notes |
| --- | --- | --- | --- | --- |
| Legendre | `FunctionSpace.legendre(n_basis, a=-1, b=1)` | $[a, b]$ | $1$ |  |
| Chebyshev | `FunctionSpace.chebyshev(n_basis, a=-1, b=1, second_kind=False)` | $[a, b]$ | $(1-x^2)^{\mp 1/2}$ | `second_kind=True` flips the sign of the exponent |
| Jacobi | `FunctionSpace.jacobi(alpha, beta, n_basis, a=-1, b=1)` | $[a, b]$ | $(1-x)^\alpha(1+x)^\beta$ |  |
| Hermite | `FunctionSpace.hermite(n_basis, loc=0, scale=1, kind="prob")` | $(-\infty, \infty)$ | $e^{-x^2/2}$ (`"prob"`) or $e^{-x^2}$ (`"phys"`) |  |
| Laguerre | `FunctionSpace.laguerre(n_basis, rate=1, start=0)` | $[\text{start}, \infty)$ | $e^{-x}$ |  |
| Fourier | `FunctionSpace.fourier(n_basis, a=-1, b=1, kind="full")` | periodic $[a, b)$ | $1$ | `kind="cosine"`/`"sine"` for one-sided trigonometric families; orthonormal by construction |
| Monomial | `FunctionSpace.monomial(n_basis, a=-1, b=1)` | $[a, b]$ | N/A (not orthogonal) | Power series |
| Lagrange | `FunctionSpace.piecewise("lagrange", degree, [a, b], nodes="lobatto")` | $[a, b]$ | N/A (not orthogonal) | Nodal (cardinal) basis: coefficients are point values at the chosen node family |
| Piecewise | `FunctionSpace.piecewise(kind, degree, breakpoints, ...)` | tiled $[a, b]$ | depends on `kind` | Nodal (`kind="lagrange"`) or modal (`kind="legendre"`) local elements, or `kind="hermite"` for $C^1$ continuity |
| B-spline | `FunctionSpace.bspline(degree, knots)` / `.clamped_bspline(degree, breakpoints)` | tiled $[a, b]$ | N/A (not orthogonal) | Smooth ($C^{\text{degree} - 1}$ by default) piecewise polynomials from a knot vector |
| Tensor product | `FunctionSpace.tensor(*spaces)` | product domain | product of factors' weights | Multivariate space built from univariate factors |

**Piecewise** is the module's finite-element basis: `kind="lagrange"` uses nodal degrees of freedom (point values, the classical FEM element), `kind="legendre"` uses modal ones, and `kind="hermite"` adds slope DOFs at each element boundary (needed for 4th-order operators like beam bending). `continuity=-1` gives a fully discontinuous (broken) basis.

The easiest way to visualize the basis functions themselves is to construct a function space and then create a `Function` using a "one-hot" unit basis vector $\mathbf{e}_i$.
For example, here are the leading Legendre basis functions:

```{code-cell} python
:tags: [remove-output]
n = 5  # Number of basis functions to use
a, b, = -1, 1  # Domain bounds

# Basic orthogonal polynomial basis on [-1, 1]
legendre = FunctionSpace.legendre(n, a, b)

x_plt = np.linspace(a, b, 100)
x = legendre.quad_rule.nodes

fig, ax = plt.subplots(1, 1, figsize=(7, 3))
for i in range(n):
    # The i-th basis function is the i-th column of the identity matrix
    e_i = np.eye(n)[:, i]
    f_i = legendre.function(e_i)
    ax.plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
    ax.plot(x, f_i(x), ".", color=ax[0].lines[-1].get_color())


ax.grid()
ax.set_ylabel(r"$\phi_i(x)$")
ax.set_title("Legendre Basis")
ax.legend(loc="upper right")
ax.set_xlabel("$x$")
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    for i in range(n):
        e_i = np.eye(n)[:, i]
        f_i = legendre.function(e_i)
        ax.plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
        ax.plot(x, f_i(x), ".", color=ax[0].lines[-1].get_color())


    ax.grid()
    ax.set_ylabel(r"$\phi_i(x)$")
    ax.set_title("Legendre Basis")
    ax.legend(loc="upper right")
    ax.set_xlabel("$x$")

    plt.savefig(plot_dir / f"approximation_0_{theme}.png")
    plt.close()
```


```{image} _plots/approximation_0_light.png
:class: only-light
```

```{image} _plots/approximation_0_dark.png
:class: only-dark
```

The dots in these plots show the nodes for the associated quadrature rule that will integrate the basis exactly.

We can use a similar approach to construct a global Lagrange basis.
Note that here the basis uses the same nodes as its quadrature rule (Legendre for both), so we can clearly see the defining Lagrange property that the $i$-th basis function takes the value 1 at node $x_i$ and 0 at all other nodes.

```{code-cell} python
:tags: [remove-output]
# Construct a Lagrange basis using Legendre nodes
# and quadrature rule using a single-element "piecewise"
# space
lagrange = FunctionSpace.piecewise(
    "lagrange",
    degree=n - 1,
    breakpoints=(a, b),
    nodes="legendre",
    continuity=-1,
)
x = lagrange.quad_rule.nodes

fig, ax = plt.subplots(1, 1, figsize=(7, 3))
for i in range(n):
    e_i = np.eye(n)[:, i]
    f_i = lagrange.function(e_i)
    ax.plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
    ax.plot(x, f_i(x), ".", color=ax[1].lines[-1].get_color())


ax.grid()
ax.set_ylabel(r"$\phi_i(x)$")
ax.set_title("Lagrange Basis")
ax.legend(loc="upper right")
ax.set_xlabel("$x$")
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    for i in range(n):
        e_i = np.eye(n)[:, i]
        f_i = lagrange.function(e_i)
        ax.plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
        ax.plot(x, f_i(x), ".", color=ax[0].lines[-1].get_color())


    ax.grid()
    ax.set_ylabel(r"$\phi_i(x)$")
    ax.set_title("Lagrange Basis")
    ax.legend(loc="upper right")
    ax.set_xlabel("$x$")

    plt.savefig(plot_dir / f"approximation_1_{theme}.png")
    plt.close()
```


```{image} _plots/approximation_1_light.png
:class: only-light
```

```{image} _plots/approximation_1_dark.png
:class: only-dark
```

We can also use the `piecewise` constructor to tile the Lagrange basis.
The standard finite element basis uses Lobatto nodes per element (one node at each basis endpoint), but a Legendre quadrature rule, so now the quadrature nodes are _not_ the same as the Lagrange nodes.
Hence, the basis functions are the classical finite element "hat" functions that take on 1 at the element boundaries, but the quadrature nodes are not co-located with the boundaries:


```{code-cell} python
:tags: [remove-output]

# Basic orthogonal polynomial basis on [-1, 1]
legendre = FunctionSpace.legendre(n, a, b)

# Construct a Lagrange basis using Legendre nodes
# and quadrature rule using a single-element "piecewise"
# space
lagrange = FunctionSpace.piecewise(
    "lagrange",
    degree=n - 1,
    breakpoints=(a, b),
    nodes="legendre",
    continuity=-1,
)

# Construct a classical P1 finite element basis
breakpoints = np.linspace(a, b, n, endpoint=True)
piecewise = FunctionSpace.piecewise(
    "lagrange",
    degree=1,
    breakpoints=breakpoints,
    nodes="lobatto",
)

x_plt = np.linspace(a, b, 100)

fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
for i in range(n):
    # The i-th basis function is the i-th column of the identity matrix
    e_i = np.eye(n)[:, i]

    # Legendre basis
    x = legendre.quad_rule.nodes
    f_i = legendre.function(e_i)
    ax[0].plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
    ax[0].plot(x, f_i(x), ".", color=ax[0].lines[-1].get_color())

    # Lagrange basis
    x = lagrange.quad_rule.nodes
    f_i = lagrange.function(e_i)
    ax[1].plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
    ax[1].plot(x, f_i(x), ".", color=ax[1].lines[-1].get_color())

    # Piecewise basis
    x = piecewise.quad_rule.nodes
    f_i = piecewise.function(e_i)
    ax[2].plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
    ax[2].plot(x, f_i(x), ".", color=ax[2].lines[-1].get_color())


ax[0].grid()
ax[0].set_ylabel("Legendre")
ax[0].legend(loc="upper right")

ax[1].grid()
ax[1].set_ylabel("Lagrange")

ax[2].grid()
ax[2].set_ylabel("Piecewise Lagrange (P1)")

ax[-1].set_xlabel("$x$")
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)
    fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for i in range(n):
        # The i-th basis function is the i-th column of the identity matrix
        e_i = np.eye(n)[:, i]

        # Legendre basis
        x = legendre.quad_rule.nodes
        f_i = legendre.function(e_i)
        ax[0].plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
        ax[0].plot(x, f_i(x), ".", color=ax[0].lines[-1].get_color())

        # Lagrange basis
        x = lagrange.quad_rule.nodes
        f_i = lagrange.function(e_i)
        ax[1].plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
        ax[1].plot(x, f_i(x), ".", color=ax[1].lines[-1].get_color())

        # Piecewise basis
        x = piecewise.quad_rule.nodes
        f_i = piecewise.function(e_i)
        ax[2].plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
        ax[2].plot(x, f_i(x), ".", color=ax[2].lines[-1].get_color())


    ax[0].grid()
    ax[0].set_ylabel("Legendre")
    ax[0].legend(loc="upper right")

    ax[1].grid()
    ax[1].set_ylabel("Lagrange")

    ax[2].grid()
    ax[2].set_ylabel("Piecewise Lagrange (P1)")

    ax[-1].set_xlabel("$x$")

    plt.savefig(plot_dir / f"approximation_0_{theme}.png")
    plt.close()
```


```{image} _plots/approximation_0_light.png
:class: only-light
```

```{image} _plots/approximation_0_dark.png
:class: only-dark
```