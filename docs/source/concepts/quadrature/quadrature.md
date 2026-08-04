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

np.random.seed(0)
```

```{code-cell} python
:tags: [remove-cell]
from pathlib import Path

plot_dir = Path.cwd() / "_plots"
plot_dir.mkdir(exist_ok=True)
```

# Quadrature

_Quadrature_ is the name given to a set of algorithms that perform approximate numerical integration of arbitrary functions.
The [`quadrature`](#archimedes.quadrature) module includes support for Gaussian quadrature implementations that are compatible with Archimedes' symbolic tracing, autodiff, and code generation.

This page gives an introduction to numerical quadrature in Archimedes, including the relationship between Gaussian quadrature rules and classical orthogonal polynomials, and how this relationship translates into the concepts of [`Measure`](#archimedes.measure.Measure) and [`QuadratureRule`](#archimedes.quadrature.QuadratureRule).

## Quickstart

Gaussian quadrature approximates a weighted integral with a discrete sum over (generally non-uniform) nodes and weights:

$$
\int_a^b f(x) \, w(x) \, dx \approx \sum_{i=1}^n w_i f(x_i),
$$

where $f(x)$ is the function to be integrated, $w(x)$ is a weight function, and $\{x_i\}, \{w_i\}$ are the nodes and weights, which are uniquely determined by the family and order of the quadrature rule.

The most common quadrature family, [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature) uses a domain of $[-1, 1]$, with uniform weight $w(x) = 1$:

$$
\int_{-1}^{1} f(x) \, dx \approx \sum_{i=1}^n w_i f(x_i),
$$

which can be shifted to an arbitrary (finite) domain $[a, b]$ by rescaling the Gauss-Legendre nodes and weights by:

$$
\begin{aligned}
x_i &\leftarrow \frac{b-a}{2} x_i + \frac{a+b}{2} \\
w_i &\leftarrow \frac{b-a}{2} w_i
\end{aligned}
$$

Definite integrals on finite domains can be calculated using Gauss-Legendre quadrature with the [`quadint`](#archimedes.quadrature.quadint) function:

```{code-cell} python
def f(x):
    return np.exp(x)


a, b = -3, 3  # Integration limits
J_ex = np.exp(b) - np.exp(a)  # Exact integral: e^b - e^a

# 5-point Gauss-Legendre quadrature rule
J_leg = arc.quadrature.quadint(f, a, b, n=5)

print(f"Exact integral:          {J_ex:.6f}")
print(f"Gauss-Legendre integral: {J_leg:.6f}")
```

Unlike [`scipy.integrate.quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html), this does not support adaptive integration with an error tolerance, nor does it support infinite or semi-infinite intervals.

However, it does support symbolic evaluation (including limits):

```{code-cell} python
# Differentiating the integral with respect to the limits of integration
def f(x):
    return np.exp(x)


@arc.compile
def integrate_f(a, b):
    return arc.quadrature.quadint(f, a, b, n=5)


dJ_da, dJ_db = arc.grad(integrate_f, argnums=(0, 1))(a, b)

# Analytical derivatives: dJ/da = -e^a, dJ/db = e^b
print(f"Analytical dJ/da: {-np.exp(a):.6f}, dJ/db: {np.exp(b):.6f}")
print(f"Computed dJ/da:   {dJ_da:.6f}, dJ/db: {dJ_db:.6f}")
```

and vector-values integrands:

```{code-cell} python
# Vector-valued integrands
def f(x):
    return np.array([np.cos(x), np.sin(x)])


a, b = 0, np.pi / 2  # Integration limits
J_vec = arc.quadrature.quadint(f, a, b, n=3)

print("Analytical integral: [1, 1]")
print(f"Computed integral:   {J_vec}")
```

Combining these, you can easily compute derivatives "under the integral sign" using the Leibnitz rule:

```{code-cell} python
# https://en.wikipedia.org/wiki/Leibniz_integral_rule#Example_2:_Variable_limits


def f(x):
    return np.cosh(x**2)


def g(x):
    # Variable limits of integration
    a = np.sin(x)
    b = np.cos(x)
    # Compute integral using Gauss-Legendre quadrature
    return arc.quadrature.quadint(f, a, b, n=5)


# Compute g'(x) using automatic differentiation
dg_dx = arc.grad(g)

x = np.linspace(0, 2 * np.pi, 100)
dg = arc.vmap(dg_dx)(x)
dg_ex = -np.cosh(np.cos(x) ** 2) * np.sin(x) - np.cosh(np.sin(x) ** 2) * np.cos(x)

print(f"Error: {np.linalg.norm(dg - dg_ex)}")
```

```{code-cell} python
:tags: [remove-output]

fig, ax = plt.subplots(1, 1, figsize=(7, 2))
ax.plot(x, dg, label="Computed")
ax.plot(x, dg_ex, "--", label="Exact")
ax.legend()
ax.grid()
ax.set_xlabel("$x$")
ax.set_ylabel("$g(x)$")
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.theme.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
    ax.plot(x, dg, label="Computed")
    ax.plot(x, dg_ex, "--", label="Exact")
    ax.legend()
    ax.grid()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$g(x)$")
    plt.savefig(plot_dir / f"quadrature_0_{theme}.png")
    plt.close()
```

<!-- TODO: composite and tensor rules -->

## Gaussian quadrature

The power of Gaussian quadrature lies in carefully chosen nodes and weights which give highly accurate approximations of integrals of polynomials (and hence arbitrary smooth functions) with relatively few sample points.
For instance, that 3e-6 error in the complicated cosh derivative-of-integral above used only _five_ sample points on the $(0, 2\pi)$ domain.

The nodes are the roots of classical orthogonal polynomials associated with the weight function (i.e. Legendre polynomials for $w(x) = 1$ on a finite interval), and the weights are derived from Lagrange interpolation of the nodal data.

Since the nodes and weights on the reference domain can be statically computed, under the hood we use SciPy's [`roots_legendre/jacobi/laguerre/hermite`](https://docs.scipy.org/doc/scipy/reference/special.html#orthogonal-polynomials) functions to do the actual math.

There are two internal abstractions that keep track of the weight function, reference domain, and reference nodes/weights.
The first is [`Measure`](#archimedes.measure.Measure), which combines a weight function with a reference interval to define families of orthogonal polynomials.
The second is [`QuadratureRule`](#archimedes.quadrature.QuadratureRule), which stores the nodes, weights, and associated `Measure`, and which is responsible for domain transformations and performing the weighted sum.

If you're not constructing exotic custom quadrature rules, you shouldn't need to interact with either of these classes directly.
Instead, there are two high-level interfaces:

1. The [`quadint`](#archimedes.quadrature.quadint) function demonstrated earlier, which takes a callable function and does Gauss-Legendre quadrature on an unweighted finite interval
2. Convenience constructors for common `QuadratureRule`s like Gauss-Radau, Clenshaw-Curtis, Gauss-Hermite, etc.

We've already seen #1 in action.
In fact, #1 is just a very thin wrapper around #2 for anyone (for example, me) who cannot keep straight the contributions of all of the French geniuses with L-names (Legendre? Lagrange? Laguerre?).

These constructors use snake-case versions of the conventional names of the rules, e.g. Clenshaw-Curtis becomes `clenshaw_curtis`, and produce a `QuadratureRule` instance.
Available options are:

| Classical name | Python function | Weight function $w(x)$ | Reference interval | Notes |
|---|---|---|---|---|
| Gauss-Legendre | `gauss_legendre(n)` | $1$ | $[-1, 1]$ | Neither endpoint included |
| Gauss-Radau | `gauss_radau(n, endpoint="left"\|"right")` | $1$ | $[-1, 1]$ | Fixes one endpoint |
| Gauss-Lobatto | `gauss_lobatto(n)` | $1$ | $[-1, 1]$ | Fixes both endpoints |
| Clenshaw-Curtis | `clenshaw_curtis(n)` | $1$ | $[-1, 1]$ | Chebyshev-Lobatto nodes |
| Gauss-Hermite (physicists') | `gauss_hermite(n, kind="phys")` | $e^{-x^2}$ | $(-\infty, \infty)$ |   |
| Gauss-Hermite (probabilists') | `gauss_hermite(n, kind="prob")` |$e^{-x^2/2}$ | $(-\infty, \infty)$ |   |
| Gauss-Laguerre | `gauss_laguerre(n)` | $e^{-x}$ | $[0, \infty)$ |   |

Once you have the `QuadratureRule` object, you can inspect the nodes and weights if you like, or just use its quadrature methods:

- `QuadratureRule.integrate(f, **kwparams)` integrates the function `f(x)` on the domain specified by `**kwparams`
- `QuadratureRule.sum(fp, **kwparams)` does the same thing but with pre-computed function data on the nodes

The equivalence between the two is literally:

```python
# This:
quad_rule.integrate(f, **kwparams)

# is the same as this:
xp = quad_rule.weights
fp = f(xp)
quad_rule.sum(fp, **kwparams)
```

The `**kwparams` define the domain and weight transformation.
For instance, `quad_rule.integrate(f, a=a, b=b)` for Gauss-Legendre (or Radau, Lobatto, Jacobi, or Clenshaw-Curtis) will transform the domain to $(a, b)$, while `quad_rule.integrate(f, mean=mu, std=sigma)` for Gauss-Hermite on an infinite domain will shift/scale the Gaussian weight function.

```python
n = 6
leg = arc.quadrature.gauss_legendre(n)
rad_left = arc.quadrature.gauss_radau(n, endpoint="left")
rad_right = arc.quadrature.gauss_radau(n, endpoint="right")
lob = arc.quadrature.gauss_lobatto(n)
cc = arc.quadrature.clenshaw_curtis(n)

zero = np.zeros_like(leg.nodes)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(leg.nodes, zero, 'o', label="Gauss-Legendre")
ax.plot(rad_left.nodes, zero + 1, 'o', label="Gauss-Radau (left)")
ax.plot(rad_right.nodes, zero + 2, 'o', label="Gauss-Radau (right)")
ax.plot(lob.nodes, zero + 3, 'o', label="Gauss-Lobatto")
ax.plot(cc.nodes, zero + 4, 'o', label="Clenshaw-Curtis")
ax.set_xlabel("Node $x_i$")
ax.set_title(f"Quadrature nodes for n={n}")
ax.legend()
ax.set_ylim(-1, 8)
ax.grid()
ax.set_yticks([])
plt.show()
```

```{code-cell} python
:tags: [remove-cell]
n = 6
leg = arc.quadrature.gauss_legendre(n)
rad_left = arc.quadrature.gauss_radau(n, endpoint="left")
rad_right = arc.quadrature.gauss_radau(n, endpoint="right")
lob = arc.quadrature.gauss_lobatto(n)
cc = arc.quadrature.clenshaw_curtis(n)

zero = np.zeros_like(leg.nodes)

for theme in ("light", "dark"):
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(leg.nodes, zero, "o", label="Gauss-Legendre")
    ax.plot(rad_left.nodes, zero + 1, "o", label="Gauss-Radau (left)")
    ax.plot(rad_right.nodes, zero + 2, "o", label="Gauss-Radau (right)")
    ax.plot(lob.nodes, zero + 3, "o", label="Gauss-Lobatto")
    ax.plot(cc.nodes, zero + 4, "o", label="Clenshaw-Curtis")
    ax.set_xlabel("Node $x_i$")
    ax.set_title(f"Quadrature nodes for n={n}")
    ax.legend()
    ax.set_ylim(-1, 8)
    ax.grid()
    ax.set_yticks([])
    plt.savefig(f"_static/nodes_{theme}.png")
```

```{image} _static/nodes_light.png
:class: only-light
```

```{image} _static/nodes_dark.png
:class: only-dark
```

```{code-cell} python
# Same example as above, but using Clenshaw-Curtis quadrature


def f(x):
    return np.exp(x)


a, b = -3, 3  # Integration limits
J_ex = np.exp(b) - np.exp(a)  # Exact integral: e^b - e^a

# 20-point Clenshaw-Curtis quadrature rule
quad_rule = arc.quadrature.clenshaw_curtis(n=20)
J_cc = quad_rule.integrate(f, a, b)

print(f"Exact integral:          {J_ex:.6f}")
print(f"Clenshaw-Curtis integral:  {J_cc:.6f}")
```

One distinct feature of the Archimedes quadrature interface is that you can optionally pass a `density=True` keyword arg to directly interpret the weight functions as probability densities.
That is, the quadrature result approximates an expectation under the corresponding probability density:

$$
\int_{\mathcal{D}} f(x) \, \rho(x) \, dx, \qquad \rho(x) =  \frac{1}{\int_{\mathcal{D}} w(x') \, dx'} w(x)
$$

For example, we can compute the expectation of $x^2$ over a normal distribution with mean $\mu$ and variance $\sigma^2$ using the probabilists' Gauss-Hermite quadrature:

```{code-cell} python
def f(x):
    return x**2


mu = 2.0
sigma = 1.5

quad_rule = arc.quadrature.gauss_hermite(n=20, kind="prob")
J = quad_rule.integrate(f, mean=mu, std=sigma, density=True)

print(f"Exact value: {mu**2 + sigma**2:.6f}")
print(f"Quadrature value: {J:.6f}")
```

This avoids needing to remember to manually divide out the sum of the weights to normalize an expectation integral.

A related difference in Archimedes is doing away with the NumPy/SciPy convention of naming the physicists' Hermite polynomials (weight function $e^{-x^2}$) plain `Hermite` and the probabilists' Hermite polynomials (weight function $e^{-x^2/2}$) `HermiteNorm` - even though it's not "normalized" in the probability density sense.
I can't be the only one who ever got tripped up by this.

Instead, in Archimedes you explicitly choose between probabilists' and physicists' Hermite families with the `kind = 'phys' | 'prob'` keyword arg, as seen above.

## Function Approximation

Gaussian quadrature is useful in its own right, but a major reason for introducing these abstractions here is as a first step towards the goal of a function approximation module.
I know, I have some open "first steps" in [other domains](../../2025/spatial.md) in other application areas to keep up with as well, but bear with me.

"Function approximation" is a broad term, which can in principle include things like deep neural networks, but in this context I mostly mean linear basis function expansions:

$$
f(x) \approx \sum_{i=1}^n c_i \phi_i(x),
$$

where $\{c_i\}_{i=1}^n$ are coefficients and $\{\phi_i\}_{i=1}^n$ are the basis functions.

Some of the more common basis expansions include:

- Monomials
- Lagrange interpolating polynomials
- Orthogonal polynomials (spectral approximation)
- B-splines
- Piecewise polynomials and lookup tables
- Polynomial chaos expansions
- Karhunen-Loeve decomposition
- Radial basis functions

The fact that all of these are linear in coefficients means that in principle they should be able to be unified under a set of operations including evaluation, mass and stiffness matrix calculation, projection, interpolation, __*quadrature*__ etc.

The design for this is still a work in progress, but the vision is something that takes inspiration from both the [Unified Form Language](https://docs.fenicsproject.org/ufl/main/manual/introduction.html) of FEniCS/Firedrake and the first-class-function models of spectral approximation libraries like [ApproxFun.jl](https://juliaapproximation.github.io/ApproxFun.jl/stable/) and its spiritual ancestor [chebfun](https://www.chebfun.org/).

If it works the way I'm envisioning, it would open up exciting possibilities in Archimedes like:

- Custom 1D FEM models
- Uncertainty quantification with polynomial chaos expansions
- Gray-box system identification
- Collocated optimal control discretizations
- Spectral methods for PDE solves

And of course, all of these would compose with the existing Archimedes capabilities in autodiff, [C code generation](../../../tutorials/codegen/codegen00.md), and [hierarchical modeling](../../../tutorials/hierarchical/hierarchical00.md).

A rough sketch of what this might look like is:

```python
class Basis(Protocol):
    domain: tuple

    def __call__(self, i: int, x, deriv: int = 0):
        """Evaluate phi[i] at x"""
        ...   # -> (len(x),)

class BasisExpansion(Protocol):
    n_basis: int
    basis: Basis

    def basis_eval(self, x, deriv: int = 0):
        """Evaluate all the basis functions at x"""
        ...   # -> (len(x), n_basis)

@arc.struct
class Function:
    coefficients: np.ndarray
    basis: BasisExpansion = arc.field(static=True)

    def __call__(self, x):
        """Interpolate the function approximation at x"""
        ...  # -> (len(x),)
```

Expressing `Function` as a `@struct`-decorated class means that if you define operations like `grad(f: Function) -> Function` to return the derivative in the same basis, and construct `dx` so that it does symbolic quadrature over the domain, then you should be able to naturally express a weak form for finite element analysis in functional form:

```python
# Bilinear form for nonlinear Poisson equation
def a(u, v, x):
    return k(x) * grad(u) * grad(v) * dx
```

"Assembling" the finite element problem is then a matter of evaluating the residual over the test basis and letting Archimedes/CasADi handle the sparse autodiff for Jacobians - no manual scatter, element bookkeeping, hand-derived tangents, etc.

You could do something similar for all of the other algorithm classes above, bringing the code much closer to how you'd write the math.

As I said, it needs some design work, but today's release of a `QuadratureRule` based on the `Measure` abstraction is the first step towards this kind of unified function approximation infrastructure.

## For More

- Check out the API docs for [quadrature](#archimedes.quadrature) to see the details
- [Subscribe](https://jaredcallaham.substack.com/embed) to the newsletter for updates and announcements
