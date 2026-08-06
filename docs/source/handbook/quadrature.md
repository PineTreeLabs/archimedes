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

# Quadrature

_Quadrature_ is the name given to a set of algorithms that perform approximate numerical integration of arbitrary functions.

This is a foundational ingredient of a number of higher-level algorithms, including PDE solving, trajectory optimization, and uncertainty quantification.
It's also a different kind of method from time-stepping ODE solvers ([`odeint`](#archimedes.odeint), for example).  Typical ODE solvers are _local_, approximating the solution over short interval at a time, whereas quadrature is _global_, approximating the integrand with an analytically-integrable function on the entire domain at once.

The [`quadrature`](#archimedes.quadrature) module includes support for (mostly) Gaussian quadrature implementations that are compatible with Archimedes' symbolic tracing, autodiff, and code generation.

This page gives an introduction to numerical quadrature in Archimedes, including the relationship between Gaussian quadrature rules and classical orthogonal polynomials, and how this relationship translates into the concepts of [`Measure`](#archimedes.measure.Measure) and [`QuadratureRule`](#archimedes.quadrature.QuadratureRule).

## Quadrature Quickstart

Gaussian quadrature approximates a weighted integral with a discrete sum over (generally non-uniform) nodes and weights:

```{math}
\int_a^b f(x) \, w(x) \, dx \approx \sum_{i=1}^n w_i f(x_i),
```

where $f(x)$ is the function to be integrated, $w(x)$ is a weight function, and $\{x_i\}, \{w_i\}$ are the nodes and weights, which are uniquely determined by the family and order of the quadrature rule.

The most common quadrature family, [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature) uses a domain of $[-1, 1]$, with uniform weight $w(x) = 1$:

```{math}
\int_{-1}^{1} f(x) \, dx \approx \sum_{i=1}^n w_i f(x_i),
```

which can be shifted to an arbitrary (finite) domain $[a, b]$ by rescaling the Gauss-Legendre nodes and weights by:

```{math}
\begin{aligned}
x_i &\leftarrow \frac{b-a}{2} x_i + \frac{a+b}{2} \\
w_i &\leftarrow \frac{b-a}{2} w_i
\end{aligned}
```

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

```{code-cell} python
:tags: [remove-cell]
# Regression check on the absolute error
assert abs(J_leg - J_ex) < 1e-3
```

Suitably constructed quadrature rules typically converge to an exact result much more quickly than, for instance, uniform trapezoidal integration:

```{code-cell} python
:tags: [hide-cell, remove-output]
n_quad = np.arange(2, 16)
J_quad = np.array([arc.quadrature.quadint(f, a, b, n=n) for n in n_quad])
e_quad = abs(J_quad - J_ex)

n_trapz = np.arange(2, 1000, 10)

def trapz(n):
    x = np.linspace(a, b, n, endpoint=True)
    return np.trapezoid(f(x), x)

J_trapz = np.array([trapz(n) for n in n_trapz])
e_trapz = abs(J_trapz - J_ex)

fig, ax = plt.subplots(1, 1, figsize=(7, 3))
ax.plot(n_quad, e_quad, '.-', label="Gauss-Legendre")
ax.plot(n_trapz, e_trapz, '.-', label="Trapezoidal")
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.grid()
ax.set_xlabel("Number of points (function evaluations)")
ax.set_ylabel("Approximation error")
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(n_quad, e_quad, '.-', label="Gauss-Legendre")
    ax.plot(n_trapz, e_trapz, '.-', label="Trapezoidal")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.grid()
    ax.set_xlabel("Number of points (function evaluations)")
    ax.set_ylabel("Approximation error")
    plt.savefig(plot_dir / f"quadrature_0_{theme}.png")
    plt.close()
```

```{image} _plots/quadrature_0_light.png
:class: only-light
```

```{image} _plots/quadrature_0_dark.png
:class: only-dark
```

Internally, the integrand evaluation is vectorized; Archimedes-traceable pure functions constructed with NumPy should generally be fine (see [Gotchas](../../gotchas.md) for more details).

Note that unlike [`scipy.integrate.quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html), this high-level `quadint` function does not support adaptive integration with an error tolerance, nor does it support infinite or semi-infinite intervals.
It is possible to define integrals over infinite or semi-infinite domains using weighted quadrature rules like [`gauss_laguerre`](#archimedes.quadrature.gauss_laguerre) or [`gauss_hermite`](#archimedes.quadrature.gauss_hermite), but not through `quadint` specifically.

## Another Quadrature Implementation?

NumPy and SciPy already implement the numerical building blocks for Gaussian-style quadrature, and SciPy's [`scipy.integrate.quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html) and relatives is a good choice for evaluating a single integral.

However, for integrals that need to be evaluated inside of simulation or optimization loops, or for any applications that need to support [codegen](../../tutorials/codegen/codegen00.md), Archimedes takes advantage of a fundamental split in the quadrature construction.
Specifically, on a fixed reference domain (e.g. $[-1, 1]$ for Legendre-based rules), **the Gaussian quadrature nodes and weights are static, precomputable data, while the integrand data is symbolic.**
In fact, internally Archimedes reuses SciPy implementations of node/weight calculations wherever possible.

In other words, the more expensive computation of the _rule_ data can be done once, offline, making the online integral evaluation a simple weighted sum, the equivalent of `np.dot(w, f(x))`, the cost of which is almost always dominated purely by the integrand evaluation `f(x)`.
Domains other than the reference domain can be used with an affine transformation of the nodes and weights, a simple operation for symbolic tracing.
This makes it possible for quadrature integrals to compose with the rest of the Archimedes infrastructure, including autodiff, codegen, and hierarchical data structures.

This "traced summation" model precludes adaptive quadrature, because adaptive rules need variable-length vectors for the nodes and weights, which are not supported in Archimedes/CasADi.
You could construct a maximum-order-limited adaptive scheme by precomputing nodes and weights for all `n < n_max` and using [control flow primitives](../../control-flow.md), but this isn't implemented out of the box since it's not a common use case.

However, Archimedes quadrature does support symbolic evaluation (including limits):

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

and vector-valued integrands:

```{code-cell} python
# Vector-valued integrands
def f(x):
    return np.array([np.cos(x), np.sin(x)])


a, b = 0, np.pi / 2  # Integration limits
J_vec = arc.quadrature.quadint(f, a, b, n=3)

print("Analytical integral: [1, 1]")
print(f"Computed integral:   {J_vec}")
```

Combining these, you can easily compute derivatives "under the integral sign" using the Leibniz rule:

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
:tags: [hide-cell, remove-output]
fig, ax = plt.subplots(1, 1, figsize=(7, 2))
ax.plot(x, dg, label="Computed")
ax.plot(x, dg_ex, "--", label="Exact")
ax.legend()
ax.grid()
ax.set_xlabel("$x$")
ax.set_ylabel("$g(x)$")
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
    ax.plot(x, dg, label="Computed")
    ax.plot(x, dg_ex, "--", label="Exact")
    ax.legend()
    ax.grid()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$g(x)$")
    plt.savefig(plot_dir / f"quadrature_1_{theme}.png")
    plt.close()
```

```{image} _plots/quadrature_1_light.png
:class: only-light
```

```{image} _plots/quadrature_1_dark.png
:class: only-dark
```

## The `quadrature` Module

The power of Gaussian quadrature lies in carefully chosen nodes and weights which give highly accurate approximations of integrals of polynomials (and hence arbitrary smooth functions) with relatively few sample points.
For instance, the complicated cosh derivative-of-integral above used only _five_ sample points.

The nodes are the roots of classical orthogonal polynomials associated with the weight function (i.e. Legendre polynomials for $w(x) = 1$ on a finite interval), and the weights are derived from Lagrange interpolation of the nodal data (see the appendix [Quadrature and Orthogonal Polynomials](#appendix-quadrature-and-orthogonal-polynomials) below).

Since the nodes and weights on the reference domain can be statically computed, under the hood we use SciPy's [`roots_legendre/jacobi/laguerre/hermite`](https://docs.scipy.org/doc/scipy/reference/special.html#orthogonal-polynomials) functions to do the actual math.

### Module Basics

There are two abstractions that keep track of the weight function, reference domain, and reference nodes/weights.
The first is [`Measure`](#archimedes.measure.Measure), which combines a weight function with a reference interval to define families of orthogonal polynomials.
The second is [`QuadratureRule`](#archimedes.quadrature.QuadratureRule), which stores the nodes, weights, and associated `Measure`, and which is responsible for domain transformations and performing the weighted sum.

## See Also

- [Approximation](approximation.md) for the higher-level function approximation system that relies on quadrature for inner products

#### `Measure`

The [`Measure`](#archimedes.measure.Measure) class defines an orthogonality measure $d\mu(x) = w(x) ~ dx$ and an associated domain $\mathcal{D}$.
This $w(x)$ is the weight function in the quadrature rule, and the quadrature nodes for this rule are the roots of the polynomials that are orthogonal with respect to this measure (see [appendix](#appendix-quadrature-and-orthogonal-polynomials)).

The `Measure` interface is roughly:

```python
class Measure:
    domain: ReferenceDomain  # UnitInterval | HalfLine | RealLine

    # Domain of support D = [a, b]
    @property
    def support(self) -> tuple[float, float]: ...

    # Weight function w(x)
    @abc.abstractmethod
    def weight(self, x: np.ndarray) -> np.ndarray: ...
```

This might seem obscure, but it is fundamental for designing numerical schemes that aren't on finite intervals, or for functions with singularities.

Measures are also a useful practical concept for working with probability distributions; in this context a properly-normalized weight function $w(x)$ _is_ the probability distribution function and the weighted integral is the mean value of $f(x)$ over that distribution (this is the core of polynomial chaos expansions, for instance).

#### `QuadratureRule`

[`QuadratureRule`](#archimedes.quadrature.QuadratureRule) is a single class (not a base class or interface) that combines a `Measure` with associated nodes, weights, and (if a composite/piecewise rule) breakpoints and elements.
This is a higher level from `Measure` because there are different rules that can be constructed on a single `Measure`.
For instance, Gauss-Lobatto and Gauss-Legendre rules both use a uniform measure but different nodes/weights; same for [composite](#composite-rules) (tiled) quadrature rules.

The key parts of `QuadratureRule` are:

```python
@arc.struct
class QuadratureRule:
    nodes: np.ndarray
    weights: np.ndarray
    measure: Measure
    breakpoints: np.ndarray | None = None  # element boundaries, if composite
    elements: np.ndarray | None = None  # owning element per node, if composite

    # Approximate the weighted integral of ``f``
    def integrate(self, f, *params, axis=-1, args=None, density=False) -> np.ndarray: ...

    # Quadrature applied to values already sampled at the nodes
    def sum(self, values, *params, axis=-1, density=False) -> np.ndarray: ...
```

#### Higher-level interface

If you're not constructing exotic custom quadrature rules, you shouldn't need to interact with either of these classes directly (if you _are_ constructing exotic custom quadrature rules, [see below](#custom-rules)).

Most applications can work with the two high-level interfaces:

1. The [`quadint`](#archimedes.quadrature.quadint) function demonstrated earlier, which takes a callable function and does Gauss-Legendre quadrature on an unweighted finite interval
2. Convenience constructors for common `QuadratureRule`s like Gauss-Radau, Clenshaw-Curtis, Gauss-Hermite, etc.

We've already seen #1 in action; in fact, #1 is just a very thin wrapper around #2 for anyone.

The convenience constructors use snake-case versions of the conventional names of the rules, e.g. Clenshaw-Curtis becomes `clenshaw_curtis`, and produce a `QuadratureRule` instance.
Available options are:

| Classical name | Python function | Weight function $w(x)$ | Reference interval | Notes |
|---|---|---|---|---|
| Gauss-Legendre | `gauss_legendre(n)` | $1$ | $[-1, 1]$ | Neither endpoint included |
| Gauss-Radau | `gauss_radau(n, endpoint="left"\|"right")` | $1$ | $[-1, 1]$ | Fixes one endpoint |
| Gauss-Lobatto | `gauss_lobatto(n)` | $1$ | $[-1, 1]$ | Fixes both endpoints |
| Clenshaw-Curtis | `clenshaw_curtis(n)` | $1$ | $[-1, 1]$ | Chebyshev-Lobatto nodes |
| Trapezoidal | `trapezoidal(n)` | $1$ | $[-1, 1]$ | Use periodic version for Fourier basis |
| Gauss-Jacobi | `gauss_jacobi(n, alpha, beta)` | $(1-x)^\alpha(1+x)^\beta$ | $[-1, 1]$ | Legendre/Chebyshev are special cases |
| Gauss-Hermite (probabilists') | `gauss_hermite(n, kind="prob")` |$e^{-x^2/2}$ | $(-\infty, \infty)$ | Default `kind` |
| Gauss-Hermite (physicists') | `gauss_hermite(n, kind="phys")` | $e^{-x^2}$ | $(-\infty, \infty)$ |   |
| Gauss-Laguerre | `gauss_laguerre(n)` | $e^{-x}$ | $[0, \infty)$ |   |

The trapezoidal rule is slightly different from the others: it is exact for trigonometric polynomials $\cos(k \pi t)$, $\sin(k \pi t)$ for $1 \leq k \leq n-1$ on $t \in [-1, 1)$ (periodic).
If `periodic=False` then it reduces to the usual trapezoidal rule, which is not a Gaussian rule in the sense we have been discussing.

Once you have the `QuadratureRule` object, you can inspect the nodes and weights if you like, or just use its quadrature methods:

- `QuadratureRule.integrate(f, **kwparams)` integrates the function `f(x)` on the domain specified by `**kwparams`
- `QuadratureRule.sum(fp, **kwparams)` does the same thing but with pre-computed function data on the nodes

The equivalence between the two is literally:

```python
# This:
quad_rule.integrate(f, **kwparams)

# is the same as this:
xp = quad_rule.nodes
fp = f(xp)
quad_rule.sum(fp, **kwparams)
```

The `**kwparams` define the domain and weight transformation.
For instance, `quad_rule.integrate(f, a=a, b=b)` for Gauss-Legendre (or Radau, Lobatto, Jacobi, Clenshaw-Curtis, or trapezoidal) will transform the domain to $(a, b)$, while `quad_rule.integrate(f, loc=mu, scale=sigma)` for Gauss-Hermite on an infinite domain will shift/scale the Gaussian weight function.

One distinct feature of the Archimedes quadrature interface is that you can optionally pass a `density=True` keyword arg to directly interpret the weight functions as probability densities.
That is, the quadrature result approximates an expectation under the corresponding probability density:

```{math}
\int_{\mathcal{D}} f(x) \, \rho(x) \, dx, \qquad \rho(x) =  \frac{1}{\int_{\mathcal{D}} w(x') \, dx'} w(x)
```

For example, we can compute the expectation of $x^2$ over a normal distribution with mean $\mu$ and variance $\sigma^2$ using the probabilists' Gauss-Hermite quadrature:

```{code-cell} python
def f(x):
    return x**2


mu = 2.0
sigma = 1.5

quad_rule = arc.quadrature.gauss_hermite(n=20)  # kind="prob" is the default
J = quad_rule.integrate(f, loc=mu, scale=sigma, density=True)

print(f"Exact value: {mu**2 + sigma**2:.6f}")
print(f"Quadrature value: {J:.6f}")
```

This avoids needing to remember to manually divide out the sum of the weights to normalize an expectation integral.

A related difference in Archimedes is doing away with the tradition of naming the physicists' Hermite polynomials (weight function $e^{-x^2}$) plain `Hermite` and the probabilists' Hermite polynomials (weight function $e^{-x^2/2}$) `HermiteNorm` - even though it's not "normalized" in the probability density sense.

Instead, in Archimedes you explicitly choose between probabilists' and physicists' Hermite families with the `kind = 'prob' | 'phys'` keyword arg, as seen above; `"prob"` is the default, since its weight is (up to normalization) the standard normal density, making `loc`/`scale` behave like an ordinary mean/standard deviation.

One subtlety to be aware of: the numeric meaning of `scale` depends on `kind`, since it's paired with the domain rather than the weight.
For `kind="prob"`, `scale` is exactly the standard deviation of the corresponding Gaussian.
For `kind="phys"`, whose weight is $e^{-x^2}$ rather than $e^{-x^2/2}$, `scale` is $\sqrt{2}$ times that standard deviation.

### Clenshaw-Curtis

Like the periodic `trapezoidal` rule, Clenshaw-Curtis quadrature is another outlier in this group.
It is not a Gaussian quadrature rule associated with a measure with nodes derived from the roots of classical orthogonal polynomials.
Instead, the Clenshaw-Curtis nodes are the *extrema* of the Chebyshev polynomials.

Like Gauss-Lobatto, the Clenshaw-Curtis nodes *include* the endpoints, but there are a couple of key practical differences that determine which is a better fit.

* Theoretically, Gauss-Lobatto has roughly twice the polynomial order of accuracy (although [in practice the gap is much smaller](TODO: REF TREFETHEN))
* The Clenshaw-Curtis nodes and weights can be cheaply and accurately computed for much larger $n$, making it more suitable for applications like large-scale PDE models (e.g. direct numerical simulation of fluid dynamics with pseudospectral methods)
* The Chebyshev-Lobatto nodes are _nested across doubling_ $n$, meaning that the nodes for `clenshaw_curtis(n)` are all also present in the set of nodes for `clenshaw_curtis(2*n)`

If these aren't relevant for your application, in general Gauss-Lobatto is preferable for its accuracy.

### Composite Rules

Quadrature rules with uniform weight $w(x) \equiv 1$ can be "tiled" into a _composite_ rule.
Mathematically, a composite rule interpolates the data onto a *piecewise polynomial* that is then integrated exactly with a piecewise quadrature rule.

For example, to construct a quadrature rule for a uniform 10-element domain with third-order Gauss-Legendre quadrature in each element:

```{code-cell} python
nel = 10
p = 3
breakpoints = np.linspace(-1, 1, nel+1, endpoint=True)
rule = arc.quadrature.composite_quad(
    arc.quadrature.gauss_legendre(n=p+1),
    breakpoints=breakpoints
)
```

```{code-cell} python
:tags: [hide-cell, remove-output]
fig, ax = plt.subplots(1, 1, figsize=(7, 2))
ax.plot(rule.nodes, 0 * rule.nodes, '.')
ax.set_yticks([])
ax.set_xticks(breakpoints)
ax.grid()
ax.set_xlabel("Reference domain $t$")
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in ("light", "dark"):
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
    ax.plot(rule.nodes, 0 * rule.nodes, '.')
    ax.set_yticks([])
    ax.set_xticks(breakpoints)
    ax.grid()
    ax.set_xlabel("Reference domain $t$")
    plt.savefig(f"_plots/quadrature_2_{theme}.png")
```

```{image} _plots/quadrature_2_light.png
:class: only-light
```

```{image} _plots/quadrature_2_dark.png
:class: only-dark
```

The composite rule does not need to use a uniform degree, nor even a uniform rule.
For example, here we add endpoints using left/right Radau rules, and locally refine two interior elements:

```{code-cell} python
el_rules = [arc.quadrature.gauss_legendre(p+1) for _ in range(nel)]
el_rules[0] = arc.quadrature.gauss_radau(p+1, "left")
el_rules[-1] = arc.quadrature.gauss_radau(p+1, "right")
el_rules[nel//2-1] = arc.quadrature.gauss_legendre(4*(p+1))
el_rules[nel//2] = arc.quadrature.gauss_legendre(4*(p+1))

rule = arc.quadrature.composite_quad(el_rules, breakpoints=breakpoints)
```

```{code-cell} python
:tags: [hide-cell, remove-output]
fig, ax = plt.subplots(1, 1, figsize=(7, 2))
ax.plot(rule.nodes, 0 * rule.nodes, '.')
ax.set_yticks([])
ax.set_xticks(breakpoints)
ax.grid()
ax.set_xlabel("Reference domain $t$")
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in ("light", "dark"):
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
    ax.plot(rule.nodes, 0 * rule.nodes, '.')
    ax.set_yticks([])
    ax.set_xticks(breakpoints)
    ax.grid()
    ax.set_xlabel("Reference domain $t$")
    plt.savefig(f"_plots/quadrature_3_{theme}.png")
```

```{image} _plots/quadrature_3_light.png
:class: only-light
```

```{image} _plots/quadrature_3_dark.png
:class: only-dark
```

### Tensor Rules

A multidimensional product rule can also be constructed via a _tensor product_ of scalar quadrature rules.

For instance, to construct a tenth-order 2D Gauss-Legendre rule:

```{code-cell} python
p = 10
dim_rules = [arc.quadrature.gauss_lobatto(p+1) for _ in range(2)]
rule = arc.quadrature.tensor_quad(*dim_rules)
```

```{code-cell} python
:tags: [hide-cell, remove-output]
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(rule.nodes[:, 0], rule.nodes[:, 1], s=2)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
plt.show()
````

```{code-cell} python
:tags: [remove-cell]

for theme in ("light", "dark"):
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(rule.nodes[:, 0], rule.nodes[:, 1], s=2)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.savefig(f"_plots/quadrature_4_{theme}.png")
```

```{image} _plots/quadrature_4_light.png
:class: only-light
```

```{image} _plots/quadrature_4_dark.png
:class: only-dark
```

Composite quadrature rules can themselves be expanded with tensor products; copying the non-uniform rule from above:

```{code-block} python
p = 3
nel = 10
breakpoints = np.linspace(-1, 1, nel+1, endpoint=True)
dim_rules = [
    arc.quadrature.composite_quad(
        arc.quadrature.gauss_legendre(p+1),
        breakpoints=breakpoints
    )
    for _ in range(2)
]
rule = arc.quadrature.tensor_quad(*dim_rules)
```

```{code-cell} python
:tags: [remove-cell]

for theme in ("light", "dark"):
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(rule.nodes[:, 0], rule.nodes[:, 1], s=2)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.savefig(f"_plots/quadrature_5_{theme}.png")
```

```{image} _plots/quadrature_5_light.png
:class: only-light
```

```{image} _plots/quadrature_5_dark.png
:class: only-dark
```

## Appendix: Quadrature and Orthogonal Polynomials

The weight functions, reference domains, and node distributions can seem to be somewhat obscure at first.
These arise from a deep connection to _classical orthogonal polynomials_, and understanding why helps select the right family for an application.

This appendix is fully optional background reading, but may help to explain this interesting connection and why it links together measures, orthogonal polynomials, and optimal quadrature rules - and the implication for numerical schemes like finite elements, pseudospectral methods, etc. that depend on these concepts.

### Integration by interpolation

The foundational idea of this kind of quadrature is that you approximate the data $f_i \equiv f(x_i)$ with a polynomial, and then integrate that polynomial exactly.
[Lagrange polynomials](https://en.wikipedia.org/wiki/Lagrange_polynomial) give you a minimum-degree polynomial that exactly interpolates a given set of data; in general $n$ data points can be exactly interpolated by an $n-1$-degree Lagrange polynomial.

In other words, if the function $f$ happens to be an $n-1$-degree polynomial, you can integrate it exactly on any set of $n$ nodes by constructing a Lagrange interpolating polynomial and integrating it analytically.
The weights are the combination of the weight function $w(x)$ evaluated at the nodes, and the (linear, pre-computable) contribution of the Lagrange polynomial from that node to the integral.

Of course, if $f$ was a polynomial we could just integrate it analytically anyway, but smooth functions can be accurately approximated with polynomials (and non-smooth functions with piecewise-polynomials), so the exactness of the polynomial degree roughly gives us the accuracy of the method for arbitrary functions.

The genius of Gaussian quadrature is to choose the node positions carefully to get much higher accuracy.
For the Gauss-Legendre method, by adding $n$ additional degrees of freedom to the algebraic problem, we boost the accuracy from $n-1$ to $2n - 1$.

Suppose $f(x)$ is a polynomial of degree $\leq 2n - 1$, and we will be interpolating at roots $x_i$.
We can construct a polynomial $q_n(x) \equiv \prod_{i=1}^n (x - x_i)$ and do [polynomial long division](https://en.wikipedia.org/wiki/Polynomial_long_division) to decompose into the node polynomial $q_n(x)$, a quotient $p(x)$, and the remainder $r(x)$ (degrees $n$, $\leq n-1$, and $\leq n-1$, respectively):

```{math}
f(x) = q_n(x) p(x) + r(x).
```

We don't know what $p(x)$ and $r(x)$ are here; they're arbitrary polynomials used to derive the conditions on the selection of roots.

If we apply the quadrature rule $\sum_{i=1}^n w_i f(x_i)$ to this, by construction $q_n(x_i) = 0$, so

```{math}
\sum_{i=1}^n w_i f(x_i) = \sum_{i=1}^n w_i r(x_i)
```

If the weights are chosen as above, then since $r(x)$ has degree $\leq n - 1$ this integral is exact over $r$, so 

```{math}
\sum_{i=1}^n w_i f(x_i) = \int_a^b w(x) \, r(x) \, dx.
```

In order for this to *also* equal the weighted integral of $f(x)$, the additional contribution from $q_n(x) p(x)$ has to vanish for *any* polynomial $p(x)$ with degree $\leq n - 1$:

```{math}
\int_a^b w(x) \, q_n(x) \, p(x) \, dx = 0.
```

This is exactly the defining property of [orthogonal polynomials](https://en.wikipedia.org/wiki/Orthogonal_polynomials).

### Orthogonal polynomials

The important thing for Gaussian quadrature is that given a weight function and a domain, you can derive a family of polynomials such that the $n$-th polynomial is orthogonal to all $n-1$ polynomials in that family with respect to that weight, exactly the property Gauss identifies for optimizing accuracy of the quadrature rule:

```{math}
\int_a^b w(x) \, q_i(x) \, q_j(x) \, dx = 0, \qquad i \neq j
```

Since any $n-1$-degree polynomial can be represented by a linear combination of $q_i(x)$, $i = 0, 1, \dots, n-1$, the quotient term from polynomial division is guaranteed to vanish.

The upshot is that **if we choose the quadrature nodes to be the roots of the appropriate orthogonal polynomial, then we get optimal quadrature accuracy**.
The "appropriate" polynomial depends on the weight function and the domain, commonly:

| Weight $w(x)$ | Domain | Orthogonal polynomials | Quadrature scheme |
|---|---|---|---|
| $1$ | $[-1,1]$ | [Legendre](https://en.wikipedia.org/wiki/Legendre_polynomials) | [Gauss–Legendre](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature) |
| $(1-x)^\alpha(1+x)^\beta$ | $[-1,1]$ | [Jacobi](https://en.wikipedia.org/wiki/Jacobi_polynomials) | [Gauss-Jacobi](https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature) |
| $e^{-x^2}$ | $(-\infty,\infty)$ | [Hermite](https://en.wikipedia.org/wiki/Hermite_polynomials) | [Gauss–Hermite](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature) |
| $e^{-x}$ | $[0,\infty)$ | [Laguerre](https://en.wikipedia.org/wiki/Laguerre_polynomials) | [Gauss–Laguerre](https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature) |

The domains (but not their finite-ness) can be adjusted by shifting and scaling the quadrature weights and nodes.

Once the nodes $x_i$ are determined, the weights are again the combination of the weight function values $w(x_i)$ and the contribution to the final integral from the associated Lagrange polynomial.

### Constrained quadrature schemes

The basic Gaussian quadrature methods place $n$ nodes to be the roots of the $n$-th order orthogonal polynomial from the appropriate family.
However, these roots don't inherently include the endpoints of the interval.

Alternatively, we can derive _constrained_ quadrature families that include one or both endpoints; the tradeoff is 1-2 fewer degrees of freedom for the polynomial interpolation and corresponding loss of exactness in polynomial quadrature:

- **Gauss-Legendre rules**: Do not include endpoints - exact to order $2n - 1$
- **Gauss-Radau rules**: Include the left or right endpoint (configurable) - exact to order $2n - 2$
- **Gauss-Lobatto rules**: Include both endpoints - exact to order $2n - 3$.

```{code-cell} python
:tags: [hide-cell, remove-output]
n = 10
leg = arc.quadrature.gauss_legendre(n)
rad_left = arc.quadrature.gauss_radau(n, endpoint="left")
rad_right = arc.quadrature.gauss_radau(n, endpoint="right")
lob = arc.quadrature.gauss_lobatto(n)

zero = np.zeros_like(leg.nodes)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(leg.nodes, zero, "o", label="Gauss-Legendre")
ax.plot(rad_left.nodes, zero + 1, "o", label="Gauss-Radau (left)")
ax.plot(rad_right.nodes, zero + 2, "o", label="Gauss-Radau (right)")
ax.plot(lob.nodes, zero + 3, "o", label="Gauss-Lobatto")
ax.set_xlabel("Node $x_i$")
ax.set_title(f"Quadrature nodes for n={n}")
ax.legend()
ax.set_ylim(-1, 6)
ax.grid()
ax.set_yticks([])
plt.show()
```


```{code-cell} python
:tags: [remove-cell]
n = 10
leg = arc.quadrature.gauss_legendre(n)
rad_left = arc.quadrature.gauss_radau(n, endpoint="left")
rad_right = arc.quadrature.gauss_radau(n, endpoint="right")
lob = arc.quadrature.gauss_lobatto(n)

zero = np.zeros_like(leg.nodes)

for theme in ("light", "dark"):
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(leg.nodes, zero, "o", label="Gauss-Legendre")
    ax.plot(rad_left.nodes, zero + 1, "o", label="Gauss-Radau (left)")
    ax.plot(rad_right.nodes, zero + 2, "o", label="Gauss-Radau (right)")
    ax.plot(lob.nodes, zero + 3, "o", label="Gauss-Lobatto")
    ax.set_xlabel("Node $x_i$")
    ax.set_title(f"Quadrature nodes for n={n}")
    ax.legend()
    ax.set_ylim(-1, 6)
    ax.grid()
    ax.set_yticks([])
    plt.savefig(f"_plots/quadrature_6_{theme}.png")
```

```{image} _plots/quadrature_6_light.png
:class: only-light
```

```{image} _plots/quadrature_6_dark.png
:class: only-dark
```


These can be useful for different applications; Gauss-Radau gives useful stability properties for implicit ODE solvers, both Gauss-Radau and Gauss-Lobatto are commonly used in pseudospectral optimal control methods, and Gauss-Lobatto is widely used for high-order spectral element methods.

The schemes themselves are derived similarly to the Gauss-Legendre case, but fixing one of the roots in the polynomial division, e.g. $q_n(x) = (x - a) q_{n-1}(x)$ for Radau.
This amounts to a different weight in the orthogonality requirement and therefore a different member of the orthogonal polynomial family.
For Gauss-Radau with an endpoint at $x = a$ and original weight $w(x) = 1$, the orthogonality condition becomes:

```{math}
\int_a^b (x - a) \, q_i(x) \, q_j(x) \, dx = 0, \qquad i \neq j
```

That is, the nodes must be placed at the roots of the polynomials that are orthogonal with respect to this inner product with weight $(x - a)$.

### Custom Rules

<!-- TODO: discretized Stieljes & Golub-Welsch extension -->