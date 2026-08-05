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

<!-- TODO: Why you need quadrature, and why odeint is different -->

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

<!-- TODO: plot convergence against np.trapz -->

Unlike [`scipy.integrate.quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html), this does not support adaptive integration with an error tolerance, nor does it support infinite or semi-infinite intervals.

<!-- TODO: note on why infinite isn't supported (see docstring) -->

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

## The `quadrature` Module

The power of Gaussian quadrature lies in carefully chosen nodes and weights which give highly accurate approximations of integrals of polynomials (and hence arbitrary smooth functions) with relatively few sample points.
For instance, the complicated cosh derivative-of-integral above used only _five_ sample points.

The nodes are the roots of classical orthogonal polynomials associated with the weight function (i.e. Legendre polynomials for $w(x) = 1$ on a finite interval), and the weights are derived from Lagrange interpolation of the nodal data (see section "Quadrature and Orthogonal Polynomials" below).

Since the nodes and weights on the reference domain can be statically computed, under the hood we use SciPy's [`roots_legendre/jacobi/laguerre/hermite`](https://docs.scipy.org/doc/scipy/reference/special.html#orthogonal-polynomials) functions to do the actual math.

<!-- TODO: Emphasize static/symbolic split -->

### Module Basics

There are two abstractions that keep track of the weight function, reference domain, and reference nodes/weights.
The first is [`Measure`](#archimedes.measure.Measure), which combines a weight function with a reference interval to define families of orthogonal polynomials.
The second is [`QuadratureRule`](#archimedes.quadrature.QuadratureRule), which stores the nodes, weights, and associated `Measure`, and which is responsible for domain transformations and performing the weighted sum.

<!-- TODO: Expand on Measure, QuadratureRule, or at least a graphic -->

<!-- TODO: Forward ref for "exotic custom quadrature rules" -->
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
| Gauss-Jacobi | `gauss_jacobi(n, alpha, beta)` | $(1-x)^\alpha(1+x)^\beta$ | $[-1, 1]$ | Legendre/Chebyshev are special cases |
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
xp = quad_rule.nodes
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

<!-- TODO: Add comment about "std" in kind="phys" -->

### Composite Rules

<!-- TODO: Write this -->

### Tensor Rules

<!-- TODO: Write this -->

### Custom Rules

<!-- Golub-Welsch extension -->

## Appendix: Quadrature and Orthogonal Polynomials

<!-- TODO: Callout that this is optional "of interest" material -->

The weight functions, reference domains, and node distributions can seem to be somewhat obscure at first.
These arise from a deep connection to _classical orthogonal polynomials_, and understanding why helps select the right family for an application.

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

$$
f(x) = q_n(x) p(x) + r(x).
$$

We don't know what $p(x)$ and $r(x)$ are here; they're arbitrary polynomials used to derive the conditions on the selection of roots.

If we apply the quadrature rule $\sum_{i=1}^n w_i f(x_i)$ to this, by construction $q_n(x_i) = 0$, so

$$
\sum_{i=1}^n w_i f(x_i) = \sum_{i=1}^n w_i r(x_i)
$$

If the weights are chosen as above, then since $r(x)$ has degree $\leq n - 1$ this integral is exact over $r$, so 

$$
\sum_{i=1}^n w_i f(x_i) = \int_a^b w(x) \, r(x) \, dx.
$$

In order for this to *also* equal the weighted integral of $f(x)$, the additional contribution from $q_n(x) p(x)$ has to vanish for *any* polynomial $p(x)$ with degree $\leq n - 1$:

$$
\int_a^b w(x) \, q_n(x) \, p(x) \, dx = 0.
$$

This is exactly the defining property of [orthogonal polynomials](https://en.wikipedia.org/wiki/Orthogonal_polynomials).

### Orthogonal polynomials

Don't go read that Wikipedia page; it's full of terms like "Lebesgue–Stieltjes integrals".
The important thing for Gaussian quadrature is that given a weight function and a domain, you can derive a family of polynomials such that the $n$-th polynomial is orthogonal to all $n-1$ polynomials in that family with respect to that weight, exactly the property Gauss identifies for optimizing accuracy of the quadrature rule:

$$
\int_a^b w(x) \, q_i(x) \, q_j(x) \, dx = 0, \qquad i \neq j
$$

Since any $n-1$-degree polynomial can be represented by a linear combination of $q_i(x)$, $i = 0, 1, \dots, n-1$, the quotient term from polynomial division is guaranteed to vanish.

The upshot is that **if we choose the quadrature nodes to be the roots of the appropriate orthogonal polynomial, then we get optimal quadrature accuracy**.
The "appropriate" polynomial depends on the weight function and the domain, commonly:

<!-- TODO: Add Wiener-Askey correspondence -->
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
n = 10
leg = arc.quadrature.gauss_legendre(n)
rad_left = arc.quadrature.gauss_radau(n, endpoint="left")
rad_right = arc.quadrature.gauss_radau(n, endpoint="right")
lob = arc.quadrature.gauss_lobatto(n)
# cc = arc.quadrature.clenshaw_curtis(n)

zero = np.zeros_like(leg.nodes)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(leg.nodes, zero, "o", label="Gauss-Legendre")
ax.plot(rad_left.nodes, zero + 1, "o", label="Gauss-Radau (left)")
ax.plot(rad_right.nodes, zero + 2, "o", label="Gauss-Radau (right)")
ax.plot(lob.nodes, zero + 3, "o", label="Gauss-Lobatto")
# ax.plot(cc.nodes, zero + 4, 'o', label="Clenshaw-Curtis")
ax.set_xlabel("Node $x_i$")
ax.set_title(f"Quadrature nodes for n={n}")
ax.legend()
ax.set_ylim(-1, 6)
ax.grid()
ax.set_yticks([])
plt.show()
```


These can be useful for different applications; Gauss-Radau gives useful stability properties for implicit ODE solvers, both Gauss-Radau and Gauss-Lobatto are commonly used in pseudospectral optimal control methods, and Gauss-Lobatto is widely used for high-order spectral element methods.

The schemes themselves are derived similarly to the Gauss-Legendre case, but fixing one of the roots in the polynomial division, e.g. $q_n(x) = (x - a) q_{n-1}(x)$ for Radau.
This amounts to a different weight in the orthogonality requirement and therefore a different member of the orthogonal polynomial family.
For Gauss-Radau with an endpoint at $x = a$ and original weight $w(x) = 1$, the orthogonality condition becomes:

$$
\int_a^b (x - a) \, q_i(x) \, q_j(x) \, dx = 0, \qquad i \neq j
$$

That is, the nodes must be placed at the roots of the polynomials that are orthogonal with respect to this inner product with weight $(x - a)$.