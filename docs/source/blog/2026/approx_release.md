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

# Quadrature and Function Approximation

**_Two new modules, endless fun_**

Jared Callaham • 1 Sep 2026

---

Archimedes development has been pretty quiet for the past few months; I've been using it for consulting projects, but haven't had much time for adding new features.
But today's announcement is a fairly substantial pair of new modules: [`quadrature`](#archimedes.quadrature) and [`approximation`](#archimedes.approximation).

The summary is that `quadrature` is for **Gaussian-style numerical approximation of weighted integrals**:

$$
\int_a^b f(x) ~ w(x) ~ dx \approx \sum_{i=1}^n w_i f(x_i),
$$

and the `approximation` is for **function representation with linear basis expansions** of the form:

$$
f(x) \approx \sum_{i=1}^n c_i \phi_i(x).
$$

These two are nicely complementary; `quadrature` provides the numerical integration used to define inner products between function spaces in `approximation`, while `approximation` implements (among other things) the orthogonal polynomial families that Gaussian quadrature is built around.

Those two lines of math are much richer than they might appear, especially in terms of their potential applications.
To get a sense of this, the rest of the post will walk through a few minimal examples covering PDE solving, trajectory optimization, system identification, and uncertainty quantification - all of which build on the same quadrature and function approximation infrastructure.

## Application examples

I'll be adding several pages to the "tutorials" section of the docs to go into more depth; these are just quick examples to get a feel for how it works and what it can be used for.

```{code-cell} python
:tags: [hide-cell]
# ruff: noqa: N802, N803, N806, N815, N816

import matplotlib.pyplot as plt
import numpy as np

import archimedes as arc
from archimedes.approximation import FunctionSpace
```

```{code-cell} python
:tags: [remove-cell]
from pathlib import Path

plot_dir = Path.cwd() / "_plots"
plot_dir.mkdir(exist_ok=True)
```

The core mechanisms are explained in the "handbook" pages for [quadrature](../../handbook/quadrature.md) and [function approximation](../../handbook/approximation.md).
To quote the "Function Approximation" page,

> There are four key abstractions:
>
> - `Basis`: the definition of the $\phi(x)$ functions
> - `FunctionSpace`: combination of a basis with a domain and associated quadrature rule, together implying an inner product
> - `BasisMatrix`: the generalized Vandermonde matrix $\boldsymbol{\Phi}$ associated with the basis and quadrature rule
> - `Function`: A coefficient vector for a particular element of a function space, defining a (piecewise) continuous function in terms of a basis expansion.
>
> The four key classes are summarized in the following table:
> 
> | Math concept    | Math notation | Code equivalent | Code convention |
> | --------------- | --------------| --------------- | --------------- |
> | Basis functions | $\{ \phi_i(x) \}_{i=1}^n$ | `Basis` | `phi` |
> | Function space  | $\operatorname{span}\{\phi_1, \dots, \phi_n\}$ | `FunctionSpace` | `V` |
> | Generalized Vandermonde matrix | $\boldsymbol{\Phi}_{ij} = \phi_j(x_i)$ | `BasisMatrix` | `Phi` |
> | Function | $f(x) = \sum_{i=1}^n c_i \phi_i(x)$ | `Function` | `f` |

### PDE solving

Linear function approximation underlies many core algorithms for solving partial differential equations, in particular finite element, spectral element, and (pseudo)spectral methods.
These mainly differ in terms of (a) which basis they use, (b) how they select quadrature points, (c) how they apply boundary conditions, and (d) how they construct a residual for numerical solution.

The "Hello, world!" of PDE solving is probably the linear Poisson equation

$$
-u''(x) = f(x), \qquad u(0) = u(1) = 0.
$$

We can construct a "manufactured solution" by choosing some analytic $u(x)$ and then deriving what $f(x)$ we'd need to achieve it.
A simple one is $u(x) = \sin(\pi x)$, leading to

$$
f(x) = \pi^2 \sin(\pi x).
$$

We'll solve this with a piecewise-linear (CG1) finite element basis:

```{code-cell} python
n_el = 6  # Number of elements
a, b = 0.0, 1.0  # Endpoints
breakpoints = np.linspace(a, b, n_el + 1)

# CG1 Lagrange space
order = 1
V = FunctionSpace.piecewise("lagrange", order, breakpoints)
```

```{code-cell} python
:tags: [hide-cell, remove-output]
# Plotting the CG1 basis functions
x, _w = V.quadrature()
x_plt = np.linspace(a, b, 501)

fig, ax = plt.subplots(1, 1, figsize=(7, 3))
for i in range(V.n_basis):
    e_i = np.eye(V.n_basis)[:, i]
    f_i = V.function(e_i)
    ax.plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
    ax.plot(x, f_i(x), ".", color=ax.lines[-1].get_color())


ax.grid()
ax.set_ylabel(r"$\phi_i(x)$")
ax.set_title("CG1 Lagrange Basis")
ax.legend(loc="upper right")
ax.set_xlabel("$x$")
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    for i in range(V.n_basis):
        e_i = np.eye(V.n_basis)[:, i]
        f_i = V.function(e_i)
        ax.plot(x_plt, f_i(x_plt), label=rf"$\phi_{i}(x)$")
        ax.plot(x, f_i(x), ".", color=ax.lines[-1].get_color())


    ax.grid()
    ax.set_ylabel(r"$\phi_i(x)$")
    ax.set_title("CG1 Lagrange Basis")
    ax.legend(loc="upper right")
    ax.set_xlabel("$x$")

    plt.savefig(plot_dir / f"approx_release_0_{theme}.png")
    plt.close()
```

```{image} _plots/approx_release_0_light.png
:class: only-light
```

```{image} _plots/approx_release_0_dark.png
:class: only-dark
```

We'll solve this in weak form with the Galerkin method.
After integrating by parts, the solution $u(x)$ satisfies the following weak form for any "test function" $v(x)$ from the same function space:

$$
\int_a^b u'(x) ~ v'(x) ~ dx = \int_a^b f(x) ~ v(x) ~ dx, \qquad \forall v \in \mathcal{V}.
$$

If we stack all the test functions $v(x)$ into the $n_q \times n$ basis matrix $\boldsymbol{\Phi}$ (for $n_q$ quadrature points), we can write this as a single linear system:

$$
\boldsymbol{\Phi}^\top \mathbf{D}^\top \mathbf{W} \mathbf{D} \mathbf{u} = \boldsymbol{\Phi}^\top \mathbf{W} \mathbf{f},
$$

where $\mathbf{u}$ and $\mathbf{f}$ denote the evaluation of $u(x)$ and $f(x)$ at the quadrature nodes, $\mathbf{D}$ is a differentiation matrix, and $\mathbf{W}$ are the diagonal quadrature weights.
Dirichlet BCs are imposed by replacing the test function rows corresponding to the boundaries with the conditions $u(a) = u_a$ and $u(b) = u_b$.

This is equivalent to the classical FEA form written in terms of mass and stiffness matrices, but in Archimedes you don't have to explicitly form mass, stiffness, derivative, or weight matrices:

```{code-cell} python
def u_ex(x):
    return np.sin(np.pi * x)


def f(x):
    return np.pi**2 * np.sin(np.pi * x)


Phi = V.basis_matrix()  # Test functions
dPhi = V.basis_matrix(deriv=1)  # Derivative of test functions
x = Phi.nodes
u0 = V.function()
left, right = V.basis.boundary_dofs()

def res(c):
    u = u0.replace(coefficients=c)

    # Evaluate the residual at the quadrature nodes and project onto the test functions
    lhs = dPhi.T @ u(x, deriv=1)
    rhs = Phi.T @ f(x)
    r = lhs - rhs

    # Dirichlet BCs - set the boundary coefficients directly for Lagrange elements
    r[left] = c[left] - 0.0
    r[right] = c[right] - 0.0
    return r
```

The weight matrices are applied automatically with matrix multiplication of the transpose `Phi.T`.

For linear PDEs, this is a single linear solve for the coefficients of $u(x)$.
More generally, this "assembly" step leads to a nonlinear residual, which can be solved with Newton-type root-finding methods:

```{code-cell} python
c0 = u0.coefficients
c_sol = arc.optimize.root(res, c0)
u_sol = u0.replace(coefficients=c_sol)

err = np.max(np.abs(u_sol(x_plt) - u_ex(x_plt)))
print(f"max |u_h - u_ex|: {err:.3e}")
```

```{code-cell} python
:tags: [hide-cell, remove-output]
fig, ax = plt.subplots(1, 1, figsize=(7, 3))
ax.plot(x_plt, u_ex(x_plt), lw=3, alpha=0.4, label="analytic")
ax.plot(x_plt, u_sol(x_plt), "--", label="FEM")
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x)$")
ax.set_title(f"Poisson FEM: {V.basis.n_elements} elements, max error {err:.1e}")
ax.legend()
ax.grid()
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.plot(x_plt, u_ex(x_plt), lw=3, alpha=0.4, label="analytic")
    ax.plot(x_plt, u_sol(x_plt), "--", label="FEM")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x)$")
    ax.set_title(f"Poisson FEM: {V.basis.n_elements} elements, max error {err:.1e}")
    ax.legend()
    ax.grid()

    plt.savefig(plot_dir / f"approx_release_1_{theme}.png")
    plt.close()
```

```{image} _plots/approx_release_1_light.png
:class: only-light
```

```{image} _plots/approx_release_1_dark.png
:class: only-dark
```

The solution error can easily be reduced by either increasing `n_el` ($h$-refinement) or `order` ($p$-refinement).

This approach can work for nonlinear PDEs as well, and with modifications a similar procedure can be applied to (pseudo)spectral and spectral element methods.
One caution, though - the Newton solve here does a direct solve for each linear subsystem.
This is fine for 1D and (possibly) small 2D problems on rectangular domains, but for PDE solving work at scale and on real geometries you'll absolutely still want a specialized library with Krylov solvers, preconditioners, support for unstructured meshes, etc. ([Firedrake](https://www.firedrakeproject.org/documentation.html) is my go-to for FEM).

### Trajectory optimization

Trajectory optimization is essentially the numerical solution of an optimal control problem: what sequence of inputs should we give a system to get it to minimize some "cost" function subject to dynamical constraints, endpoint constraints, bounds, and potentially other (e.g. path) constraints.
There are many kinds of trajectory optimization and the most common depends on the application domain.

For a quick demo, we'll look at the classic "block push" (double integrator) problem using a Legendre-Gauss-Lobatto pseudospectral collocation method.

The "block push" problem is

$$
\min_{u(t), x} \int_0^1 u^2(\tau) ~ d \tau, \qquad \text{subject to} \qquad
\begin{cases}
\dot{x} &= f(t, x, u) \\ 
x(0) &= \begin{bmatrix} 0 & 0 \end{bmatrix}^T \\
x(1) &= \begin{bmatrix} 1 & 0 \end{bmatrix}^T \\
\end{cases}
$$

where the dynamics are the double-integrator

$$
f(t, x, u) = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} x + \begin{bmatrix} 0 \\ 1 \end{bmatrix} u.
$$

Physically, this is a unit mass on a frictionless table, acted on by a force $u$, with the state $x$ representing position and velocity.
We'd like to move it from rest at position 0 to rest at position 1 with the minimal effort.

Pseudospectral trajectory optimization represents the dynamic state $x(t)$ and input controls $u(t)$ with basis expansions:

$$
x(t) \approx \sum_{i=0}^n x_i \, \ell_i(t), \qquad
u(t) \approx \sum_{i=0}^n u_i \, \ell_i(t),
$$

where $\ell_i(x)$ are the [Lagrange basis functions](https://en.wikipedia.org/wiki/Lagrange_polynomial) evaluated at the Gauss-Lobatto quadrature nodes.

This is exactly our linear basis expansion representation, so we can reuse the same function approximation infrastructure, now combined with constrained optimization using [`minimize`](#archimedes.minimize).

Collocation is imposed by differentiating the interpolant using the  $(n+1) \times (n+1)$ differentiation matrix $D_{ij} = \ell_j'(t_i)$ and requiring it to match the dynamics at every node:

$$
\sum_{j=0}^n D_{ij} \, x_j = f(x_i, u_i), \quad i = 0, \ldots, n,
\qquad x_0 = x(t_0), \quad x_n = x(t_f).
$$

This is the "defect" constraint. Boundary conditions, path constraints, etc. can simply be concatenated to form the full constraint vector:

```{code-cell} python
:tags: [hide-output]
t0, tf = 0.0, 1.0
x0, xf = np.array([0.0, 0.0]), np.array([1.0, 0.0])
p = 6  # polynomial degree

quad_rule = arc.quadrature.composite_quad(arc.quadrature.gauss_lobatto(p + 1), [-1.0, 1.0])
V = FunctionSpace.piecewise(
    "lagrange", p, breakpoints=[t0, tf], nodes="lobatto", quad_rule=quad_rule
)
tp, w = V.quadrature()

def f(x, u):
    return np.array([x[1], u[0]], like=x)


def obj(params):
    u_fn = V.function(params["u"])
    return u_fn.dot(u_fn)


def constr(params):
    xp, up = params["x"], params["u"]  # Values at nodes
    x_fn = V.function(xp)

    # State derivative at nodes, from differentiating interpolant
    x_dot = x_fn(tp, deriv=1)

    # Vectorize the dynamics evaluation over the nodes
    x_dot_eval = arc.vmap(f, in_axes=(0, 0))(xp, up)
    defect = x_dot - x_dot_eval

    # Concatenate boundary values
    return np.concatenate([defect, xp[:1] - x0, xp[-1:] - xf]).ravel()


x_guess = x0 + (tp[:, None] - t0) * (xf - x0) / (tf - t0)
init = {"x": x_guess, "u": np.zeros((p + 1, 1))}

res = arc.minimize(obj, x0=init, constr=constr)
sol = res.x
x_opt, u_opt = V.function(sol["x"]), V.function(sol["u"])
```

The block-push problem has a [simple analytic solution](https://epubs.siam.org/doi/10.1137/16M1062569):

```{code-cell} python
def x_ex(t):
    return 3 * t**2 - 2 * t**3


def u_ex(t):
    return 6 - 12 * t


t_plt = np.linspace(t0, tf, 1000)
x_plt = x_opt(t_plt)
u_plt = u_opt(t_plt)

print(f"Max absolute error: {max(abs(x_plt[:, 0] - x_ex(t_plt))):.4e}")
```

```{code-cell} python
:tags: [hide-cell, remove-output]

fig, ax = plt.subplots(2, 1, figsize=(7, 3), sharex=True)
ax[0].plot(t_plt, x_plt[:, 0])
ax[0].scatter(tp, x_opt(tp)[:, 0], c=ax[0].lines[0].get_color(), label="Optimal trajectory")
ax[0].plot(t_plt, x_ex(t_plt), "--", lw=2, label="Exact solution")
ax[0].legend()
ax[0].grid()
ax[0].set_ylabel(r"$x$")
ax[1].plot(t_plt, u_plt)
ax[1].scatter(tp[:-1], u_opt(tp[:-1]), c=ax[1].lines[0].get_color())
ax[1].plot(t_plt, u_ex(t_plt), "--", lw=2)
ax[1].grid()
ax[1].set_ylabel(r"$u$")
ax[1].set_xlabel(r"$t$")

plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)

    fig, ax = plt.subplots(2, 1, figsize=(7, 3), sharex=True)
    ax[0].plot(t_plt, x_plt[:, 0])
    ax[0].scatter(tp, x_opt(tp)[:, 0], c=ax[0].lines[0].get_color(), label="Optimal trajectory")
    ax[0].plot(t_plt, x_ex(t_plt), "--", lw=2, label="Exact solution")
    ax[0].legend()
    ax[0].grid()
    ax[0].set_ylabel(r"$x$")
    ax[1].plot(t_plt, u_plt)
    ax[1].scatter(tp[:-1], u_opt(tp[:-1]), c=ax[1].lines[0].get_color())
    ax[1].plot(t_plt, u_ex(t_plt), "--", lw=2)
    ax[1].grid()
    ax[1].set_ylabel(r"$u$")
    ax[1].set_xlabel(r"$t$")

    plt.savefig(plot_dir / f"approx_release_2_{theme}.png")
    plt.close()
```

```{image} _plots/approx_release_2_light.png
:class: only-light
```

```{image} _plots/approx_release_2_dark.png
:class: only-dark
```

Similar recipes can be used for Hermite-Simpson trajectory optimization, $hp$-adaptive pseudospectral collocation on multi-element meshes, and other trajectory optimization formulations.

Eventually the plan is for Archimedes to provide some built-in functionality so you can just pass objective and constraint functions without hand-rolling the discretization, but for now the core quadrature and interpolation/differentiation machinery is there for you to write custom algorithms.

And of course, this is all compatible with the [codegen system](../../tutorials/codegen/codegen00.md), so you can either deploy the optimized state/control functions and interpolate them online in a feedforward/feedback scheme, or re-solve the optimal control problem online for a model-predictive control scheme. (Although note that CasADi only supports codegen for certain NLP solvers: SQP but not IPOPT).

### System identification



### Uncertainty quantification


## A Little History

Archimedes actually began life as a trajectory optimization project I called `coco` (Collocated Control). I was trying to write my own version of the legendary GPOPS-ii algorithm based on Rao et al's papers.
In fact, it actually started in JAX and then moved to Julia before landing on Casadi (but that's a story for another day).

The original `coco` code is still on the Archimedes GitHub, and I still think it was a respectable optimal control code.
So after all this time, why hasn't there been a single trajectory optimization example published in Archimedes?

The short answer is: I was waiting for this very day.

The longer answer has two parts.
One is just priorities; as satisfying as it is to see a trajectory optimization work, there [are a lot of great open-source trajectory optimization codes already].
More importantly, a [smooth path to hardware](TODO) for basic control algorithms and real-time simulation is much more impactful for most practical applications.  As a friend with a lot of experience in automotive controls says, "nobody optimizes anything; in real life it's all just state machines and lookup tables".

But more relevant here, the second reason was the software design equivalent of writer's block.  CoCo included a lot of one-off implementations of things like Gauss-Radau quadrature, barycentric interpolation, and collocation on spectral elements.  My feeling was that these 