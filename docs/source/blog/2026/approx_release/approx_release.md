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

Jared Callaham • 21 Sep 2026

---

Archimedes development has been pretty quiet for the past few months; I've been using it for consulting projects, but haven't had much time for adding new features.
But today's announcement is a fairly substantial pair of new modules: [`quadrature`](#archimedes.quadrature) and [`approximation`](#archimedes.approximation).

These are available as of `v0.5.0`, and if you're a current Archimedes user you should be able to upgrade with:

```bash
pip install -U archimedes
```

The `quadrature` module is for **Gaussian-style numerical approximation of weighted integrals**:

$$
\int_a^b f(x) ~ w(x) ~ dx \approx \sum_{i=1}^n w_i f(x_i),
$$

and `approximation` is for **function representation with linear basis expansions** of the form:

$$
f(x) \approx \sum_{i=1}^n c_i \phi_i(x).
$$

Currently supported basis families include:

- [Orthogonal polynomials](#archimedes.approximation.OrthogonalPolynomialBasis), including Legendre, Laguerre, Jacobi, Chebyshev, Hermite, and custom measures.
- [Cubic Hermite polynomials](#archimedes.approximation.CubicHermiteBasis)
- [Fourier series expansions](#archimedes.approximation.FourierBasis)
- [Lagrange polynomials](#archimedes.approximation.LagrangeBasis)
- [Monomials](#archimedes.approximation.MonomialBasis)
- [B-splines](#archimedes.approximation.BSplineBasis)
- Piecewise tiling of (most of) the above families (i.e. finite or spectral elements)
- Tensor bases for multivariate functions, supporting arbitrary combinations of the above per dimension
- Custom bases constructed by constraining or concatenating other bases (e.g. bubble + vertex functions)

These two modules are nicely complementary; `quadrature` provides the numerical integration used to define inner products between function spaces in `approximation`, while `approximation` implements (among other things) the orthogonal polynomial families that Gaussian quadrature is built around.

Those two lines of math are richer than they might appear, especially in terms of their potential applications.
To get a sense of this, the rest of the post will walk through a few minimal examples covering PDE solving, trajectory optimization, system identification, and uncertainty quantification - all of which build on the same quadrature and function approximation infrastructure.

## Application examples

I'll be adding several pages to the "tutorials" section of the docs to go into more depth; these are just quick examples to get a feel for how it works and what it can be used for.

```{code-cell} python
:tags: [hide-cell]
# ruff: noqa: N802, N803, N806, N815, N816

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import archimedes as arc
from archimedes.approximation import FunctionSpace
```

```{code-cell} python
:tags: [remove-cell]
from pathlib import Path

plot_dir = Path.cwd() / "_plots"
plot_dir.mkdir(exist_ok=True)
```

The core mechanisms are explained in the "handbook" pages for [quadrature](../../../handbook/quadrature.md) and [function approximation](../../../handbook/approximation.md).
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
c_sol = arc.root(res, c0)
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

Trajectory optimization is essentially the numerical solution of an optimal control problem: what sequence of inputs should we give a system to get it to minimize some cost or objective function subject to dynamical constraints, endpoint constraints, bounds, and potentially other (e.g. path) constraints.
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

quad_rule = arc.quadrature.composite_quad(
    arc.quadrature.gauss_lobatto(p + 1), [-1.0, 1.0]
)
V = FunctionSpace.piecewise(
    "lagrange", p, breakpoints=[t0, tf], nodes="lobatto", quad_rule=quad_rule
)
tp, w = V.quadrature()


def f(x, u):  # noqa: F811
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
ax[0].scatter(
    tp, x_opt(tp)[:, 0], c=ax[0].lines[0].get_color(), label="Optimal trajectory"
)
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
    ax[0].scatter(
        tp, x_opt(tp)[:, 0], c=ax[0].lines[0].get_color(), label="Optimal trajectory"
    )
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

And of course, this is all compatible with the [codegen system](../../../tutorials/codegen/codegen00.md), so you can either deploy the optimized state/control functions and interpolate them online in a feedforward/feedback scheme, or re-solve the optimal control problem online for a model-predictive control scheme. (Although note that CasADi only supports codegen for certain NLP solvers: SQP but not IPOPT).

### System identification

One less obvious place where flexible function approximation can be useful is in _gray-box system identification_.
The "gray-box" designation here basically means "partially known physics", and we fill in the gaps with "black box" components.
The `approximation` module can supply black box components ranging from simple linear lookup tables to B-splines and orthogonal polynomials.

As a simple example, here's some data generated by simulating a spring-mass system with a complex nonlinear friction law combining Coulomb, Stribeck, and viscous friction.
This will be the topic of a full upcoming tutorial or post; this is the quick version.

The basic known physics is Newton's law: for a known external forcing $u(t)$, the system evolves according to

$$
m \dot{v} + F_f(v) + kx = u(t).
$$

The free parameters $m$ and $k$ can either be measured directly or inferred via standard [parameter estimation](../battery_sysid/battery_sysid.md).
If we don't know the friction function $F_f(v)$, we can parameterize it with a basis expansion just as before:

$$
F_f(v) \approx \sum_{i=1}^n c_i \phi_i(v).
$$

The trick, of course, is choosing *which* basis expansion to use, and which other constraints will result in a physically meaningful approximation.
That's a rabbit hole we won't go down today, but you'll see in a moment what happens when we *don't* treat those issues carefully enough.

The data comes from a "chirp" response:

```{code-cell} python
:tags: [hide-output]
df = pd.read_csv("chirp_response.csv")

data = arc.sysid.Timeseries(
    ts=df["t"].values, ys=df[["x"]].values.T, us=df[["u"]].values.T
)
```

```{code-cell} python
:tags: [hide-cell, remove-output]
fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax[0].plot(data.ts, data.us[0])
ax[0].set_ylabel("Force [N]")
ax[0].grid()
ax[1].plot(data.ts, data.ys[0])
ax[1].set_ylabel("Position [m]")
ax[1].grid()
ax[-1].set_xlabel("Time [s]")
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)

    fig, ax = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    ax[0].plot(data.ts, data.us[0])
    ax[0].set_ylabel("Force [N]")
    ax[0].grid()
    ax[1].plot(data.ts, data.ys[0])
    ax[1].set_ylabel("Position [m]")
    ax[1].grid()
    ax[-1].set_xlabel("Time [s]")

    plt.savefig(plot_dir / f"approx_release_3_{theme}.png")
    plt.close()
```

```{image} _plots/approx_release_3_light.png
:class: only-light
```

```{image} _plots/approx_release_3_dark.png
:class: only-dark
```

The chirp response shows the characteristic stick-slip pattern at low frequencies, giving way to a more typical resonance peak and then damping.

The following code fits a "gray-box" model to this data; see [the parameter estimation tutorial](../../../tutorials/sysid/parameter-estimation.md) and the [Li-ion battery modeling](../battery_sysid/battery_sysid.md) blog post for more background on the prediction error method and system identification.
We'll use a piecewise cubic Hermite polynomial for a smooth but flexible function approximation.

```{code-cell} python
:tags: [hide-output]
dt = data.ts[1] - data.ts[0]

# Piecewise-cubic Hermite spline
n_el = 2
bkpts = np.linspace(0, 1, n_el + 1)
V = FunctionSpace.piecewise("hermite", 3, bkpts, continuity=1)


@arc.struct
class ModelParameters:
    c: np.ndarray  # Friction coefficients
    k: float  # Spring constant
    m: float  # Mass


# Dynamics model (discretized with RK4)
@arc.discretize(dt=dt, method="rk4", n_steps=1)
def dyn(t, y, u, p):
    x, v = y
    coeffs, k, m = p.c, p.k, p.m

    # Evaluate friction model, imposing symmetry
    friction = V.function(coeffs)
    F = friction(abs(np.atleast_1d(v))).squeeze()
    F = np.where(v >= 0, F, -F)

    # Known physics from Newton's laws
    F_net = -F - k * x + u
    return np.hstack([v, F_net / m])


# Observation model
def obs(t, y, u, p):
    return y[0]  # Observe position


nx, ny = 2, 1  # State and output dimensions

# Set up noise estimates
noise_var = 0.5 * np.var(np.diff(data.ys[0]))
R = noise_var * np.eye(ny)  # Measurement noise
Q = 1e-2 * noise_var * np.eye(nx)  # Process noise

# Extended Kalman Filter for predictions
ekf = arc.observers.ExtendedKalmanFilter(dyn, obs, Q, R)


params_guess = ModelParameters(
    c=np.zeros(V.n_basis),
    k=25.0,
    m=1.0,
)
P0 = 1e-3 * np.eye(nx)  # Initial state covariance estimate
result = arc.sysid.pem(ekf, data, params_guess, x0=np.zeros(2), P0=P0)
```

Now we can run the model forward to see how well it matches the data.
Of course, in a real setting, we'd want to do this against _held-out cross-validation_ data, but this works for a demo.
Since we wrote the dynamics function as a discrete RK4 stepper for compatibility with parameter estimation, we'll just run this as a plain for-loop:

```{code-cell} python
xs_pred = np.zeros((nx, len(data.ts)))
xs_pred[:, 0] = np.zeros(nx)  # Initial state guess
for i in range(1, len(data.ts)):
    t = data.ts[i - 1]
    u = data.us[:, i - 1]
    xs_pred[:, i] = dyn(t, xs_pred[:, i - 1], u, result.p)
```

```{code-cell} python
:tags: [hide-cell, remove-output]
fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax[0].plot(data.ts, data.ys[0], label="Data")
ax[0].plot(data.ts, xs_pred[0], "--", alpha=0.8, label="Predicted")
ax[0].set_ylabel("Position [m]")
ax[0].grid()
ax[0].legend()
vs_fd = np.gradient(data.ys[0], data.ts)
ax[1].plot(data.ts, vs_fd, label="Data")
ax[1].plot(data.ts, xs_pred[1], "--", alpha=0.8, label="Predicted")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Velocity [m/s]")
ax[1].grid()
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)

    fig, ax = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    ax[0].plot(data.ts, data.ys[0], label="Data")
    ax[0].plot(data.ts, xs_pred[0], "--", alpha=0.8, label="Predicted")
    ax[0].set_ylabel("Position [m]")
    ax[0].grid()
    ax[0].legend()
    vs_fd = np.gradient(data.ys[0], data.ts)
    ax[1].plot(data.ts, vs_fd, label="Data")
    ax[1].plot(data.ts, xs_pred[1], "--", alpha=0.8, label="Predicted")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Velocity [m/s]")
    ax[1].grid()

    plt.savefig(plot_dir / f"approx_release_4_{theme}.png")
    plt.close()
```

```{image} _plots/approx_release_4_light.png
:class: only-light
```

```{image} _plots/approx_release_4_dark.png
:class: only-dark
```

Looks pretty nice!
Although one major caveat: this quick-and-dirty parameter estimation run is clearly overfit, with not enough data towards the high-velocity end of the curve.
The model is well-behaved for smaller velocities (where it had more training data), but then oscillates wildly and even predicts negative friction at high velocities.


```{code-cell} python
:tags: [hide-cell, remove-output]
est_friction = V.function(result.p.c)

nodes = V.basis_matrix().nodes

v_plt = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(1, 1, figsize=(7, 3))
(line,) = ax.plot(v_plt, est_friction(v_plt), "--", alpha=0.8, label="Approximation")
ax.plot(
    nodes, est_friction(nodes), ".", color=line.get_color(), label="Quadrature Nodes"
)
ax.legend()
ax.set_xlabel("Velocity [m/s]")
ax.set_ylabel("Friction Force [N]")
ax.set_xlim([0.0, 1.0])
# ax.set_ylim([-0.1, 2.0])
ax.grid()
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    (line,) = ax.plot(
        v_plt, est_friction(v_plt), "--", alpha=0.8, label="Approximation"
    )
    ax.plot(
        nodes,
        est_friction(nodes),
        ".",
        color=line.get_color(),
        label="Quadrature Nodes",
    )
    ax.legend()
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Friction Force [N]")
    ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([-0.1, 2.0])
    ax.grid()

    plt.savefig(plot_dir / f"approx_release_5_{theme}.png")
    plt.close()
```

```{image} _plots/approx_release_5_light.png
:class: only-light
```

```{image} _plots/approx_release_5_dark.png
:class: only-dark
```

This is (a) why you need cross-validation, and (b) why building in physics-motivated or at least heuristic constraints is crucial for nonlinear system identification.
But again, that's not the point for today, which is that the `approximation` module gives you a wide menu of flexible expansion bases that compose cleanly with simulation, optimization, and code generation.

### Uncertainty quantification

One other application that might not be immediately obvious is uncertainty quantification.
[Polynomial chaos](https://en.wikipedia.org/wiki/Polynomial_chaos) in particular relies on function approximations and quadrature in a fundamental way.

The basic idea is to represent a random variable as a function of other random variables using - you guessed it - a linear basis expansion.
From Wikipedia directly, for a random variable $Y$ that depends on other random variables $X$, the polynomial chaos expansion (PCE) is simply:

$$
Y = f(X) \approx \sum_{i=1}^n c_i \Psi_i(X).
$$

The expansion functions are usually orthogonal polynomials that are chosen according to the "Wiener-Askey scheme".
For example, if a variable is normally distributed, you'd use Hermite basis functions (because the weight is Gaussian).
For a uniform variable you'd use Legendre (because the weight is uniform), and so on.
See [the handbook page on quadrature and measures](../../../handbook/quadrature.md) for details.

For a scalar $X$, the basis functions are the normal flavor we've been working with.
For multiple random inputs, you'd typically use a [`TensorBasis`](#archimedes.approximation.TensorBasis) to combine per-input bases constructed based on the input distributions.

The coefficients are determined by L2 projection - literally, evaluating the deterministic function $f(X)$ at quadrature nodes and summing with quadrature weights.
That's what makes PCE so efficient; you can construct a probabilistic model *without* randomization or Monte Carlo that converges exponentially quickly in the number of samples.
The drawback is that the number of basis functions _grows_ exponentially in dimension, so beyond functions of a few variables you'd need something like Smolyak sparse quadrature, and beyond a few dozen variables... you might be back to Monte Carlo.

Once you have the expansion coefficients, you can trivially compute the mean and variance according to:

$$
\mathbb{E}[Y] = c_0, \qquad \mathrm{Var}[Y] = \sum_{i=1}^n c_i^2.
$$

Higher moments can also be evaluated as needed, although the formulas are not as simple.

In any case, `approximation` and `quadrature` make PCE almost trivial to implement.
Let's take a simple example with a closed-form solution introduced by the landmark [Xiu & Karniadakis (2002) paper](https://epubs.siam.org/doi/10.1137/S1064827501387826) that introduced the Wiener-Askey version of PCE that's dominant today.

The problem is a scalar linear ODE $\dot{y} = -k y$, with initial condition $y(0) = 1$ and an uncertain rate constant $k \sim \mathcal{N}(\mu_k, \sigma_k^2)$.
The analytic solution at a fixed time $t_f$ is

$$
y = f(k) = e^{-k t_f},
$$

which is itself a random (lognormal) variable with analytic mean and variance

$$
\mathbb{E}[Y] = e^{-\mu_k t_f + \tfrac12 \sigma_k^2 t_f^2}, \qquad \mathrm{Var}[Y] = e^{-2 \mu_k t_f + \sigma_k^2 t_f^2} \left( e^{\sigma_k^2 t_f^2} - 1 \right).
$$

Usually this isn't available, but we can use it to cross-check the convergence of the numerical approximation.

```{code-cell} python
mu_k = 1.0  # Nominal decay rate [1/s]
sigma_k = 0.3  # Rate-constant uncertainty [1/s]
tf = 3.0  # Evaluation time [s]


def f_decay(k):
    return np.exp(-k * tf)


# Closed-form mean and variance
lam = sigma_k * tf
mu_y_ex = np.exp(-mu_k * tf + 0.5 * lam**2)
var_y_ex = np.exp(-2 * mu_k * tf + lam**2) * (np.exp(lam**2) - 1)

print(f"Exact mean: {mu_y_ex:.6f}")
print(f"Exact std:  {np.sqrt(var_y_ex):.6f}")
```

Constructing a polynomial chaos approximation of $y$ is simple to implement.
Since $k$ is Gaussian, we'll use a (probabilists') Hermite basis, and since this is a probability distribution we have to add the `density=True` flag to ensure normalization:

```{code-cell} python
p = 6
V = FunctionSpace.hermite(
    n_basis=p + 1,
    loc=mu_k,
    scale=sigma_k,
    density=True,
)

# The quadrature points are where the projection will sample
x, _w = V.quadrature()
print(f"Number of PCE sample points: {len(x)}")

# Project the forward map onto the Hermite basis
# This is the PCE approximation, then the coefficients give us
# the statistical information.
f_decay_pce = V.project(f_decay)
c = f_decay_pce.coefficients

mu_y_pce = c[0]
var_y_pce = np.sum(c[1:] ** 2)
mu_y_err = abs(mu_y_pce - mu_y_ex)
sigma_y_err = abs(np.sqrt(var_y_pce) - np.sqrt(var_y_ex))
print(f"error on mean: {mu_y_err:.3e}")
print(f"error on std:  {sigma_y_err:.3e}")
```

```{code-cell} python
:tags: [remove-cell]
assert mu_y_err < 1e-8
assert sigma_y_err < 1e-5
```

Pretty good error for a handful of samples!
How does this compare to Monte Carlo?

```{code-cell} python
:tags: [hide-cell, remove-output]

# Polynomial chaos statistics

orders = np.arange(9)
pce_mean_err = np.zeros_like(orders, dtype=float)

for p in orders:
    V = FunctionSpace.hermite(n_basis=p + 1, loc=mu_k, scale=sigma_k, density=True)
    f_decay_pce = V.project(f_decay)
    c = f_decay_pce.coefficients

    pce_mean_err[p] = abs(c[0] - mu_y_ex)


# Monte Carlo statistics
rng = np.random.default_rng(0)
n_samples_max = 1_000_000
k_samples = rng.normal(mu_k, sigma_k, n_samples_max)
y_samples = f_decay(k_samples)

n_conv = np.logspace(1, 6, num=13, dtype=int)
mc_mean_err = np.array([abs(np.mean(y_samples[:n]) - mu_y_ex) for n in n_conv])

# Compare convergence rates
fig, ax = plt.subplots(1, 1, figsize=(7, 3))
ax.loglog(n_conv, mc_mean_err, ".-", label="Monte Carlo")
ax.loglog(orders + 1, np.maximum(pce_mean_err, 1e-16), ".-", label="PCE (Hermite)")
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel("Absolute error in mean")
ax.set_title("PCE vs. Monte Carlo convergence")
ax.legend()
ax.grid()
plt.show()
```

```{code-cell} python
:tags: [remove-cell]

for theme in {"light", "dark"}:
    arc.set_theme(theme)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.loglog(n_conv, mc_mean_err, ".-", label="Monte Carlo")
    ax.loglog(orders + 1, np.maximum(pce_mean_err, 1e-16), ".-", label="PCE (Hermite)")
    ax.set_xlabel("Number of function evaluations")
    ax.set_ylabel("Absolute error in mean")
    ax.set_title("PCE vs. Monte Carlo convergence")
    ax.legend()
    ax.grid()

    plt.savefig(plot_dir / f"approx_release_6_{theme}.png")
    plt.close()
```

```{image} _plots/approx_release_6_light.png
:class: only-light
```

```{image} _plots/approx_release_6_dark.png
:class: only-dark
```

It's like magic... at least for a univariate function.
Again, the number of quadrature points scales exponentially with the dimension of the space $\sim n^d$ (number of random inputs), so Monte Carlo becomes competitive again after a point.

Still, PCE is useful for cases where function evaluations are expensive, or for applications like optimization under uncertainty where you need tight statistical convergence quickly.
And with the `quadrature` and `approximation` modules, you get it almost for free!

## A Little History

Now that we've seen some of the applications for the quadrature and function approximation infrastructure, I want to zoom out and explain a bit why I'm excited about these new modules, as nerdy as that is.

Archimedes actually began life as a trajectory optimization project I called `coco` (Collocated Control).
I was trying to write my own version of the legendary [GPOPS-ii](https://gpops2.com/) algorithm based on Rao et al's papers.
In fact, it actually started in JAX and then moved to Julia before landing on CasADi (but that's a whole other story).

The original `coco` code is still on the [Archimedes GitHub](https://github.com/PineTreeLabs/archimedes/tree/64afb74e923c3a91b0d97b17e2ac03ceb8c99904/src/archimedes/experimental/coco), and I still think it was a respectable optimal control code.
So after all this time, why hasn't there been a single trajectory optimization example published in Archimedes?

One reason was just priority; as satisfying as it is to see a trajectory optimization work, there are a lot of great open-source trajectory optimization codes already: [acados](https://docs.acados.org/), [Dymos](https://openmdao.github.io/dymos/), and [PSOPT](https://www.psopt.net/), to name a few.
As a consequence, I thought that a [smooth path to hardware](../../../tutorials/deployment/deployment00.md) for basic control algorithms and real-time simulation could be more impactful than one more trajopt code.
As a friend with a lot of experience in automotive controls says, "nobody optimizes anything; in real life it's all just state machines and lookup tables".

But more relevant here, the second reason was the coding equivalent of writer's block.  `coco` included a lot of one-off implementations of things like Gauss-Radau quadrature, barycentric interpolation, and collocation on spectral elements.
My feeling was that these were the tip of some bigger iceberg, but I couldn't quite figure out what that should be or how to implement it in a useful, modular way.
Eventually, I realized that the unifying theme was the _function approximation via linear basis expansion_ we've been discussing throughout this post.

My hope is that implementing these as modular and composable building blocks will support the use cases from the "Applications" section above (including trajectory optimization), but also allow you to build whatever oddball solver or algorithm you want without fully reinventing the numerical wheel.

If you do build something cool with this (and it's not sensitive or proprietary), feel free to share on the [Discussions](https://github.com/PineTreeLabs/archimedes/discussions) page! It would be great to see what ends up being useful, painful, or any other feedback.

## Read On

There's a lot of math and code that is barely covered here.
For more background on what the new modules do and why, check out the "handbook" pages on [quadrature](../../../handbook/quadrature.md) and [function approximation](../../../handbook/approximation.md).

For more on system identification and how it works in Archimedes, start with the [parameter estimation tutorial](../../../tutorials/sysid/parameter-estimation.md) and [Li-ion battery modeling blog post](../battery_sysid/battery_sysid.md).

More deep-dive posts and tutorials to follow!

---

:::{admonition} About the Author
:class: blog-author-bio

**Jared Callaham** is the creator of Archimedes and principal at Pine Tree Labs.
He is a consulting engineer on modeling, simulation, optimization, and control systems with a particular focus on applications in aerospace engineering.

*Have questions or feedback? [Open a discussion on GitHub](https://github.com/pinetreelabs/archimedes/discussions)*
:::