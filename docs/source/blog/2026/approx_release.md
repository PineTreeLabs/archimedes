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

The headline summary is that `quadrature` is for **Gaussian-style numerical approximation of weighted integrals**:

$$
\int_a^b f(x) ~ w(x) ~ dx \approx \sum_{i=1}^n w_i f(x_i),
$$

and the `approximation` is for **function representation with linear basis expansions** of the form:

$$
f(x) \approx \sum_{i=1}^n c_i \phi_i(x).
$$

These two are nicely complementary; `quadrature` provides the numerical integration used to define inner products between function spaces in `approximation`, while `approximation` implements (among other things) the orthogonal polynomial families that Gaussian quadrature is built around.

Those two lines of math are much richer than they might appear, especially in terms of their potential applications.
I want to give a few of these example applications, but first a little backstory on why 

<!-- 
Examples:

- PDE solve
- LGL trajopt
- sysid?
- PCE?
-->

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