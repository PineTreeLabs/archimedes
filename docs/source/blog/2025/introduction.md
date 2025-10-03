---
title: Introducing Archimedes
description: Public beta release announcement
author: Jared Callaham
---

# [Introducing Archimedes]{.hidden-title}

```{image} _static/rocket_intro.png
:alt: Introducing Archimedes
```

**_A Python toolkit for hardware engineering_**

Jared Callaham • 30 Sep 2025

---

A great engineer (controls being no exception) has to be part hacker, part master craftsman.

You have to be a hacker because things rarely "just work" in the real world without a little... creativity.
But you can't _only_ be a hacker; developing complex systems in aerospace, automotive, robotics, and similar industries demands a disciplined, methodical approach.
You need tools that let you be both fast _and_ rigorous.

<!--

Not clear how the _rigor_ translates here. Speed -> Python, but rigor -> C?  Be explicit about where the rigor comes from.

-->

Modern deep learning frameworks solved this years ago — you can develop in PyTorch or JAX and deploy anywhere.
But those tools were built for neural net models, GPUs, and cloud deployments, not dynamics models, MCUs, and HIL testing.

That's where Archimedes comes in.
The goal is to build an open-source "PyTorch for hardware" that gives you the productivity of Python with the deployability of C.
In short, Archimedes is a Python framework that lets you develop and analyze algorithms in NumPy and automatically generate optimized C code for embedded systems.

```{image} _static/dev_workflow.png
:class: only-light
```

```{image} _static/dev_workflow_dark.png
:class: only-dark
```

## The Linchpin: Python → C Code Generation

Archimedes started with the question, **"What would you need to actually do practical control systems development in Python?"**

As a high-level language, it's hard to beat Python on design principles like progressive disclosure, flexibility, and scalability.
The numerical ecosystem (NumPy, SciPy, Matplotlib, Pandas, PyTorch, etc.) is also excellent.
The problem is that **none of it can deploy to typical embedded systems.**

If you need to deploy to hardware today, you have a few basic options:

1. Work in a high-level language like Python or Julia and manually translate algorithms to C code
2. Work entirely in a low-level language like C/C++ or Rust
3. Adopt an expensive vendor-locked ecosystem that supports automatic code generation

While running Python itself on a microcontroller is growing in popularity for educational and hobby applications, there's no real future for pure Python in real-time mission-critical deployments.

However, if you could do seamless C code generation from standard NumPy code, you could layer on simulation and optimization tools, building blocks for physics modeling, testing frameworks, and other features of comprehensive controls engineering toolchains.
But without the code generation, there will always be a gulf between the software and the hardware deployment.

:::::{dropdown}  **Kalman Filter Comparison**

Below are two implementations of a Kalman filter, an algorithm that combines noisy sensor measurements with a prediction model to estimate system state.
This is what's behind GPS navigation, spacecraft guidance, and sensor fusion in millions of devices.

On the left is hand-written C code, and on the right is a NumPy version that can be used to generate an equivalent function.

Here we'll show an implementation for the common case of a single sensor, which avoids having to use a library for matrix inversion in C (though Archimedes does support operations like Cholesky factorization).

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card}  **Handwritten C**

```c
#include <stdint.h>
#define N_STATES        4

typedef struct {
    float H[N_STATES];  // Measurement matrix (1 x n)
    float R;            // Measurement noise covariance (scalar)
} kf_params_t;

typedef struct {
    float x[N_STATES];            // State estimate
    float P[N_STATES][N_STATES]; // Estimate covariance
} kf_state_t;

typedef struct {
    float K[N_STATES];              // Kalman gain (n x 1)
    float M[N_STATES][N_STATES];    // I - K * H temporary
    float MP[N_STATES][N_STATES];   // M * P temporary
    float MPMT[N_STATES][N_STATES]; // M * P * M^T temporary
    float KRKT[N_STATES][N_STATES]; // K * R * K^T temporary
} kf_work_t;

/**
 * Kalman filter update step (scalar measurement case)
 *
 * Mathematical formulation:
 *   y = z - H·x                      (innovation)
 *   S = H·P·H^T + R                  (innovation covariance)
 *   K = P·H^T·S^(-1)                 (Kalman gain)
 *   x' = x + K·y                     (state update)
 *   P' = (I-KH)·P·(I-KH)^T + K·R·K^T (Joseph form covariance)
 *
 * @param z: Latest measurement
 * @param kf_state: Pointer to Kalman filter state struct
 * @param kf_params: Pointer to Kalman filter parameters struct
 * @param kf_work: Pointer to Kalman filter work struct (for temporaries)
 * @return: 0 on success, -1 on error
 */
int kalman_update(float z, kf_state_t *kf_state,
                  const kf_params_t *kf_params,
                  kf_work_t *kf_work) {
    #ifdef DEBUG
    if (!kf_state || !kf_params || !kf_work)
        return -1;
    #endif
    size_t i, j, k;

    // Innovation: y = z - H * x
    float y = z;
    for (i = 0; i < N_STATES; i++)
        y -= kf_params->H[i] * kf_state->x[i];

    // Innovation covariance: S = H * P * H^T + R
    float S = kf_params->R;

    // Compute P * H^T (mv_mult)
    // Using K as temporary storage here
    for (i = 0; i < N_STATES; i++) {
        kf_work->K[i] = 0.0f;
        for (j = 0; j < N_STATES; j++) {
            kf_work->K[i] += kf_state->P[i][j] * kf_params->H[j];
        }
    }
    for (i = 0; i < N_STATES; i++)
        S += kf_params->H[i] * kf_work->K[i];

    // Kalman gain: K = P * H^T / S
    for (i = 0; i < N_STATES; i++)
        kf_work->K[i] /= S;

    // Update state with feedback from new measurement: x = x + K * y
    for (i = 0; i < N_STATES; i++)
        kf_state->x[i] += kf_work->K[i] * y;

    // Joseph form update: P = (I - K * H) * P * (I - K * H)^T + K * R * K^T
    // First compute M = I - K * H
    for (i = 0; i < N_STATES; i++) {
        for (j = 0; j < N_STATES; j++) {
            if (i == j)
                kf_work->M[i][j] = 1.0f - kf_work->K[i] * kf_params->H[j];
            else
                kf_work->M[i][j] = -kf_work->K[i] * kf_params->H[j];
        }
    }

    // Compute M * P
    for (i = 0; i < N_STATES; i++) {
        for (j = 0; j < N_STATES; j++) {
            kf_work->MP[i][j] = 0.0f;
            for (k = 0; k < N_STATES; k++) {
                kf_work->MP[i][j] += kf_work->M[i][k] * kf_state->P[k][j];
            }
        }
    }

    // Compute (M * P) * M^T
    for (i = 0; i < N_STATES; i++) {
        for (j = 0; j < N_STATES; j++) {
            kf_work->MPMT[i][j] = 0.0f;
            for (k = 0; k < N_STATES; k++) {
                kf_work->MPMT[i][j] += kf_work->MP[i][k] * kf_work->M[j][k];
            }
        }
    }

    // Compute K * R * K^T
    for (i = 0; i < N_STATES; i++) {
        for (j = 0; j < N_STATES; j++) {
            kf_work->KRKT[i][j] = kf_work->K[i] * kf_params->R * kf_work->K[j];
        }
    }

    // Final covariance update: P = MPMT + KRKT
    for (i = 0; i < N_STATES; i++) {
        for (j = 0; j < N_STATES; j++) {
            kf_state->P[i][j] = kf_work->MPMT[i][j] + kf_work->KRKT[i][j];
        }
    }

    return 0;
}
```

:::

:::{grid-item-card} **Archimedes Codegen**

```python
@arc.compile
def kalman_update(x, P, z, H, R):
    """Update state estimate with new measurement"""
    I = np.eye(len(x))
    R = np.atleast_2d(R)  # Ensure R is 2D for matrix operations

    y = np.atleast_1d(z - H @ x)  # Innovation
    S = H @ P @ H.T + R  # Innovation covariance  
    K = P @ H.T / S  # Kalman gain (scalar S)

    # Update state with feedback from new measurement
    x_new = x + K * y

    # Joseph form covariance update
    P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    
    return x_new, P_new

# Generate optimized C code:
return_names = ("x_new", "P_new")
args = (x, P, z, H, R)
arc.codegen(kalman_update, args, return_names=return_names)
```

:::
::::

Neither of these implementations is optimized, but it gives a sense of what it looks like to work in either environment.
Of course, for production hand-written code, you'd likely also use optimized linear algebra libraries like CMSIS-DSP and numerical strategies like Cholesky factorization or a square-root form for stability.
But the extra numerical features are only a few extra lines in NumPy, while the hand-written C version becomes more and more complex.

:::::

## Beyond Codegen

While Python → C code generation is foundational to Archimedes, there's much more you can do.

Archimedes has a primarily _functional_ style, meaning that much of the core functionality is exposed through function _decorators_ that transform the function you wrote into a modified function according to the purpose of the decorator.

This design was heavily influenced by JAX and PyTorch, but don't worry if you haven't used these frameworks before.

### Compilation

Archimedes can "compile" a Python function into a C++ _computational graph_, meaning that when you call the compiled function, the entire numerical code gets executed in C++ rather than interpreted Python.

```python
import numpy as np
import archimedes as arc

@arc.compile
def rotate(x, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ], like=x)
    return R @ x

rotate(np.array([1.0, 0.0]), 0.1)
```

These graphs can embed [ODE solves](#archimedes.odeint), [constrained nonlinear optimization](#archimedes.minimize), and more.
For complicated functions you can achieve dramatic speedups over pure Python.

What this does is dynamically building up a C++ representation of the calculation by feeding the function _symbolic arrays_ that match the shape and dtype of your numerical inputs.
Then, instead of operating on the numerical arrays, your code operates on these symbolic replacements, using NumPy's _array dispatch_ mechanism to redirect to CasADi whenever you call NumPy functions.
By the end of this "tracing" step, CasADi has a full view of what the function does and can reproduce it in efficient C++.
Then, any time you call that function again, the C++ equivalent is what actually gets executed.

This is not "Just-In-Time" (JIT) compilation in the sense used by Julia/JAX/Numba, where the Python code is literally compiled down to highly optimized platform-specific machine code.
We'll show some benchmarking in a separate post, but generally what you can expect is that these JIT-compiled frameworks will be somewhat faster than pre-compiled CasADi (and hence, Archimedes).
However, by avoiding the overhead and "unrolling" of true JIT compilation, we get a massive reduction in compilation time for the kind of complex functions typical of advanced controls applications.

For more on how this works (and when it doesn't), see the [What is Archimedes?](../../about.md) and [Under the Hood](../../under-the-hood.md) documentation pages.

### Simulation

Archimedes provides a SciPy-like interface to the powerful and robust [CVODES](https://computing.llnl.gov/projects/sundials/cvodes) solver from SUNDIALS.

```python
xs = arc.odeint(dynamics, (0, 10), x0, t_eval=ts)
```



### Optimization & Root-finding

```python
# Find trim conditions automatically
x_trim = arc.root(residual, x0)
```

### Automatic differentiation

```python
# Linearize for stability analysis
A = arc.jac(dynamics, argnums=1)(t, x_eq, u)
```

- Linearization (stability analysis, controller design)
- Sensitivity analysis
- Gradients for optimization
- Jacobians for implicit ODE/DAE solvers

Note sparse-mode support


### Structured data types

## Why Archimedes?

Existing vision/philosophy

### Why _not_ [language/framework X]?

I'm not here to sell you on a language or framework.
If you're happy with your current development tools, then by all means stick with them.
I know that Julia, JAX, Numba, PyTorch, MATLAB/Simulink, pure CasADi, Rust, C++, Fortran, etc., etc. all have their superfans, and all of these have places where they definitively shine.

What I'll try to do here is give an honest and concise assessment of where Archimedes falls among all of these to give you a sense of whether it might be right for you.

To be clear, what we're talking about is a workflow involving some combination of:

- complex physics modeling and simulation
- advanced control algorithm development
- constrained nonlinear optimization
- deployment to an embedded real-time controller

**Julia**

**JAX/Numba**

**PyTorch**

**MATLAB/Simulink**

**Pure CasADi**

**C/C++/Rust**

I probably shouldn't lump languages this different into the same category, but these are all low-level "systems" languages.

The advantage to working with these directly is that what you write what gets run, giving you complete control over the code.
If you're good, you can write powerful frameworks with high-quality, reliable code that solve _exactly_ your problem.

The downside is that for mere mortals, it's hard to do this well - and doubly hard on a demanding development timeline.
It can also be more complicated to integrate with high-level analysis and design tools - not impossible, just more complicated than working directly in Python.

## Get Started

CTAs to tutorials, quickstart, GitHub

---

:::{admonition} About the Author
:class: blog-author-bio

**Jared Callaham** is the creator of Archimedes and principal at Pine Tree Labs.

*Have questions or feedback? [Open a discussion on GitHub](https://github.com/jcallaham/archimedes/discussions)*
:::