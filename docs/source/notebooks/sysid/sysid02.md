# System ID: Core Concepts

Before diving into example system identification workflows, it may be helpful to understand the basic mathematical formulation of the parameter estimation problem.

On the other hand, if you'd rather dive into the code and learn what's happening under the hood later on, then feel free to skip this part and come back another time.

### Classifications

System identification problems can be broadly categorized by the type of model and by the mathematical approach to parameter estimation.
Obvious distrinctions include linear vs. nonlinear models, and state-space vs. frequency domain representations.
Models can also be categorized using Wiener's "box" terminology describing how much theoretical knowledge the model encodes:

- **White-box models** are based on first-principles modeling, although they may include some uncertain parameters (e.g. inertia or damping terms).
- **Gray-box models** combine some level of physics with data-driven approximations for unknown or complex relationships (e.g. efficiency curves or turbomachinery maps).
- **Black-box models** approximate the dynamics with purely empirical representations like NARMAX, SINDy, or neural ODE models.

<!-- TODO: Include "box" graphic here -->
<!-- https://en.wikipedia.org/wiki/System_identification#/media/File:System_identification_methods.png -->

## Mathematical formulation

### State-space models

We will consider problems of the general discrete-time stochastic state-space form:

\begin{align}
x_{k+1} &= f(t_k, x_k, u_k; p) + w_k \\
y_k &= h(t_k, x_k, u_k; p) + v_k,
\end{align}

where $t$ is the time, $x$ are the system states, $u$ are the control inputs, and $p$ are the system parameters.
The process noise $w$ and measurement noise $v$ are assumed to be Gaussian-distributed with covariance matrices $Q$ and $R$, respectively.
We will refer to $f$ as the "dynamics" function and $h$ as the "observation" function.

Commonly, the most natural representation of physical dynamics is as a continuous ordinary differential equation or differential algebraic equation; in these cases the model can be transformed to a discrete-time system using the [`discretize`](#archimedes.discretize) function, which applies a zero-order hold to inputs and uses implicit or explicit Runge-Kutta methods to advance the differential equation.

This kind of model structure can naturally represent gray-box models, and therefore also white-box (by leaving out data-driven components) and black-box (by leaving out any physics modeling).

### The prediction error method

In this series we will focus on one particular approach, called the "Prediction Error Method" (PEM), often associated with the work by [Lennart Ljung](https://www.mit.bme.hu/system/files/oktatas/targyak/9132/Ljung_L_System_Identification_Theory_for_User-ed2.pdf).
Specifically, we will use a nonlinear PEM construction that can handle:

- Linear or nonlinear models
- Any level of known physics
- Input/output or output-only time series
- Stochastic systems with process and measurement noise
- Bounds on the system parameters
- Data from multiple experiments
- General nonlinear constraints

The basic PEM formulation is the nonlinear least-squares problem as given above, but with a "predictor" model $\phi$:

\begin{align}
\hat{x}_{k+1} &= \phi(t_k, \hat{x}_k, u_k; p) \\
\hat{y}_k &= h(t_k, \hat{x}_k, u_k; p).
\end{align}

As implemented here, the method uses a [_Kalman filter_](https://en.wikipedia.org/wiki/Kalman_filter) for the predictor model.
The Kalman filter maintains an estimate $\hat{x}$ of the full state of the system, which is typically not fully measured in practice.
This internal estimate is then used to produce the output predictions $\hat{y}$, which are minimized in a least-squares sense.

### Kalman filtering

You may be familiar with Kalman filtering in the context of control systems, where they are typically used for full-state estimation.
This enables design of state-feedback controllers like a linear-quadratic regulator, which can be more effective for partially observed systems than output-only feedback.

The filters used in the prediction error method are identical to the state estimation formulation.
For linear systems the Kalman filter gives an _optimal_ state estimate accounting for the Gaussian noise, and under some assumptions can be proven to converge to the true state.

The key idea of the filter is to maintain a state estimate $\hat{x}$ that is updated via a feedback mechanism with incoming measurements.
For each new measurement $y_k$, the _innovation_ $e_k = y_k - \hat{y}_k$ is calculated and used to correct the state estimate by

$$
\hat{x}_{k+1} = f(t_k, \hat{x}_k, u_k) + K_k e_k.
$$

The Kalman gain matrix $K_k$ is generally time-varying and helps to account for the noise and estimation errors in an optimal way.

For nonlinear systems, the Kalman filter can be generalized to the _Extended_ Kalman filter (EKF) by simply linearizing about the current state estimate.
This loses the strict guarantees of the linear filter, but is still highly robust and widely used in practical control applications.
For more strongly nonlinear systems, an _Unscented_ Kalman filter may perform even better.

These filters are useful in system identification for several reasons:

1. They provide a principled way of quantifying uncertainty and dealing with sources of noise.
2. They can use measurements to "steer" predictions towards the true value, smoothing out the cost landscape
3. The probabilistic prediction models allow for alternative formulations like maximum likelihood estimation

<!-- TODO: Compare forward prediction vs Kalman estimate for second-order system -->

### Solving the optimization problem

The full PEM formulation of the system ID problem is

\begin{gather}
\min_p \sum_{k=1}^N ||y_k - \hat{y}_k||^2 \\
\hat{x}_{k+1} = \phi(t_k, \hat{x}_k, u_k; p) \\
\hat{y}_k = h(t_k, \hat{x}_k, u_k; p).
\end{gather}

Bounds may also be added on the parameters, and the objective may be further customized as needed (see for example [Part 4](
../../generated/notebooks/sysid/sysid04) of this series).

This is a nonlinear optimization problem which can be solved using least-squares methods like the Gauss-Newton or Levenberg-Marquardt algorithms.
Alternatively, it can be treated as a nonlinear programming problem and solved with general-purpose solvers like BFGS or IPOPT.
However, all of these require the gradient (and possibly the Hessian) of the objective function, which is difficult and error-prone to manually implement as a result of the complex recursive structure of the PEM objective.

Fortunately, Archimedes can leverage the highly efficient automatic differentiation capabilities of CasADi for gradient and (if needed) Hessian calculations.
This means that, as long as you implement your models within the scope of supported operations in Archimedes, you won't have to think about derivatives at all.

Automatic differentiation is handled through the interfaces in the [`optimize`](#archimedes.optimize) module, including [`least_squares`](#archimedes.optimize.least_squares) and [`minimize`](#archimedes.optimize.minimize).

The least-squares interface includes wrappers for the SciPy methods `"lm"` (the MINPACK implementation of Levenberg-Marquardt) and `"trf"` (a trust-region reflective algorithm that supports bounds), as well as `"hess-lm"`, a custom Levenberg-Marquardt implementation that supports bounds by switching to the [OSQP](https://osqp.org/) quadratic programming solver.

Similarly, the `minimize` interface supports both CasADi solvers (SQP and IPOPT) and provides a wrapper to SciPy optimizers like BFGS, automatically constructing gradient and Hessian functions as necessary.

## Code implementation

There are two interfaces for solving system ID problems with the prediction error method: a high-level [`pem`](#archimedes.sysid.pem) function that will set up the optimization problem and call an appropriate solver, or a set of lower-level interfaces that can be used for additional customization.

The high-level [`pem`](#archimedes.sysid.pem) interface requires a predictor model (e.g. a Kalman filter with supplied dynamics and observation functions), a dataset to train on, and an initial guess for the parameters to be estimated.
If known, the initial state can also be provided (otherwise it will also be estimated as part of the optimization).
Upper and lower bounds can also be supplied for each parameter (or set to `np.inf` if unbounded).

```python
result = pem(ekf, data, params_guess, x0=x0, bounds=bounds, method=method)
```

The function will construct an appropriate objective function and call the optimization method specified via the keyword arg (e.g. `"lm"` or `"bfgs"`).

If more control and customization is needed, you can manually create a [`PEMObjective`](#archimedes.sysid.PEMObjective) object that can calculate residuals.  This can then be combined with other objective functions or nonlinear constraints to create advanced system ID workflows.
A simple example of using this low-level interface with multi-experiment data will be covered in [Part 4](
../../generated/notebooks/sysid/sysid04) of this series.