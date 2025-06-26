# Calibrating Models with Data

**You have a model of the dynamics of your system.  You have test data from your system.  How do you get them to match?**

With Archimedes, the answer can be surprisingly simple. In this tutorial, you'll learn to automatically calibrate physics-based models with experimental data using just a few lines of code.
Then you can leverage those calibrated models for control design and optimization.

## What You'll Accomplish

By the end of this tutorial, you'll be able to:

- **Fit dynamic models to time series data** using modern optimization methods
- **Handle nonlinear systems** with complex physics and multiple parameters  
- **Incorporate physical constraints** and bounds on parameters
- **Scale to multiple experiments** and different types of data

We'll work through three progressively sophisticated examples:

1. **Second-order oscillator**: Learn the basics with a linear system and step response data
2. **Nonlinear Duffing oscillator**: Add complexity with nonlinear dynamics and explore advanced features
3. **Hammerstein-Wiener system**: Tackle a realistic example with multiple data sources and nonlinear constraints


## The System Identification Challenge

System identification refers to algorithms for fitting dynamical systems models to data. For example, if we have measurements $y_k$ and a model generating predictions $\hat{y}_k(p)$ with parameters $p$, we want to solve:

$$\min_p \sum_{k=1}^N ||y_k - \hat{y}_k(p)||^2$$

This is a nonlinear least-squares problem, but the recursive nature of dynamical systems makes it much more challenging than standard parameter fitting.

Traditional approaches require either manual parameter tuning, expensive Monte Carlo searches, or specialized commercial software. Archimedes provides a modern alternative that's:

- **Automatic**: Gradients computed via automatic differentiation
- **Flexible**: Works with any physics model you can code in NumPy
- **Fast**: Leverages state-of-the-art optimization algorithms
- **Integrated**: Seamless workflow from identification to control design to deployment

## Prerequisites

To follow this tutorial, you'll need:
- Basic familiarity with Python and NumPy
- A Python environment with Archimedes installed
- An understanding of how to define dynamics models in Archimedes

You won't need prior expertise with system identification or optimization methods.

## Tutorial overview

In this tutorial we will start with some fundamentals of system identification, and then walk through three increasingly complicated examples of system ID workflows.

First we see the basic process of calibrating parameters for a second-order oscillator responding to a step function input.
This will give an idea of the basic interfaces and some "cookbook" code for arranging data.

In the second example we will identify parameters for the nonlinear Duffing oscillator.
The fundamental process is no different than the linear second-order system, but we will take the opportunity to explore some more advanced capabilities like filter selection, PyTree parameter structuring, and adding physical bounds.

The third example is a nonlinear Hammerstein-Wiener model with a known structure.
Again, the basic process could be simple, but here we show some of the drawbacks of calibrating with limited data, and mock up a more realistic workflow that combines steady-state and transient responses to accurately identify the model.
This in turn requires using a lower-level interface and setting up a custom optimization problem with nonlinear constraints and additional terms in the objective function.

In this series there are a few important topics we will not cover:

- **Data wrangling:** How should you clean up and preprocess you data?
- **Model parameterization:** How should the system be modeled mathematically and implemented numerically?
- **Noise modeling:** How do you estimate the noise covariance matrices needed for the predictive filters?
- **Validation:** How do you make sure the model is correct?

The reason we won't cover these topics is that they are highly domain- and even problem-specific and there is usually no one-size-fits-all answer.
However, the final part of the series will offer a set of practical recommendations and strategies for fitting reliable models you can have high confidence in.

For more thorough community-based discussions, feel free to post questions or ideas on the [GitHub Discussions page](https://github.com/pinetreelabs/archimedes/discussions).
For commercial or consulting inquiries, email us at [info@archimedes.sh](mailto:info@archimedes.sh).


```{toctree}
:maxdepth: 1
sysid00
../../generated/notebooks/sysid/sysid01
sysid02
../../generated/notebooks/sysid/sysid03
../../generated/notebooks/sysid/sysid04
sysid05
   
```