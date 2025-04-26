# Experimental modules

Everything in this directory is _experimental_, meaning:

1. It is not subject to the requirements on code coverage, type checking, etc.
2. Nothing is exposed above the level of the `experimental` module
3. All APIs should be considered unstable

## Contents

- **`aero`**: Generic tools for aerospace simulations, including 6dof flight vehicle dynamics, sensor models, and environment models
- **`coco`**: A module for "collocated control" - a pseudospectral optimal control (trajectory optimization) solver
- **`nn`**: Simple neural network functionality, including an ADAM optimizer
- **`state_estimation`**: Implementations of Kalman filter-type algorithms, including extended and unscented variants
- **`sysid`**: Implementation of a nonlinear prediction error method (PEM) for parameter estimation
- **`discretize`**: RK4 and Radau5 methods for converting continuous-time ODEs to discrete forward maps
- **`events`**: Preliminary tools for handling periodic events.  Will be superseded by the "task" framework.
- **`lqr`**: Functionality for designing finite-horizon LQR controllers.