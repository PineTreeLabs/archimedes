
## Basis implementations

- Cubic Hermite
- Chebyshev
- Fourier
- Monomials
- RBF
- B-splines
- PCHIP
- Karhunen-Loeve
- Support for POD/eigenfunction expansions/model reduction?
- Custom Basis for Galerkin (zero on boundaries)?
- SEM bubble/vertex bases?
- Timoshenko?


## Examples:

1. Polynomial chaos expansion - Hohmann maneuver
2. Gray-box system ID
3. Spectral/pseudospectral PDE solves (+ method of lines) - scalar PDEs (convection-diffusion, KdV, similar)
4. 1D FEM - nonlinear Poisson, Euler-Bernoulli, fluid flow?
5. Boundary value problems - Sturm-Liouville, Blasius
6. Trajectory optimization (pseudospectral/Hermite-Simpson) - block push problem, Cart-Pole swingup
7. Lagrangian continuum mechanics - elastic pendulum

## Challenge/motivating problems

1. Fluid flow w/ spectral/pseudospectral methods: Taylor-Couette & Rayleigh-Benard + weakly nonlinear analysis, chaotic thermosiphon (+ Lorenz phase-space plots)
2. Cosserat rods (PyElastica), rope/tether problems
3. Multistage trajectory optimization and/or low-thrust orbit transfer (see old examples/trajopt notebooks)
4. Aeroelasticity: FEM wing model + Theodorsen-type aero (or state-space approximations)
5. Inverted flag model (coupled FSI - research problem)