
## Basis implementations

- Cubic Hermite
- Chebyshev
- Fourier
- Monomials
- Bernstein polynomials
- RBF
- B-splines
- PCHIP
- Karhunen-Loeve
- Support for POD/eigenfunction expansions/model reduction?
- Quintic Hermite / higher-order C1 elements (natural extension of CubicHermiteBasis)
- Sinc basis (Boyd ch. 16) -- alternative to Hermite/Laguerre on unbounded domains
- Rational Chebyshev (Boyd ch. 17, TL/TB) -- alternative to Hermite/Laguerre on semi-infinite/infinite domains
- SEM bubble/vertex bases?
- Constrained/derived bases: build a basis as a linear combination of another
  basis (e.g. Chebyshev, Legendre) whose functions satisfy given linear
  constraints (Dirichlet/Neumann/periodic BCs) by construction, via the null
  space of a constraint matrix evaluated at the reference domain. See
  `_constrained.py` sketch -- open question is whether this belongs as a
  built-in wrapper or is better left to worked examples.

Not a basis (moved to Examples): Timoshenko beam theory doesn't need its own
basis family -- it's normally discretized with plain (often mixed-order)
Lagrange elements for the independently-interpolated deflection/rotation
fields, unlike Euler-Bernoulli which needs C1 (Hermite) elements.


## Examples:

1. Polynomial chaos expansion - Hohmann maneuver
2. Gray-box system ID
3. Spectral/pseudospectral PDE solves (+ method of lines) - scalar PDEs (convection-diffusion, KdV, similar)
4. 1D FEM - nonlinear Poisson, Euler-Bernoulli, Timoshenko beam (shear locking), fluid flow?
5. Boundary value problems - Sturm-Liouville, Blasius
6. Trajectory optimization (pseudospectral/Hermite-Simpson) - block push problem, Cart-Pole swingup
7. Lagrangian continuum mechanics - elastic pendulum

## Challenge/motivating problems

1. Fluid flow w/ spectral/pseudospectral methods: Taylor-Couette & Rayleigh-Benard + weakly nonlinear analysis, chaotic thermosiphon (+ Lorenz phase-space plots)
2. Cosserat rods (PyElastica), rope/tether problems
3. Multistage trajectory optimization and/or low-thrust orbit transfer (see old examples/trajopt notebooks)
4. Aeroelasticity: FEM wing model + Theodorsen-type aero (or state-space approximations)
5. Inverted flag model (coupled FSI - research problem)