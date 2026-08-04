
## Basis implementations

- ~~Cubic Hermite~~
- ~~Chebyshev~~
- ~~Fourier~~
- ~~Monomials~~
- Bernstein polynomials
- ~~RBF~~ (deferred - should live in interpolation, not function approximation)
- B-splines
- ~~PCHIP~~ (deferred - this is an interpolation function that produces a standard cubic Hermite polynomial)
- Karhunen-Loeve
- Support for POD/eigenfunction expansions/model reduction?
- Quintic Hermite / higher-order C1 elements (natural extension of CubicHermiteBasis)
- Sinc basis (Boyd ch. 16) -- alternative to Hermite/Laguerre on unbounded domains
- Rational Chebyshev (Boyd ch. 17, TL/TB) -- alternative to Hermite/Laguerre on semi-infinite/infinite domains
- ~~SEM bubble/vertex bases - more generally, constrained/derived and concatenated bases~~

## Examples:

1. ~~Polynomial chaos expansion - Hohmann maneuver~~
2. Gray-box system ID: nonlinear friction?
3. Spectral/pseudospectral PDE solves (+ method of lines) - scalar PDEs (convection-diffusion, KdV, similar)
4. 1D FEM - nonlinear Poisson, Euler-Bernoulli, Timoshenko beam (shear locking), fluid flow?
5. Boundary value problems - Blasius
6. ~~Eigenvalue problem - Sturm-Liouville with custom basis~~
7. Trajectory optimization (pseudospectral/Hermite-Simpson) - block push problem, Cart-Pole swingup
8. Lagrangian continuum mechanics - elastic pendulum

## Challenge/motivating problems

1. Fluid flow w/ spectral/pseudospectral methods: Taylor-Couette & Rayleigh-Benard + weakly nonlinear analysis, chaotic thermosiphon (+ Lorenz phase-space plots)
2. Cosserat rods (PyElastica), rope/tether problems
3. Multistage trajectory optimization and/or low-thrust orbit transfer (see old examples/trajopt notebooks)
4. Aeroelasticity: FEM wing model + Theodorsen-type aero (or state-space approximations)
5. Inverted flag model (coupled FSI - research problem)