# Changelog

All notable changes will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with pre-1.0 development conventions.
In particular, the API is still evolving and may change between minor versions, although we'll aim to document such changes here.

## [Unreleased]

## [0.1.0] - 2025-04-26

### Added
- Initial unofficial "public" release
- Core functionality:
  - NumPy-compatible symbolic arrays
  - Function compilation with `@arc.compile`
  - Automatic differentiation (`grad`, `jac`, `hess`)
  - ODE solvers and integration
  - Optimization and root-finding capabilities
  - PyTree data structures
  - C code generation
- Examples for:
  - Basic usage and function compilation
  - ODE integration (Lotka-Volterra, pendulum)
  - Optimization (Rosenbrock problem)
  - Root-finding and implicit functions
- Documentation:
  - Installation guide
  - Getting started tutorials
  - API reference
  - Conceptual framework explanation
  - "Under the hood" technical details
  - Common pitfalls and gotchas
  - Extended tutorials for "multirotor dynamics" and "deploying to hardware"
