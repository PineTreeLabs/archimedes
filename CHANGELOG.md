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

## [0.2.0a1] - 2025-08-25

New version tag for release to PyPI (previous "Archimedes" project tagged with v0.1.0).

## [0.2.0a3] - 2025-09-26

### Changes

- Added support for autogenerating C structs from tree data types ([PR #78](https://github.com/PineTreeLabs/archimedes/pull/78))
- Added experimental `Rotation` class ([PR #83](https://github.com/PineTreeLabs/archimedes/pull/83))
- Added `StructConfig` and `UnionConfig` for managing configuration of complex struct hierarchies ([PR #84](https://github.com/PineTreeLabs/archimedes/pull/84))
- Refactored `FlightVehicle` to `RigidBody` ([PR #86](https://github.com/PineTreeLabs/archimedes/pull/86))
- Added new tutorial series on end-to-end controls development workflow w/ HIL proof-of-concept ([PR #88](https://github.com/PineTreeLabs/archimedes/pull/88))
- Added experimental `IIRFilter` class ([PR #88](https://github.com/PineTreeLabs/archimedes/pull/88))
- Moved images, data, notebooks, etc. to LFS ([PR #88](https://github.com/PineTreeLabs/archimedes/pull/88))
- Migrated "PyTree" to "struct" terminology ([Issue #89](https://github.com/PineTreeLabs/archimedes/issues/89))
- Renamed `@pytree_node` decorator to `@struct` ([Issue #89](https://github.com/PineTreeLabs/archimedes/issues/89))

## [0.3.0] - 2025-10-06
- Added MyPy to CI checks
- Fixed all type checking errors
- Use `Rotation` for attitude in `RigidBody`
- Revised "Hierarchical Modeling" tutorial
- Add blog to website
- Convert all notebooks to MyST

## [0.3.1]
- Bugfix for `Rotation.as_euler` for rotations with odd permutations
- Move `RigidBody` and related functionality to `spatial` module
- Move `spatial` module out of `experimental` (+codecov, documentation)