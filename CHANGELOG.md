# Changelog

All notable changes will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with pre-1.0 development conventions.
In particular, the API is still evolving and may change between minor versions, although we'll aim to document such changes here.

## [Unreleased]
- Add support for Gaussian quadrature in `archimedes.quadrature`
- Added `Measure` classes in `archimedes.measure` (currently only used for quadrature weights)
- Added `archimedes.experimental.approximation`: `Basis`/`FunctionSpace`/`Function` for linear basis expansions, with orthogonal polynomial, Lagrange, and piecewise (C⁻¹/C⁰) bases
- `FunctionSpace.quad_rule` is now optional, defaulting to `basis.default_quadrature()`; an explicit rule is validated against the basis rather than silently accepted (a piecewise basis integrated with an unaligned rule gave a quietly incorrect mass matrix)
- `QuadratureRule` records the `breakpoints` a composite rule was tiled across (`None` for a plain rule)
- **Fix incorrect matrix norms**: `np.linalg.norm(A, ord=...)` for 2-D `A` returned CasADi's *entrywise* norms instead of NumPy's *induced operator* norms, silently giving wrong values for `ord=1` and `ord=inf` (`ord=None`/`'fro'` were correct). `ord=1` and `ord=inf` now match NumPy; `ord=2` (spectral norm) raises `NotImplementedError` pending a symbolic SVD, where it previously returned the Frobenius norm
- **Fix `np.roll(a, shift)` silently changing shape**: with `axis=None` and 2-D `a`, the result was returned flattened instead of restored to `a.shape`
- **Fix `np.append(a, b, axis=0)` silently changing shape**: for 1-D inputs the result had shape `(n, 1)` instead of `(n,)`
- Fix `np.sum` with a negative `axis`, which raised `UnboundLocalError` instead of reducing (and gave `UnboundLocalError` rather than `AxisError` when out of bounds)
- Support `keepdims` in `np.sum`; previously any value, including the default `keepdims=False`, raised `TypeError`
- Support a matrix right-hand side in `np.linalg.solve`, as NumPy does; solving `m` right-hand sides now uses a single factorization instead of `m`
- Support `np.ndim`, `np.cumsum`, and `np.cumprod` for symbolic arrays, and the `axis` argument of `np.squeeze`
- Accept NumPy integer scalars (e.g. `np.int64`) as `axis` arguments wherever a Python `int` was previously required

## [0.4.6] - 2026-07-18
- Bump tornado, mistune, click, setuptools, soupsieve, tornado to resolve vulnerabilities (not expected to be relevant to Archimedes specifically)

## [0.4.5] - 2026-06-2
- Support `np.polyval` for symbolic arrays/compiled functions (uses Horner's method, same as NumPy)
- Support autodiff of `compile(buffered=True)`
- Bump nbconvert and pytest versions [PR #154](https://github.com/PineTreeLabs/archimedes/pull/154)
- Extract struct/pytree functionality into a new standalone package [structree](https://github.com/PineTreeLabs/structree)

## [0.4.4] - 2026-02-27
- Bump control requirement to 0.10.2 (required for Lynx diagram export compatibility)
- Bump pip version in uv.lock to 26.0 to resolve vulnerability GHSA-6vgw-5pg2-w6jp
- Bump nbconvert to 7.17.0 to resolve vulnerability GHSA-xm59-rqc7-hhvf
- Enable finite differencing for `callback` functions
- Fix bug in `switch` where args were passed directly during `Function` creation
- Implement DFS for sorting struct order in codegen (fixes struct definition bug)
- Remove Python 3.10+ features to allow for force-installing into legacy environments (3.9 + NumPy 1.20, in particular)

## [0.4.3] - 2025-12-23
- Raise error instead of silently incorrect result on `__bool__` evaluation ([Issue #128](https://github.com/PineTreeLabs/archimedes/issues/128))
- Fix transpose error for nonsquare matrices in buffered compile ([Issue #130](https://github.com/PineTreeLabs/archimedes/issues/130))
- Add support for `np.block`
- Add logic for creating additional struct types in codegen if they have differently-sized children ([Issue #135](https://github.com/PineTreeLabs/archimedes/issues/135))
- Add acronym handling to snake case conversion in codegen (`IIRFilter` -> `iir_filter_t`, not `i_i_r_filter_t`) - ([Issue #136](https://github.com/PineTreeLabs/archimedes/issues/136))
- Bump filelock to 3.20.1 to resolve vulnerabilities ([Issue #138](https://github.com/PineTreeLabs/archimedes/issues/138))

## [0.4.2] - 2025-12-11
- Bump urllib3 to 2.6.0 to resolve vulnerabilities ([Issue #126](https://github.com/PineTreeLabs/archimedes/issues/126))

## [0.4.1] - 2025-11-26
- Performance improvements to `@compile` calls from Python runtime
- Add `buffered` mode to `@compile`
- Support addition and scalar multiplication for `Quaternion`
- Bugfix for non-commutative broadcasting (PR [#122](https://github.com/PineTreeLabs/archimedes/pull/122) and [#123](https://github.com/PineTreeLabs/archimedes/pull/123))

## [0.4.0] - 2025-11-09
- Overhaul `spatial` module: `Attitude` protocol, low-level functions, wrapper classes, `RigidBody` singleton ([Issue #114](https://github.com/PineTreeLabs/archimedes/issues/114))
- Bugfix for struct type name resolution with inner classes ([Issue #115](https://github.com/PineTreeLabs/archimedes/issues/115))

## [0.3.2] - 2025-11-01
- Bump pip to 25.3 to resolve vulnerability ([Issue #103](https://github.com/PineTreeLabs/archimedes/issues/103))

## [0.3.1] - 2025-10-15
- Bugfix for `Rotation.as_euler` for rotations with odd permutations
- Move `RigidBody` and related functionality to `spatial` module
- Move `spatial` module out of `experimental` (+codecov, documentation)

## [0.3.0] - 2025-10-06
- Added MyPy to CI checks
- Fixed all type checking errors
- Use `Rotation` for attitude in `RigidBody`
- Revised "Hierarchical Modeling" tutorial
- Add blog to website
- Convert all notebooks to MyST

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

## [0.2.0a1] - 2025-08-25

New version tag for release to PyPI (previous "Archimedes" project tagged with v0.1.0).

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
