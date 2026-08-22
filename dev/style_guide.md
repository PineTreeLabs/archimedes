# Archimedes style guide

The goal of this style guide is to provide and enforce a clear and consistent style across Archimedes written content, including code, documentation, tutorials, and blog posts.

## Naming conventions

Archimedes is math-heavy code, and standard Python naming rules (PEP 8, enforced here via Ruff's `N` rules) actively hurt readability for code that implements a specific equation. This section covers when to break from `snake_case` and how to name the resulting symbols consistently.

### Mathematical notation exception

**Use a mathematical symbol** (`M`, `F`, `R`, single/double letters, capitalized where the mathematical convention capitalizes it) when:

* The code directly implements a documented equation.
* The name corresponds to a well-known symbol (`F` for force, `I` for inertia, `M` for mass matrix).
* Matching the paper/textbook notation makes the code easier to check against the theory.

**Use a descriptive `snake_case` name** when:

* The quantity has no standard symbol.
* The code is organizational/infrastructure (config, I/O, dispatch) rather than computational.
* The mathematical context isn't obvious from the surrounding code.

```python
def dynamics(x, u):
    q, v = x[:n], x[n:]        # generalized position, velocity
    M = compute_mass_matrix(q)
    C = compute_coriolis(q, v)
    tau = compute_forces(q, v, u)
    a = np.linalg.solve(M, tau - C @ v)
    return np.concatenate([v, a])
```

Prefer scoping directory-level exceptions (`N802`/`N803`/`N806`/`N815`/`N816`) using `per-file-ignores` to local `# noqa` comments or per-file `select`/`ignore` overrides for these codes.

### Monogram notation

For frame-dependent physical quantities (positions, velocities, rotations, forces), Archimedes follows the [Drake monogram notation convention](https://drake.mit.edu/doxygen_cxx/group__multibody__notation__basics.html). The general pattern is:

```
Quantity_ReferenceTarget_ExpressedIn
```

* **Quantity**: a single letter for the physical quantity — `p` position, `v` translational velocity, `w` angular velocity, `R` rotation matrix, `F` force, `M` moment/torque, and so on.
* **Reference/Target**: the frame(s) involved, as single capital letters. `R_BA` is the rotation matrix that re-expresses a vector given in frame `A` into frame `B`, so that `v_B = R_BA @ v_A`.
* **Expressed-in**: an optional trailing frame subscript when the expressed-in frame isn't the same as the reference frame, e.g. `v_WB_B` — the velocity of `B` relative to `W`, expressed in the `B` frame. When the reference and expressed-in frames are the same, the trailing subscript is dropped (`v_A` means "expressed in `A`," not "expressed in some unstated frame").
* A descriptive middle token is fine when it disambiguates which quantity of that type you mean: `F_aero_B`, `M_prop_B`, `r_CM_B` (a force, a moment, and a position, each expressed in frame `B`).

Full semantics - points vs. frames vs. bodies, spatial vectors, defaults for dropping subscripts - are in the Drake link above.

### Time derivatives

Use the `_dot` / `_ddot` suffix for first and second time derivatives, regardless of whether the base name is a monogram symbol or a descriptive name: `x_dot`, `q_dot`, `v_WB_dot`.
Don't use `xdot` (no separator) or `x_t`.

## Documentation

Docs and docstrings follow three layers: `numpydoc` for docstring structure, Google's developer style guide for prose, and a small set of house rules for things neither covers.
These rules **do not apply** to long-form documentation like blog posts or tutorials.

### Docstring structure (numpydoc)

Follow [`numpydoc`](https://numpydoc.readthedocs.io/en/latest/format.html) conventions, as used by NumPy, SciPy, pandas, and scikit-learn:

* Section order: short summary, deprecation warning (if any), extended summary, `Parameters`, `Returns`, `Yields`/`Receives`, `Other Parameters`, `Raises`, `Warns`, `Warnings`, `See Also`, `Notes`, `References`, `Examples`.
* Triple double-quotes; wrap prose at roughly 75 characters.
* The one-line summary uses **imperative, present-tense mood**: "Compute the norm," not "Computes the norm" or "This will compute...".
* `Parameters`/`Returns`/`Attributes` entries are noun-phrase fragments (`x : array_like` / `Input array.`), not full sentences or active-voice commands.
* Minimize LaTeX; prefer plain Python pseudocode where an equation isn't essential.
* Terminology: "indices" (not "indexes"), "matrices" (not "matrixes").

### Prose style (Google developer documentation style guide)

For narrative docs — README, `docs/source/*.md`, dev guides, and the extended-summary/`Notes` prose inside docstrings — follow the [Google developer documentation style guide](https://developers.google.com/style), with the *Chicago Manual of Style* as a tiebreaker for anything it doesn't cover. This mirrors [NumPy's own policy](https://numpy.org/doc/stable/dev/howto-docs.html).

Rules from Google's guide:

* **Active voice, second person, present tense**: "The function returns a string," not "A string will be returned by the function." ([Highlights](https://developers.google.com/style/highlights))
* **No `-ing` word as the first word of a heading** — write "Compile a function," not "Compiling functions." An `-ing` word later in a heading is fine. ([Headings and titles](https://developers.google.com/style/headings))
* **Sentence case** for headings and titles, not title case.
* **Imperative voice** to introduce procedural steps: "Run the tests with...", not "You should run the tests with...".

### Additional rules (adapted from ASD-STE100)

Neither guide above addresses these, so we add them as house rules, borrowed from the aerospace [ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/) standard:

* **Sentence length**: cap prose sentences (in `Notes`, extended summaries, and narrative docs) at roughly 25–30 words; split run-ons into two sentences.
* **One idea per sentence**: avoid stacking two or more actions/claims with "and" / "as well as" / a semicolon.
* **No bare pronoun openers**: don't start a sentence with "This/It [verb]..." unless the antecedent is the single preceding noun. Repeat the noun instead — "This dtype is used for..." not "This is used for...".
* **Avoid filler/formal words** that have plainer substitutes:

  | Avoid | Prefer |
  |---|---|
  | leverage | use |
  | utilize | use |
  | seamless(ly) | smoothly / directly |
  | robust | reliable / handles X well |
  | comprehensive | complete / full |
  | essentially | (usually just cut it) |
  | encapsulate(s) | contains / wraps |
  | facilitate | help / allow |
  | straightforwardly | directly |

* **Consistent terminology**: pick one name per concept and use it everywhere.

## Security

* Do not use `assert` statements in package code, since these are flagged as low-severity issues by Bandit.  It's fine to use them in tests.
