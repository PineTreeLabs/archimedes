

## Formatting

To ignore case conventions for examples that use this quasi-mathematical notation, add the following line to the top of the file (or the first code block in a Jupyter notebook):

```python
# ruff: noqa: N802, N803, N806, N815, N816
```

## Documentation

Docs and docstrings follow three layers: `numpydoc` for docstring structure, Google's developer style guide for prose, and a small set of house rules for things neither covers.

### Docstring structure (numpydoc)

Follow [`numpydoc`](https://numpydoc.readthedocs.io/en/latest/format.html) conventions, as used by NumPy, SciPy, pandas, and scikit-learn:

* Section order: short summary, deprecation warning (if any), extended summary, `Parameters`, `Returns`, `Yields`/`Receives`, `Other Parameters`, `Raises`, `Warns`, `Warnings`, `See Also`, `Notes`, `References`, `Examples`.
* Triple double-quotes; wrap prose at roughly 75 characters.
* The one-line summary uses **imperative, present-tense mood**: "Compute the norm," not "Computes the norm" or "This will compute...".
* `Parameters`/`Returns`/`Attributes` entries are noun-phrase fragments (`x : array_like` / `Input array.`), not full sentences — don't force these into active-voice commands; that's not what the convention asks for.
* Minimize LaTeX; prefer plain Python pseudocode where an equation isn't essential.
* Terminology: "indices" (not "indexes"), "matrices" (not "matrixes").

### Prose style (Google developer documentation style guide)

For narrative docs — README, `docs/source/*.md`, dev guides, and the extended-summary/`Notes` prose inside docstrings — follow the [Google developer documentation style guide](https://developers.google.com/style), with the *Chicago Manual of Style* as a tiebreaker for anything it doesn't cover. This mirrors [NumPy's own policy](https://numpy.org/doc/stable/dev/howto-docs.html).

Rules pulled directly from Google's guide:

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

* **Consistent terminology**: pick one name per concept and use it everywhere (e.g. "compiled function," not "compiled function" in one doc and "symbolic function" in another).

## Security

* Do not use `assert` statements in package code, since these are flagged as low-severity issues by Bandit.  It's fine to use them in tests.