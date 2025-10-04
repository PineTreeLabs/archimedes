# Testing Guide

## Local testing

You can run a version of the CI test workflow locally as follows.

First, set up an environment using UV and installing development dependencies as described above.
Then you can run the basic unit tests with:

```bash
uv run pytest
```

We require 100% code coverage of the tests to help ensure reliability.  To print a test test coverage report to the terminal run

```bash
uv run pytest --cov=archimedes --cov-report=term-missing
```

Alternatively, generate a detailed report with

```bash
uv run pytest --cov=archimedes --cov-report=html
```

Linting and formatting is done with [ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff format src test examples docs
uv run ruff check --fix src test examples docs
```

We also have a CI test for static type checking with [mypy](https://mypy-lang.org/):

```bash
uv run mypy src/
```

Finally, to build the documentation locally, run

```bash
cd docs
make clean && make html
```

This will scrape API documentation from the docstrings, parse and execute MyST Markdown files, and then create the HTML website from the outputs.
Any tests embedded in the MyST files will also run as part of this workflow.

The outputs will be cached in `.jupyter_cache/` and can be checked for linting with

```bash
uv run ruff check .jupyter_cache
uv run ruff format --diff .jupyter_cache
```

Unfortunately, because the MyST files themselves are neither Jupyter notebooks nor standard Python code, these cached notebooks can't directly be auto-fixed.
Instead, the source `.md` files have to be edited following the feedback from the linter.

### Security scanning

First, scan the project requirements for known vulnerabilities:

```bash
uv export --no-emit-project --format requirements-txt > requirements.txt
uv run pip-audit -r requirements.txt --disable-pip
rm requirements.txt
```

Then run [Bandit](https://bandit.readthedocs.io/) to do a static analysis of the Archimides code itself:

```bash
uv run bandit -r src
```

### License annotation

We use [REUSE](https://reuse.software/) to track licensing.  By default all files tracked by git are licensed under the project license.  To confirm compliance with the REUSE standard, run

```bash
uv run reuse lint
```

## `pip-audit` vulnerabilities

If a security vulnerability is found that does not raise concerns for typical Archimedes usage patterns and cannot be resolved by temporarily rolling back the dependency, the "Security Scan" workflow can be configured to temporarily ignore the vulnerability:

```yaml
    - name: Run pip-audit
      run: |
        # === TEMPORARY SECURITY EXCEPTIONS ===
        #
        # CVE-2025-XXXX (short description)
        # - ID: GHSA-XXXX-XXXX-XXX
        # - Affects:
        # - Resolution:
        # - Tracking: https://github.com/PineTreeLabs/archimedes/issues/YYY
        # - Impact:
        EXPIRY_DATE="YYYY-MM-DD"
        if [[ $(date +%Y-%m-%d) > $EXPIRY_DATE ]]; then
          echo "ERROR: pip-audit exception has expired. Review issue #YYY."
          exit 1
        fi
        uv export --no-emit-project --format requirements-txt > requirements.txt
        uv run pip-audit -r requirements.txt --disable-pip --ignore-vuln GHSA-XXXX-XXXX-XXX
```

The expiration date should typically be 90 days or less, but can be extended if a resolution is not possible in that window.