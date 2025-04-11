

## Formatting

To ignore case conventions for examples that use this quasi-mathematical notation, add the following line to the top of the file (or the first code block in a Jupyter notebook):

```python
# ruff: noqa: N802, N803, N806, N815, N816
```

## Security

* Do not use `assert` statements in package code, since these are flagged as low-severity issues by Bandit.  It's fine to use them in tests.