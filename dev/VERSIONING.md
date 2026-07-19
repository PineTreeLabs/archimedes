# Versioning and Release

**In the last PR before a new release**: bump the version in `pyproject.toml` with:

```bash
uv version --bump <major,minor,patch>
```

Ensure `CHANGELOG.md` is updated as needed, and then finish the PR as usual.

Tag the new version and push with:

```bash
git tag v0.X.X
git push origin --tags
```

This will trigger a GitHub action to publish to PyPI.