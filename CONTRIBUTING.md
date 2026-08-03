# Contributing

Create changes on a branch from `dev` and open a pull request back to `dev`. Release pull
requests promote tested changes from `dev` to `main`.

## Local setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Required checks

```bash
ruff check .
ruff format --check .
pytest
python -m build
python -m twine check dist/*
```

Tests must not call the live ProsodyAI API. Use `httpx.MockTransport` for API contract tests and
temporary directories for audio-path security tests. Never commit API keys or customer audio.

Changes to the public client must remain aligned with the checked-in ProsodyAI OpenAPI contract.

## Promotion and release automation

A successful push CI run on `dev` creates or updates the `dev` to `main` pull request when
`PROSODYAI_RELEASE_TOKEN` is configured. The workflow never merges or enables auto-merge. If
the secret is absent, it exits successfully and leaves promotion manual. Limit this token to
`ProsodyAI/langchain` with read access to contents and write access to pull requests.

Protect `main` with the four CI matrix checks as required checks. The promotion workflow only
maintains the pull request and must not be treated as an approval or merge boundary.

A successful push CI run on `main` sends a `prosodyai_langchain_main_updated` repository
dispatch to `ProsodyAI/prosodyai`. The payload includes `gitlink_path=packages/langchain` and the
exact green commit SHA. Configure `PROSODYAI_ROOT_DISPATCH_TOKEN` as a token limited to the root
repository with permission to create repository dispatches. If the secret is absent, the
workflow exits successfully and the root repository's polling fallback remains responsible for
the gitlink.

Publishing is deliberately separate from merging. To release:

1. Update both `project.version` in `pyproject.toml` and `__version__` in
   `prosodyai_langchain/__init__.py` on `dev`.
2. Let the green promotion pull request merge that version into `main`.
3. Create an explicit `vX.Y.Z` tag at the merged `main` commit and push the tag.
4. Manually run the `Publish` workflow, provide that existing tag, and enable its confirmation.
5. Approve the protected `pypi` environment after reviewing the resolved tag, version, and SHA.

The workflow rejects a run not dispatched from `main`, a tag that does not exactly match
`project.version`, or a tag not contained in `main`. The `pypi` environment must require
reviewers, restrict deployments to `main`, and be registered as the PyPI Trusted Publisher for
`.github/workflows/release.yml`. Repository variable `PYPI_PUBLISH_ENABLED` must also equal
`true`. Pushing a tag alone never publishes a package.
