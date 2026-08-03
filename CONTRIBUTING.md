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
