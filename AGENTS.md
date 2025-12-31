# AGENTS.md

## Repository Conventions
- Python 3.14, uses .venv/bin/python; package manager: `uv`.
  - Activate virtual env
    `source .venv/bin/activate`
  - upate dependencies for a subprojectm using app_api as an example:
    `cd app_api; uv sync --active --extra dev`
  - To run test cases
    `uv run --active pytest`
- Lint & format with Ruff.
- Always run before committing:
  1. `uvx ruff check --fix .`
  2. `uvx ruff format .`

## Default Tasks
- **Style fix**: Run `make fix && make format` (Makefile targets exist).
- **Pre-commit**: If missing, install with `uvx pre-commit install`.

## Pull Request Policy
- CI requires `uvx ruff check .` and `uvx ruff format --check .` to pass.

