# Common dev operations for excitertools.
# Run `just` (or `just --list`) to see available recipes.

# Show available recipes.
default:
    @just --list

# Sync the venv with locked dev/test dependencies.
sync:
    uv sync --group test

# Run lint checks.
lint:
    uv run --group test ruff check

# Run the test suite, including module doctests.
test:
    uv run --group test pytest excitertools tests/

# Run tests with coverage (lcov output for Coveralls).
coverage:
    uv run --group test pytest --cov=excitertools --cov-report=lcov:coverage.lcov excitertools tests/

# Regenerate README.rst from the module docstrings.
docs:
    uv run --group test python regenerate_readme.py -m excitertools/__init__.py > README.rst

# Build the sdist and wheel locally.
build:
    uv build

# Bump + tag + push a release. Pushing a v* tag triggers PyPI publish.
# Usage: `just release` for patch, or `just release minor` / `just release major`.
release bump="patch":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "error: working tree is dirty; commit or stash first" >&2
        exit 1
    fi
    uv version --bump {{bump}}
    new_version=$(uv version --short)
    git commit -am "Bump to ${new_version}"
    git tag "v${new_version}"
    git push --follow-tags
    echo "released v${new_version}"
