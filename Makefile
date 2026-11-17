.PHONY: install install-serve install-train check test lint format

install:            ## dev tools only (runs anywhere, no GPU)
	uv sync --extra dev

install-serve:      ## project 1 on a GPU box
	uv sync --extra dev --extra serve

install-train:      ## projects 2-4 on a GPU box
	uv sync --extra dev --extra train

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q

check: lint
	uv run mypy
	uv run pytest -q
