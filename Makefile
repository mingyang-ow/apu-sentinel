.PHONY: setup baseline train evaluate test lint

CONFIG ?= local

setup:
	uv sync --extra cpu
	uv run pre-commit install

baseline:
	uv run python scripts/baseline.py

train:
	uv run python scripts/train.py --config $(CONFIG)

evaluate:
	uv run python scripts/evaluate.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
