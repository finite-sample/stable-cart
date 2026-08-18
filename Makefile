# Makefile for stable-cart project
# Simple, focused targets for essential development tasks

.PHONY: help install test lint format check docs doctest clean coverage ci-docker

# Default target
help:
	@echo "Available targets:"
	@echo "  install     Install development dependencies"
	@echo "  test        Run all tests"
	@echo "  lint        Run ruff lint and format checks"
	@echo "  format      Apply ruff formatting"
	@echo "  check       Run preen conformance checks"
	@echo "  docs        Build documentation with warnings as errors"
	@echo "  doctest     Run documentation examples"
	@echo "  coverage    Run tests with coverage report"
	@echo "  ci-docker   Run CI pipeline in Docker container"
	@echo "  clean       Clean up generated files"

# Development setup
install:
	uv sync --all-groups

# Testing
test:
	uv run pytest tests/ -v

coverage:
	uv run pytest tests/ -v --cov=stable_cart --cov-report=term-missing

# Linting and formatting
lint:
	uv run ruff check stable_cart/ tests/ examples/
	uv run ruff format --check stable_cart/ tests/ examples/

format:
	uv run ruff format stable_cart/ tests/ examples/
	uv run ruff check --fix stable_cart/ tests/ examples/

check:
	uv run codespell
	uv run --with preen preen check --strict --skip codespell

docs:
	uv run sphinx-build -W --keep-going -b html docs docs/_build/html

doctest:
	uv run sphinx-build -W --keep-going -b doctest docs docs/_build/doctest

# Docker-based CI (simple and clean)
ci-docker:
	docker run --rm -v $$(pwd):/app -w /app python:3.11 bash -c \
		"pip install uv && uv sync --all-groups && make lint && make test"

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/ coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
