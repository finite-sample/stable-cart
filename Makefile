# Makefile for stable-cart project
# Simple, focused targets for essential development tasks

.PHONY: help install test lint format check clean coverage benchmark quick-benchmark stability-benchmark ci-docker

# Default target
help:
	@echo "Available targets:"
	@echo "  install     Install development dependencies"
	@echo "  test        Run all tests"
	@echo "  lint        Run ruff lint and format checks"
	@echo "  format      Apply ruff formatting"
	@echo "  check       Run preen conformance checks"
	@echo "  coverage    Run tests with coverage report"
	@echo "  benchmark         Run comprehensive benchmark (all datasets)"
	@echo "  quick-benchmark   Run quick benchmark (key datasets, fast)"
	@echo "  stability-benchmark  Run stability-focused benchmark"
	@echo "  ci-docker   Run CI pipeline in Docker container"
	@echo "  clean       Clean up generated files"

# Development setup
install:
	uv sync --all-groups

# Testing
test:
	uv run pytest tests/ -v

coverage:
	uv run pytest tests/ -v --cov=stable_cart --cov-report=term-missing --cov-report=html

# Linting and formatting
lint:
	uv run ruff check stable_cart/ tests/
	uv run ruff format --check stable_cart/ tests/

format:
	uv run ruff format stable_cart/ tests/
	uv run ruff check --fix stable_cart/ tests/

check:
	preen check

# Docker-based CI (simple and clean)
ci-docker:
	docker run --rm -v $$(pwd):/app -w /app python:3.11 bash -c \
		"pip install uv && uv sync --all-groups && make lint && make test"

# Benchmarking
benchmark:
	PYTHONPATH=. uv run python scripts/comprehensive_benchmark.py --datasets comprehensive

quick-benchmark:
	PYTHONPATH=. uv run python scripts/comprehensive_benchmark.py --datasets quick --quick

stability-benchmark:
	PYTHONPATH=. uv run python scripts/comprehensive_benchmark.py --datasets stability_showcase

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/ coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
