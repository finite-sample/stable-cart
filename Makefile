# Makefile for stable-cart project
# Provides convenient targets for local development and CI simulation

.PHONY: help install test lint ci-local ci-matrix docs-local clean coverage format check-format

# Default target
help:
	@echo "Available targets:"
	@echo "  install        Install development dependencies"
	@echo "  test          Run all tests"
	@echo "  lint          Run linting checks (black + flake8)"
	@echo "  format        Apply code formatting with black"
	@echo "  check-format  Check code formatting without applying changes"
	@echo "  coverage      Run tests with coverage report"
	@echo "  ci-local      Run CI workflow locally with act (single Python version)"
	@echo "  ci-matrix     Run CI workflow for all Python versions with act"
	@echo "  docs-local    Build documentation locally with act"
	@echo "  benchmark     Run benchmark scripts"
	@echo "  clean         Clean up generated files"

# Development setup
install:
	python3 -m pip install --upgrade pip
	pip install -e ".[dev]"

# Testing
test:
	python3 -m pytest tests/ -v

coverage:
	python3 -m pytest tests/ -v --cov=stable_cart --cov-report=term-missing --cov-report=html

# Linting and formatting
lint:
	@echo "Running linting checks (as in CI)..."
	python3 -m black --check stable_cart/ tests/
	python3 -m flake8 stable_cart/ tests/ --max-line-length=100 --extend-ignore=E203,W503

format:
	@echo "Applying code formatting..."
	python3 -m black stable_cart/ tests/

check-format:
	@echo "Checking code formatting..."
	python3 -m black --check --diff stable_cart/ tests/

# Local CI simulation with act
ci-local:
	@echo "Running CI locally with act (Python 3.11)..."
	@if [ ! -f .actrc ]; then echo "❌ .actrc not found. Please run from project root."; exit 1; fi
	scripts/ci/ci-local.sh

ci-matrix:
	@echo "Running CI matrix testing with act (all Python versions)..."
	@if [ ! -f .actrc ]; then echo "❌ .actrc not found. Please run from project root."; exit 1; fi
	scripts/ci/test-matrix.sh

docs-local:
	@echo "Building documentation locally with act..."
	@if [ ! -f .actrc ]; then echo "❌ .actrc not found. Please run from project root."; exit 1; fi
	scripts/ci/docs-local.sh

# Alternative local CI targets that use local tools instead of act
ci-lint:
	@echo "Running linting exactly as in CI..."
	scripts/ci/lint-ci.sh

ci-test:
	@echo "Running tests locally (mimics CI test step)..."
	@if python3 -c "import pytest_cov" 2>/dev/null; then \
		python3 -m pytest tests/ -v --cov=stable_cart --cov-report=xml; \
	else \
		echo "⚠️  pytest-cov not installed, running tests without coverage..."; \
		python3 -m pytest tests/ -v; \
	fi

# Benchmarking
benchmark:
	@echo "Running benchmark scripts..."
	PYTHONPATH=. python3 scripts/benchmark_less_greedy.py

# Utility targets
clean:
	@echo "Cleaning up generated files..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -f bench_out/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker requirements check
check-docker:
	@if ! docker info >/dev/null 2>&1; then \
		echo "❌ Docker is not running. Please start Docker Desktop first."; \
		exit 1; \
	fi
	@echo "✅ Docker is running"

# Pre-commit style checks
pre-commit: lint test
	@echo "✅ All pre-commit checks passed!"

# Full CI simulation pipeline
ci-full: check-docker ci-lint ci-test
	@echo "🎉 Full CI simulation completed successfully!"