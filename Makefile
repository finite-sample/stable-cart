# Makefile for stable-cart project
# Simple, focused targets for essential development tasks

.PHONY: help install test lint format clean coverage benchmark ci-docker

# Default target
help:
	@echo "Available targets:"
	@echo "  install     Install development dependencies"
	@echo "  test        Run all tests"
	@echo "  lint        Run linting checks (black + flake8)"
	@echo "  format      Apply code formatting with black"
	@echo "  coverage    Run tests with coverage report"
	@echo "  benchmark   Run benchmark scripts"
	@echo "  ci-docker   Run CI pipeline in Docker container"
	@echo "  clean       Clean up generated files"

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
	python3 -m black --check stable_cart/ tests/
	python3 -m flake8 stable_cart/ tests/ --max-line-length=100 --extend-ignore=E203,W503

format:
	python3 -m black stable_cart/ tests/

# Docker-based CI (simple and clean)
ci-docker:
	docker run --rm -v $$(pwd):/app -w /app python:3.11 bash -c \
		"pip install -e .[dev] && make lint && make test"

# Benchmarking
benchmark:
	PYTHONPATH=. python3 scripts/benchmark_less_greedy.py

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/ coverage.xml
	rm -f bench_out/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete