#!/bin/bash
# Local linting script that mimics CI linting exactly
# Runs the same linting commands as the GitHub Actions workflow

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[LINT-CI]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ SUCCESS:${NC} $1"
}

print_error() {
    echo -e "${RED}❌ ERROR:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  WARNING:${NC} $1"
}

print_status "Running linting checks exactly as in CI..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run this from the project root."
    exit 1
fi

# Run black check (exactly as in CI)
print_status "Running black --check stable_cart/ tests/"
if python3 -m black --check stable_cart/ tests/; then
    print_success "Black formatting check passed"
else
    print_error "Black formatting check failed"
    print_warning "Run 'python3 -m black stable_cart/ tests/' to fix formatting"
    exit 1
fi

# Run flake8 (exactly as in CI)
print_status "Running flake8 stable_cart/ tests/ --max-line-length=100 --extend-ignore=E203,W503"
if python3 -m flake8 stable_cart/ tests/ --max-line-length=100 --extend-ignore=E203,W503; then
    print_success "Flake8 style check passed"
else
    print_error "Flake8 style check failed"
    exit 1
fi

print_success "All linting checks passed! 🎉"
print_status "Your code is ready for CI"