#!/bin/bash
# Test all Python versions in the CI matrix locally
# Simulates the complete matrix testing from GitHub Actions

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[MATRIX-TEST]${NC} $1"
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

# Python versions from the CI matrix
PYTHON_VERSIONS=("3.11" "3.12" "3.13")
FAILED_VERSIONS=()
PASSED_VERSIONS=()

print_status "Starting matrix testing for Python versions: ${PYTHON_VERSIONS[*]}"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker Desktop first."
    exit 1
fi

for version in "${PYTHON_VERSIONS[@]}"; do
    print_status "Testing Python ${version}..."
    
    if act push --job test --matrix python-version:${version} --quiet; then
        print_success "Python ${version} passed"
        PASSED_VERSIONS+=("${version}")
    else
        print_error "Python ${version} failed"
        FAILED_VERSIONS+=("${version}")
    fi
    
    echo "" # Add spacing between tests
done

# Summary
echo "=================================="
print_status "Matrix Testing Summary"
echo "=================================="

if [ ${#PASSED_VERSIONS[@]} -gt 0 ]; then
    print_success "Passed versions: ${PASSED_VERSIONS[*]}"
fi

if [ ${#FAILED_VERSIONS[@]} -gt 0 ]; then
    print_error "Failed versions: ${FAILED_VERSIONS[*]}"
    print_error "Matrix testing failed!"
    exit 1
else
    print_success "All Python versions passed! 🎉"
fi