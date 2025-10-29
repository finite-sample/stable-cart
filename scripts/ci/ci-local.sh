#!/bin/bash
# Local CI simulation script using act
# Mimics the GitHub Actions CI workflow locally

set -e

echo "🚀 Running local CI simulation with act..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[CI-LOCAL]${NC} $1"
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

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Check if act is installed
if ! command -v act &> /dev/null; then
    print_error "act is not installed. Please install it with: brew install act"
    exit 1
fi

print_status "Docker is running ✓"
print_status "act is available ✓"

# Default values
PYTHON_VERSION=${1:-"3.11"}
WORKFLOW=${2:-"ci.yml"}
EVENT=${3:-"push"}

print_status "Running CI workflow with Python ${PYTHON_VERSION}..."

# Run the CI workflow with act
if act ${EVENT} --job test --matrix python-version:${PYTHON_VERSION} --workflows .github/workflows/${WORKFLOW}; then
    print_success "CI workflow completed successfully!"
else
    print_error "CI workflow failed!"
    exit 1
fi

print_status "CI simulation complete"