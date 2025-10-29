#!/bin/bash
# Local documentation building script using act
# Mimics the GitHub Actions docs workflow locally

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[DOCS-LOCAL]${NC} $1"
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

# Default values
EVENT=${1:-"push"}

print_status "Building documentation locally with act..."

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

# Run the docs workflow build job only (skip deployment)
print_status "Running documentation build job..."

if act ${EVENT} --job build --workflows .github/workflows/docs.yml; then
    print_success "Documentation build completed successfully!"
    print_status "Documentation artifacts should be available in the container"
    print_warning "Note: GitHub Pages deployment is skipped in local mode"
else
    print_error "Documentation build failed!"
    exit 1
fi

print_status "Documentation simulation complete"