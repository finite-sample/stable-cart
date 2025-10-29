#!/bin/bash
# Utility functions and wrappers for act (GitHub Actions local simulation)
# Source this file to get helper functions: source scripts/ci/act-utils.sh

# Colors for output
export GREEN='\033[0;32m'
export BLUE='\033[0;34m'
export RED='\033[0;31m'
export YELLOW='\033[1;33m'
export NC='\033[0m' # No Color

# Utility functions
print_status() {
    echo -e "${BLUE}[ACT-UTILS]${NC} $1"
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
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop first."
        return 1
    fi
    return 0
}

# Check if act is installed
check_act() {
    if ! command -v act &> /dev/null; then
        print_error "act is not installed. Please install it with: brew install act"
        return 1
    fi
    return 0
}

# Check prerequisites
check_prerequisites() {
    check_docker && check_act
}

# List all available workflows
act_list() {
    print_status "Available workflows:"
    act --list
}

# Run CI workflow for a specific Python version
act_ci() {
    local python_version=${1:-"3.11"}
    print_status "Running CI workflow for Python ${python_version}..."
    
    if ! check_prerequisites; then
        return 1
    fi
    
    act push --job test --matrix python-version:${python_version}
}

# Run CI workflow for all Python versions
act_ci_matrix() {
    print_status "Running CI workflow for all Python versions..."
    
    if ! check_prerequisites; then
        return 1
    fi
    
    local versions=("3.11" "3.12" "3.13")
    local failed_versions=()
    local passed_versions=()
    
    for version in "${versions[@]}"; do
        print_status "Testing Python ${version}..."
        
        if act push --job test --matrix python-version:${version} --quiet; then
            print_success "Python ${version} passed"
            passed_versions+=("${version}")
        else
            print_error "Python ${version} failed"
            failed_versions+=("${version}")
        fi
    done
    
    # Summary
    echo "=================================="
    print_status "Matrix Testing Summary"
    echo "=================================="
    
    if [ ${#passed_versions[@]} -gt 0 ]; then
        print_success "Passed versions: ${passed_versions[*]}"
    fi
    
    if [ ${#failed_versions[@]} -gt 0 ]; then
        print_error "Failed versions: ${failed_versions[*]}"
        return 1
    else
        print_success "All Python versions passed! 🎉"
        return 0
    fi
}

# Run documentation build
act_docs() {
    print_status "Building documentation with act..."
    
    if ! check_prerequisites; then
        return 1
    fi
    
    act push --job build --workflows .github/workflows/docs.yml
}

# Clean up act containers and images
act_cleanup() {
    print_status "Cleaning up act containers and images..."
    
    # Stop and remove act containers
    docker ps -a --filter "label=act" --format "{{.ID}}" | xargs -r docker rm -f
    
    # Remove unused act images (optional)
    read -p "Remove unused Docker images? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker image prune -f
        print_success "Docker cleanup completed"
    else
        print_status "Skipped image cleanup"
    fi
}

# Show act configuration
act_config() {
    print_status "Current act configuration:"
    if [ -f ".actrc" ]; then
        echo "--- .actrc ---"
        cat .actrc
    else
        print_warning ".actrc file not found"
    fi
    
    echo ""
    print_status "Act version: $(act --version)"
    print_status "Docker status:"
    if check_docker; then
        print_success "Docker is running"
    fi
}

# Interactive menu for act operations
act_menu() {
    while true; do
        echo ""
        echo "=================================="
        echo "    Act GitHub Actions Simulator"
        echo "=================================="
        echo "1. Run CI (single Python version)"
        echo "2. Run CI matrix (all Python versions)"
        echo "3. Build documentation"
        echo "4. List workflows"
        echo "5. Show configuration"
        echo "6. Cleanup containers"
        echo "7. Exit"
        echo ""
        read -p "Select an option [1-7]: " choice
        
        case $choice in
            1)
                read -p "Enter Python version (default: 3.11): " version
                act_ci ${version:-3.11}
                ;;
            2)
                act_ci_matrix
                ;;
            3)
                act_docs
                ;;
            4)
                act_list
                ;;
            5)
                act_config
                ;;
            6)
                act_cleanup
                ;;
            7)
                print_status "Goodbye!"
                break
                ;;
            *)
                print_error "Invalid option. Please choose 1-7."
                ;;
        esac
    done
}

# Help function
act_help() {
    echo "Act GitHub Actions Simulator - Utility Functions"
    echo ""
    echo "Available functions:"
    echo "  act_ci [version]       - Run CI workflow for specific Python version"
    echo "  act_ci_matrix          - Run CI workflow for all Python versions"
    echo "  act_docs               - Build documentation"
    echo "  act_list               - List all available workflows"
    echo "  act_config             - Show current configuration"
    echo "  act_cleanup            - Clean up containers and images"
    echo "  act_menu               - Interactive menu"
    echo "  act_help               - Show this help"
    echo ""
    echo "Prerequisites:"
    echo "  - Docker Desktop must be running"
    echo "  - act must be installed (brew install act)"
    echo ""
    echo "Usage:"
    echo "  source scripts/ci/act-utils.sh"
    echo "  act_ci 3.11"
    echo "  act_ci_matrix"
    echo "  act_menu"
}

# Show help if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    act_help
fi