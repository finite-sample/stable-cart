# Local CI/CD with Act

This document explains how to run GitHub Actions workflows locally using [act](https://github.com/nektos/act) for the stable-cart project.

## Overview

The stable-cart project uses GitHub Actions for CI/CD with three main workflows:
- **CI Workflow** (`ci.yml`): Testing across Python 3.11, 3.12, 3.13
- **Documentation** (`docs.yml`): Building and deploying Sphinx documentation  
- **Publishing** (`python-publish.yml`): PyPI package publishing

Act allows you to run these workflows locally for faster feedback and debugging.

## Prerequisites

### Required Software
1. **Docker Desktop** - Must be running for act to work
   ```bash
   # Check if Docker is running
   docker version
   ```

2. **Act** - GitHub Actions local runner
   ```bash
   # Install on macOS
   brew install act
   
   # Verify installation
   act --version
   ```

### Project Setup
The project includes act configuration and helper scripts:
- `.actrc` - Act configuration file
- `scripts/ci/` - Local CI simulation scripts
- `Makefile` - Convenient targets for common operations

## Quick Start

### Using Makefile Targets (Recommended)

```bash
# Run linting checks (fast, no Docker required)
make ci-lint

# Run tests locally (fast, no Docker required)  
make ci-test

# Run full CI simulation with act (requires Docker)
make ci-local

# Run CI matrix testing for all Python versions (requires Docker)
make ci-matrix

# Build documentation locally (requires Docker)
make docs-local

# Show all available targets
make help
```

### Using Scripts Directly

```bash
# Run CI for single Python version
./scripts/ci/ci-local.sh 3.11

# Run CI matrix for all Python versions
./scripts/ci/test-matrix.sh

# Run linting exactly as in CI
./scripts/ci/lint-ci.sh

# Build documentation
./scripts/ci/docs-local.sh
```

### Using Act Directly

```bash
# List all workflows
act --list

# Run CI workflow for Python 3.11
act push --job test --matrix python-version:3.11

# Run documentation build
act push --job build --workflows .github/workflows/docs.yml
```

## Act Utilities

Source the utilities for interactive functions:

```bash
# Load act utility functions
source scripts/ci/act-utils.sh

# Interactive menu
act_menu

# Run CI for specific version
act_ci 3.11

# Run full matrix testing
act_ci_matrix

# Show configuration
act_config

# Clean up containers
act_cleanup
```

## Configuration

### .actrc Configuration

The project includes a `.actrc` file with optimized settings:

```bash
# Use larger runner images with more tools
-P ubuntu-latest=catthehacker/ubuntu:act-latest

# Apple M-series compatibility
--container-architecture linux/amd64

# Reuse containers for speed
--reuse

# Environment variables
--env PYTHONUNBUFFERED=1
--env PIP_DISABLE_PIP_VERSION_CHECK=1
```

### Workflow Matrix Testing

The CI workflow tests multiple Python versions:
- Python 3.11
- Python 3.12  
- Python 3.13

Each version runs the full test suite including:
- Black code formatting check
- Flake8 style checking
- Pytest with coverage reporting

## Troubleshooting

### Common Issues

1. **Docker not running**
   ```bash
   Error: Cannot connect to the Docker daemon
   ```
   **Solution:** Start Docker Desktop first

2. **Act command timeout**
   ```bash
   Command timed out after 2m
   ```
   **Solution:** First run takes longer due to image downloads. Subsequent runs are faster with `--reuse`

3. **Container architecture warnings**
   ```bash
   WARNING: You are using Apple M-series chip
   ```
   **Solution:** This is handled by `--container-architecture linux/amd64` in `.actrc`

4. **Missing coverage in local tests**
   ```bash
   pytest-cov not installed
   ```
   **Solution:** Install dev dependencies: `make install` or `pip install -e ".[dev]"`

### Performance Tips

1. **Use reusable containers**
   - Act is configured with `--reuse` to speed up subsequent runs
   - First run downloads containers and can take several minutes
   - Subsequent runs are much faster

2. **Local-first development**
   - Use `make ci-lint` and `make ci-test` for fast feedback
   - Use act only when you need full CI environment simulation

3. **Cleanup when needed**
   - Run `act_cleanup` or `make clean` to free up disk space
   - Docker containers and images can accumulate over time

## Advanced Usage

### Custom Workflows

Test specific workflows:
```bash
# Test only the docs workflow
act push --workflows .github/workflows/docs.yml

# Test with specific event
act pull_request --job test
```

### Environment Variables

Add environment variables for testing:
```bash
# Create .env file for act
echo "DEBUG=1" > .env
act push --job test
```

### Secrets Simulation

For workflows requiring secrets:
```bash
# Create .secrets file
echo "CODECOV_TOKEN=test-token" > .secrets
act push --job test
```

## Integration with Development Workflow

### Pre-commit Hooks

Add to your development workflow:
```bash
# Before committing
make ci-lint
make ci-test

# Full CI simulation before pushing
make ci-local
```

### VS Code Integration

Add tasks to `.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "CI Local",
            "type": "shell",
            "command": "make",
            "args": ["ci-local"],
            "group": "test"
        }
    ]
}
```

## Comparison: Local vs Act vs GitHub Actions

| Method | Speed | Accuracy | Requirements | Use Case |
|--------|--------|----------|--------------|----------|
| Local (`make ci-lint`, `make ci-test`) | ⚡ Fast | 🟡 Good | Python only | Quick feedback |
| Act (`make ci-local`) | 🐌 Slower | 🟢 Excellent | Docker + Act | CI debugging |
| GitHub Actions | 🐌 Slowest | 🟢 Perfect | Git push | Final validation |

**Recommendation:** Use local commands for development, act for pre-push validation, and GitHub Actions for final CI.

## Supported Workflows

| Workflow | Local Support | Act Support | Notes |
|----------|---------------|-------------|-------|
| CI Testing | ✅ | ✅ | Full matrix testing available |
| Linting | ✅ | ✅ | Identical to CI commands |
| Documentation | ⚠️ | ✅ | Requires Sphinx dependencies |
| Package Publishing | ❌ | ⚠️ | Simulation only, no actual publishing |

## Further Reading

- [Act Documentation](https://github.com/nektos/act)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Project Contributing Guidelines](../README.md)