# CI Scripts

This directory contains scripts for local CI/CD simulation using [act](https://github.com/nektos/act).

## Scripts Overview

| Script | Purpose | Requirements |
|--------|---------|--------------|
| `ci-local.sh` | Run CI workflow for single Python version | Docker + act |
| `test-matrix.sh` | Run CI matrix for all Python versions | Docker + act |
| `lint-ci.sh` | Run linting checks exactly as in CI | Python only |
| `docs-local.sh` | Build documentation locally | Docker + act |
| `act-utils.sh` | Utility functions and interactive menu | Docker + act |

## Quick Usage

```bash
# Fast local checks (no Docker required)
./scripts/ci/lint-ci.sh

# Full CI simulation (Docker required)
./scripts/ci/ci-local.sh

# Test all Python versions (Docker required)
./scripts/ci/test-matrix.sh

# Interactive menu
source scripts/ci/act-utils.sh && act_menu
```

## Prerequisites

1. **Docker Desktop** must be running for act-based scripts
2. **Act** must be installed: `brew install act`
3. **Python dev dependencies**: `pip install -e ".[dev]"`

## Integration

These scripts are integrated with the project Makefile:

```bash
make ci-local    # -> scripts/ci/ci-local.sh
make ci-matrix   # -> scripts/ci/test-matrix.sh  
make ci-lint     # -> scripts/ci/lint-ci.sh
make docs-local  # -> scripts/ci/docs-local.sh
```

## Configuration

Scripts use the project's `.actrc` configuration file for consistent behavior across all act operations.

For detailed documentation, see [docs/LOCAL_CI.md](../../docs/LOCAL_CI.md).