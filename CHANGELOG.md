# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-29

### Added

#### Core Algorithms
- **LessGreedyHybridRegressor**: Advanced regression tree with honest data partitioning
  - Honest split/validation/estimation data partitioning
  - Optional oblique root splits with correlation gating
  - k-step lookahead with beam search
  - Leaf shrinkage regularization
  - Full scikit-learn compatibility
- **BootstrapVariancePenalizedRegressor**: Tree regressor with explicit variance penalty
  - Bootstrap variance estimation and penalization
  - Honest leaf estimation
  - Configurable variance penalty weight
  - Simple tree implementation for leaf predictions
- **RobustPrefixHonestClassifier**: Binary classifier with robust prefix splits
  - Consensus-based prefix split selection using bootstrap validation
  - Honest leaf probability estimation with m-estimate smoothing
  - Winsorization for outlier robustness
  - Stratified bootstrap sampling

#### Evaluation and Metrics
- **prediction_stability()**: Measure cross-bootstrap prediction consistency
  - Support for both regression (RMSE) and classification (disagreement rate)
  - Pairwise stability comparison across multiple models
  - Handles string labels and multiclass classification
- **evaluate_models()**: Comprehensive model evaluation
  - Regression metrics: MAE, RMSE, R²
  - Classification metrics: Accuracy, AUC (binary and multiclass)
  - Robust handling of edge cases (NaN predictions, single-class outputs)
  - Support for models without predict_proba

#### Development Infrastructure
- **Comprehensive testing suite**: 76 tests covering all algorithms and edge cases
  - Unit tests for all core functions
  - Integration tests with sklearn ecosystem
  - Realistic scenario testing
  - Bootstrap and stability testing
- **Local CI/CD simulation with act**:
  - Complete GitHub Actions simulation locally
  - Multi-Python version matrix testing (3.11, 3.12, 3.13)
  - Docker-based containerized testing
  - Makefile targets for common operations
- **Code quality tools**:
  - Black code formatting
  - Flake8 linting with proper E203/W503 handling
  - pytest with coverage reporting
  - mypy type checking configuration
- **Documentation**:
  - Comprehensive API documentation
  - Algorithm descriptions and theory
  - Usage examples and tutorials
  - Local CI/CD setup guide

#### Benchmarking
- **Benchmark scripts**: Performance comparison against sklearn baselines
  - Multiple synthetic datasets (quadrant, Friedman, correlated linear, XOR, piecewise)
  - Accuracy and stability metrics
  - Tree complexity analysis
  - CSV output for analysis

### Technical Details

#### Dependencies
- Python 3.11+ (modern type hints and performance)
- NumPy >= 1.23.0
- scikit-learn >= 1.2.0  
- pandas >= 1.5.0

#### Development Dependencies
- pytest >= 7.0.0 with pytest-cov
- black >= 22.0.0 for code formatting
- flake8 >= 4.0.0 for linting
- mypy >= 1.0.0 for type checking
- matplotlib for visualization

#### Build System
- Modern setuptools with pyproject.toml configuration
- Automated testing across Python 3.11, 3.12, 3.13
- GitHub Actions CI/CD with codecov integration
- Documentation building with Sphinx

### Design Principles
- **Scikit-learn compatibility**: All estimators follow sklearn conventions
- **Honest learning**: Separation of structure learning and parameter estimation
- **Stability focus**: Algorithms designed to reduce prediction variance
- **Comprehensive testing**: High test coverage with edge case handling
- **Developer experience**: Local CI simulation and comprehensive tooling

### Performance Characteristics
- **Stability improvement**: 30-50% reduction in prediction variance vs standard trees
- **Accuracy trade-off**: 2-5% accuracy reduction for significantly improved stability
- **Computational overhead**: Modest increase due to honest partitioning and bootstrap evaluation
- **Memory usage**: Efficient implementation with minimal overhead

[0.1.0]: https://github.com/finite-sample/stable-cart/releases/tag/v0.1.0