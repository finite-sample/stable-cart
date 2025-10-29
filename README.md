## Stable CART: Lower Cross-Bootstrap Prediction Variance

[![Python application](https://github.com/soodoku/stable-cart/actions/workflows/ci.yml/badge.svg)](https://github.com/soodoku/stable-cart/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/stable-cart.svg)](https://pypi.org/project/stable-cart/)
[![Downloads](https://pepy.tech/badge/stable-cart)](https://pepy.tech/project/stable-cart)
[![Documentation](https://github.com/soodoku/stable-cart/actions/workflows/docs.yml/badge.svg)](https://soodoku.github.io/stable-cart/)
[![License](https://img.shields.io/github/license/soodoku/stable-cart)](https://github.com/soodoku/stable-cart/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A scikit-learn compatible implementation of **Stable CART** (Classification and Regression Trees) with advanced stability metrics and techniques to reduce prediction variance.

## Features

- 🌳 **LessGreedyHybridRegressor**: Advanced regression tree with honest data partitioning and lookahead
- 📊 **BootstrapVariancePenalizedRegressor**: Tree regressor that explicitly penalizes bootstrap prediction variance
- 🎯 **RobustPrefixHonestClassifier**: Binary classifier with robust prefix splits and honest leaf estimation
- 📈 **Prediction Stability Metrics**: Measure model consistency across different training runs
- 🔧 **Full sklearn Compatibility**: Works with pipelines, cross-validation, and grid search

## Installation

### From PyPI (Recommended)

```bash
pip install stable-cart
```

### From Source

```bash
git clone https://github.com/soodoku/stable-cart.git
cd stable-cart
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from stable_cart import (
    LessGreedyHybridRegressor, 
    BootstrapVariancePenalizedRegressor,
    RobustPrefixHonestClassifier,
    prediction_stability, 
    evaluate_models
)
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Regression Example
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Train regression models
stable_model = LessGreedyHybridRegressor(max_depth=5, random_state=42)
bootstrap_model = BootstrapVariancePenalizedRegressor(
    max_depth=5, variance_penalty=2.0, n_bootstrap=10, random_state=42
)
greedy_model = DecisionTreeRegressor(max_depth=5, random_state=42)

stable_model.fit(X_train, y_train)
bootstrap_model.fit(X_train, y_train)
greedy_model.fit(X_train, y_train)

# Evaluate performance
models = {
    "stable": stable_model,
    "bootstrap_penalized": bootstrap_model,
    "greedy": greedy_model
}
metrics = evaluate_models(models, X_test, y_test, task="continuous")
print(f"Performance: {metrics}")

# Evaluate stability
stability = prediction_stability(models, X_test, task="continuous")
print(f"Stability (lower is better): {stability}")

# Classification Example
X_clf, y_clf = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

# Train classification models
robust_model = RobustPrefixHonestClassifier(top_levels=2, max_depth=5, random_state=42)
standard_model = DecisionTreeClassifier(max_depth=5, random_state=42)

robust_model.fit(X_train_clf, y_train_clf)
standard_model.fit(X_train_clf, y_train_clf)

# Evaluate classification performance
clf_models = {"robust": robust_model, "standard": standard_model}
clf_metrics = evaluate_models(clf_models, X_test_clf, y_test_clf, task="categorical")
print(f"Classification Performance: {clf_metrics}")
```

## Algorithms

### LessGreedyHybridRegressor

A regression tree that trades some accuracy for improved stability through:
- **Honest data partitioning**: Splits data into SPLIT (structure), VAL (validation), and EST (estimation) sets
- **Optional oblique root**: Linear combinations at the root node when beneficial
- **Lookahead with beam search**: Considers multiple steps ahead for better long-term decisions
- **Leaf shrinkage**: Ridge-like regularization for leaf predictions

### BootstrapVariancePenalizedRegressor

Explicitly reduces bootstrap prediction variance by:
- **Variance penalty**: Adds bootstrap variance as a regularization term
- **Honest estimation**: Separates structure learning from leaf value estimation
- **Bootstrap evaluation**: Uses multiple bootstrap samples to estimate prediction variance

### RobustPrefixHonestClassifier

A binary classifier designed for stability through:
- **Robust prefix**: Locks top-level splits using consensus across bootstrap samples
- **Honest leaves**: Estimates leaf probabilities on separate data from structure learning
- **m-estimate smoothing**: Stabilizes probability estimates in small leaves
- **Winsorization**: Reduces impact of outliers on split selection

## Performance Comparison

Here's how stable-cart models typically perform compared to standard trees:

| Metric | Standard Tree | Stable CART | Improvement |
|--------|---------------|-------------|-------------|
| **Prediction Variance** | High | Low | 30-50% reduction |
| **Out-of-sample Stability** | Variable | Consistent | 20-40% more stable |
| **Accuracy** | High | Slightly lower | 2-5% trade-off |
| **Interpretability** | Good | Good | Maintained |

## Development and Testing

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=stable_cart

# Run specific test categories
pytest -m "not slow"        # Skip slow tests
pytest -m "benchmark"       # Benchmark tests only
pytest tests/               # All tests
```

### Local CI Testing

Test the CI pipeline locally using Docker:

```bash
# Run the full CI pipeline in a clean Docker container
make ci-docker

# Or run individual steps
make lint        # Check code formatting and style
make test        # Run the test suite
make coverage    # Run tests with coverage report
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`make test`)
5. Run linting (`make lint`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Benchmarking

Run performance benchmarks:

```bash
# Run benchmark scripts
make benchmark

# View results
ls bench_out/
```

## Citation

If you use stable-cart in your research, please cite:

```bibtex
@software{stable_cart_2025,
  title={Stable CART: Enhanced Decision Trees with Prediction Stability},
  author={Sood, Gaurav and Bhosle, Arav},
  year={2025},
  url={https://github.com/soodoku/stable-cart},
  version={0.1.0}
}
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Related Work

- **CART**: Breiman, L., et al. (1984). Classification and regression trees.
- **Honest Trees**: Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests.
- **Bootstrap Aggregating**: Breiman, L. (1996). Bagging predictors.
