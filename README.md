# Stable CART Package

[![Python application](https://github.com/finite-sample/stable-cart/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/stable-cart/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/stable-cart.svg)](https://pypi.org/project/stable-cart/)
[![Downloads](https://pepy.tech/badge/stable-cart)](https://pepy.tech/project/stable-cart)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://finite-sample.github.io/stable-cart/)
[![License](https://img.shields.io/github/license/finite-sample/stable-cart)](https://github.com/finite-sample/stable-cart/blob/main/LICENSE)

A scikit-learn compatible implementation of **Stable CART** (Classification and Regression Trees) with advanced stability metrics and techniques to reduce prediction variance.

## Features

- 🌳 **LessGreedyHybridRegressor**: Advanced regression tree with stability-enhancing techniques
- 📊 **BootstrapVariancePenalizedRegressor**: Tree regressor that explicitly penalizes bootstrap prediction variance
- 📈 **Prediction Stability Metrics**: Measure model consistency across different training runs
- 🔧 **Full sklearn Compatibility**: Works with pipelines, cross-validation, and grid search

## Installation

### From Source

```bash
git clone https://github.com/yourusername/stable-cart.git
cd stable-cart
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from stable_cart import LessGreedyHybridRegressor, BootstrapVariancePenalizedRegressor
from stable_cart import prediction_stability, accuracy
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Generate data
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
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
metrics = accuracy(models, X_test, y_test, task="continuous")
print(f"Performance: {metrics}")

# Evaluate stability
stability = prediction_stability(models, X_test, task="continuous")
print(f"Stability (lower is better): {stability}")
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=stable_cart

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest tests/unit/     # Unit tests only
pytest tests/e2e/      # End-to-end tests only
```

## License

MIT License - see LICENSE file for details.
