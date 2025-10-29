## Stable CART: Lower Cross-Bootstrap Prediction Variance

[![Python application](https://github.com/finite-sample/stable-cart/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/stable-cart/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/stable-cart.svg)](https://pypi.org/project/stable-cart/)
[![Downloads](https://pepy.tech/badge/stable-cart)](https://pepy.tech/project/stable-cart)
[![Documentation](https://github.com/finite-sample/stable-cart/actions/workflows/docs.yml/badge.svg)](https://finite-sample.github.io/stable-cart/)
[![License](https://img.shields.io/github/license/finite-sample/stable-cart)](https://github.com/finite-sample/stable-cart/blob/main/LICENSE)
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
git clone https://github.com/finite-sample/stable-cart.git
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

**🎯 When to use**: When you need stable predictions but can't afford the complexity of ensembles

**💡 Core intuition**: Like a careful decision-maker who considers multiple options before choosing, rather than going with the first good option. Standard CART makes greedy choices at each split - this algorithm looks ahead and thinks more carefully.

**⚖️ Trade-offs**: 
- ✅ **Gain**: 30-50% more stable predictions across different training runs
- ✅ **Gain**: Better generalization with honest estimation  
- ❌ **Cost**: ~5% accuracy reduction, slightly higher training time

**🔧 How it works**:
- **Honest data partitioning**: Separates data for structure learning vs. prediction estimation (like training a model on one set but tuning on another)
- **Lookahead with beam search**: Considers multiple future splits before deciding (not just immediate gain)
- **Optional oblique root**: Can use linear combinations at the top when it helps capture the main pattern
- **Leaf shrinkage**: Prevents overfitting by regularizing final predictions

### BootstrapVariancePenalizedRegressor

**🎯 When to use**: When prediction consistency is more important than squeezing out every bit of accuracy

**💡 Core intuition**: Like choosing a reliable car over a faster but unpredictable one. This algorithm explicitly optimizes for models that give similar predictions even when trained on slightly different data samples.

**⚖️ Trade-offs**:
- ✅ **Gain**: Most consistent predictions across bootstrap samples
- ✅ **Gain**: Excellent for scenarios where you retrain models frequently  
- ❌ **Cost**: Moderate training time increase due to bootstrap evaluation
- ❌ **Cost**: May sacrifice some accuracy for consistency

**🔧 How it works**:
- **Variance penalty**: During training, penalizes splits that lead to high prediction variance across bootstrap samples
- **Honest estimation**: Builds tree structure on one data subset, estimates leaf values on another (prevents overfitting)
- **Bootstrap evaluation**: Tests each potential split on multiple bootstrap samples to measure stability before deciding

### RobustPrefixHonestClassifier

**🎯 When to use**: For binary classification where you need reliable probability estimates and stable decision boundaries

**💡 Core intuition**: Like making the big strategic decisions first with a committee consensus, then fine-tuning details with fresh information. This classifier locks in the most important splits using agreement across multiple bootstrap samples, then uses separate data to estimate probabilities.

**⚖️ Trade-offs**:
- ✅ **Gain**: Very stable decision boundaries across different training runs
- ✅ **Gain**: Reliable probability estimates (great for risk assessment)
- ✅ **Gain**: Robust to outliers and data noise
- ❌ **Cost**: Limited to binary classification only
- ❌ **Cost**: May be conservative in capturing complex patterns

**🔧 How it works**:
- **Robust prefix**: Uses multiple bootstrap samples to find splits that consistently matter, then locks those in
- **Honest leaves**: After structure is fixed, estimates class probabilities on completely separate data
- **m-estimate smoothing**: Prevents overconfident predictions in regions with little data
- **Winsorization**: Caps extreme feature values to reduce outlier influence

## Choosing the Right Algorithm

### 🤔 Decision Guide

**Start here**: What's your primary concern?

```
📊 Regression Tasks:
├── Need maximum stability? → BootstrapVariancePenalizedRegressor
├── Want balanced stability + flexibility? → LessGreedyHybridRegressor  
└── Just need sklearn DecisionTree baseline? → Standard CART

🎯 Classification Tasks:
├── Binary classification + need probability estimates? → RobustPrefixHonestClassifier
├── Multi-class classification? → Standard CART (stable methods coming soon!)
└── Just need sklearn DecisionTree baseline? → Standard CART
```

### 📋 Use Case Comparison

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| **Financial risk models** | RobustPrefixHonest | Stable probability estimates crucial |
| **A/B testing analysis** | BootstrapVariancePenalized | Consistency across samples matters most |
| **Medical diagnosis support** | RobustPrefixHonest | Reliable probabilities + robust to outliers |
| **Demand forecasting** | LessGreedyHybrid | Balance of accuracy + stability |
| **Real-time recommendations** | Standard CART | Speed over stability |
| **Research/prototyping** | LessGreedyHybrid | Good general-purpose stable option |

### ⚡ Quick Selection Rules

**Choose BootstrapVariancePenalizedRegressor when**:
- You retrain models frequently with new data
- Prediction consistency is more important than peak accuracy
- You have sufficient training time

**Choose LessGreedyHybridRegressor when**:
- You want stability without major accuracy loss
- You need a general-purpose stable regressor
- Training time is somewhat constrained

**Choose RobustPrefixHonestClassifier when**:
- You have binary classification
- You need trustworthy probability estimates
- Your data may have outliers

**Stick with Standard CART when**:
- You need maximum speed
- You have very large datasets (>100k samples)
- Stability is not a concern
- You need multi-class classification (for now)

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

Run comprehensive benchmarks comparing CART vs stable-CART methods:

```bash
# Quick benchmark (4 key datasets, fast execution)
make quick-benchmark

# Comprehensive benchmark (all datasets)
make benchmark

# Stability-focused benchmark (datasets highlighting variance differences)
make stability-benchmark

# Custom benchmark
python scripts/comprehensive_benchmark.py --datasets friedman1,breast_cancer --models CART,LessGreedyHybrid --quick

# View results
ls benchmark_results/
cat benchmark_results/comprehensive_benchmark_report.md
```

## Citation

If you use stable-cart in your research, please cite:

```bibtex
@software{stable_cart_2025,
  title={Stable CART: Enhanced Decision Trees with Prediction Stability},
  author={Sood, Gaurav and Bhosle, Arav},
  year={2025},
  url={https://github.com/finite-sample/stable-cart},
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
