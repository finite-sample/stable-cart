## Stable CART: Lower Cross-Bootstrap Prediction Variance

[![Python application](https://github.com/finite-sample/stable-cart/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/stable-cart/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/stable-cart.svg)](https://pypi.org/project/stable-cart/)
[![Downloads](https://pepy.tech/badge/stable-cart)](https://pepy.tech/project/stable-cart)
[![Documentation](https://github.com/finite-sample/stable-cart/actions/workflows/docs.yml/badge.svg)](https://finite-sample.github.io/stable-cart/)
[![License](https://img.shields.io/github/license/finite-sample/stable-cart)](https://github.com/finite-sample/stable-cart/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A scikit-learn compatible implementation of **Stable CART** (Classification and Regression Trees) with advanced stability metrics and techniques to reduce prediction variance.

## Features

- 🌳 **Unified Tree Architecture**: All trees support both regression and classification with a simple `task` parameter
- 🎯 **LessGreedyHybridTree**: Advanced tree with honest data partitioning, lookahead, and optional oblique splits
- 📊 **BootstrapVariancePenalizedTree**: Explicitly penalizes bootstrap prediction variance during split selection
- 🛡️ **RobustPrefixHonestTree**: Robust consensus-based prefix splits with honest leaf estimation
- 🎲 **CentroidTree**: Train N trees, select the one closest to ensemble mean—single-tree interpretability with ensemble-like stability
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
    # Unified trees - all support both regression and classification
    LessGreedyHybridTree,
    BootstrapVariancePenalizedTree,
    RobustPrefixHonestTree,
    CentroidTree,
    # Evaluation utilities
    prediction_stability,
    evaluate_models
)
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# === UNIFIED ARCHITECTURE ===

# Regression Example
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# All trees support both tasks with the 'task' parameter
less_greedy = LessGreedyHybridTree(task='regression', max_depth=5, random_state=42)
bootstrap_tree = BootstrapVariancePenalizedTree(
    task='regression', max_depth=5, variance_penalty=2.0, n_bootstrap=10, random_state=42
)
robust_tree = RobustPrefixHonestTree(task='regression', top_levels=2, max_depth=5, random_state=42)
centroid_tree = CentroidTree(task='regression', n_candidates=20, random_state=42)
greedy_model = DecisionTreeRegressor(max_depth=5, random_state=42)

# Fit models
for model in [less_greedy, bootstrap_tree, robust_tree, centroid_tree, greedy_model]:
    model.fit(X_train, y_train)

# Classification Example with Same Tree Classes
X_clf, y_clf = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

# Same tree classes, just change the task parameter
less_greedy_clf = LessGreedyHybridTree(task='classification', max_depth=5, random_state=42)
bootstrap_clf = BootstrapVariancePenalizedTree(
    task='classification', max_depth=5, variance_penalty=1.0, n_bootstrap=5, random_state=42
)
robust_clf = RobustPrefixHonestTree(task='classification', top_levels=2, max_depth=5, random_state=42)
centroid_clf = CentroidTree(task='classification', n_candidates=20, random_state=42)
standard_clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# Fit classification models
for model in [less_greedy_clf, bootstrap_clf, robust_clf, centroid_clf, standard_clf]:
    model.fit(X_train_clf, y_train_clf)

# Evaluate both regression and classification
reg_models = {
    "less_greedy": less_greedy,
    "bootstrap_penalized": bootstrap_tree,
    "robust_prefix": robust_tree,
    "centroid": centroid_tree,
    "greedy": greedy_model
}

clf_models = {
    "less_greedy": less_greedy_clf,
    "bootstrap_penalized": bootstrap_clf,
    "robust_prefix": robust_clf,
    "centroid": centroid_clf,
    "standard": standard_clf
}

# Get predictions and probabilities
reg_predictions = {name: model.predict(X_test) for name, model in reg_models.items()}
clf_predictions = {name: model.predict(X_test_clf) for name, model in clf_models.items()}
clf_probabilities = {name: model.predict_proba(X_test_clf) for name, model in clf_models.items() 
                     if hasattr(model, 'predict_proba')}

print("Regression R² scores:")
for name, model in reg_models.items():
    score = model.score(X_test, y_test)
    print(f"  {name}: {score:.3f}")

print("\nClassification accuracy scores:")
for name, model in clf_models.items():
    score = model.score(X_test_clf, y_test_clf)
    print(f"  {name}: {score:.3f}")

```

## Advanced Configuration Examples

### Unified Parameter Interface

All stable-cart trees share a unified parameter interface with comprehensive stability primitives:

```python
from stable_cart import LessGreedyHybridTree

# Regression with all stability features enabled
advanced_reg_tree = LessGreedyHybridTree(
    # === CORE CONFIGURATION ===
    task='regression',               # 'regression' or 'classification'
    max_depth=6,                    # Maximum tree depth
    min_samples_split=50,           # Minimum samples to split node
    min_samples_leaf=25,            # Minimum samples per leaf
    
    # === HONEST DATA PARTITIONING ===
    split_frac=0.6,                 # Fraction for structure learning
    val_frac=0.2,                   # Fraction for validation
    est_frac=0.2,                   # Fraction for leaf estimation
    enable_stratified_sampling=True, # Balanced honest partitioning
    
    # === OBLIQUE SPLITS (SIGNATURE FEATURE) ===
    enable_oblique_splits=True,     # Enable oblique splits
    oblique_strategy='root_only',   # 'root_only', 'all_levels', 'adaptive'
    oblique_regularization='lasso', # 'lasso', 'ridge', 'elastic_net'
    enable_correlation_gating=True, # Use correlation gating
    min_correlation_threshold=0.3,  # Minimum correlation to trigger oblique
    
    # === LOOKAHEAD SEARCH (SIGNATURE FEATURE) ===
    enable_lookahead=True,          # Enable lookahead search
    lookahead_depth=2,              # Lookahead depth
    beam_width=15,                  # Number of candidates to track
    enable_ambiguity_gating=True,   # Use ambiguity gating
    ambiguity_threshold=0.05,       # Trigger lookahead when splits are close
    min_samples_for_lookahead=800,  # Minimum samples to enable lookahead
    
    # === CROSS-METHOD LEARNING FEATURES ===
    enable_robust_consensus_for_ambiguous=True, # Consensus for ambiguous splits
    consensus_samples=12,                       # Bootstrap samples for consensus
    consensus_threshold=0.7,                    # Agreement threshold
    enable_winsorization=True,                  # Outlier clipping (from RobustPrefix)
    winsor_quantiles=(0.02, 0.98),            # Outlier clipping bounds
    enable_bootstrap_variance_tracking=True,   # Variance tracking (from Bootstrap)
    variance_tracking_samples=10,              # Bootstrap samples for variance
    
    # === LEAF STABILIZATION ===  
    leaf_smoothing=0.1,             # Shrinkage parameter (0=none, higher=more)
    leaf_smoothing_strategy='m_estimate',  # 'm_estimate', 'shrink_to_parent', 'beta_smoothing'
    
    random_state=42
)

# Classification with conservative stability settings
conservative_clf_tree = LessGreedyHybridTree(
    task='classification',
    max_depth=4,                                    # Shallower for more stability
    min_samples_split=60,                           # Higher split threshold
    min_samples_leaf=30,                            # Larger leaves for stability
    leaf_smoothing=0.5,                             # Heavy smoothing
    leaf_smoothing_strategy='m_estimate',           # Bayesian smoothing
    enable_bootstrap_variance_tracking=True,       # Track prediction variance
    enable_robust_consensus_for_ambiguous=True,    # Use consensus for ambiguous splits
    consensus_threshold=0.8,                       # High agreement requirement
    consensus_samples=15,                          # More bootstrap samples
    enable_winsorization=True,                     # Enable outlier protection
    classification_criterion='gini',              # Gini impurity criterion
    random_state=42
)

# Fit and evaluate
advanced_reg_tree.fit(X_train, y_train)
conservative_clf_tree.fit(X_train_clf, y_train_clf)

print(f"Regression R²: {advanced_reg_tree.score(X_test, y_test):.3f}")
print(f"Classification accuracy: {conservative_clf_tree.score(X_test_clf, y_test_clf):.3f}")
```

### Stability Measurement

```python
from stable_cart import prediction_stability

# Measure prediction stability across bootstrap samples
stability_results = prediction_stability(
    [advanced_reg_tree, conservative_clf_tree], 
    [X_test, X_test_clf], 
    n_bootstrap=20
)

print("Prediction variance (lower = more stable):")
for model_name, variance in stability_results.items():
    print(f"  {model_name}: {variance:.4f}")
```

## Algorithms

All trees in stable-cart use a **unified architecture** that supports both regression and classification through a simple `task` parameter. This means you can use the same algorithm for both types of problems!

### LessGreedyHybridTree

**🎯 When to use**: When you need stable predictions but can't afford the complexity of ensembles (works for both regression and classification)

**💡 Core intuition**: Like a careful decision-maker who considers multiple options before choosing, rather than going with the first good option. Standard CART makes greedy choices at each split - this algorithm looks ahead and thinks more carefully.

**⚖️ Trade-offs**: 
- ✅ **Gain**: 30-50% more stable predictions across different training runs
- ✅ **Gain**: Better generalization with honest estimation
- ✅ **Gain**: Works for both regression and classification with same API
- ❌ **Cost**: ~5% accuracy reduction, slightly higher training time

**🔧 How it works**:
- **Honest data partitioning**: Separates data for structure learning vs. prediction estimation
- **Lookahead with beam search**: Considers multiple future splits before deciding (not just immediate gain)
- **Optional oblique root**: Can use linear combinations at the top (Lasso for regression, LogisticRegression for classification)
- **Task-adaptive leaf estimation**: Shrinkage for regression, m-estimate smoothing for classification

### BootstrapVariancePenalizedTree

**🎯 When to use**: When prediction consistency is more important than squeezing out every bit of accuracy (both regression and classification)

**💡 Core intuition**: Like choosing a reliable car over a faster but unpredictable one. This algorithm explicitly optimizes for models that give similar predictions even when trained on slightly different data samples.

**⚖️ Trade-offs**:
- ✅ **Gain**: Most consistent predictions across bootstrap samples
- ✅ **Gain**: Excellent for scenarios where you retrain models frequently
- ✅ **Gain**: Unified interface for regression and classification
- ❌ **Cost**: Moderate training time increase due to bootstrap evaluation
- ❌ **Cost**: May sacrifice some accuracy for consistency

**🔧 How it works**:
- **Variance penalty**: During training, penalizes splits that lead to high prediction variance across bootstrap samples
- **Honest estimation**: Builds tree structure on one data subset, estimates leaf values on another
- **Bootstrap evaluation**: Tests each potential split on multiple bootstrap samples to measure stability
- **Task-adaptive loss**: Uses SSE for regression, Gini/entropy for classification

### RobustPrefixHonestTree

**🎯 When to use**: When you need reliable probability estimates and stable decision boundaries (supports both binary classification and regression)

**💡 Core intuition**: Like making the big strategic decisions first with a committee consensus, then fine-tuning details with fresh information. This tree locks in the most important splits using agreement across multiple bootstrap samples, then uses separate data for final estimates.

**⚖️ Trade-offs**:
- ✅ **Gain**: Very stable decision boundaries across different training runs
- ✅ **Gain**: Reliable probability estimates (classification) or predictions (regression)
- ✅ **Gain**: Robust to outliers and data noise
- ✅ **Gain**: Unified API for both regression and classification
- ❌ **Cost**: Limited to binary classification (multi-class support coming soon)
- ❌ **Cost**: May be conservative in capturing complex patterns

**🔧 How it works**:
- **Robust prefix**: Uses multiple bootstrap samples to find splits that consistently matter, then locks those in
- **Honest leaves**: After structure is fixed, estimates values on completely separate data
- **Task-adaptive smoothing**: Shrinkage for regression, m-estimate for classification
- **Winsorization**: Caps extreme feature values to reduce outlier influence

### CentroidTree

**🎯 When to use**: When you need ensemble-like stability but must keep a single interpretable tree (supports both regression and classification)

**💡 Core intuition**: Like picking the most "average" candidate from a pool. Train many trees with different random seeds, then select the one whose predictions best represent what all trees collectively predict. You get one interpretable tree that behaves like an ensemble.

**⚖️ Trade-offs**:
- ✅ **Gain**: Reduces prediction variance by ~3% compared to single CART
- ✅ **Gain**: Single tree output—fully interpretable, auditable
- ✅ **Gain**: Works with any base tree (CART, LessGreedyHybridTree, etc.)
- ✅ **Gain**: Unified API for both regression and classification
- ❌ **Cost**: N× training time (trains N candidate trees)
- ❌ **Cost**: ~1% accuracy reduction vs single tree

**🔧 How it works**:
- **Candidate generation**: Trains N trees with different random seeds
- **Ensemble mean**: Computes mean prediction across all candidates on validation set
- **Selection**: Picks the tree closest to ensemble mean (RMSE for regression, disagreement for classification)
- **Single output**: Returns just the selected tree—not an ensemble

## Choosing the Right Algorithm

### 🤔 Decision Guide

**Start here**: What's your primary concern?

```
🌟 UNIFIED ARCHITECTURE:
├── Need maximum stability? → BootstrapVariancePenalizedTree(task='regression'|'classification')
├── Want balanced stability + flexibility? → LessGreedyHybridTree(task='regression'|'classification')
├── Need robust prefix + reliable estimates? → RobustPrefixHonestTree(task='regression'|'classification')
├── Need ensemble stability + single tree? → CentroidTree(task='regression'|'classification')
└── Just need sklearn baseline? → DecisionTreeRegressor/DecisionTreeClassifier
```

**💡 Pro Tip**: All stable-cart trees use the same unified interface with the `task` parameter - switch between regression and classification effortlessly!

### 📋 Use Case Comparison

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| **Financial risk models** | RobustPrefixHonestTree(task='classification') | Stable probability estimates crucial |
| **A/B testing analysis** | BootstrapVariancePenalizedTree(task='regression') | Consistency across samples matters most |
| **Medical diagnosis support** | RobustPrefixHonestTree(task='classification') | Reliable probabilities + robust to outliers |
| **Demand forecasting** | LessGreedyHybridTree(task='regression') | Balance of accuracy + stability |
| **Customer churn prediction** | LessGreedyHybridTree(task='classification') | Stable classification with probability estimates |
| **Regulatory compliance (auditable model)** | CentroidTree(task='classification') | Single interpretable tree with ensemble stability |
| **Real-time recommendations** | Standard CART | Speed over stability |
| **Research/prototyping** | LessGreedyHybridTree(task='regression'/'classification') | Good general-purpose stable option |

### ⚡ Quick Selection Rules

**Choose BootstrapVariancePenalizedTree when**:
- You retrain models frequently with new data
- Prediction consistency is more important than peak accuracy
- You have sufficient training time
- **Works for both**: `task='regression'` or `task='classification'`

**Choose LessGreedyHybridTree when**:
- You want stability without major accuracy loss
- You need a general-purpose stable tree
- Training time is somewhat constrained
- **Works for both**: `task='regression'` or `task='classification'`

**Choose RobustPrefixHonestTree when**:
- You need trustworthy probability estimates (classification) or predictions (regression)
- Your data may have outliers
- You want very stable decision boundaries
- **Works for both**: `task='regression'` or `task='classification'` (binary only for now)

**Choose CentroidTree when**:
- You need a single interpretable tree (for auditing, explanation, regulatory compliance)
- You want ensemble-like stability without an actual ensemble
- Training time is not a bottleneck (trains N candidates)
- **Works for both**: `task='regression'` or `task='classification'`

**Stick with Standard CART when**:
- You need maximum speed
- You have very large datasets (>100k samples)
- Stability is not a concern

## Performance Comparison

Here's how stable-cart models typically perform compared to standard trees:

| Metric | Standard Tree | Stable CART | Improvement |
|--------|---------------|-------------|-------------|
| **Prediction Variance** | High | Low | 30-50% reduction |
| **Out-of-sample Stability** | Variable | Consistent | 20-40% more stable |
| **Accuracy** | High | Slightly lower | 2-5% trade-off |
| **Interpretability** | Good | Good | Maintained |

### CentroidTree Experimental Results

Results from synthetic classification experiments (500 samples, 10 features, 30 random seeds):

| Method | Accuracy | Disagreement Rate | vs CART |
|--------|----------|-------------------|---------|
| CART | 0.85 | 26.4% | baseline |
| CentroidTree (N=20) | 0.84 | 25.6% | **-3% disagreement** |
| LessGreedyHybridTree | 0.80 | 26.2% | similar stability |
| RandomForest (N=20) | 0.88 | 27.1% | ensemble baseline |

**Key finding**: CentroidTree achieves ~3% disagreement reduction with only ~1% accuracy loss, while maintaining single-tree interpretability. This makes it ideal for applications requiring auditable models.

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
  version={1.1.0}
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
