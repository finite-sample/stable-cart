"""Unit tests for BootstrapVariancePenalizedRegressor."""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stable_cart.bootstrap_variance_tree import BootstrapVariancePenalizedRegressor, SimpleTree


# Tolerance for floating-point comparisons
TOL = 1e-6


@pytest.fixture
def small_regression_data():
    """Small regression dataset for basic tests."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([1, 2, 3, 4, 5, 6])
    return X, y


@pytest.fixture
def regression_data():
    """Larger regression dataset for testing."""
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    return X, y


# -------------------------------
# Test BootstrapVariancePenalizedRegressor
# -------------------------------


def test_bootstrap_variance_regressor_basic_fit(small_regression_data):
    """Test basic fitting functionality."""
    X, y = small_regression_data
    model = BootstrapVariancePenalizedRegressor(
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        variance_penalty=1.0,
        n_bootstrap=3,
        random_state=42,
    )
    model.fit(X, y)

    assert model.tree_["type"] in ["split", "leaf"]
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert model.fit_time_sec_ >= 0
    assert model.bootstrap_evaluations_ >= 0


def test_bootstrap_variance_regressor_no_penalty(regression_data):
    """Test with zero variance penalty (should behave like standard tree)."""
    X, y = regression_data
    model = BootstrapVariancePenalizedRegressor(
        max_depth=3, variance_penalty=0.0, n_bootstrap=0, random_state=42
    )
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == y.shape
    assert model.bootstrap_evaluations_ == 0  # No bootstrap evaluations when penalty=0


def test_bootstrap_variance_regressor_with_penalty(regression_data):
    """Test with non-zero variance penalty."""
    X, y = regression_data
    model = BootstrapVariancePenalizedRegressor(
        max_depth=3, variance_penalty=2.0, n_bootstrap=5, bootstrap_max_depth=1, random_state=42
    )
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == y.shape
    assert model.bootstrap_evaluations_ > 0  # Should have bootstrap evaluations


def test_bootstrap_variance_regressor_sklearn_compatibility(regression_data):
    """Test sklearn API compatibility."""
    X, y = regression_data

    # Test with cross-validation
    model = BootstrapVariancePenalizedRegressor(
        max_depth=3, variance_penalty=1.0, n_bootstrap=3, random_state=42
    )
    scores = cross_val_score(model, X, y, cv=3, scoring="r2")
    assert len(scores) == 3
    assert all(isinstance(s, (int, float)) for s in scores)

    # Test in pipeline
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                BootstrapVariancePenalizedRegressor(
                    max_depth=3, variance_penalty=0.5, n_bootstrap=3, random_state=42
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    predictions = pipe.predict(X)
    assert predictions.shape == y.shape


def test_bootstrap_variance_regressor_score_method(regression_data):
    """Test the score method."""
    X, y = regression_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = BootstrapVariancePenalizedRegressor(
        max_depth=3, variance_penalty=1.0, n_bootstrap=3, random_state=42
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    assert isinstance(score, (int, float))
    assert -1 <= score <= 1  # R² score range


def test_bootstrap_variance_regressor_count_leaves(small_regression_data):
    """Test leaf counting."""
    X, y = small_regression_data
    model = BootstrapVariancePenalizedRegressor(
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        variance_penalty=1.0,
        n_bootstrap=3,
        random_state=42,
    )
    model.fit(X, y)

    leaf_count = model.count_leaves()
    assert isinstance(leaf_count, int)
    assert leaf_count >= 1


def test_bootstrap_variance_regressor_empty_data():
    """Test error handling with empty data."""
    X = np.array([]).reshape(0, 2)
    y = np.array([])

    model = BootstrapVariancePenalizedRegressor(random_state=42)

    with pytest.raises(ValueError, match="X and y must contain at least one sample"):
        model.fit(X, y)


def test_bootstrap_variance_regressor_invalid_fractions():
    """Test error handling with invalid data fractions."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)

    model = BootstrapVariancePenalizedRegressor(
        split_frac=0.8, val_frac=0.15, est_frac=0.1, random_state=42  # Doesn't sum to 1
    )

    with pytest.raises(AssertionError, match="split_frac \\+ val_frac \\+ est_frac must sum to 1"):
        model.fit(X, y)


# -------------------------------
# Test SimpleTree helper class
# -------------------------------


def test_simple_tree_basic_functionality(small_regression_data):
    """Test basic SimpleTree functionality."""
    X, y = small_regression_data
    tree = SimpleTree(max_depth=2, min_samples_leaf=1)
    tree.fit(X, y)

    preds = tree.predict(X)
    assert preds.shape == y.shape
    assert tree.tree_ is not None


def test_simple_tree_depth_limiting():
    """Test that SimpleTree respects depth limits."""
    X, y = make_regression(n_samples=50, n_features=3, random_state=42)

    # Very shallow tree
    tree = SimpleTree(max_depth=1, min_samples_leaf=5)
    tree.fit(X, y)

    preds = tree.predict(X)
    assert preds.shape == y.shape


def test_simple_tree_small_data():
    """Test SimpleTree with very small datasets."""
    X = np.array([[1], [2]])
    y = np.array([1, 2])

    tree = SimpleTree(max_depth=1, min_samples_leaf=1)
    tree.fit(X, y)

    preds = tree.predict(X)
    assert preds.shape == y.shape


# -------------------------------
# Test parameter effects
# -------------------------------


def test_variance_penalty_effect(regression_data):
    """Test that different variance penalties affect bootstrap evaluations."""
    X, y = regression_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # No penalty model
    model_no_penalty = BootstrapVariancePenalizedRegressor(
        max_depth=3, variance_penalty=0.0, n_bootstrap=0, random_state=42
    )
    model_no_penalty.fit(X_train, y_train)

    # High penalty model
    model_high = BootstrapVariancePenalizedRegressor(
        max_depth=3, variance_penalty=5.0, n_bootstrap=5, random_state=42
    )
    model_high.fit(X_train, y_train)

    # The high penalty model should have conducted bootstrap evaluations
    assert model_no_penalty.bootstrap_evaluations_ == 0
    assert model_high.bootstrap_evaluations_ > 0

    # Both models should be able to make predictions
    preds_no_penalty = model_no_penalty.predict(X_test)
    preds_high = model_high.predict(X_test)
    assert preds_no_penalty.shape == y_test.shape
    assert preds_high.shape == y_test.shape


def test_bootstrap_samples_effect(regression_data):
    """Test that different numbers of bootstrap samples affect computation."""
    X, y = regression_data

    # Few bootstrap samples
    model_few = BootstrapVariancePenalizedRegressor(
        max_depth=3, variance_penalty=1.0, n_bootstrap=2, random_state=42
    )
    model_few.fit(X, y)

    # Many bootstrap samples
    model_many = BootstrapVariancePenalizedRegressor(
        max_depth=3, variance_penalty=1.0, n_bootstrap=10, random_state=42
    )
    model_many.fit(X, y)

    # More bootstrap samples should result in more evaluations
    assert model_many.bootstrap_evaluations_ >= model_few.bootstrap_evaluations_
