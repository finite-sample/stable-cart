"""Unit tests for stable_cart package."""

import pytest
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Import from the installed package
from stable_cart.evaluation import prediction_stability, accuracy
from stable_cart.less_greedy_tree import (
    GreedyCARTExact,
    LessGreedyHybridRegressor,
    _sse,
    _ComparableFloat,
)


# Tolerance for floating-point comparisons
TOL = 1e-6

# -------------------------------
# Fixtures
# -------------------------------


@pytest.fixture
def small_regression_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    return X, y


@pytest.fixture
def regression_models_and_data():
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    X_tr, X_oos, y_tr, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    models = {
        "dt1": DecisionTreeRegressor(random_state=0).fit(X_tr, y_tr),
        "dt2": DecisionTreeRegressor(random_state=1).fit(X_tr, y_tr),
    }
    return models, X_oos, "continuous"


@pytest.fixture
def classification_models_and_data():
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
    X_tr, X_oos, y_tr, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    models = {
        "dt1": DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr),
        "dt2": DecisionTreeClassifier(random_state=1).fit(X_tr, y_tr),
    }
    return models, X_oos, "categorical"


# -------------------------------
# Test backward compatibility
# -------------------------------


# -------------------------------
# Test utilities
# -------------------------------


def test_sse():
    y = np.array([1.0, 2.0, 3.0])
    assert _sse(y) == pytest.approx(2.0, abs=TOL)  # var=2/3, n*var=2
    y = np.array([5.0])
    assert _sse(y) == pytest.approx(0.0, abs=TOL)
    y = np.array([])
    assert _sse(y) == pytest.approx(0.0, abs=TOL)


def test_comparable_float():
    cf = _ComparableFloat(1.0)
    assert cf == pytest.approx(1.0)
    assert cf > pytest.approx(0.999)
    assert cf < pytest.approx(1.001)
    assert cf >= pytest.approx(1.0)
    assert cf <= pytest.approx(1.0)


# -------------------------------
# Test GreedyCARTExact
# -------------------------------


def test_greedy_cart_exact_fit_predict(small_regression_data):
    X, y = small_regression_data
    model = GreedyCARTExact(max_depth=2, min_samples_split=2, min_samples_leaf=1)
    model.fit(X, y)
    assert model.tree_["type"] in ["split", "leaf"]
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert model.fit_time_sec_ >= 0
    assert model.splits_scanned_ >= 0


def test_greedy_cart_exact_sklearn_compatibility():
    """Test sklearn API compatibility."""
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = make_regression(n_samples=100, n_features=5, random_state=42)

    # Test with cross-validation
    model = GreedyCARTExact(max_depth=3)
    scores = cross_val_score(model, X, y, cv=3, scoring="r2")
    assert len(scores) == 3

    # Test in pipeline
    pipe = Pipeline([("scaler", StandardScaler()), ("model", GreedyCARTExact(max_depth=3))])
    pipe.fit(X, y)
    predictions = pipe.predict(X)
    assert predictions.shape == y.shape


# -------------------------------
# Test LessGreedyHybridRegressor
# -------------------------------


def test_less_greedy_hybrid_fit_predict(small_regression_data):
    X, y = small_regression_data
    model = LessGreedyHybridRegressor(
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        split_frac=0.5,
        val_frac=0.25,
        est_frac=0.25,
        random_state=42,
    )
    model.fit(X, y)
    assert model.tree_["type"] in ["split", "split_oblique", "leaf"]
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert model.fit_time_sec_ >= 0


def test_less_greedy_hybrid_sklearn_compatibility():
    """Test sklearn API compatibility."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.base import clone

    X, y = make_regression(n_samples=100, n_features=5, random_state=42)

    # Test cloning
    model = LessGreedyHybridRegressor(max_depth=3, random_state=42)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()

    # Test GridSearchCV
    param_grid = {"max_depth": [2, 3], "leaf_shrinkage_lambda": [0.0, 0.1]}
    grid = GridSearchCV(model, param_grid, cv=3, scoring="r2")
    grid.fit(X, y)
    assert hasattr(grid, "best_params_")
    assert hasattr(grid, "best_score_")


# -------------------------------
# Test evaluation functions
# -------------------------------


def test_prediction_stability_continuous(regression_models_and_data):
    models, X_oos, task = regression_models_and_data
    scores = prediction_stability(models, X_oos, task)
    assert len(scores) == len(models)
    assert all(isinstance(v, float) for v in scores.values())
    assert all(v >= 0 for v in scores.values())


def test_prediction_stability_categorical(classification_models_and_data):
    models, X_oos, task = classification_models_and_data
    scores = prediction_stability(models, X_oos, task)
    assert len(scores) == len(models)
    assert all(0 <= v <= 1 for v in scores.values())


def test_accuracy_continuous():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    model = DecisionTreeRegressor(random_state=42).fit(X, y)
    models = {"dt": model}

    results = accuracy(models, X, y, task="continuous")
    assert len(results) == 1
    metrics = results["dt"]
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1


def test_accuracy_categorical():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    model = DecisionTreeClassifier(random_state=42).fit(X, y)
    models = {"dt": model}

    results = accuracy(models, X, y, task="categorical")
    assert len(results) == 1
    metrics = results["dt"]
    assert "acc" in metrics
    assert 0 <= metrics["acc"] <= 1
    if "auc" in metrics:
        assert 0 <= metrics["auc"] <= 1
