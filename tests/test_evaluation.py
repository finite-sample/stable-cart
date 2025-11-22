"""Updated tests for evaluation.py with improved function names and edge cases."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stable_cart.evaluation import evaluate_models, prediction_stability

# Tolerance for floating-point comparisons
TOL = 1e-6


# -------------------------------
# Fixtures
# -------------------------------


@pytest.fixture
def regression_models_and_data():
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    X_tr, X_oos, y_tr, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    models = {
        "dt1": DecisionTreeRegressor(random_state=0).fit(X_tr, y_tr),
        "dt2": DecisionTreeRegressor(random_state=1).fit(X_tr, y_tr),
        "dt3": DecisionTreeRegressor(random_state=2).fit(X_tr, y_tr),
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


@pytest.fixture
def multiclass_models_and_data():
    X, y = make_classification(
        n_samples=300, n_features=10, n_classes=3, n_informative=5, random_state=42
    )
    X_tr, X_oos, y_tr, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    models = {
        "dt1": DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr),
        "dt2": DecisionTreeClassifier(random_state=1).fit(X_tr, y_tr),
    }
    return models, X_oos, y, "categorical"


# -------------------------------
# Test prediction_stability
# -------------------------------


def test_prediction_stability_continuous(regression_models_and_data):
    """Test prediction stability for continuous task."""
    models, X_oos, task = regression_models_and_data
    scores = prediction_stability(models, X_oos, task)

    assert len(scores) == len(models)
    assert all(isinstance(v, float) for v in scores.values())
    assert all(v >= 0 for v in scores.values())
    # RMSE should be finite
    assert all(np.isfinite(v) for v in scores.values())


def test_prediction_stability_categorical(classification_models_and_data):
    """Test prediction stability for categorical task."""
    models, X_oos, task = classification_models_and_data
    scores = prediction_stability(models, X_oos, task)

    assert len(scores) == len(models)
    # Disagreement should be between 0 and 1
    assert all(0 <= v <= 1 for v in scores.values())


def test_prediction_stability_multiclass(multiclass_models_and_data):
    """Test prediction stability for multi-class classification."""
    models, X_oos, y_oos, task = multiclass_models_and_data
    scores = prediction_stability(models, X_oos, task)

    assert len(scores) == len(models)
    assert all(0 <= v <= 1 for v in scores.values())


def test_prediction_stability_single_model_error():
    """Test that single model raises error."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    models = {"dt1": DecisionTreeRegressor(random_state=0).fit(X, y)}

    with pytest.raises(ValueError, match="at least 2 models"):
        prediction_stability(models, X, task="continuous")


def test_prediction_stability_invalid_task_error():
    """Test that invalid task raises error."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    models = {
        "dt1": DecisionTreeRegressor(random_state=0).fit(X, y),
        "dt2": DecisionTreeRegressor(random_state=1).fit(X, y),
    }

    with pytest.raises(ValueError, match="must be 'categorical' or 'continuous'"):
        prediction_stability(models, X, task="invalid_task")


def test_prediction_stability_identical_models():
    """Test stability when models are identical (should be ~0)."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    model = DecisionTreeRegressor(random_state=42).fit(X, y)

    # Same model, different names
    models = {"model1": model, "model2": model}
    scores = prediction_stability(models, X, task="continuous")

    # Should have very low stability score (models are identical)
    assert all(v < 1e-6 for v in scores.values())


def test_prediction_stability_with_string_labels():
    """Test that string labels are handled correctly."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]] * 10)
    y = np.array(["cat", "cat", "dog", "dog"] * 10)

    # Fit models with string labels
    models = {
        "dt1": DecisionTreeClassifier(random_state=0).fit(X, y),
        "dt2": DecisionTreeClassifier(random_state=1).fit(X, y),
    }

    scores = prediction_stability(models, X, task="categorical")

    # Should work without errors
    assert len(scores) == 2
    assert all(0 <= v <= 1 for v in scores.values())


# -------------------------------
# Test evaluate_models (new name)
# -------------------------------


def test_evaluate_models_continuous():
    """Test evaluate_models for regression."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    model = DecisionTreeRegressor(random_state=42).fit(X, y)
    models = {"dt": model}

    results = evaluate_models(models, X, y, task="continuous")

    assert len(results) == 1
    metrics = results["dt"]
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1


def test_evaluate_models_categorical():
    """Test evaluate_models for classification."""
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    model = DecisionTreeClassifier(random_state=42).fit(X, y)
    models = {"dt": model}

    results = evaluate_models(models, X, y, task="categorical")

    assert len(results) == 1
    metrics = results["dt"]
    assert "acc" in metrics
    assert 0 <= metrics["acc"] <= 1
    if "auc" in metrics:
        assert 0 <= metrics["auc"] <= 1


def test_evaluate_models_multiclass_auc():
    """Test that multiclass AUC is computed correctly."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    models = {"dt": model}

    results = evaluate_models(models, X_test, y_test, task="categorical")

    metrics = results["dt"]
    assert "acc" in metrics
    assert "auc" in metrics  # Should compute multiclass AUC
    assert 0 <= metrics["auc"] <= 1


def test_evaluate_models_without_predict_proba():
    """Test with model that doesn't have predict_proba."""
    from sklearn.svm import LinearSVC

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = LinearSVC(random_state=42).fit(X, y)
    models = {"svm": model}

    results = evaluate_models(models, X, y, task="categorical")

    # Should still work, just without AUC
    assert "acc" in results["svm"]
    # AUC should be absent (not an error)
    assert "auc" not in results["svm"] or results["svm"]["auc"] is None


def test_evaluate_models_multiple_models():
    """Test with multiple models."""
    X, y = make_regression(n_samples=150, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "shallow": DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train),
        "deep": DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_train, y_train),
    }

    results = evaluate_models(models, X_test, y_test, task="continuous")

    assert len(results) == 2
    assert "shallow" in results
    assert "deep" in results
    # Deep tree should typically have better R² on training data
    # (but we're testing on test data, so this isn't guaranteed)


def test_evaluate_models_invalid_task():
    """Test that invalid task raises error."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    model = DecisionTreeRegressor(random_state=42).fit(X, y)
    models = {"dt": model}

    with pytest.raises(ValueError, match="must be 'categorical' or 'continuous'"):
        evaluate_models(models, X, y, task="invalid_task")


def test_evaluate_models_perfect_predictions():
    """Test with perfect predictions."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    # Simple model that memorizes perfectly
    model = DecisionTreeRegressor(random_state=42).fit(X, y)
    models = {"perfect": model}

    results = evaluate_models(models, X, y, task="continuous")

    # Should have perfect metrics
    assert results["perfect"]["mae"] == pytest.approx(0.0, abs=TOL)
    assert results["perfect"]["rmse"] == pytest.approx(0.0, abs=TOL)
    assert results["perfect"]["r2"] == pytest.approx(1.0, abs=TOL)


# -------------------------------
# Test backward compatibility
# -------------------------------


# -------------------------------
# Test edge cases
# -------------------------------


def test_evaluate_models_single_sample():
    """Test with very small dataset."""
    X = np.array([[1, 2]])
    y = np.array([1.0])

    model = DecisionTreeRegressor(random_state=42).fit(X, y)
    models = {"dt": model}

    results = evaluate_models(models, X, y, task="continuous")

    # Should work even with 1 sample
    assert "mae" in results["dt"]
    assert results["dt"]["mae"] == pytest.approx(0.0, abs=TOL)


def test_evaluate_models_all_same_predictions():
    """Test with model that predicts constant."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    # Create a dummy model that always predicts mean
    class ConstantModel:
        def __init__(self, value):
            self.value = value

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), self.value)

    model = ConstantModel(2.5)
    models = {"constant": model}

    results = evaluate_models(models, X, y, task="continuous")

    # R² should be 0 or negative for constant predictions
    assert results["constant"]["r2"] <= 0.01


def test_prediction_stability_three_models():
    """Test stability with three models."""
    X, y = make_regression(n_samples=150, n_features=5, random_state=42)

    models = {
        "dt1": DecisionTreeRegressor(random_state=0).fit(X, y),
        "dt2": DecisionTreeRegressor(random_state=1).fit(X, y),
        "dt3": DecisionTreeRegressor(random_state=2).fit(X, y),
    }

    scores = prediction_stability(models, X, task="continuous")

    # Should compute pairwise stability for all three
    assert len(scores) == 3
    assert all(v >= 0 for v in scores.values())


def test_evaluate_models_nan_predictions():
    """Test handling of NaN predictions."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    # Create a model that produces NaN
    class NaNModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), np.nan)

    model = NaNModel()
    model.fit(X, y)
    models = {"nan_model": model}

    results = evaluate_models(models, X, y, task="continuous")

    # Metrics with NaN predictions should be NaN
    assert np.isnan(results["nan_model"]["mae"])
    assert np.isnan(results["nan_model"]["rmse"])


# -------------------------------
# Test comprehensive scenarios
# -------------------------------


def test_full_evaluation_workflow():
    """Test complete workflow: train, evaluate, check stability."""
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train multiple models
    models = {
        "shallow": DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train),
        "medium": DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train),
        "deep": DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train, y_train),
    }

    # Evaluate performance
    performance = evaluate_models(models, X_test, y_test, task="categorical")

    # Check stability
    stability = prediction_stability(models, X_test, task="categorical")

    # All metrics should be valid
    assert len(performance) == 3
    assert len(stability) == 3

    for name in models.keys():
        assert "acc" in performance[name]
        assert 0 <= performance[name]["acc"] <= 1
        assert 0 <= stability[name] <= 1

    # Deeper trees often less stable (higher disagreement)
    # But this isn't guaranteed, so just check they're all valid
    assert all(0 <= v <= 1 for v in stability.values())
