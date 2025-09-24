import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import evalutation as ev  # Intentional misspelling per original


def test_prediction_stability_needs_at_least_two_models():
    """Test that prediction_stability raises ValueError with fewer than two models."""
    X = np.random.RandomState(0).normal(size=(20, 4))
    m1 = DecisionTreeClassifier().fit(X, (X[:, 0] > 0).astype(int))
    with pytest.raises(ValueError, match="Need at least 2 models"):
        ev.prediction_stability({"only_one": m1}, X, task="categorical")


def test_prediction_stability_invalid_task_raises():
    """Test that prediction_stability raises ValueError for invalid task."""
    X = np.random.RandomState(0).normal(size=(10, 2))
    m1 = DecisionTreeClassifier().fit(X, np.zeros(10))
    m2 = DecisionTreeClassifier().fit(X, np.zeros(10))
    with pytest.raises(ValueError, match="task must be 'categorical' or 'continuous'"):
        ev.prediction_stability({"m1": m1, "m2": m2}, X, task="invalid")


def test_accuracy_invalid_task_raises():
    """Test that accuracy raises ValueError for invalid task."""
    X = np.random.RandomState(0).normal(size=(10, 2))
    y = np.zeros(10)
    m1 = DecisionTreeRegressor().fit(X, y)
    with pytest.raises(ValueError, match="task must be 'categorical' or 'continuous'"):
        ev.accuracy({"m1": m1}, X, y, task="wrong")


def test_prediction_stability_empty_X_raises():
    """Test that prediction_stability raises an error for empty X_oos."""
    X = np.random.RandomState(0).normal(size=(10, 2))
    m1 = DecisionTreeClassifier().fit(X, np.zeros(10))
    m2 = DecisionTreeClassifier().fit(X, np.zeros(10))
    with pytest.raises(ValueError):  # Raised by numpy during predict
        ev.prediction_stability({"m1": m1, "m2": m2}, np.empty((0, 2)), task="categorical")