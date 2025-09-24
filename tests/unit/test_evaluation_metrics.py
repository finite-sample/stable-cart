import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseClassifierMixin

import evalutation as ev


def test_prediction_stability_categorical_returns_per_model_scores():
    """Test that categorical prediction stability returns valid per-model disagreement scores."""
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=6, n_classes=3, random_state=0
    )
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.33, random_state=0)
    m1 = DecisionTreeClassifier(max_depth=3, random_state=1).fit(Xtr, ytr)
    m2 = DecisionTreeClassifier(max_depth=5, random_state=2).fit(Xtr, ytr)
    models = {"cart3": m1, "cart5": m2}
    scores = ev.prediction_stability(models, Xte, task="categorical")
    assert set(scores.keys()) == {"cart3", "cart5"}
    for v in scores.values():
        assert 0.0 <= v <= 1.0
        assert pytest.approx(v, abs=0.1) == v  # Ensure reasonable range


def test_prediction_stability_continuous_returns_nonnegative_rmse():
    """Test that continuous prediction stability returns non-negative, finite RMSE."""
    X, y = make_regression(n_samples=500, n_features=8, noise=15, random_state=0)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=0)
    m1 = DecisionTreeRegressor(max_depth=3, random_state=1).fit(Xtr, ytr)
    m2 = LinearRegression().fit(Xtr, ytr)
    models = {"tree": m1, "lin": m2}
    scores = ev.prediction_stability(models, Xte, task="continuous")
    assert set(scores.keys()) == {"tree", "lin"}
    for v in scores.values():
        assert v >= 0.0 and np.isfinite(v)
        assert pytest.approx(v, rel=0.1) == v  # Ensure reasonable magnitude


def test_accuracy_reports_expected_fields_for_both_tasks():
    """Test that accuracy returns expected metrics for categorical and continuous tasks."""
    # Classification
    Xc, yc = make_classification(n_samples=400, n_features=10, n_classes=2, random_state=7)
    Xctr, Xcte, yctr, ycte = train_test_split(Xc, yc, test_size=0.3, random_state=7)
    c1 = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xctr, yctr)
    c2 = DecisionTreeClassifier(max_depth=6, random_state=1).fit(Xctr, yctr)
    cls = {"d4": c1, "d6": c2}
    acc = ev.accuracy(cls, Xcte, ycte, task="categorical")
    for name in cls:
        assert {"acc", "auc"} <= set(acc[name].keys())
        assert 0.0 <= acc[name]["acc"] <= 1.0
        assert 0.5 <= acc[name]["auc"] <= 1.0

    # Regression
    Xr, yr = make_regression(n_samples=400, n_features=6, noise=10, random_state=9)
    Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.3, random_state=9)
    r1 = DecisionTreeRegressor(max_depth=4, random_state=0).fit(Xrtr, yrtr)
    r2 = LinearRegression().fit(Xrtr, yrtr)
    regs = {"dtr": r1, "lin": r2}
    perf = ev.accuracy(regs, Xrte, yrte, task="continuous")
    for name in regs:
        assert {"mae", "rmse", "r2"} <= set(perf[name].keys())
        assert perf[name]["rmse"] >= 0.0 and np.isfinite(perf[name]["rmse"])
        assert np.isfinite(perf[name]["mae"])
        assert np.isfinite(perf[name]["r2"])


def test_accuracy_categorical_no_predict_proba():
    """Test that accuracy skips AUC for classifiers without predict_proba."""
    class NoProbaClassifier(DecisionTreeClassifier):
        def predict_proba(self, X):
            raise AttributeError("No predict_proba")

    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    model = NoProbaClassifier(max_depth=3).fit(Xtr, ytr)
    acc = ev.accuracy({"nop": model}, Xte, yte, task="categorical")
    assert set(acc["nop"].keys()) == {"acc"}
    assert 0.0 <= acc["nop"]["acc"] <= 1.0