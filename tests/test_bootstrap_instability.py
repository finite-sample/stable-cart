"""Tests for bootstrap_instability.

The checks here are identities with exactly right answers, so a violation is a
defect rather than a judgement call.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stable_cart import bootstrap_instability


class ConstantRegressor(BaseEstimator, RegressorMixin):
    """Predicts a fixed value, ignoring the training data entirely."""

    def fit(self, X, y):
        """Ignore the data and return self."""
        return self

    def predict(self, X):
        """Return the same constant for every row."""
        return np.zeros(len(X))


@pytest.fixture
def regression_data():
    """A small regression problem split into train and eval."""
    X, y = make_regression(n_samples=300, n_features=6, noise=1.0, random_state=0)
    return X[:200], y[:200], X[200:]


@pytest.fixture
def classification_data():
    """A small classification problem split into train and eval."""
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, random_state=0
    )
    return X[:200], y[:200], X[200:]


def test_a_model_ignoring_its_training_data_is_perfectly_stable(regression_data):
    """The degenerate case that makes this metric dangerous read alone."""
    X_train, y_train, X_eval = regression_data

    result = bootstrap_instability(
        ConstantRegressor, X_train, y_train, X_eval, task="continuous", random_state=0
    )

    assert result["instability_mean"] == 0.0
    assert result["instability_max"] == 0.0


def test_averaging_is_more_stable_than_a_single_tree(regression_data):
    """Bagging must beat its own base learner; if not, the harness is wrong."""
    X_train, y_train, X_eval = regression_data

    single = bootstrap_instability(
        lambda: DecisionTreeRegressor(max_depth=8, random_state=0),
        X_train,
        y_train,
        X_eval,
        task="continuous",
        random_state=0,
    )
    bagged = bootstrap_instability(
        lambda: BaggingRegressor(
            DecisionTreeRegressor(max_depth=8), n_estimators=20, random_state=0
        ),
        X_train,
        y_train,
        X_eval,
        task="continuous",
        random_state=0,
    )

    assert bagged["instability_mean"] < single["instability_mean"]


def test_deterministic_given_random_state(regression_data):
    """Same seed twice, identical numbers."""
    X_train, y_train, X_eval = regression_data
    kwargs = {"task": "continuous", "random_state": 7}

    a = bootstrap_instability(
        lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
        X_train,
        y_train,
        X_eval,
        **kwargs,
    )
    b = bootstrap_instability(
        lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
        X_train,
        y_train,
        X_eval,
        **kwargs,
    )

    assert a == b


def test_classification_disagreement_is_a_fraction(classification_data):
    """The categorical statistic is a rate, so it lives in [0, 1]."""
    X_train, y_train, X_eval = classification_data

    result = bootstrap_instability(
        lambda: DecisionTreeClassifier(max_depth=8, random_state=0),
        X_train,
        y_train,
        X_eval,
        task="categorical",
        random_state=0,
    )

    assert 0.0 <= result["instability_mean"] <= 1.0
    assert 0.0 <= result["instability_max"] <= 1.0
    assert result["instability_mean"] <= result["instability_max"]


def test_shallow_trees_are_more_stable_than_deep_ones(regression_data):
    """Regularization reduces instability — the direction the metric must capture."""
    X_train, y_train, X_eval = regression_data

    shallow = bootstrap_instability(
        lambda: DecisionTreeRegressor(max_depth=2, random_state=0),
        X_train,
        y_train,
        X_eval,
        task="continuous",
        random_state=0,
    )
    deep = bootstrap_instability(
        lambda: DecisionTreeRegressor(max_depth=None, random_state=0),
        X_train,
        y_train,
        X_eval,
        task="continuous",
        random_state=0,
    )

    assert shallow["instability_mean"] < deep["instability_mean"]


def test_rejects_bad_arguments(regression_data):
    """Invalid task or too few resamples fail loudly."""
    X_train, y_train, X_eval = regression_data

    with pytest.raises(ValueError, match="task must be"):
        bootstrap_instability(
            ConstantRegressor, X_train, y_train, X_eval, task="nonsense"
        )
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_instability(
            ConstantRegressor, X_train, y_train, X_eval, n_bootstrap=1
        )
