"""Tests for bootstrap_instability.

The checks here are identities with exactly right answers, so a violation is a
defect rather than a judgment call.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stable_cart import bootstrap_instability, bootstrap_predictions
from stable_cart.evaluation import _pairwise_standard_error


def _brute_pairwise_jackknife(predictions, distance):
    n_draws = len(predictions)
    pair_distances = np.zeros((n_draws, n_draws))
    for left in range(n_draws):
        for right in range(left + 1, n_draws):
            value = distance(predictions[left], predictions[right])
            pair_distances[left, right] = value
            pair_distances[right, left] = value
    total = np.sum(np.triu(pair_distances, k=1))
    leave_one_out = np.array(
        [
            (total - np.sum(pair_distances[index]))
            / ((n_draws - 1) * (n_draws - 2) / 2)
            for index in range(n_draws)
        ]
    )
    return np.sqrt(
        (n_draws - 1) / n_draws * np.sum((leave_one_out - np.mean(leave_one_out)) ** 2)
    )


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


def test_classification_uses_the_ordinary_pairs_bootstrap():
    """Class prevalence must vary rather than being frozen by stratification."""
    X = np.zeros((100, 1))
    y = np.array([0] * 50 + [1] * 50)

    raw = bootstrap_predictions(
        lambda: DummyClassifier(strategy="prior"),
        X,
        y,
        X[:1],
        task="categorical",
        prediction_method="predict_proba",
        n_bootstrap=100,
        random_state=0,
    )

    prevalence = raw["bootstrap"][:, 0, 1]
    assert np.min(prevalence) < 0.5 < np.max(prevalence)
    assert raw["pairwise"][0] > 0.0


def test_probability_alignment_allows_a_class_missing_from_a_resample():
    X = np.zeros((22, 1))
    y = np.array([0] * 20 + [1, 2])

    raw = bootstrap_predictions(
        lambda: DummyClassifier(strategy="prior"),
        X,
        y,
        X[:2],
        task="categorical",
        prediction_method="predict_proba",
        n_bootstrap=100,
        random_state=0,
    )

    assert raw["bootstrap"].shape == (100, 2, 3)
    assert np.allclose(raw["bootstrap"].sum(axis=2), 1.0)
    assert np.any(raw["bootstrap"][:, :, 1:] == 0.0)


def test_one_class_draws_are_redrawn_for_common_classifiers():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 19 + [1])

    raw = bootstrap_predictions(
        lambda: LogisticRegression(),
        X,
        y,
        X[:2],
        task="categorical",
        prediction_method="predict_proba",
        n_bootstrap=20,
        random_state=0,
    )

    assert raw["bootstrap"].shape == (20, 2, 2)
    assert raw["n_rejected_resamples"] > 0
    assert raw["n_fit_attempts"] == 21
    assert raw["n_resample_attempts"] == 20 + raw["n_rejected_resamples"]


def test_multiclass_probabilities_keep_all_classes():
    """Probability instability uses aligned vectors, not maximum confidence."""
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=6,
        n_classes=3,
        random_state=0,
    )

    raw = bootstrap_predictions(
        lambda: DecisionTreeClassifier(max_depth=5, random_state=0),
        X[:220],
        y[:220],
        X[220:],
        task="categorical",
        prediction_method="predict_proba",
        n_bootstrap=12,
        random_state=0,
    )

    assert raw["original"].shape == (80, 3)
    assert raw["bootstrap"].shape == (12, 80, 3)
    assert raw["classes"].tolist() == [0, 1, 2]
    assert np.allclose(raw["bootstrap"].sum(axis=2), 1.0)
    assert np.allclose(raw["pairwise"], 2 * raw["per_point"])

    expected_se = _brute_pairwise_jackknife(
        raw["bootstrap"],
        lambda left, right: np.mean(np.sum((left - right) ** 2, axis=1)),
    )
    assert raw["pairwise_standard_error"] == pytest.approx(expected_se)


def test_pairwise_standard_error_matches_all_pairs_for_numeric_and_labels():
    rng = np.random.default_rng(1)
    numeric = rng.normal(size=(7, 5))
    labels = np.array(
        [
            ["a", "a", "b", "c"],
            ["a", "b", "b", "c"],
            ["b", "b", "b", "c"],
            ["b", "a", "b", "a"],
            ["a", "a", "a", "a"],
        ]
    )

    expected_numeric = _brute_pairwise_jackknife(
        numeric, lambda left, right: np.mean((left - right) ** 2)
    )
    expected_labels = _brute_pairwise_jackknife(
        labels, lambda left, right: np.mean(left != right)
    )

    assert _pairwise_standard_error(numeric, numeric=True) == pytest.approx(
        expected_numeric
    )
    assert _pairwise_standard_error(labels, numeric=False) == pytest.approx(
        expected_labels
    )


def test_pairwise_standard_error_is_stable_under_a_large_common_offset():
    rng = np.random.default_rng(2)
    predictions = 1e12 + rng.normal(size=(10, 100))
    expected = _brute_pairwise_jackknife(
        predictions, lambda left, right: np.mean((left - right) ** 2)
    )

    assert _pairwise_standard_error(predictions, numeric=True) == pytest.approx(
        expected
    )


def test_preserves_dataframe_columns_and_mixed_dtypes():
    """A procedure using named columns must receive a DataFrame on every refit."""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {
            "kind": np.tile(["a", "b", "c"], 30),
            "value": np.linspace(-2.0, 2.0, 90),
        }
    )
    y = np.tile([0, 1, 1], 30)

    result = bootstrap_predictions(
        lambda: make_pipeline(
            ColumnTransformer(
                [
                    ("category", OneHotEncoder(handle_unknown="ignore"), ["kind"]),
                    ("numeric", StandardScaler(), ["value"]),
                ]
            ),
            LogisticRegression(max_iter=1000),
        ),
        X.iloc[:60],
        y[:60],
        X.iloc[60:],
        task="categorical",
        prediction_method="predict_proba",
        n_bootstrap=6,
        random_state=0,
    )

    assert result["bootstrap"].shape == (6, 30, 2)


def test_preserves_sparse_feature_matrices():
    """Sparse input must not be coerced into a scalar object array."""
    sparse = pytest.importorskip("scipy.sparse")
    X, y = make_regression(n_samples=100, n_features=5, random_state=0)
    X_sparse = sparse.csr_matrix(X)

    result = bootstrap_predictions(
        lambda: DecisionTreeRegressor(max_depth=3, random_state=0),
        X_sparse[:70],
        y[:70],
        X_sparse[70:],
        n_bootstrap=6,
        random_state=0,
    )

    assert result["bootstrap"].shape == (6, 30)


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
    with pytest.raises(ValueError, match="prediction_method"):
        bootstrap_predictions(
            ConstantRegressor,
            X_train,
            y_train,
            X_eval,
            prediction_method="decision_function",
        )
    with pytest.raises(ValueError, match="only meaningful"):
        bootstrap_predictions(
            ConstantRegressor,
            X_train,
            y_train,
            X_eval,
            prediction_method="predict_proba",
        )
    with pytest.raises(ValueError, match="same number of rows"):
        bootstrap_predictions(ConstantRegressor, X_train, y_train[:-1], X_eval)
    with pytest.raises(ValueError, match="X_eval must not be empty"):
        bootstrap_predictions(ConstantRegressor, X_train, y_train, X_eval[:0])
    with pytest.raises(ValueError, match="one-dimensional"):
        bootstrap_predictions(ConstantRegressor, X_train, y_train[:, None], X_eval)
