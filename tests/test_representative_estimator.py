"""Executable specification and limiting cases for RepresentativeEstimator."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stable_cart import RepresentativeEstimator


@pytest.fixture
def regression_data():
    return make_regression(n_samples=260, n_features=6, noise=4, random_state=0)


@pytest.fixture
def classification_data():
    return make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )


@pytest.mark.parametrize("metric", ["rmse", "mae", "correlation"])
def test_regression_scores_recompute_from_stored_candidate_predictions(
    metric, regression_data
):
    X, y = regression_data
    model = RepresentativeEstimator(
        estimator=DecisionTreeRegressor(max_depth=3),
        task="regression",
        n_candidates=7,
        proximity_metric=metric,
        random_state=3,
    ).fit(X, y)
    predictions = model.validation_predictions_.astype(float)
    centroid = predictions.mean(axis=0)
    if metric == "rmse":
        expected = np.sqrt(np.mean((predictions - centroid) ** 2, axis=1))
    elif metric == "mae":
        expected = np.mean(np.abs(predictions - centroid), axis=1)
    else:
        expected = np.array(
            [model._correlation_distance(row, centroid) for row in predictions]
        )

    assert np.allclose(model.ensemble_predictions_, centroid)
    assert np.allclose(model.candidate_scores_, expected)
    assert model.selected_index_ == int(np.argmin(expected))
    assert model.get_selected_estimator() is model.candidates_[model.selected_index_]


@pytest.mark.parametrize("metric", ["disagreement", "probability_mse", "auto"])
def test_multiclass_scores_recompute_from_stored_candidates(
    metric, classification_data
):
    X, y = classification_data
    model = RepresentativeEstimator(
        estimator=DecisionTreeClassifier(max_depth=3),
        task="classification",
        n_candidates=7,
        proximity_metric=metric,
        random_state=4,
    ).fit(X, y)

    if model.proximity_metric_ == "probability_mse":
        probabilities = model.validation_probabilities_
        assert probabilities is not None
        centroid = probabilities.mean(axis=0)
        expected = np.mean((probabilities - centroid) ** 2, axis=(1, 2))
    else:
        centroid = model._column_modes(model.validation_predictions_)
        expected = np.mean(model.validation_predictions_ != centroid, axis=1)

    assert np.allclose(model.candidate_scores_, expected)
    assert model.selected_index_ == int(np.argmin(expected))
    predicted_probabilities = model.predict_proba(X)
    assert predicted_probabilities.shape == (len(X), 3)
    assert np.allclose(predicted_probabilities.sum(axis=1), 1)
    assert set(model.predict(X)) <= set(y)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_candidate_performance_scores_recompute_on_selection_split(
    task, regression_data, classification_data
):
    X, y = regression_data if task == "regression" else classification_data
    seed = 9
    model = RepresentativeEstimator(
        task=task,
        n_candidates=5,
        random_state=seed,
    ).fit(X, y)
    _, X_validation, _, y_validation = train_test_split(
        X,
        y,
        test_size=model.validation_fraction,
        random_state=seed,
        stratify=y if task == "classification" else None,
    )
    metric = accuracy_score if task == "classification" else r2_score
    expected = [
        metric(y_validation, candidate.predict(X_validation))
        for candidate in model.candidates_
    ]

    assert np.allclose(model.candidate_performance_scores_, expected)


def test_one_candidate_selects_that_candidate_with_zero_distance(regression_data):
    X, y = regression_data
    model = RepresentativeEstimator(
        task="regression", n_candidates=1, random_state=0
    ).fit(X, y)

    assert model.selected_index_ == 0
    assert model.candidate_scores_[0] == pytest.approx(0)
    assert model.get_selected_estimator() is model.candidates_[0]


def test_squared_centroid_rule_is_exactly_the_prediction_medoid(regression_data):
    X, y = regression_data
    model = RepresentativeEstimator(
        estimator=DecisionTreeRegressor(max_depth=3),
        task="regression",
        n_candidates=8,
        proximity_metric="rmse",
        random_state=7,
    ).fit(X, y)
    predictions = model.validation_predictions_.astype(float)
    pairwise_squared_distance = np.mean(
        (predictions[:, None, :] - predictions[None, :, :]) ** 2,
        axis=2,
    )
    medoid_scores = pairwise_squared_distance.mean(axis=1)

    assert model.selected_index_ == int(np.argmin(medoid_scores))


def test_no_bootstrap_collapses_deterministic_candidate_pool(regression_data):
    X, y = regression_data
    model = RepresentativeEstimator(
        estimator=Ridge(),
        task="regression",
        n_candidates=5,
        bootstrap_candidates=False,
        random_state=0,
    ).fit(X, y)

    assert np.all(model.validation_predictions_ == model.validation_predictions_[0])
    assert np.allclose(model.candidate_scores_, 0)
    assert model.selected_index_ == 0


def test_constant_correlation_is_not_mistaken_for_a_perfect_match():
    centroid = np.ones(5)

    assert RepresentativeEstimator._correlation_distance(np.ones(5), centroid) == 0
    assert np.isinf(
        RepresentativeEstimator._correlation_distance(np.zeros(5), centroid)
    )


class LabelOnlyClassifier(BaseEstimator):
    """Small classifier used to verify the no-probability branch."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.threshold_ = float(np.median(X[:, 0]))
        return self

    def predict(self, X):
        return self.classes_[(X[:, 0] > self.threshold_).astype(int)]


def test_auto_classification_falls_back_to_disagreement_without_probabilities():
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=2, random_state=0
    )
    model = RepresentativeEstimator(
        estimator=LabelOnlyClassifier(),
        task="classification",
        n_candidates=5,
        proximity_metric="auto",
        random_state=0,
    ).fit(X, y)

    assert model.proximity_metric_ == "disagreement"
    assert model.validation_probabilities_ is None
    assert not hasattr(model, "predict_proba")
    with pytest.raises(AttributeError):
        model.predict_proba(X)


def test_explicit_probability_metric_requires_probabilities():
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=2, random_state=0
    )

    with pytest.raises(ValueError, match="every candidate"):
        RepresentativeEstimator(
            estimator=LabelOnlyClassifier(),
            task="classification",
            proximity_metric="probability_mse",
            random_state=0,
        ).fit(X, y)


def test_decision_tree_can_be_used_as_a_base_estimator(regression_data):
    X, y = regression_data
    model = RepresentativeEstimator(
        estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=5),
        task="regression",
        n_candidates=3,
        random_state=0,
    ).fit(X, y)

    assert isinstance(model.get_selected_estimator(), DecisionTreeRegressor)
    assert model.predict(X).shape == (len(X),)


@pytest.mark.parametrize(
    ("task", "estimator"),
    [("regression", Ridge()), ("classification", LogisticRegression(max_iter=1000))],
)
def test_selector_is_not_tree_specific(task, estimator):
    if task == "regression":
        X, y = make_regression(n_samples=180, n_features=5, random_state=0)
    else:
        X, y = make_classification(n_samples=180, n_features=5, random_state=0)
    model = RepresentativeEstimator(
        estimator=estimator,
        task=task,
        n_candidates=4,
        random_state=0,
    ).fit(X, y)

    assert type(model.get_selected_estimator()) is type(estimator)
    assert model.predict(X).shape == (len(X),)


def test_dataframe_column_names_survive_fit_and_predict():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {
            "kind": np.tile(["a", "b", "c"], 40),
            "value": np.linspace(-2.0, 2.0, 120),
        }
    )
    y = np.linspace(0.0, 1.0, len(X))
    estimator = make_pipeline(
        ColumnTransformer(
            [
                ("category", OneHotEncoder(handle_unknown="ignore"), ["kind"]),
                ("numeric", StandardScaler(), ["value"]),
            ]
        ),
        Ridge(),
    )

    model = RepresentativeEstimator(
        estimator=estimator,
        task="regression",
        n_candidates=4,
        random_state=12,
    ).fit(X, y)

    assert model.feature_names_in_.tolist() == ["kind", "value"]
    assert model.predict(X.iloc[:7]).shape == (7,)


def test_nested_random_states_are_seeded_reproducibly(regression_data):
    X, y = regression_data
    estimator = make_pipeline(
        StandardScaler(),
        DecisionTreeRegressor(splitter="random", max_depth=4),
    )

    first = RepresentativeEstimator(
        estimator=estimator,
        task="regression",
        n_candidates=6,
        random_state=13,
    ).fit(X, y)
    second = RepresentativeEstimator(
        estimator=estimator,
        task="regression",
        n_candidates=6,
        random_state=13,
    ).fit(X, y)

    assert np.array_equal(first.validation_predictions_, second.validation_predictions_)
    assert np.array_equal(first.predict(X), second.predict(X))
    assert first.selected_index_ == second.selected_index_
    assert all(
        candidate.get_params()["decisiontreeregressor__random_state"] is not None
        for candidate in first.candidates_
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task": "ranking"}, "task"),
        ({"n_candidates": 0}, "positive integer"),
        ({"validation_fraction": 0}, "strictly between"),
        ({"validation_fraction": 1}, "strictly between"),
        ({"proximity_metric": "bogus"}, "Unknown"),
        (
            {"task": "classification", "proximity_metric": "rmse"},
            "not valid",
        ),
        (
            {"task": "regression", "proximity_metric": "disagreement"},
            "not valid",
        ),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message, regression_data):
    X, y = regression_data

    with pytest.raises(ValueError, match=message):
        RepresentativeEstimator(**kwargs).fit(X, y)


def test_regression_does_not_expose_classification_predictions(regression_data):
    X, y = regression_data
    model = RepresentativeEstimator(task="regression", n_candidates=3).fit(X, y)

    assert not hasattr(model, "predict_proba")
