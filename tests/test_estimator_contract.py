"""Shared scikit-learn contract for RepresentativeEstimator."""

import inspect

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score

import stable_cart
from stable_cart import RepresentativeEstimator


def build(task):
    return RepresentativeEstimator(task=task, random_state=4, n_candidates=5)


def test_supported_estimator_is_exported():
    assert "RepresentativeEstimator" in stable_cart.__all__


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_clone_cross_validation_determinism_and_shapes(task):
    if task == "regression":
        X, y = make_regression(n_samples=180, n_features=5, noise=2, random_state=2)
    else:
        X, y = make_classification(
            n_samples=180, n_features=5, n_informative=3, random_state=2
        )
    estimator = build(task)
    twin = clone(estimator)

    first = estimator.fit(X, y).predict(X)
    second = twin.fit(X, y).predict(X)

    assert first.shape == (len(X),)
    assert np.array_equal(first, second)
    assert len(cross_val_score(build(task), X, y, cv=3)) == 3


def test_constructor_parameters_round_trip():
    estimator = RepresentativeEstimator()
    parameters = inspect.signature(RepresentativeEstimator.__init__).parameters

    assert set(estimator.get_params()) == set(parameters) - {"self"}
    clone(estimator)
    estimator.set_params(**estimator.get_params())


def test_not_fitted_and_wrong_width_errors():
    X, y = make_regression(n_samples=100, n_features=4, random_state=0)
    estimator = build("regression")

    with pytest.raises(NotFittedError):
        estimator.predict(X)
    estimator.fit(X, y)
    with pytest.raises(ValueError, match="features"):
        estimator.predict(X[:, :-1])


def test_dynamic_sklearn_tags_report_the_configured_task():
    classifier = RepresentativeEstimator(task="classification")
    regressor = RepresentativeEstimator(task="regression")

    assert is_classifier(classifier)
    assert not is_regressor(classifier)
    assert is_regressor(regressor)
    assert not is_classifier(regressor)


def test_score_accepts_sample_weights():
    X, y = make_regression(n_samples=140, n_features=4, random_state=3)
    model = build("regression").fit(X, y)
    weights = np.linspace(1, 2, len(y))

    assert np.isfinite(model.score(X, y, sample_weight=weights))
