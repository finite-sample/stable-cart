"""The contract every exported estimator must keep, checked on all of them.

Until 2.0 the shipped estimators had almost no tests of their own: the suite
exercised three unexported legacy modules of the same names, which no user could
reach through the public API. This module replaces that with one parameterised
contract applied to everything ``stable_cart`` exports, so a regression in any
estimator fails the same test rather than none.

The contract is deliberately about *behaviour a user can rely on* — shapes,
determinism, cloning, error signalling, and clearing the trivial baseline — not
about internal structure, which differs by design between the estimators.
"""

import inspect

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from stable_cart import (
    BootstrapVariancePenalizedTree,
    CentroidTree,
    LessGreedyHybridTree,
    RobustPrefixHonestTree,
    StableTree,
)

ESTIMATORS = [
    LessGreedyHybridTree,
    BootstrapVariancePenalizedTree,
    RobustPrefixHonestTree,
    CentroidTree,
    StableTree,
]

# Parameters each estimator renames on the way to its base class, with a value
# that differs from the default. These are the ones ``set_params`` used to drop.
_PROBES = {
    LessGreedyHybridTree: {
        "enable_gain_margin_logic": False,
        "enable_robust_consensus_for_ambiguous": False,
    },
    BootstrapVariancePenalizedTree: {
        "enable_gain_margin_logic": False,
        "enable_robust_consensus": False,
        "variance_penalty": 8.0,
        "n_bootstrap": 3,
    },
    RobustPrefixHonestTree: {
        "top_levels": 1,
        "smoothing": 25.0,
        "enable_threshold_binning": False,
        "enable_gain_margin_logic": False,
    },
    CentroidTree: {"n_candidates": 5, "bootstrap_candidates": False},
    StableTree: {"consensus_threshold": 0.9, "leaf_shrinkage": 20.0},
}


def build(cls, task, **kwargs):
    """Construct an estimator at a size every one of them supports."""
    params = {"task": task, "random_state": 0, **kwargs}
    if cls is CentroidTree:
        params.setdefault("n_candidates", 8)
    else:
        params.setdefault("max_depth", 4)
    return cls(**params)


@pytest.fixture(scope="module")
def regression_data():
    """A regression problem with real signal."""
    X, y = make_regression(
        n_samples=400, n_features=6, n_informative=4, noise=5.0, random_state=0
    )
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture(scope="module")
def classification_data():
    """A binary problem with real signal."""
    X, y = make_classification(
        n_samples=400, n_features=6, n_informative=4, random_state=0
    )
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda c: c.__name__)
class TestRegressionContract:
    """What a user may assume when ``task='regression'``."""

    def test_fit_returns_self_and_predicts_one_value_per_row(
        self, cls, regression_data
    ):
        X_train, X_test, y_train, _ = regression_data
        model = build(cls, "regression")

        assert model.fit(X_train, y_train) is model
        pred = model.predict(X_test)

        assert pred.shape == (len(X_test),)
        assert np.all(np.isfinite(pred))

    def test_beats_predicting_the_mean(self, cls, regression_data):
        """A tree that cannot beat the mean on easy data is not fitting."""
        X_train, X_test, y_train, y_test = regression_data
        model = build(cls, "regression").fit(X_train, y_train)

        baseline = np.mean((y_test - np.mean(y_train)) ** 2)
        mse = np.mean((y_test - model.predict(X_test)) ** 2)

        assert mse < baseline

    def test_same_seed_gives_identical_predictions(self, cls, regression_data):
        """Reproducibility is the premise of every stability claim here."""
        X_train, X_test, y_train, _ = regression_data

        first = build(cls, "regression").fit(X_train, y_train).predict(X_test)
        second = build(cls, "regression").fit(X_train, y_train).predict(X_test)

        assert np.array_equal(first, second)

    def test_score_is_r_squared(self, cls, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = build(cls, "regression").fit(X_train, y_train)

        pred = model.predict(X_test)
        expected = 1.0 - np.sum((y_test - pred) ** 2) / np.sum(
            (y_test - np.mean(y_test)) ** 2
        )

        assert model.score(X_test, y_test) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda c: c.__name__)
class TestClassificationContract:
    """What a user may assume when ``task='classification'``."""

    def test_predicts_labels_that_were_seen_in_training(self, cls, classification_data):
        X_train, X_test, y_train, _ = classification_data
        model = build(cls, "classification").fit(X_train, y_train)

        pred = model.predict(X_test)

        assert pred.shape == (len(X_test),)
        assert set(np.unique(pred)) <= set(np.unique(y_train))

    def test_beats_the_majority_class(self, cls, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = build(cls, "classification").fit(X_train, y_train)

        majority = np.bincount(y_train).argmax()
        baseline = np.mean(y_test == majority)

        assert model.score(X_test, y_test) > baseline

    def test_probabilities_are_a_distribution(self, cls, classification_data):
        X_train, X_test, y_train, _ = classification_data
        model = build(cls, "classification").fit(X_train, y_train)
        if not hasattr(model, "predict_proba"):
            pytest.skip(f"{cls.__name__} does not expose predict_proba")

        proba = model.predict_proba(X_test)

        assert proba.shape == (len(X_test), 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda c: c.__name__)
class TestSklearnContract:
    """The parts scikit-learn's own machinery depends on."""

    def test_clone_reproduces_the_configuration(self, cls, regression_data):
        """``clone`` is how cross_val_score and GridSearchCV copy an estimator."""
        X_train, X_test, y_train, _ = regression_data
        model = build(cls, "regression")

        twin = clone(model)

        assert twin.get_params() == model.get_params()
        assert np.array_equal(
            model.fit(X_train, y_train).predict(X_test),
            twin.fit(X_train, y_train).predict(X_test),
        )

    def test_set_params_accepts_every_constructor_parameter(self, cls):
        """A parameter ``get_params`` reports must be settable, or grid search breaks."""
        model = build(cls, "regression")

        for name, value in model.get_params().items():
            model.set_params(**{name: value})

        assert model.get_params()["task"] == "regression"

    def test_get_params_echoes_what_the_constructor_was_given(self, cls):
        """A parameter must come back out under the name it went in under.

        Three estimators remapped parameters on the way to the base class and
        reported the base class's default instead. ``clone`` then built a
        differently configured twin, so ``cross_val_score`` silently evaluated a
        model the caller had not asked for.
        """
        defaults = inspect.signature(cls.__init__).parameters
        reported = build(cls, "regression").get_params()

        for name, param in defaults.items():
            if name in ("self", "task", "random_state", "max_depth", "n_candidates"):
                continue
            assert reported[name] == param.default, f"{name} does not round-trip"

    @pytest.mark.parametrize("task", ["regression", "classification"])
    def test_set_params_does_what_the_constructor_does(self, cls, task, request):
        """The two ways of configuring an estimator must agree.

        ``GridSearchCV`` uses ``set_params``, so a parameter honoured only by the
        constructor turns a search over that parameter into a search over one
        point — without any error to say so. Measured on 1.1.0:
        ``set_params(enable_gain_margin_logic=False)`` changed no prediction
        while the constructor argument changed many.
        """
        X_train, X_test, y_train, _ = request.getfixturevalue(f"{task}_data")
        moved = False

        for name, alternative in _PROBES.get(cls, {}).items():
            via_constructor = (
                build(cls, task, **{name: alternative})
                .fit(X_train, y_train)
                .predict(X_test)
            )
            model = build(cls, task)
            model.set_params(**{name: alternative})
            via_set_params = model.fit(X_train, y_train).predict(X_test)

            assert np.array_equal(via_constructor, via_set_params), (
                f"set_params({name}={alternative!r}) disagrees with the constructor"
            )
            baseline = build(cls, task).fit(X_train, y_train).predict(X_test)
            moved = moved or not np.array_equal(baseline, via_set_params)

        if _PROBES.get(cls) and not moved:
            pytest.skip("no probe moved a prediction on this task")

    def test_records_the_number_of_features_it_saw(self, cls, regression_data):
        X_train, _X_test, y_train, _ = regression_data

        model = build(cls, "regression").fit(X_train, y_train)

        assert model.n_features_in_ == X_train.shape[1]

    def test_predict_before_fit_raises(self, cls, regression_data):
        X_train, _X_test, _y_train, _ = regression_data

        with pytest.raises(NotFittedError):
            build(cls, "regression").predict(X_train)

    def test_empty_training_data_raises(self, cls):
        model = build(cls, "regression")

        with pytest.raises(ValueError, match=r"0 sample|at least one"):
            model.fit(np.empty((0, 3)), np.empty(0))

    def test_predicting_with_the_wrong_width_raises(self, cls, regression_data):
        """Silently predicting from a differently shaped matrix is the worst outcome."""
        X_train, X_test, y_train, _ = regression_data
        model = build(cls, "regression").fit(X_train, y_train)

        with pytest.raises(ValueError, match="features"):
            model.predict(X_test[:, :-1])
