"""Tests for the accuracy-stability frontier."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stable_cart import pareto_front, stability_frontier
from stable_cart.frontier import _score


@pytest.fixture
def regression_data():
    """A regression problem with signal and noise."""
    return make_regression(
        n_samples=400, n_features=6, n_informative=4, noise=3.0, random_state=0
    )


class TestParetoFront:
    """The frontier is the part users act on, so its definition must be exact."""

    def test_keeps_only_non_dominated_points(self):
        """A point beaten on both axes is dropped."""
        points = [
            {"score": 0.9, "instability": 1.0},  # dominates the third
            {"score": 0.7, "instability": 0.2},  # best stability
            {"score": 0.8, "instability": 2.0},  # dominated
        ]

        front = pareto_front(points)

        assert {(p["score"], p["instability"]) for p in front} == {
            (0.9, 1.0),
            (0.7, 0.2),
        }

    def test_returns_accuracy_descending(self):
        """Ordered so the first row is the most accurate option."""
        points = [
            {"score": 0.5, "instability": 0.1},
            {"score": 0.9, "instability": 0.9},
        ]

        assert [p["score"] for p in pareto_front(points)] == [0.9, 0.5]

    def test_a_single_point_is_its_own_frontier(self):
        assert pareto_front([{"score": 0.5, "instability": 0.5}]) == [
            {"score": 0.5, "instability": 0.5}
        ]


def test_constant_regression_targets_use_standard_r_squared_semantics():
    target = np.ones(3)

    assert _score(target, target, "continuous") == 1.0
    assert _score(np.zeros(3), target, "continuous") == 0.0


class TestFrontier:
    """End-to-end behavior on real estimators."""

    def test_finds_the_known_direction_of_the_tradeoff(self, regression_data):
        """Deeper trees are less stable; the sweep must recover that ordering."""
        X, y = regression_data

        result = stability_frontier(
            lambda **kw: DecisionTreeRegressor(
                min_samples_leaf=10, random_state=0, **kw
            ),
            {"max_depth": [1, 3, 8]},
            X,
            y,
            n_bootstrap=10,
            random_state=0,
        )

        by_depth = {p["params"]["max_depth"]: p for p in result["points"]}
        assert by_depth[1]["instability"] < by_depth[8]["instability"]

    def test_bagging_is_more_stable_than_its_base_learner(self, regression_data):
        """The sanity check that catches a broken harness."""
        X, y = regression_data
        common = {"X": X, "y": y, "n_bootstrap": 10, "random_state": 0}

        single = stability_frontier(
            lambda **kw: DecisionTreeRegressor(random_state=0, **kw),
            {"max_depth": [8]},
            **common,
        )
        bagged = stability_frontier(
            lambda **kw: BaggingRegressor(
                DecisionTreeRegressor(max_depth=8), n_estimators=15, random_state=0
            ),
            {},
            **common,
        )

        assert bagged["points"][0]["instability"] < single["points"][0]["instability"]

    def test_reports_what_the_answer_cost(self, regression_data):
        """A tool nobody waits for is not a capability, so the cost is reported."""
        X, y = regression_data

        result = stability_frontier(
            lambda **kw: DecisionTreeRegressor(random_state=0, **kw),
            {"max_depth": [2, 4]},
            X,
            y,
            n_bootstrap=6,
            random_state=0,
        )

        assert result["n_fits"] == 2 * (6 + 1)
        assert result["seconds"] > 0

    def test_a_model_that_ignores_its_data_is_perfectly_stable(self, regression_data):
        """Zero instability at useless accuracy — the trap, made visible.

        Both numbers are reported together for exactly this reason: a model that
        never looks at its training data scores perfectly on stability. Note the
        predictor must be genuinely data-independent; an unsplittable *tree* still
        moves, because its single leaf value is estimated from the sample.
        """
        X, y = regression_data

        class AlwaysZero:
            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        result = stability_frontier(
            lambda **kw: AlwaysZero(), {}, X, y, n_bootstrap=5, random_state=0
        )

        assert result["points"][0]["mape"] == pytest.approx(0.0)
        assert result["points"][0]["instability"] == pytest.approx(0.0)
        assert result["points"][0]["score"] < 0.1

    def test_even_a_stump_moves_because_its_leaf_is_estimated(self, regression_data):
        """The finer point behind the previous test.

        A tree forced to a single leaf is not a fixed constant: the leaf value is
        a sample mean, so it varies across resamples. Anything claiming a tree is
        perfectly stable has stopped measuring the leaf component.
        """
        X, y = regression_data

        result = stability_frontier(
            lambda **kw: DecisionTreeRegressor(random_state=0, **kw),
            {"max_depth": [1], "min_samples_leaf": [len(X)]},
            X,
            y,
            n_bootstrap=5,
            random_state=0,
        )

        assert result["points"][0]["mape"] > 0.0

    def test_works_for_classification(self):
        """Categorical outcomes use disagreement, not squared error."""
        X, y = make_classification(
            n_samples=300, n_features=6, n_informative=4, random_state=0
        )

        result = stability_frontier(
            lambda **kw: DecisionTreeClassifier(random_state=0, **kw),
            {"max_depth": [2, 6]},
            X,
            y,
            task="categorical",
            n_bootstrap=8,
            random_state=0,
        )

        assert len(result["points"]) == 2
        assert all(0.0 <= p["score"] <= 1.0 for p in result["points"])

    def test_multiclass_instability_is_invariant_to_label_names(self):
        """Class codes are names, so their spacing cannot affect instability."""
        X, y = make_classification(
            n_samples=360,
            n_features=8,
            n_informative=6,
            n_classes=3,
            random_state=0,
        )
        labels = np.array(["cedar", "oak", "sequoia"])[y]
        far_apart = np.array([0, 1, 100])[y]
        kwargs = {
            "estimator_factory": lambda **kw: DecisionTreeClassifier(
                random_state=0, **kw
            ),
            "param_grid": {"max_depth": [4]},
            "X": X,
            "task": "categorical",
            "n_bootstrap": 30,
            "random_state": 7,
        }

        named = stability_frontier(y=labels, **kwargs)["points"][0]
        numeric = stability_frontier(y=far_apart, **kwargs)["points"][0]

        assert named["score"] == pytest.approx(numeric["score"])
        assert named["instability"] == pytest.approx(numeric["instability"])
        assert named["mape"] == pytest.approx(numeric["mape"])

    def test_probability_frontier_uses_the_full_multiclass_vector(self):
        X, y = make_classification(
            n_samples=360,
            n_features=8,
            n_informative=6,
            n_classes=3,
            random_state=0,
        )

        result = stability_frontier(
            lambda **kw: DecisionTreeClassifier(random_state=0, **kw),
            {"max_depth": [4]},
            X,
            y,
            task="categorical",
            prediction_method="predict_proba",
            n_bootstrap=20,
            random_state=0,
        )

        point = result["points"][0]
        assert point["pairwise"] > 0.0
        assert point["mape"] > 0.0

    def test_two_families_land_on_the_same_axes(self, regression_data):
        """The comparison the package exists to enable."""
        X, y = regression_data
        common = {"X": X, "y": y, "n_bootstrap": 8, "random_state": 0}

        cart = stability_frontier(
            lambda **kw: DecisionTreeRegressor(max_depth=4, random_state=0, **kw),
            {"ccp_alpha": [0.0, 1.0]},
            **common,
        )
        bagged = stability_frontier(
            lambda **kw: BaggingRegressor(random_state=0, **kw),
            {"n_estimators": [3, 8]},
            **common,
        )

        combined = pareto_front(cart["points"] + bagged["points"])
        assert combined, "the two families must be comparable on one frontier"

    def test_rejects_bad_arguments(self, regression_data):
        """Fail loudly on nonsense."""
        X, y = regression_data
        factory = lambda **kw: DecisionTreeRegressor(**kw)  # noqa: E731

        with pytest.raises(ValueError, match="task must be"):
            stability_frontier(factory, {}, X, y, task="nonsense")
        with pytest.raises(ValueError, match="at least 2"):
            stability_frontier(factory, {}, X, y, n_bootstrap=1)

    def test_flattens_single_column_validation_targets(self, regression_data):
        X, y = regression_data
        split = 300
        kwargs = {
            "estimator_factory": lambda **kw: DecisionTreeRegressor(
                random_state=0, **kw
            ),
            "param_grid": {"max_depth": [3]},
            "X": X[:split],
            "y": y[:split],
            "X_eval": X[split:],
            "n_bootstrap": 4,
            "random_state": 0,
        }

        vector = stability_frontier(y_eval=y[split:], **kwargs)
        column = stability_frontier(y_eval=y[split:, None], **kwargs)

        assert column["points"][0]["score"] == pytest.approx(
            vector["points"][0]["score"]
        )

    def test_rejects_multioutput_validation_targets(self, regression_data):
        X, y = regression_data

        with pytest.raises(ValueError, match="y should be a 1d array"):
            stability_frontier(
                lambda **kw: DecisionTreeRegressor(random_state=0, **kw),
                {},
                X[:300],
                y[:300],
                X_eval=X[300:],
                y_eval=np.column_stack([y[300:], y[300:]]),
                n_bootstrap=2,
            )


def test_identical_points_appear_once():
    """Two configurations landing on the same point are one operating point.

    A knob that does nothing on this data produces duplicates; counting them
    twice would overstate how many distinct choices a family offers.
    """
    points = [
        {"score": 0.9, "instability": 1.0, "params": {"a": 1}},
        {"score": 0.9, "instability": 1.0, "params": {"a": 2}},
        {"score": 0.5, "instability": 0.1, "params": {"a": 3}},
    ]

    assert len(pareto_front(points)) == 2
