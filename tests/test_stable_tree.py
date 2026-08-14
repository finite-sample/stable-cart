"""Tests for StableTree.

Each mechanism gets a control that fails when the mechanism is disabled, so a
passing suite means the averaging is actually happening rather than the class
merely being a decision tree with extra parameters — which is how the previous
four estimators shipped ~20 inert parameters each.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score

from stable_cart import StableTree
from stable_cart.stability_utils import _find_candidate_splits


@pytest.fixture
def regression_data():
    """A regression problem with a clear signal."""
    return make_regression(
        n_samples=600, n_features=8, n_informative=5, noise=1.0, random_state=0
    )


@pytest.fixture
def classification_data():
    """A binary classification problem with a clear signal."""
    return make_classification(
        n_samples=600, n_features=8, n_informative=5, random_state=0
    )


class TestSplitAveraging:
    """The split decision must be an average, not one sample's argmax."""

    def test_single_replicate_reproduces_the_greedy_split(self, regression_data):
        """n_consensus=1 uses the node itself, so the root must be the greedy pick.

        This is the negative control for the averaging: switch it off and the
        estimator has to agree with the ordinary argmax-gain split on the full
        sample.
        """
        X, y = regression_data
        tree = StableTree(
            task="regression",
            n_consensus=1,
            consensus_threshold=0.0,
            max_depth=1,
            min_samples_leaf=20,
            random_state=0,
        ).fit(X, y)

        greedy = max(
            _find_candidate_splits(X, y, max_candidates=40), key=lambda c: c.gain
        )

        assert tree.tree_["feature"] == greedy.feature_idx
        assert tree.tree_["threshold"] == pytest.approx(greedy.threshold)

    def test_averaged_cut_point_differs_from_any_single_replicate(
        self, regression_data
    ):
        """With many replicates the cut point is a median, not one sample's value."""
        X, y = regression_data
        averaged = StableTree(
            task="regression",
            n_consensus=32,
            consensus_threshold=0.0,
            max_depth=1,
            min_samples_leaf=20,
            random_state=0,
        ).fit(X, y)
        single = StableTree(
            task="regression",
            n_consensus=1,
            consensus_threshold=0.0,
            max_depth=1,
            min_samples_leaf=20,
            random_state=0,
        ).fit(X, y)

        assert averaged.tree_["threshold"] != single.tree_["threshold"]

    def test_support_is_recorded_on_every_split(self, regression_data):
        """Each split carries the share of replicates that voted for its feature."""
        X, y = regression_data
        tree = StableTree(
            task="regression",
            n_consensus=16,
            consensus_threshold=0.0,
            max_depth=3,
            min_samples_leaf=20,
            random_state=0,
        ).fit(X, y)

        supports = tree.split_supports()

        assert supports, "a fitted tree with depth 3 should have splits"
        assert all(0.0 < s <= 1.0 for s in supports)


class TestConsensusThreshold:
    """A split the data cannot reproduce should not be made."""

    def test_raising_the_threshold_never_grows_the_tree(self, regression_data):
        """Monotonicity: a stricter bar can only remove splits."""
        X, y = regression_data
        sizes = []
        for level in (0.0, 0.2, 0.4, 0.6, 0.9):
            tree = StableTree(
                task="regression",
                n_consensus=16,
                consensus_threshold=level,
                max_depth=4,
                min_samples_leaf=20,
                random_state=0,
            ).fit(X, y)
            sizes.append(tree.get_n_leaves())

        assert sizes == sorted(sizes, reverse=True), f"not monotone: {sizes}"

    def test_an_unreachable_threshold_yields_a_stump(self, regression_data):
        """Demanding unanimity on noisy data leaves a single leaf."""
        X, y = regression_data
        tree = StableTree(
            task="regression",
            n_consensus=16,
            consensus_threshold=1.01,
            max_depth=4,
            min_samples_leaf=20,
            random_state=0,
        ).fit(X, y)

        assert tree.get_n_leaves() == 1
        assert tree.predict(X[:5]) == pytest.approx(np.full(5, y.mean()), rel=1e-6)


class TestLeafShrinkage:
    """The leaf half of the variance budget."""

    def test_no_shrinkage_leaves_the_sample_mean(self, regression_data):
        """leaf_shrinkage=0 must not move the leaf value."""
        X, y = regression_data
        tree = StableTree(
            task="regression",
            n_consensus=8,
            consensus_threshold=0.0,
            leaf_shrinkage=0.0,
            max_depth=2,
            min_samples_leaf=20,
            random_state=0,
        ).fit(X, y)

        for value, rows in tree._leaf_values_and_rows(X, y):
            assert value == pytest.approx(rows.mean())

    def test_shrinkage_pulls_predictions_toward_the_global_mean(self, regression_data):
        """Heavier shrinkage must compress the spread of predictions."""
        X, y = regression_data
        common = {
            "task": "regression",
            "n_consensus": 8,
            "consensus_threshold": 0.0,
            "max_depth": 3,
            "min_samples_leaf": 20,
            "random_state": 0,
        }
        none = StableTree(leaf_shrinkage=0.0, **common).fit(X, y).predict(X)
        heavy = StableTree(leaf_shrinkage=50.0, **common).fit(X, y).predict(X)

        assert np.std(heavy) < np.std(none)


class TestContract:
    """The sklearn contract the four incumbents all break."""

    def test_deterministic_given_random_state(self, regression_data):
        """Same seed twice, bit-identical predictions."""
        X, y = regression_data
        a = StableTree(task="regression", random_state=7).fit(X, y).predict(X)
        b = StableTree(task="regression", random_state=7).fit(X, y).predict(X)

        assert np.array_equal(a, b)

    def test_every_init_parameter_is_stored_under_its_own_name(self):
        """What broke clone() on RobustPrefixHonestTree."""
        import inspect

        estimator = StableTree()
        for name in inspect.signature(StableTree.__init__).parameters:
            if name != "self":
                assert hasattr(estimator, name), f"{name} not stored"

    def test_clone_and_cross_validate(self, regression_data):
        """Must survive sklearn's meta-estimators."""
        X, y = regression_data
        estimator = StableTree(task="regression", max_depth=3, random_state=0)

        clone(estimator)
        scores = cross_val_score(estimator, X, y, cv=3)

        assert len(scores) == 3

    def test_sets_n_features_in(self, regression_data):
        """Missing on all four incumbents."""
        X, y = regression_data
        assert StableTree(task="regression").fit(X, y).n_features_in_ == X.shape[1]

    def test_no_fitted_attributes_before_fit(self):
        """Fitted state must not exist on a fresh estimator."""
        estimator = StableTree()
        underscored = [
            a for a in vars(estimator) if a.endswith("_") and not a.startswith("_")
        ]

        assert underscored == []

    def test_classification_predicts_known_labels(self, classification_data):
        """Classification returns labels from the training set."""
        X, y = classification_data
        tree = StableTree(task="classification", max_depth=3, random_state=0).fit(X, y)

        assert set(np.unique(tree.predict(X))) <= set(np.unique(y))
        assert tree.predict_proba(X).shape == (len(X), 2)

    def test_predict_before_fit_raises(self, regression_data):
        """Unfitted use fails loudly."""
        X, _ = regression_data
        with pytest.raises(Exception, match=r"not fitted|NotFitted"):
            StableTree().predict(X)
