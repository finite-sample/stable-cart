"""Executable specification and falsification checks for BootstrapSplitTree."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError

from experiments.bootstrap_split_tree import BootstrapSplitTree
from experiments.split_candidates import _all_split_gains, _find_candidate_splits


@pytest.fixture
def regression_data():
    return make_regression(
        n_samples=400, n_features=6, n_informative=4, noise=3, random_state=0
    )


def test_one_vote_is_exact_greedy_limiting_case(regression_data):
    X, y = regression_data
    model = BootstrapSplitTree(
        task="regression",
        max_depth=1,
        min_samples_leaf=10,
        min_samples_split=2,
        n_consensus=1,
        consensus_threshold=0,
        random_state=0,
    ).fit(X, y)
    reference = _find_candidate_splits(
        X,
        y,
        task="regression",
        max_candidates=model.max_candidates,
        min_samples_leaf=model.min_samples_leaf,
    )[0]

    assert model.tree_["feature"] == reference.feature_idx
    assert model.tree_["threshold"] == pytest.approx(reference.threshold)
    assert model.tree_["support"] == 1


def test_root_is_exact_median_vote_projected_to_admissible_grid(regression_data):
    X, y = regression_data
    model = BootstrapSplitTree(
        task="regression",
        max_depth=1,
        min_samples_leaf=10,
        min_samples_split=2,
        n_consensus=13,
        consensus_threshold=0,
        random_state=9,
    ).fit(X, y)
    rng = np.random.default_rng(9)
    votes = {}
    for _ in range(13):
        indices = rng.integers(0, len(y), len(y))
        best = _find_candidate_splits(
            X[indices],
            y[indices],
            task="regression",
            max_candidates=model.max_candidates,
            min_samples_leaf=model.min_samples_leaf,
        )[0]
        votes.setdefault(best.feature_idx, []).append(best.threshold)
    feature = max(votes, key=lambda index: (len(votes[index]), -index))
    median = np.median(votes[feature])
    admissible, _ = _all_split_gains(
        X[:, feature],
        y,
        task="regression",
        min_samples_leaf=model.min_samples_leaf,
    )
    expected = admissible[
        np.flatnonzero(
            np.abs(admissible - median) == np.min(np.abs(admissible - median))
        )[0]
    ]

    assert model.tree_["feature"] == feature
    assert model.tree_["support"] == pytest.approx(len(votes[feature]) / 13)
    assert model.tree_["threshold"] == pytest.approx(expected)


def test_every_fitted_split_obeys_minimum_leaf_size(regression_data):
    X, y = regression_data
    model = BootstrapSplitTree(
        task="regression",
        max_depth=4,
        min_samples_leaf=17,
        min_samples_split=2,
        n_consensus=8,
        consensus_threshold=0,
        random_state=0,
    ).fit(X, y)

    def walk(node, rows):
        if node["type"] == "leaf":
            return
        left = rows[:, node["feature"]] <= node["threshold"]
        assert min(left.sum(), (~left).sum()) >= 17
        walk(node["left"], rows[left])
        walk(node["right"], rows[~left])

    walk(model.tree_, X)


def test_vote_threshold_can_veto_a_split(regression_data):
    X, y = regression_data
    permissive = BootstrapSplitTree(
        max_depth=3,
        min_samples_leaf=10,
        min_samples_split=2,
        n_consensus=16,
        consensus_threshold=0,
        random_state=5,
    ).fit(X, y)
    strict = BootstrapSplitTree(
        max_depth=3,
        min_samples_leaf=10,
        min_samples_split=2,
        n_consensus=16,
        consensus_threshold=1,
        random_state=5,
    ).fit(X, y)

    assert strict.get_n_leaves() <= permissive.get_n_leaves()
    assert all(0 <= support <= 1 for support in permissive.split_supports())
    assert sum(permissive.stop_reasons_.values()) == permissive.get_n_leaves()


def test_leaf_shrinkage_matches_declared_pseudocount_formula():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.array([0] * 6 + [10] * 6, dtype=float)
    model = BootstrapSplitTree(
        max_depth=1,
        min_samples_leaf=1,
        min_samples_split=2,
        n_consensus=1,
        consensus_threshold=0,
        leaf_shrinkage=6,
    ).fit(X, y)

    left = model.tree_["left"]
    right = model.tree_["right"]
    assert left["value"] == pytest.approx((6 * 0 + 6 * 5) / 12)
    assert right["value"] == pytest.approx((6 * 10 + 6 * 5) / 12)


def test_binary_label_renaming_does_not_change_partitions():
    X, y = make_classification(
        n_samples=300, n_features=5, n_informative=3, random_state=0
    )
    numeric = BootstrapSplitTree(
        task="classification",
        max_depth=3,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=7,
    ).fit(X, y)
    labels = np.where(y == 0, "control", "treated")
    renamed = BootstrapSplitTree(
        task="classification",
        max_depth=3,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=7,
    ).fit(X, labels)

    expected = np.where(numeric.predict(X) == 0, "control", "treated")
    assert np.array_equal(renamed.predict(X), expected)
    assert renamed.split_supports() == numeric.split_supports()


def test_binary_probabilities_are_consistent_with_predictions():
    X, y = make_classification(
        n_samples=250, n_features=5, n_informative=3, random_state=2
    )
    model = BootstrapSplitTree(
        task="classification",
        max_depth=3,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=2,
    ).fit(X, y)
    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert np.array_equal(
        model.classes_[np.argmax(probabilities, axis=1)], model.predict(X)
    )


def test_multiclass_is_rejected_explicitly():
    X, y = make_classification(
        n_samples=180,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=0,
    )

    with pytest.raises(ValueError, match="Only binary"):
        BootstrapSplitTree(task="classification").fit(X, y)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("task", "ranking"),
        ("max_depth", -1),
        ("min_samples_leaf", 0),
        ("min_samples_split", 1),
        ("n_consensus", 0),
        ("consensus_threshold", 1.1),
        ("leaf_shrinkage", -1),
        ("max_candidates", 0),
    ],
)
def test_invalid_parameters_are_rejected(parameter, value, regression_data):
    X, y = regression_data

    with pytest.raises(ValueError, match=parameter):
        BootstrapSplitTree(**{parameter: value}).fit(X, y)


def test_classification_api_is_not_exposed_for_regression(regression_data):
    X, y = regression_data
    model = BootstrapSplitTree(task="regression").fit(X, y)

    assert not hasattr(model, "predict_proba")


def test_inspection_methods_require_fit():
    with pytest.raises(NotFittedError):
        BootstrapSplitTree().get_n_leaves()
    with pytest.raises(NotFittedError):
        BootstrapSplitTree().split_supports()
