"""Tests for axis-aligned split candidate generation.

Regression tests for the candidate scan, which previously evaluated only the
first ``max_candidates // n_features`` midpoints of each feature's sorted unique
values — i.e. only the low tail — and so could not find a split in the middle or
upper range of a feature.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier

from stable_cart import BaseStableTree
from stable_cart.stability_utils import _evaluate_split_gain, _find_candidate_splits


def test_finds_split_in_upper_range_of_feature():
    """The single correct threshold sits at the feature's midpoint, not its low tail."""
    X = np.arange(100, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 50).astype(int)

    candidates = _find_candidate_splits(X, y, max_candidates=20)

    assert candidates, "expected at least one candidate"
    best = max(candidates, key=lambda c: c.gain)
    assert best.threshold == pytest.approx(49.5)


def test_candidates_are_not_confined_to_the_low_tail():
    """Candidate thresholds must span the feature, not cluster at the minimum."""
    X = np.arange(200, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 150).astype(int)

    thresholds = [c.threshold for c in _find_candidate_splits(X, y, max_candidates=20)]

    assert max(thresholds) > 100, (
        f"all candidates below 100; highest was {max(thresholds)}"
    )


def test_depth_one_tree_matches_sklearn_on_a_separable_feature():
    """A depth-1 fit on perfectly separable data must reach the same stump as CART."""
    X = np.arange(100, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 50).astype(int)

    stable = BaseStableTree(
        task="classification",
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=1,
        enable_honest_estimation=False,
        enable_validation_checking=False,
        algorithm_focus="speed",
        random_state=0,
    ).fit(X, y)
    cart = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)

    assert stable.tree_["threshold"] == pytest.approx(cart.tree_.threshold[0])
    assert stable.score(X, y) == pytest.approx(cart.score(X, y))


@pytest.mark.parametrize("min_samples_leaf", [1, 5, 20, 50])
def test_every_candidate_respects_the_leaf_size_constraint(min_samples_leaf):
    """A candidate the tree cannot use is worse than no candidate.

    The generator used to ignore leaf sizes entirely, so it could propose a split
    putting three rows on one side. A tree that then rejects it stops growing at
    that node instead of taking the next admissible split — measured as the sole
    reason StableTree stopped on diabetes (4 of 4 leaves) and wine (3 of 3).
    """
    X, y = make_regression(
        n_samples=300, n_features=5, n_informative=3, noise=1.0, random_state=1
    )

    candidates = _find_candidate_splits(
        X, y, max_candidates=40, min_samples_leaf=min_samples_leaf
    )

    assert candidates, "expected candidates"
    for candidate in candidates:
        left = int((X[:, candidate.feature_idx] <= candidate.threshold).sum())
        right = len(X) - left
        assert min(left, right) >= min_samples_leaf, (
            f"candidate leaves {min(left, right)} rows, below {min_samples_leaf}"
        )


def test_impossible_leaf_size_yields_no_candidates():
    """Asking for more rows per leaf than exist returns nothing, not garbage."""
    X, y = make_regression(n_samples=100, n_features=4, noise=1.0, random_state=2)

    assert _find_candidate_splits(X, y, max_candidates=20, min_samples_leaf=60) == []


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_candidate_gains_match_the_reference_implementation(task):
    """Vectorized gains must equal the per-mask reference for every candidate."""
    if task == "regression":
        X, y = make_regression(
            n_samples=120, n_features=4, n_informative=3, noise=1.0, random_state=3
        )
    else:
        X, y = make_classification(
            n_samples=120, n_features=4, n_informative=3, n_redundant=0, random_state=3
        )

    for candidate in _find_candidate_splits(X, y, max_candidates=40):
        left_mask = X[:, candidate.feature_idx] <= candidate.threshold
        expected = _evaluate_split_gain(y, left_mask)
        assert candidate.gain == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert np.array_equal(candidate.left_indices, np.where(left_mask)[0])
        assert np.array_equal(candidate.right_indices, np.where(~left_mask)[0])
