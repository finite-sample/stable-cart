"""Exact reference tests for exhaustive split scoring."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from experiments.split_candidates import _evaluate_split_gain, _find_candidate_splits


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_vectorized_gains_equal_mask_reference(task):
    if task == "regression":
        X, y = make_regression(n_samples=120, n_features=4, noise=2, random_state=1)
    else:
        X, y = make_classification(
            n_samples=120,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            random_state=1,
        )

    candidates = _find_candidate_splits(
        X, y, task=task, max_candidates=80, min_samples_leaf=5
    )

    assert candidates
    for candidate in candidates:
        left = X[:, candidate.feature_idx] <= candidate.threshold
        expected = _evaluate_split_gain(y, left, task=task)
        assert candidate.gain == pytest.approx(expected, rel=1e-10, abs=1e-12)
        assert np.array_equal(candidate.left_indices, np.flatnonzero(left))
        assert np.array_equal(candidate.right_indices, np.flatnonzero(~left))
        assert min(left.sum(), (~left).sum()) >= 5


def test_integer_regression_is_not_silently_scored_as_classification():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1, 8, 8, 8, 9, 9, 9], dtype=int)

    regression = _find_candidate_splits(X, y, task="regression", max_candidates=20)
    classification = _find_candidate_splits(
        X, y, task="classification", max_candidates=20
    )

    assert regression[0].gain != pytest.approx(classification[0].gain)
    left = X[:, 0] <= regression[0].threshold
    assert regression[0].gain == pytest.approx(
        _evaluate_split_gain(y, left, task="regression")
    )


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_unique_depth_one_optimum_matches_sklearn(task):
    X = np.arange(100, dtype=float).reshape(-1, 1)
    if task == "regression":
        y = np.where(X[:, 0] < 50, -3.0, 7.0)
        reference = DecisionTreeRegressor(max_depth=1, random_state=0)
    else:
        y = (X[:, 0] >= 50).astype(int)
        reference = DecisionTreeClassifier(max_depth=1, random_state=0)
    candidate = _find_candidate_splits(
        X, y, task=task, max_candidates=20, min_samples_leaf=1
    )[0]
    reference.fit(X, y)

    assert candidate.feature_idx == reference.tree_.feature[0]
    assert candidate.threshold == pytest.approx(reference.tree_.threshold[0])


def test_candidates_span_the_feature_and_obey_the_global_limit():
    X = np.arange(200, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 150).astype(int)
    candidates = _find_candidate_splits(X, y, task="classification", max_candidates=20)

    assert len(candidates) == 20
    assert max(candidate.threshold for candidate in candidates) > 100


def test_impossible_leaf_size_returns_no_candidate():
    X, y = make_regression(n_samples=100, n_features=3, random_state=0)

    assert (
        _find_candidate_splits(
            X,
            y,
            task="regression",
            max_candidates=20,
            min_samples_leaf=51,
        )
        == []
    )


def test_large_target_translation_preserves_split_and_gain():
    X = np.arange(12, dtype=float)[:, None]
    y = np.repeat([0.0, 1.0], 6)
    original = _find_candidate_splits(X, y, task="regression")
    shifted = _find_candidate_splits(X, y + 1e8, task="regression")
    assert shifted[0].threshold == original[0].threshold == 5.5
    assert shifted[0].gain == pytest.approx(original[0].gain)
