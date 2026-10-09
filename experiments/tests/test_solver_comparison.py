"""Enumerated objectives and invariance checks for the solver comparison."""

import numpy as np
import pytest

from experiments.solver_comparison import (
    binarize,
    binarizer_edges,
    fit_arms,
    summarize_predictions,
)


def test_cutpoints_are_learned_from_fitting_data_only():
    X = np.arange(12, dtype=float)[:, None]
    edges = binarizer_edges(X, 3)
    original = [v.copy() for v in edges]
    binarize(np.array([[-1e9], [1e9]]), edges)
    assert all(np.array_equal(a, b) for a, b in zip(edges, original, strict=True))
    assert np.allclose(edges[0], [2.75, 5.5, 8.25])


def test_identical_arms_have_zero_paired_accuracy_difference():
    predictions = [[0, 1, 1], [1, 1, 0], [0, 0, 0]]
    summary = summarize_predictions(
        {"gosdt": predictions, "cart_binary": predictions}, [0, 1, 0]
    )
    assert summary["gosdt_minus_cart_binary_accuracy_pp"] == {
        "mean": 0.0,
        "ci": [0.0, 0.0],
    }


def test_solver_objective_matches_exhaustive_depth_two_search():
    pytest.importorskip(
        "gosdt", reason="Run with the separately locked solver environment"
    )
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (20, 1))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    _, _, metadata, certificate = fit_arms(X, y, X, 1, 2, 0.05, 5)

    def optimum(indices, depth):
        labels = y[indices]
        loss = min(labels.sum(), len(labels) - labels.sum()) / len(y) + 0.05
        if depth:
            for feature in range(X.shape[1]):
                left = indices[X[indices, feature] == 0]
                right = indices[X[indices, feature] == 1]
                if len(left) and len(right):
                    loss = min(
                        loss, optimum(left, depth - 1) + optimum(right, depth - 1)
                    )
        return loss

    actual = (
        1 - metadata["gosdt"]["train_accuracy"] + 0.05 * metadata["gosdt"]["leaves"]
    )
    assert actual == pytest.approx(optimum(np.arange(len(y)), 2))
    assert certificate["certified"]


def test_constant_solver_solution_remains_constant_in_cart_arms():
    pytest.importorskip(
        "gosdt", reason="Run with the separately locked solver environment"
    )
    X = np.arange(80, dtype=float)[:, None]
    y = (X[:, 0] >= 40).astype(int)
    _, _, metadata, certificate = fit_arms(X, y, X, 2, 2, 0.6, 5)
    assert certificate["leaf_budget"] == 1
    assert all(model["leaves"] == 1 for model in metadata.values())
