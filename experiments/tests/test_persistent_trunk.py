"""Correctness tests for the research-only persistent-partition workflows."""

import numpy as np
import pytest

from experiments.persistent_trunk import (
    PersistentTrunkTree,
    RefreshedLeafTree,
    terminal_nodes,
    tree_factory,
)


def _data(task, seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    if task == "regression":
        y = 4 * (X[:, 0] > 0) + 2 * (X[:, 1] > 0) + rng.normal(size=n)
    else:
        scores = np.column_stack((2 * X[:, 0], -2 * X[:, 0], 1.5 * X[:, 1]))
        y = np.argmax(scores + rng.normal(scale=0.3, size=scores.shape), axis=1)
    return X, y


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_persistent_predictions_reconstruct_from_routes_and_subtrees(task):
    X_reference, y_reference = _data(task, seed=1)
    X_update, y_update = _data(task, seed=2)
    X_test, _ = _data(task, seed=3, n=80)
    trunk = tree_factory(task, max_depth=1, min_samples_leaf=10, random_state=4).fit(
        X_reference, y_reference
    )
    model = PersistentTrunkTree(
        trunk, task=task, subtree_depth=3, min_samples_leaf=10, random_state=5
    ).fit(X_update, y_update)

    signature = model.trunk_signature()
    routes = model.trunk_.apply(X_test)
    prediction = model.predict(X_test)
    for node, subtree in model.subtrees_.items():
        mask = routes == node
        assert np.array_equal(prediction[mask], subtree.predict(X_test[mask]))
    assert model.trunk_signature() == signature


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_refreshed_values_are_independent_leaf_reconstructions(task):
    X_reference, y_reference = _data(task, seed=4)
    X_update, y_update = _data(task, seed=5)
    tree = tree_factory(task, max_depth=4, min_samples_leaf=10, random_state=6).fit(
        X_reference, y_reference
    )
    model = RefreshedLeafTree(tree, task=task).fit(X_update, y_update)
    signature = model.partition_signature()
    routes = tree.apply(X_update)

    for node in terminal_nodes(tree):
        values = y_update[routes == node]
        if not len(values):
            continue
        if task == "regression":
            assert model.values_[int(node)] == pytest.approx(np.mean(values))
        else:
            expected = np.asarray([np.mean(values == label) for label in tree.classes_])
            assert np.allclose(model.values_[int(node)], expected)
    assert model.partition_signature() == signature


def test_multiclass_probabilities_align_and_unseen_branch_uses_fallback():
    X_reference, y_reference = _data("classification", seed=7)
    trunk = tree_factory(
        "classification", max_depth=1, min_samples_leaf=10, random_state=8
    ).fit(X_reference, y_reference)
    reference_routes = trunk.apply(X_reference)
    kept_node = np.unique(reference_routes)[0]
    update_mask = reference_routes == kept_node
    X_update = X_reference[update_mask]
    y_update = y_reference[update_mask]
    model = PersistentTrunkTree(
        trunk,
        task="classification",
        subtree_depth=2,
        min_samples_leaf=5,
        random_state=9,
    ).fit(X_update, y_update)

    probabilities = model.predict_proba(X_reference)
    unseen = reference_routes != kept_node
    expected = np.asarray(
        [np.mean(y_update == label) for label in trunk.classes_], dtype=float
    )
    assert probabilities.shape == (len(X_reference), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[unseen], expected)
    assert np.array_equal(
        model.predict(X_reference), trunk.classes_[np.argmax(probabilities, axis=1)]
    )


def test_row_order_and_label_renaming_preserve_partitions():
    X_reference, y_reference = _data("classification", seed=10)
    X_update, y_update = _data("classification", seed=11)
    trunk = tree_factory(
        "classification", max_depth=1, min_samples_leaf=10, random_state=12
    ).fit(X_reference, y_reference)
    forward = PersistentTrunkTree(
        trunk,
        task="classification",
        subtree_depth=3,
        min_samples_leaf=10,
        random_state=13,
    ).fit(X_update, y_update)
    reverse = PersistentTrunkTree(
        trunk,
        task="classification",
        subtree_depth=3,
        min_samples_leaf=10,
        random_state=13,
    ).fit(X_update[::-1], y_update[::-1])
    assert np.array_equal(forward.predict(X_reference), reverse.predict(X_reference))

    renamed_reference = np.asarray(["low", "middle", "high"])[y_reference]
    renamed_update = np.asarray(["low", "middle", "high"])[y_update]
    renamed_trunk = tree_factory(
        "classification", max_depth=1, min_samples_leaf=10, random_state=12
    ).fit(X_reference, renamed_reference)
    renamed = PersistentTrunkTree(
        renamed_trunk,
        task="classification",
        subtree_depth=3,
        min_samples_leaf=10,
        random_state=13,
    ).fit(X_update, renamed_update)
    inverse = {"low": 0, "middle": 1, "high": 2}
    remapped = np.asarray([inverse[value] for value in renamed.predict(X_reference)])
    assert np.array_equal(forward.predict(X_reference), remapped)


def test_persistent_trunk_requires_a_fitted_depth_one_tree():
    X, y = _data("regression")
    deep = tree_factory(
        "regression", max_depth=2, min_samples_leaf=10, random_state=0
    ).fit(X, y)

    with pytest.raises(ValueError, match="depth-one"):
        PersistentTrunkTree(deep, task="regression")
