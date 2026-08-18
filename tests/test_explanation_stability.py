"""Tests for the explanation-stability metrics.

Where possible the expected value is worked out by hand from a tree built to have
a known structure, so a failure means the metric is wrong rather than that a
number moved.
"""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stable_cart import RepresentativeEstimator
from stable_cart.explanation_stability import (
    explanation_instability,
    path_agreement,
    root_agreement,
    split_feature_paths,
    split_features,
)


def stump_on(feature, n=200, seed=0):
    """A depth-1 tree guaranteed to split on the named feature."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, feature] > 0).astype(int)
    return DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)


def test_split_features_reads_a_known_stump():
    """A stump on feature 2 tests exactly {2: 1}."""
    assert split_features(stump_on(2), max_depth=3) == {2: 1}


def test_identical_structures_have_zero_instability():
    """Two fits that read the same must score 0 — the metric's fixed point."""
    trees = [stump_on(1, seed=0), stump_on(1, seed=1)]

    assert explanation_instability(trees)["jaccard_mean"] == 0.0
    assert root_agreement(trees) == 1.0


def test_disjoint_structures_have_maximal_instability():
    """No shared feature means Jaccard distance exactly 1."""
    trees = [stump_on(0), stump_on(3)]

    assert explanation_instability(trees)["jaccard_mean"] == 1.0
    assert root_agreement(trees) == 0.5


def test_jaccard_matches_a_hand_computed_value():
    """Three stumps on features 0, 0, 1.

    Pairs: (0,0) distance 0; (0,1) distance 1; (0,1) distance 1.
    Mean = 2/3. Root agreement = 2/3.
    """
    trees = [stump_on(0, seed=0), stump_on(0, seed=1), stump_on(1, seed=2)]
    result = explanation_instability(trees)

    assert result["jaccard_mean"] == pytest.approx(2 / 3)
    assert result["jaccard_max"] == 1.0
    assert root_agreement(trees) == pytest.approx(2 / 3)
    assert result["distinct_structures"] == pytest.approx(2 / 3)


def test_features_are_a_multiset_not_a_set():
    """A feature tested twice weighs more than one tested once.

    The DGP is ``X0>0 ? (X0>1 ? 2 : 1) : 0``, so at depth 2 the left branch is
    already pure and never splits: the tree has a root plus one child split, both
    on feature 0. Tree A tests {0:1}, tree B tests {0:2}. As *sets* both are {0}
    and the distance would be 0; as multisets the intersection is 1 and the union
    2, giving 1/2.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = np.where(X[:, 0] > 0, np.where(X[:, 0] > 1, 2.0, 1.0), 0.0)

    shallow = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)
    deeper = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)

    assert split_features(shallow) == {0: 1}
    assert split_features(deeper) == {0: 2}
    assert explanation_instability([shallow, deeper])["jaccard_mean"] == pytest.approx(
        1 - 1 / 2
    )


def test_max_depth_limits_what_counts():
    """Only splits at or above max_depth enter the comparison."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 4))
    y = X[:, 0] + 0.5 * X[:, 1] + 0.25 * X[:, 2]
    tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)

    shallow = split_features(tree, max_depth=0)
    deep = split_features(tree, max_depth=3)

    assert sum(shallow.values()) == 1
    assert sum(deep.values()) > sum(shallow.values())


def test_path_agreement_is_one_for_identical_trees():
    """Identical structures route every row identically."""
    trees = [stump_on(1, seed=0), stump_on(1, seed=1)]
    X = np.random.default_rng(3).normal(size=(50, 4))

    assert path_agreement(trees, X) == 1.0


def test_feature_paths_follow_sklearn_missing_value_routing():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    X[rng.choice(len(X), 5, replace=False), 0] = np.nan
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    rows = np.array([[np.nan, 2.0], [-2.0, np.nan], [0.0, np.nan]])

    expected = []
    for row in rows:
        nodes = tree.decision_path(row.reshape(1, -1)).indices
        expected.append(
            tuple(
                int(tree.tree_.feature[node])
                for node in nodes
                if node < tree.tree_.node_count and tree.tree_.feature[node] >= 0
            )
        )

    assert split_feature_paths(tree, rows) == expected


def test_metrics_read_tree_selected_by_representative_estimator():
    """A supported wrapper exposes the selected fitted tree structure."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(240, 4))
    y = X[:, 2] + rng.normal(scale=0.1, size=len(X))
    model = RepresentativeEstimator(
        estimator=DecisionTreeRegressor(max_depth=3),
        task="regression",
        n_candidates=4,
        random_state=6,
    ).fit(X, y)

    assert split_features(model)
    assert path_agreement([model, model], X[:12]) == 1.0


def test_rejects_too_few_trees():
    """Pairwise metrics need a pair."""
    with pytest.raises(ValueError, match="at least 2"):
        explanation_instability([stump_on(0)])
    with pytest.raises(ValueError, match="at least 2"):
        path_agreement([stump_on(0)], np.zeros((3, 4)))
