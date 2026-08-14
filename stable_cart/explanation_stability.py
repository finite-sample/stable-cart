"""Measure whether a tree's *explanation* holds still when the data is perturbed.

A single decision tree is chosen because it can be read. Its deliverable is the
structure — which feature is tested at the root, what is asked next — and that is
what a user shows a colleague or a regulator. Prediction stability does not
measure it. The two can move independently: when two features carry the same
information, a tree can flip between them on every resample while its predictions
barely change, and conversely a tree can keep its shape while its leaf estimates
wander.

These functions measure the structural half. Read them beside a prediction
measure such as :func:`stable_cart.bootstrap_instability`, never alone — a stump
that always tests the same feature scores perfectly here and may still be useless.

Supports both scikit-learn trees and this package's dict-structured trees.
"""

from collections import Counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "split_features",
    "split_feature_paths",
    "explanation_instability",
    "root_agreement",
    "path_agreement",
]


def _sklearn_splits(tree: Any, max_depth: int) -> list[tuple[int, int]]:
    """Collect (depth, feature) for a fitted sklearn tree, breadth-first."""
    inner = tree.tree_
    out: list[tuple[int, int]] = []
    queue = [(0, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > max_depth or inner.feature[node] < 0:
            continue
        out.append((depth, int(inner.feature[node])))
        queue.append((int(inner.children_left[node]), depth + 1))
        queue.append((int(inner.children_right[node]), depth + 1))
    return out


def _dict_splits(node: Any, max_depth: int, depth: int = 0) -> list[tuple[int, int]]:
    """Collect (depth, feature) for this package's dict-structured trees."""
    if not isinstance(node, dict) or depth > max_depth:
        return []
    if node.get("type") == "leaf" or node.get("feature_idx") is None:
        return []
    out = [(depth, int(node["feature_idx"]))]
    out.extend(_dict_splits(node.get("left"), max_depth, depth + 1))
    out.extend(_dict_splits(node.get("right"), max_depth, depth + 1))
    return out


def _splits(model: Any, max_depth: int) -> list[tuple[int, int]]:
    """Collect (depth, feature) pairs from either tree representation.

    Parameters
    ----------
    model
        A fitted tree, or an object exposing one.
    max_depth
        Deepest level to include; the root is level 0.

    Returns
    -------
    list[tuple[int, int]]
        (depth, feature index) for every internal node down to max_depth.

    Raises
    ------
    TypeError
        If no readable tree structure can be found on the object.
    """
    inner = getattr(model, "tree_", None)
    if inner is None:
        inner = getattr(getattr(model, "selected_tree_", None), "tree_", None)
        if inner is None:
            raise TypeError(f"no readable tree structure on {type(model).__name__}")
        return _sklearn_splits(model.selected_tree_, max_depth)
    if isinstance(inner, dict):
        return _dict_splits(inner, max_depth)
    if hasattr(inner, "feature"):
        return _sklearn_splits(model, max_depth)
    raise TypeError(f"unrecognised tree structure on {type(model).__name__}")


def split_features(model: Any, max_depth: int = 3) -> Counter:
    """
    Return the multiset of features tested down to a given depth.

    A multiset rather than a set: a feature tested at three different nodes is a
    more central part of the explanation than one tested once, and collapsing that
    to a set would hide it.

    Parameters
    ----------
    model
        A fitted tree.
    max_depth
        Deepest level to include; the root is level 0.

    Returns
    -------
    Counter
        Feature index -> number of nodes testing it.
    """
    return Counter(feature for _, feature in _splits(model, max_depth))


def split_feature_paths(model: Any, X: NDArray[np.floating]) -> list[tuple[int, ...]]:
    """
    Return the feature sequence each row is tested against on its way to a leaf.

    This is the explanation an individual case receives — "you were declined
    because of X, then Y" — so it is the right unit when the audience is the
    subject of a decision rather than the modeller.

    Parameters
    ----------
    model
        A fitted tree.
    X
        Rows to route through the tree.

    Returns
    -------
    list[tuple[int, ...]]
        One tuple of feature indices per row.
    """
    inner = getattr(model, "tree_", None)
    paths = []

    if isinstance(inner, dict):
        for row in np.asarray(X, dtype=float):
            node, path = inner, []
            while isinstance(node, dict) and node.get("type") != "leaf":
                feature = node.get("feature_idx")
                if feature is None:
                    break
                path.append(int(feature))
                node = (
                    node["left"]
                    if row[int(feature)] <= node["threshold"]
                    else node["right"]
                )
            paths.append(tuple(path))
        return paths

    sk = inner if inner is not None and hasattr(inner, "feature") else None
    if sk is None:
        raise TypeError(f"no readable tree structure on {type(model).__name__}")
    for row in np.asarray(X, dtype=float):
        node, path = 0, []
        while sk.feature[node] >= 0:
            feature = int(sk.feature[node])
            path.append(feature)
            node = (
                int(sk.children_left[node])
                if row[feature] <= sk.threshold[node]
                else int(sk.children_right[node])
            )
        paths.append(tuple(path))
    return paths


def _jaccard_distance(a: Counter, b: Counter) -> float:
    """Multiset Jaccard distance: 1 - |intersection| / |union|."""
    if not a and not b:
        return 0.0
    intersection = sum((a & b).values())
    union = sum((a | b).values())
    return 1.0 - intersection / union if union else 0.0


def explanation_instability(models: list, max_depth: int = 3) -> dict[str, float]:
    """
    How much the structure changes across independently fitted trees.

    The headline number is the mean pairwise Jaccard distance between the
    multisets of features tested down to ``max_depth``: 0 when every fit reads the
    same, 1 when no two fits share a single tested feature.

    Parameters
    ----------
    models
        Fitted trees, each from a different training sample. At least two.
    max_depth
        Deepest level to include; the root is level 0.

    Returns
    -------
    dict[str, float]
        ``jaccard_mean`` and ``jaccard_max`` over all pairs, and
        ``distinct_structures`` — the number of distinct feature multisets seen,
        divided by the number of fits.

    Raises
    ------
    ValueError
        If fewer than two models are supplied.
    """
    if len(models) < 2:
        raise ValueError("explanation instability needs at least 2 fitted trees")

    features = [split_features(m, max_depth) for m in models]
    distances = [
        _jaccard_distance(features[i], features[j])
        for i in range(len(features))
        for j in range(i + 1, len(features))
    ]
    signatures = {tuple(sorted(f.items())) for f in features}

    return {
        "jaccard_mean": float(np.mean(distances)),
        "jaccard_max": float(np.max(distances)),
        "distinct_structures": len(signatures) / len(models),
    }


def root_agreement(models: list) -> float:
    """
    Fraction of fits that test the most common root feature.

    1.0 means every fit opens the same way. This is the single most visible part
    of an explanation, and the part a reader remembers.

    Parameters
    ----------
    models
        Fitted trees, each from a different training sample.

    Returns
    -------
    float
        Modal root-feature frequency, in [0, 1].

    Raises
    ------
    ValueError
        If no models are supplied.
    """
    if not models:
        raise ValueError("root agreement needs at least 1 fitted tree")

    roots = []
    for model in models:
        splits = _splits(model, max_depth=0)
        roots.append(splits[0][1] if splits else -1)
    return max(Counter(roots).values()) / len(roots)


def path_agreement(models: list, X: NDArray[np.floating]) -> float:
    """
    Fraction of rows routed through the same feature sequence by most fits.

    Averaged over rows: for each row, the share of fits agreeing with that row's
    modal path. This is explanation stability from the perspective of the
    individual being explained to.

    Parameters
    ----------
    models
        Fitted trees, each from a different training sample.
    X
        Rows to route.

    Returns
    -------
    float
        Mean modal-path agreement, in [0, 1].

    Raises
    ------
    ValueError
        If fewer than two models are supplied.
    """
    if len(models) < 2:
        raise ValueError("path agreement needs at least 2 fitted trees")

    per_model = [split_feature_paths(m, X) for m in models]
    agreements = []
    for row in range(len(np.asarray(X))):
        counts = Counter(paths[row] for paths in per_model)
        agreements.append(max(counts.values()) / len(models))
    return float(np.mean(agreements))
