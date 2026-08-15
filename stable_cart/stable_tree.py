"""A single decision tree whose splits are averaged rather than taken from one sample.

The problem this solves: you have to ship *one* readable tree, and you want its
predictions to move as little as possible when the training data shifts.

Averaging is the only mechanism that decisively reduces a tree's prediction
variance — a random forest cuts it by 62-92% on this package's benchmark — but
averaging *predictions* leaves you with a forest, not a tree. This class applies
the averaging one level earlier, to the **split decision**, so the output is
still a single tree you can read.

Two mechanisms, one per component of the variance:

**Averaged split selection.** At each node, bootstrap the node's rows
``n_consensus`` times, take the best split in each replicate, elect the feature
by vote, and set the cut point to the *median* of the cut points from replicates
that chose it. Cut-point variance is the dominant structural instability —
Geurts and Wehenkel (ECML 2000) found it high even at large sample sizes, and
this package's own measurements put threshold agreement at 2-22% while the root
feature agrees 100% of the time. Averaging the cut point attacks the part that
actually moves.

**A reproducibility floor.** If the winning feature is chosen by fewer than
``consensus_threshold`` of the replicates, no split is made and the node becomes
a leaf. A split the data cannot reproduce is not one worth showing to a reviewer.

**Leaf shrinkage.** Leaf estimation is 40-90% of prediction variance when noise
is high, so leaf values are optionally shrunk toward the parent.

Every parameter here changes predictions; none is decorative.
"""

from collections import Counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .stability_utils import (
    _find_candidate_splits,
    check_predict_input,
    stabilize_leaf_estimate,
)

__all__ = ["StableTree"]


class StableTree(BaseEstimator):
    r"""
    A decision tree whose split decisions are averaged over bootstrap replicates.

    Parameters
    ----------
    task
        Prediction task.
    max_depth
        Maximum tree depth.
    min_samples_leaf
        Minimum rows in a leaf.
    min_samples_split
        Minimum rows required to consider splitting a node.
    n_consensus
        Bootstrap replicates used to elect each split. ``1`` disables the
        averaging and uses the node's own rows, which reduces the estimator to
        ordinary greedy selection — the negative control.
    consensus_threshold
        Minimum share of replicates that must agree on a feature before the split
        is made. Higher values yield smaller, more reproducible trees; a value
        above 1 is unreachable and produces a stump.
    leaf_shrinkage
        Strength of shrinkage of leaf values toward the parent. ``0`` leaves the
        sample mean untouched.
    max_candidates
        Candidate splits scored per replicate.
    random_state
        Seed for the bootstrap replicates.

    Attributes
    ----------
    tree\_
        Nested dict describing the fitted tree.
    n_features_in\_
        Number of features seen during fit.
    stop_reasons\_
        Why each node stopped growing, as a Counter. Answers "why is my tree so
        small?" — the common answers being ``max_depth`` and
        ``no_reproducible_split``.
    classes\_
        Class labels, for classification only.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=500, n_features=6, random_state=0)
    >>> tree = StableTree(task="regression", random_state=0).fit(X, y)
    >>> tree.split_supports()  # how reproducible each split is
    """

    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        min_samples_split: int = 40,
        n_consensus: int = 16,
        consensus_threshold: float = 0.3,
        leaf_shrinkage: float = 0.0,
        max_candidates: int = 40,
        random_state: int | None = None,
    ):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_consensus = n_consensus
        self.consensus_threshold = consensus_threshold
        self.leaf_shrinkage = leaf_shrinkage
        self.max_candidates = max_candidates
        self.random_state = random_state

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> "StableTree":
        """
        Fit the tree.

        Parameters
        ----------
        X
            Training features of shape (n_samples, n_features).
        y
            Training targets of shape (n_samples,).

        Returns
        -------
        StableTree
            The fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        rng = np.random.default_rng(self.random_state)

        self.n_features_in_ = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
            y_work = np.searchsorted(self.classes_, y).astype(float)
        else:
            y_work = y.astype(float)

        self.stop_reasons_: Counter = Counter()
        self.tree_ = self._build(X, y_work, depth=0, parent=y_work, rng=rng)
        return self

    def _elect_split(self, X: NDArray[Any], y: NDArray[Any], rng: np.random.Generator):
        """
        Elect a split by vote across bootstrap replicates.

        Parameters
        ----------
        X
            Feature rows at this node.
        y
            Targets at this node.
        rng
            Random generator.

        Returns
        -------
        tuple | None
            ``(feature, threshold, support)``, or None when no feature clears the
            reproducibility floor.
        """
        n = len(y)
        votes: dict[int, list[float]] = {}

        for _ in range(self.n_consensus):
            if self.n_consensus == 1:
                Xb, yb = X, y
            else:
                idx = rng.integers(0, n, n)
                Xb, yb = X[idx], y[idx]

            candidates = _find_candidate_splits(
                Xb, yb, self.max_candidates, self.min_samples_leaf
            )
            if not candidates:
                continue
            best = max(candidates, key=lambda c: c.gain)
            votes.setdefault(int(best.feature_idx), []).append(float(best.threshold))

        if not votes:
            return None

        feature = max(votes, key=lambda f: len(votes[f]))
        support = len(votes[feature]) / self.n_consensus
        if support < self.consensus_threshold:
            return None

        # The median cut point across replicates that chose this feature: the
        # averaging that this class exists to perform.
        threshold = float(np.median(votes[feature]))

        # Each replicate's cut point is admissible in *its own* resample, but their
        # median need not be admissible here — averaging does not preserve the
        # leaf-size constraint. Clip it into the admissible band rather than
        # abandoning a node over an arithmetic artifact.
        column = np.sort(X[:, feature])
        k = self.min_samples_leaf
        if 0 < k <= len(column) - k:
            low, high = column[k - 1], column[len(column) - k - 1]
            if low < high:
                threshold = float(np.clip(threshold, low, high))

        return feature, threshold, support

    def _leaf(self, y, parent):
        """Build a leaf, shrinking its value toward the parent when asked."""
        value = float(np.mean(y))
        if self.leaf_shrinkage > 0 and len(parent) > 0:
            value = float(
                stabilize_leaf_estimate(
                    y,
                    parent,
                    strategy="shrink_to_parent",
                    smoothing=self.leaf_shrinkage,
                    task="regression",
                )
            )
        return {"type": "leaf", "value": value, "n": len(y)}

    def _build(self, X, y, depth, parent, rng):
        """Recursively build the tree."""
        if depth >= self.max_depth:
            self.stop_reasons_["max_depth"] += 1
            return self._leaf(y, parent)
        if len(y) < self.min_samples_split:
            self.stop_reasons_["min_samples_split"] += 1
            return self._leaf(y, parent)
        if len(np.unique(y)) < 2:
            self.stop_reasons_["pure_node"] += 1
            return self._leaf(y, parent)

        elected = self._elect_split(X, y, rng)
        if elected is None:
            self.stop_reasons_["no_reproducible_split"] += 1
            return self._leaf(y, parent)

        feature, threshold, support = elected
        left_mask = X[:, feature] <= threshold
        if (
            left_mask.sum() < self.min_samples_leaf
            or (~left_mask).sum() < self.min_samples_leaf
        ):
            # Unreachable now that candidates are generated under the leaf-size
            # constraint. It fired on 4 of 4 nodes on diabetes before that fix,
            # abandoning nodes where an admissible split existed, so the guard
            # stays and a test asserts it never triggers.
            self.stop_reasons_["leaf_size_rejected"] += 1
            return self._leaf(y, parent)

        return {
            "type": "split",
            "feature": feature,
            "threshold": threshold,
            "support": support,
            "n": len(y),
            "left": self._build(X[left_mask], y[left_mask], depth + 1, y, rng),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1, y, rng),
        }

    def _route(self, row, node):
        """Follow one row to its leaf value."""
        while node["type"] == "split":
            node = (
                node["left"]
                if row[node["feature"]] <= node["threshold"]
                else node["right"]
            )
        return node["value"]

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Predict for each row.

        Parameters
        ----------
        X
            Features of shape (n_samples, n_features).

        Returns
        -------
        NDArray[Any]
            Predicted values, or class labels for classification.
        """
        X = check_predict_input(self, X, "tree_")
        raw = np.array([self._route(row, self.tree_) for row in X])

        if self.task == "classification":
            return self.classes_[
                np.clip(np.rint(raw).astype(int), 0, len(self.classes_) - 1)
            ]
        return raw

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.floating]:
        """
        Predict class probabilities.

        Parameters
        ----------
        X
            Features of shape (n_samples, n_features).

        Returns
        -------
        NDArray[np.floating]
            Probabilities of shape (n_samples, n_classes).

        Raises
        ------
        ValueError
            If the estimator was not fitted for classification.
        """
        check_is_fitted(self, "tree_")
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification")

        X = check_array(X, accept_sparse=False)
        raw = np.clip(np.array([self._route(row, self.tree_) for row in X]), 0.0, 1.0)
        return np.column_stack([1.0 - raw, raw])

    def score(self, X: NDArray[Any], y: NDArray[Any]) -> float:
        """
        Return R² for regression or accuracy for classification.

        Parameters
        ----------
        X
            Evaluation features.
        y
            True targets.

        Returns
        -------
        float
            The score.
        """
        pred = self.predict(X)
        if self.task == "classification":
            return float(np.mean(pred == y))
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def get_n_leaves(self) -> int:
        """
        Count the leaves of the fitted tree.

        Returns
        -------
        int
            Number of leaves.
        """
        check_is_fitted(self, "tree_")

        def count(node):
            return (
                1
                if node["type"] == "leaf"
                else count(node["left"]) + count(node["right"])
            )

        return count(self.tree_)

    def split_supports(self) -> list[float]:
        """
        Reproducibility of each split, as the share of replicates electing it.

        This is the number a reviewer should see: a split at 0.95 was chosen by
        almost every resample, one at 0.35 is close to a coin flip.

        Returns
        -------
        list[float]
            One support value per internal node, root first.
        """
        check_is_fitted(self, "tree_")

        def walk(node):
            if node["type"] == "leaf":
                return []
            return [node["support"], *walk(node["left"]), *walk(node["right"])]

        return walk(self.tree_)

    def _leaf_values_and_rows(self, X, y):
        """Yield (leaf value, training rows routed there) — used by tests."""
        check_is_fitted(self, "tree_")
        X = check_array(X, accept_sparse=False)
        y = np.asarray(y, dtype=float)
        out = []

        def walk(node, mask):
            if node["type"] == "leaf":
                out.append((node["value"], y[mask]))
                return
            left = mask & (X[:, node["feature"]] <= node["threshold"])
            walk(node["left"], left)
            walk(node["right"], mask & ~left)

        walk(self.tree_, np.ones(len(X), dtype=bool))
        return out
