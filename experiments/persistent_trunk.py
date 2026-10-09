"""Research-only persistent-partition tree workflows.

The candidate freezes a fitted depth-one CART partition and relearns a separate
subtree inside each side. The comparison workflow freezes a complete CART
partition and refreshes only its terminal predictions. Neither class is part of
the :mod:`stable_cart` package.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

Task = Literal["classification", "regression"]
Tree = DecisionTreeClassifier | DecisionTreeRegressor


def tree_factory(
    task: Task,
    *,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> Tree:
    """Create the explicit CART specification used by the study."""
    common = {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "ccp_alpha": 0.0,
        "random_state": random_state,
    }
    if task == "classification":
        return DecisionTreeClassifier(criterion="gini", **common)
    return DecisionTreeRegressor(criterion="squared_error", **common)


def terminal_nodes(tree: Tree) -> NDArray[np.int_]:
    """Return the terminal node identifiers of a fitted tree."""
    return np.flatnonzero(tree.tree_.children_left == -1)


def aligned_probabilities(
    estimator: DecisionTreeClassifier,
    X: ArrayLike,
    classes: NDArray,
) -> NDArray[np.float64]:
    """Align a classifier's probability columns to a declared class order."""
    observed = estimator.predict_proba(X)
    aligned = np.zeros((len(observed), len(classes)), dtype=float)
    positions = {label: index for index, label in enumerate(classes)}
    for source, label in enumerate(estimator.classes_):
        aligned[:, positions[label]] = observed[:, source]
    return aligned


def _class_distribution(y: NDArray, classes: NDArray) -> NDArray[np.float64]:
    counts = np.asarray(
        [np.count_nonzero(y == label) for label in classes], dtype=float
    )
    if counts.sum() == 0:
        return np.full(len(classes), 1.0 / len(classes))
    return counts / counts.sum()


class PersistentTrunkTree:
    """Freeze a reference trunk and relearn one subtree within each trunk leaf."""

    def __init__(
        self,
        trunk: Tree,
        *,
        task: Task,
        subtree_depth: int = 3,
        min_samples_leaf: int = 10,
        random_state: int = 0,
    ) -> None:
        if trunk.tree_.max_depth != 1:
            raise ValueError("trunk must be a fitted depth-one tree")
        self.trunk_ = deepcopy(trunk)
        self.task = task
        self.subtree_depth = subtree_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.classes_ = (
            np.asarray(self.trunk_.classes_)
            if task == "classification"
            else np.empty(0)
        )
        self.trunk_signature_ = self.trunk_signature()

    def trunk_signature(self) -> tuple[int, float]:
        """Return the frozen root feature and threshold."""
        return int(self.trunk_.tree_.feature[0]), float(self.trunk_.tree_.threshold[0])

    def fit(self, X: ArrayLike, y: ArrayLike) -> PersistentTrunkTree:
        """Relearn descendant subtrees without modifying the reference trunk."""
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        if self.trunk_signature() != self.trunk_signature_:
            raise RuntimeError("the frozen trunk was modified")
        routes = self.trunk_.apply(X_array)
        self.subtrees_: dict[int, Tree] = {}
        if self.task == "classification":
            self.fallback_ = _class_distribution(y_array, self.classes_)
        else:
            self.fallback_ = float(np.mean(y_array))
        for node in terminal_nodes(self.trunk_):
            mask = routes == node
            if not np.any(mask):
                continue
            subtree = tree_factory(
                self.task,
                max_depth=self.subtree_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + int(node),
            )
            self.subtrees_[int(node)] = subtree.fit(X_array[mask], y_array[mask])
        if self.trunk_signature() != self.trunk_signature_:
            raise RuntimeError("fit changed the frozen trunk")
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict probabilities aligned to the reference class order."""
        if self.task != "classification":
            raise AttributeError("predict_proba is available only for classification")
        X_array = np.asarray(X)
        routes = self.trunk_.apply(X_array)
        predictions = np.tile(self.fallback_, (len(X_array), 1))
        for node, subtree in self.subtrees_.items():
            mask = routes == node
            if np.any(mask):
                predictions[mask] = aligned_probabilities(
                    subtree, X_array[mask], self.classes_
                )
        return predictions

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict labels or regression values."""
        if self.task == "classification":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        X_array = np.asarray(X)
        routes = self.trunk_.apply(X_array)
        predictions = np.full(len(X_array), self.fallback_, dtype=float)
        for node, subtree in self.subtrees_.items():
            mask = routes == node
            if np.any(mask):
                predictions[mask] = subtree.predict(X_array[mask])
        return predictions


class RefreshedLeafTree:
    """Preserve a complete reference partition and refresh terminal values."""

    def __init__(self, tree: Tree, *, task: Task) -> None:
        self.tree_ = deepcopy(tree)
        self.task = task
        self.classes_ = (
            np.asarray(self.tree_.classes_) if task == "classification" else np.empty(0)
        )
        self.partition_signature_ = self.partition_signature()

    def partition_signature(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Return every frozen split feature and threshold."""
        internal = self.tree_.tree_.children_left != -1
        features = tuple(int(value) for value in self.tree_.tree_.feature[internal])
        thresholds = tuple(
            float(value) for value in self.tree_.tree_.threshold[internal]
        )
        return features, thresholds

    def fit(self, X: ArrayLike, y: ArrayLike) -> RefreshedLeafTree:
        """Refresh terminal predictions without changing any split."""
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        if self.partition_signature() != self.partition_signature_:
            raise RuntimeError("the frozen partition was modified")
        routes = self.tree_.apply(X_array)
        self.values_: dict[int, float | NDArray[np.float64]] = {}
        if self.task == "classification":
            self.fallback_ = _class_distribution(y_array, self.classes_)
        else:
            self.fallback_ = float(np.mean(y_array))
        for node in terminal_nodes(self.tree_):
            values = y_array[routes == node]
            if len(values) == 0:
                continue
            if self.task == "classification":
                self.values_[int(node)] = _class_distribution(values, self.classes_)
            else:
                self.values_[int(node)] = float(np.mean(values))
        if self.partition_signature() != self.partition_signature_:
            raise RuntimeError("fit changed the frozen partition")
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict refreshed probabilities in the reference class order."""
        if self.task != "classification":
            raise AttributeError("predict_proba is available only for classification")
        X_array = np.asarray(X)
        routes = self.tree_.apply(X_array)
        predictions = np.tile(self.fallback_, (len(X_array), 1))
        for node, value in self.values_.items():
            predictions[routes == node] = value
        return predictions

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict refreshed labels or regression values."""
        X_array = np.asarray(X)
        routes = self.tree_.apply(X_array)
        if self.task == "classification":
            return self.classes_[np.argmax(self.predict_proba(X_array), axis=1)]
        predictions = np.full(len(X_array), self.fallback_, dtype=float)
        for node, value in self.values_.items():
            predictions[routes == node] = value
        return predictions
