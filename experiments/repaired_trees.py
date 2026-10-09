"""Narrow research implementations of two repaired tree ideas.

These estimators are deliberately outside :mod:`stable_cart`. They exist to
test the signature mechanisms that the deleted package classes advertised,
without the unrelated oblique, lookahead, smoothing, and preprocessing layers
that made the old implementations uninterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

Task = Literal["regression", "classification"]


@dataclass(slots=True)
class _Candidate:
    feature: int
    threshold: float
    gain: float
    variance: float = 0.0
    score: float = 0.0
    support: float = 0.0


@dataclass(slots=True)
class _Node:
    prediction: float | NDArray[np.float64]
    depth: int
    n_structure: int
    n_estimation: int
    feature: int | None = None
    threshold: float | None = None
    gain: float | None = None
    variance: float | None = None
    score: float | None = None
    support: float | None = None
    vote_supports: tuple[float, ...] | None = None
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        """Return whether this node has no split."""
        return self.feature is None


def _relative_gain(
    y: NDArray[np.float64], left: NDArray[np.bool_], task: Task
) -> float:
    if not np.any(left) or np.all(left):
        return 0.0

    if task == "regression":
        parent = float(np.var(y))
        if parent <= 0.0:
            return 0.0
        weighted = (
            np.count_nonzero(left) * float(np.var(y[left]))
            + np.count_nonzero(~left) * float(np.var(y[~left]))
        ) / len(y)
    else:
        n_classes = int(np.max(y)) + 1

        def gini(values: NDArray[np.float64]) -> float:
            counts = np.bincount(values.astype(int), minlength=n_classes)
            probabilities = counts / len(values)
            return float(1.0 - probabilities @ probabilities)

        parent = gini(y)
        if parent <= 0.0:
            return 0.0
        weighted = (
            np.count_nonzero(left) * gini(y[left])
            + np.count_nonzero(~left) * gini(y[~left])
        ) / len(y)

    return max(0.0, (parent - weighted) / parent)


def _candidate_thresholds(
    values: NDArray[np.float64], max_candidates: int | None
) -> NDArray[np.float64]:
    unique = np.unique(values)
    if len(unique) < 2:
        return np.empty(0, dtype=float)
    thresholds = unique[:-1] + (unique[1:] - unique[:-1]) / 2.0
    if max_candidates is not None and len(thresholds) > max_candidates:
        positions = np.linspace(0, len(thresholds) - 1, max_candidates)
        thresholds = thresholds[np.unique(np.rint(positions).astype(int))]
    return thresholds


def _candidates(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    task: Task,
    min_samples_leaf: int,
    max_candidates_per_feature: int | None,
) -> list[_Candidate]:
    found: list[_Candidate] = []
    for feature in range(X.shape[1]):
        for threshold in _candidate_thresholds(
            X[:, feature], max_candidates_per_feature
        ):
            left = X[:, feature] <= threshold
            if (
                np.count_nonzero(left) < min_samples_leaf
                or np.count_nonzero(~left) < min_samples_leaf
            ):
                continue
            gain = _relative_gain(y, left, task)
            if gain > 0.0:
                found.append(
                    _Candidate(
                        feature=feature,
                        threshold=float(threshold),
                        gain=gain,
                        score=gain,
                    )
                )
    return found


def _candidate_key(candidate: _Candidate) -> tuple[float, int, float]:
    return (-candidate.score, candidate.feature, candidate.threshold)


def _greedy(candidates: list[_Candidate]) -> _Candidate | None:
    return min(candidates, key=_candidate_key) if candidates else None


def _bootstrap_gains(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    candidates: list[_Candidate],
    *,
    task: Task,
    bootstrap_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
    gains = np.zeros((len(bootstrap_indices), len(candidates)), dtype=float)
    features = np.asarray([candidate.feature for candidate in candidates], dtype=int)
    thresholds = np.asarray(
        [candidate.threshold for candidate in candidates], dtype=float
    )
    for bootstrap_number, indices in enumerate(bootstrap_indices):
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        left = X_bootstrap[:, features] <= thresholds
        n_left = left.sum(axis=0).astype(float)
        n_right = len(y_bootstrap) - n_left
        valid = (n_left > 0) & (n_right > 0)
        if not np.any(valid):
            continue
        left_float = left.astype(float)
        if task == "regression":
            y_bootstrap = y_bootstrap - y_bootstrap[0]
            parent = float(np.var(y_bootstrap))
            if parent <= 0.0:
                continue
            total_sum = float(np.sum(y_bootstrap))
            total_squares = float(y_bootstrap @ y_bootstrap)
            left_sum = y_bootstrap @ left_float
            left_squares = (y_bootstrap**2) @ left_float
            right_sum = total_sum - left_sum
            right_squares = total_squares - left_squares
            child_sse = np.zeros(len(candidates), dtype=float)
            child_sse[valid] = (
                left_squares[valid]
                - left_sum[valid] ** 2 / n_left[valid]
                + right_squares[valid]
                - right_sum[valid] ** 2 / n_right[valid]
            )
            relative = 1.0 - child_sse / (len(y_bootstrap) * parent)
        else:
            n_classes = int(np.max(y_bootstrap)) + 1
            one_hot = np.eye(n_classes)[y_bootstrap.astype(int)]
            left_counts = one_hot.T @ left_float
            total_counts = one_hot.sum(axis=0)[:, None]
            right_counts = total_counts - left_counts
            parent_probabilities = total_counts[:, 0] / len(y_bootstrap)
            parent = float(1.0 - parent_probabilities @ parent_probabilities)
            if parent <= 0.0:
                continue
            weighted = np.zeros(len(candidates), dtype=float)
            weighted[valid] = (
                n_left[valid]
                * (1.0 - np.sum((left_counts[:, valid] / n_left[valid]) ** 2, axis=0))
                + n_right[valid]
                * (1.0 - np.sum((right_counts[:, valid] / n_right[valid]) ** 2, axis=0))
            ) / len(y_bootstrap)
            relative = (parent - weighted) / parent
        gains[bootstrap_number] = np.where(valid, np.maximum(0.0, relative), 0.0)
    return gains


class _ResearchTree(BaseEstimator):
    """Shared fitting and prediction machinery for the research trees."""

    def __init__(
        self,
        *,
        task: Task,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        max_candidates_per_feature: int | None,
        random_state: int | None,
    ) -> None:
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates_per_feature = max_candidates_per_feature
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        if self.max_depth < 0:
            raise ValueError("max_depth must be nonnegative")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if self.min_samples_split < 2 * self.min_samples_leaf:
            raise ValueError(
                "min_samples_split must be at least twice min_samples_leaf"
            )
        if (
            self.max_candidates_per_feature is not None
            and self.max_candidates_per_feature < 1
        ):
            raise ValueError("max_candidates_per_feature must be positive or None")

    def _prepare_target(self, y: NDArray[Any]) -> NDArray[np.float64]:
        if self.task == "regression":
            if not np.issubdtype(np.asarray(y).dtype, np.number):
                raise ValueError("regression requires a numeric target")
            return np.asarray(y, dtype=float)

        target_type = type_of_target(y)
        if target_type not in {"binary", "multiclass"}:
            raise ValueError("classification requires a binary or multiclass target")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        return encoded.astype(float)

    def _ordered_rows(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
        keys = tuple(X[:, feature] for feature in range(X.shape[1] - 1, -1, -1))
        order = np.lexsort(keys)
        return X[order], y[order], order

    def _leaf_prediction(
        self,
        y_estimation: NDArray[np.float64],
        fallback: NDArray[np.float64],
    ) -> float | NDArray[np.float64]:
        values = y_estimation if len(y_estimation) else fallback
        if self.task == "regression":
            return float(np.mean(values))
        counts = np.bincount(values.astype(int), minlength=len(self.classes_))
        return counts.astype(float) / np.sum(counts)

    def _predict_values(self, X: NDArray[Any]) -> NDArray[Any]:
        check_is_fitted(self, "tree_")
        checked = validate_data(self, X, reset=False)
        values: list[float | NDArray[np.float64]] = []
        for row in checked:
            node = self.tree_
            while not node.is_leaf:
                node = node.left if row[node.feature] <= node.threshold else node.right
            values.append(node.prediction)
        return np.asarray(values)

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Predict regression values or class labels."""
        values = self._predict_values(X)
        if self.task == "regression":
            return values.astype(float)
        return self.classes_[np.argmax(values, axis=1)]

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.float64]:
        """Predict class probabilities for a classification tree."""
        if self.task != "classification":
            raise AttributeError("predict_proba is available only for classification")
        return self._predict_values(X).astype(float)

    def score(self, X: NDArray[Any], y: NDArray[Any]) -> float:
        """Return accuracy for classification and R-squared for regression."""
        predicted = self.predict(X)
        if self.task == "classification":
            return float(accuracy_score(y, predicted))
        return float(r2_score(y, predicted))


class BootstrapVariancePenalizedTree(_ResearchTree):
    """Tree that penalizes bootstrap variance of every candidate split's gain."""

    def __init__(
        self,
        *,
        task: Task = "regression",
        max_depth: int = 4,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        variance_penalty: float = 1.0,
        n_bootstrap: int = 32,
        max_candidates_per_feature: int | None = 32,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            task=task,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_candidates_per_feature=max_candidates_per_feature,
            random_state=random_state,
        )
        self.variance_penalty = variance_penalty
        self.n_bootstrap = n_bootstrap

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.variance_penalty < 0:
            raise ValueError("variance_penalty must be nonnegative")
        if self.n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> BootstrapVariancePenalizedTree:
        """Fit the variance-penalized tree."""
        self._validate_parameters()
        X_checked, y_checked = validate_data(self, X, y, ensure_min_samples=2)
        target = self._prepare_target(y_checked)
        X_ordered, y_ordered, self.row_order_ = self._ordered_rows(
            X_checked.astype(float), target
        )
        self._rng_ = np.random.default_rng(self.random_state)
        self.tree_ = self._build(X_ordered, y_ordered, depth=0)
        return self

    def _select(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> _Candidate | None:
        candidates = _candidates(
            X,
            y,
            task=self.task,
            min_samples_leaf=self.min_samples_leaf,
            max_candidates_per_feature=self.max_candidates_per_feature,
        )
        if not candidates or self.variance_penalty == 0.0:
            return _greedy(candidates)

        indices = self._rng_.integers(0, len(X), size=(self.n_bootstrap, len(X)))
        gains = _bootstrap_gains(
            X,
            y,
            candidates,
            task=self.task,
            bootstrap_indices=indices,
        )
        variances = np.var(gains, axis=0, ddof=1)
        for candidate, variance in zip(candidates, variances, strict=True):
            candidate.variance = float(variance)
            candidate.score = (
                candidate.gain - self.variance_penalty * candidate.variance
            )
        selected = _greedy(candidates)
        return selected if selected is not None and selected.score > 0.0 else None

    def _build(
        self, X: NDArray[np.float64], y: NDArray[np.float64], depth: int
    ) -> _Node:
        prediction = self._leaf_prediction(y, y)
        node = _Node(prediction, depth, len(y), len(y))
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return node
        selected = self._select(X, y)
        if selected is None:
            return node
        left = X[:, selected.feature] <= selected.threshold
        node.feature = selected.feature
        node.threshold = selected.threshold
        node.gain = selected.gain
        node.variance = selected.variance
        node.score = selected.score
        node.left = self._build(X[left], y[left], depth + 1)
        node.right = self._build(X[~left], y[~left], depth + 1)
        return node


class RobustPrefixHonestTree(_ResearchTree):
    """Honest tree with bootstrap-consensus selection in a fixed prefix."""

    def __init__(
        self,
        *,
        task: Task = "regression",
        max_depth: int = 4,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        prefix_levels: int = 2,
        consensus_threshold: float = 0.5,
        n_bootstrap: int = 32,
        estimation_fraction: float = 0.5,
        max_candidates_per_feature: int | None = 32,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            task=task,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_candidates_per_feature=max_candidates_per_feature,
            random_state=random_state,
        )
        self.prefix_levels = prefix_levels
        self.consensus_threshold = consensus_threshold
        self.n_bootstrap = n_bootstrap
        self.estimation_fraction = estimation_fraction

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.prefix_levels < 0:
            raise ValueError("prefix_levels must be nonnegative")
        if not 0.0 <= self.consensus_threshold <= 1.0:
            raise ValueError("consensus_threshold must be in [0, 1]")
        if self.n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")
        if not 0.0 < self.estimation_fraction < 1.0:
            raise ValueError("estimation_fraction must be strictly between 0 and 1")

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> RobustPrefixHonestTree:
        """Fit the consensus-prefix tree and honest leaf estimates."""
        self._validate_parameters()
        X_checked, y_checked = validate_data(self, X, y, ensure_min_samples=4)
        target = self._prepare_target(y_checked)
        X_ordered, y_ordered, order = self._ordered_rows(
            X_checked.astype(float), target
        )
        self.row_order_ = order
        self._rng_ = np.random.default_rng(self.random_state)
        assignment = self._rng_.permutation(len(X_ordered))
        n_estimation = min(
            len(X_ordered) - 1,
            max(1, round(self.estimation_fraction * len(X_ordered))),
        )
        estimation = np.zeros(len(X_ordered), dtype=bool)
        estimation[assignment[:n_estimation]] = True
        structure = ~estimation
        self.structure_indices_ = order[structure]
        self.estimation_indices_ = order[estimation]
        self._global_estimation_target_ = y_ordered[estimation]
        self.tree_ = self._build(
            X_ordered[structure],
            y_ordered[structure],
            X_ordered[estimation],
            y_ordered[estimation],
            depth=0,
        )
        return self

    def _consensus_select(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[_Candidate | None, tuple[float, ...]]:
        candidates = _candidates(
            X,
            y,
            task=self.task,
            min_samples_leaf=self.min_samples_leaf,
            max_candidates_per_feature=self.max_candidates_per_feature,
        )
        if not candidates:
            return None, ()
        indices = self._rng_.integers(0, len(X), size=(self.n_bootstrap, len(X)))
        gains = _bootstrap_gains(
            X,
            y,
            candidates,
            task=self.task,
            bootstrap_indices=indices,
        )
        votes = np.zeros(len(candidates), dtype=int)
        for row in gains:
            winner = min(
                range(len(candidates)),
                key=lambda index: (
                    -row[index],
                    candidates[index].feature,
                    candidates[index].threshold,
                ),
            )
            votes[winner] += 1
        supports = votes / self.n_bootstrap
        for candidate, support in zip(candidates, supports, strict=True):
            candidate.support = float(support)
        eligible = [
            candidate
            for candidate in candidates
            if candidate.support >= self.consensus_threshold
        ]
        if not eligible:
            return None, tuple(float(value) for value in supports)
        selected = min(
            eligible,
            key=lambda candidate: (
                -candidate.support,
                -candidate.gain,
                candidate.feature,
                candidate.threshold,
            ),
        )
        return selected, tuple(float(value) for value in supports)

    def _build(
        self,
        X_structure: NDArray[np.float64],
        y_structure: NDArray[np.float64],
        X_estimation: NDArray[np.float64],
        y_estimation: NDArray[np.float64],
        depth: int,
    ) -> _Node:
        prediction = self._leaf_prediction(
            y_estimation, self._global_estimation_target_
        )
        node = _Node(
            prediction,
            depth,
            len(y_structure),
            len(y_estimation),
        )
        if (
            depth >= self.max_depth
            or len(y_structure) < self.min_samples_split
            or len(np.unique(y_structure)) == 1
        ):
            return node
        candidates = _candidates(
            X_structure,
            y_structure,
            task=self.task,
            min_samples_leaf=self.min_samples_leaf,
            max_candidates_per_feature=self.max_candidates_per_feature,
        )
        vote_supports: tuple[float, ...] | None = None
        if depth < self.prefix_levels:
            selected, vote_supports = self._consensus_select(X_structure, y_structure)
        else:
            selected = _greedy(candidates)
        if selected is None:
            node.vote_supports = vote_supports
            return node

        left_structure = X_structure[:, selected.feature] <= selected.threshold
        left_estimation = X_estimation[:, selected.feature] <= selected.threshold
        node.feature = selected.feature
        node.threshold = selected.threshold
        node.gain = selected.gain
        node.support = selected.support if depth < self.prefix_levels else None
        node.vote_supports = vote_supports
        node.left = self._build(
            X_structure[left_structure],
            y_structure[left_structure],
            X_estimation[left_estimation],
            y_estimation[left_estimation],
            depth + 1,
        )
        node.right = self._build(
            X_structure[~left_structure],
            y_structure[~left_structure],
            X_estimation[~left_estimation],
            y_estimation[~left_estimation],
            depth + 1,
        )
        return node


__all__ = ["BootstrapVariancePenalizedTree", "RobustPrefixHonestTree"]
