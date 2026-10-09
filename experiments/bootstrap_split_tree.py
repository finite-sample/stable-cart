"""A single experimental tree built from bootstrap-aggregated split decisions."""

from collections import Counter
from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import ClassifierTags, RegressorTags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from experiments.split_candidates import (
    Task,
    _all_split_gains,
    _find_candidate_splits,
    check_predict_input,
)

__all__ = ["BootstrapSplitTree"]


class BootstrapSplitTree(BaseEstimator):
    """Build one tree by aggregating split choices across bootstrap replicates.

    At each node, each bootstrap replicate votes with its best greedy split. The
    winning feature is the one with the most votes, and its threshold is the
    median threshold among votes for that feature. The median is projected onto
    the nearest split that satisfies ``min_samples_leaf`` in the full node.

    This is an experimental algorithm. The procedure above is verified by exact
    reference tests; it is not a guarantee that the resulting tree is more stable
    than CART on a particular data-generating process.

    Parameters
    ----------
    task
        ``"regression"`` or binary ``"classification"``.
    max_depth
        Maximum depth. Zero produces a constant tree.
    min_samples_leaf
        Minimum observations in each child.
    min_samples_split
        Minimum observations required to attempt a split.
    n_consensus
        Number of bootstrap votes per node. One uses the node itself and is the
        exact greedy limiting case.
    consensus_threshold
        Minimum fraction of all replicates voting for the winning feature.
    leaf_shrinkage
        Parent-mean pseudo-count used to shrink each leaf mean. Zero disables
        shrinkage.
    max_candidates
        Maximum split candidates retained within each replicate.
    random_state
        Random seed.
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

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> "BootstrapSplitTree":
        """Fit the experimental tree and return ``self``."""
        self._validate_parameters()
        X, y = cast(
            tuple[NDArray[Any], NDArray[Any]],
            cast(Any, validate_data)(self, X=X, y=y, accept_sparse=False, reset=True),
        )

        if self.task == "classification":
            self.classes_ = np.unique(y)
            target_type = type_of_target(y, input_name="y", raise_unknown=True)
            if target_type != "binary" or len(self.classes_) != 2:
                raise ValueError(
                    "Only binary classification is supported. "
                    f"The type of the target is {target_type}; observed "
                    f"{len(self.classes_)} class(es)."
                )
            y_work = np.searchsorted(self.classes_, y).astype(float)
        else:
            try:
                y_work = y.astype(float)
            except (TypeError, ValueError) as error:
                raise ValueError("Regression targets must be numeric.") from error

        self.stop_reasons_: Counter[str] = Counter()
        rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build(X, y_work, depth=0, parent=y_work, rng=rng)
        return self

    def _validate_parameters(self) -> None:
        """Reject configurations with undefined algorithmic meaning."""
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'.")
        self._require_integer("max_depth", self.max_depth, minimum=0)
        self._require_integer("min_samples_leaf", self.min_samples_leaf, minimum=1)
        self._require_integer("min_samples_split", self.min_samples_split, minimum=2)
        self._require_integer("n_consensus", self.n_consensus, minimum=1)
        self._require_integer("max_candidates", self.max_candidates, minimum=1)
        if (
            not isinstance(self.consensus_threshold, Real)
            or not 0 <= self.consensus_threshold <= 1
        ):
            raise ValueError("consensus_threshold must be between 0 and 1.")
        if not isinstance(self.leaf_shrinkage, Real) or self.leaf_shrinkage < 0:
            raise ValueError("leaf_shrinkage must be nonnegative.")

    @staticmethod
    def _require_integer(name: str, value: Any, *, minimum: int) -> None:
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or value < minimum
        ):
            raise ValueError(
                f"{name} must be an integer greater than or equal to {minimum}."
            )

    def _elect_split(
        self, X: NDArray[Any], y: NDArray[np.floating], rng: np.random.Generator
    ) -> tuple[int, float, float] | None:
        """Aggregate replicate-level greedy choices into one admissible split."""
        votes: dict[int, list[float]] = {}
        for _ in range(self.n_consensus):
            if self.n_consensus == 1:
                X_sample, y_sample = X, y
            else:
                indices = rng.integers(0, len(y), len(y))
                X_sample, y_sample = X[indices], y[indices]
            candidates = _find_candidate_splits(
                X_sample,
                y_sample,
                task=cast(Task, self.task),
                max_candidates=self.max_candidates,
                min_samples_leaf=self.min_samples_leaf,
            )
            if candidates:
                best = candidates[0]
                votes.setdefault(best.feature_idx, []).append(best.threshold)

        if not votes:
            return None
        feature = max(votes, key=lambda index: (len(votes[index]), -index))
        support = len(votes[feature]) / self.n_consensus
        if support < self.consensus_threshold:
            return None

        admissible, _ = _all_split_gains(
            X[:, feature],
            y,
            task=cast(Task, self.task),
            min_samples_leaf=self.min_samples_leaf,
        )
        if admissible.size == 0:
            return None
        median_vote = float(np.median(votes[feature]))
        distance = np.abs(admissible - median_vote)
        threshold = float(admissible[np.flatnonzero(distance == distance.min())[0]])
        return feature, threshold, support

    def _leaf(
        self, y: NDArray[np.floating], parent: NDArray[np.floating]
    ) -> dict[str, Any]:
        """Return a leaf whose mean may be shrunk toward its parent mean."""
        leaf_mean = float(np.mean(y))
        if self.leaf_shrinkage:
            value = (
                len(y) * leaf_mean + self.leaf_shrinkage * float(np.mean(parent))
            ) / (len(y) + self.leaf_shrinkage)
        else:
            value = leaf_mean
        return {"type": "leaf", "value": float(value), "n": len(y)}

    def _build(self, X, y, depth, parent, rng):
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
        if min(int(left_mask.sum()), int((~left_mask).sum())) < self.min_samples_leaf:
            raise RuntimeError(
                "Internal error: elected split violates min_samples_leaf."
            )
        return {
            "type": "split",
            "feature": feature,
            "threshold": threshold,
            "support": support,
            "n": len(y),
            "left": self._build(X[left_mask], y[left_mask], depth + 1, y, rng),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1, y, rng),
        }

    @staticmethod
    def _route(row: NDArray[Any], node: dict[str, Any]) -> float:
        while node["type"] == "split":
            node = (
                node["left"]
                if row[node["feature"]] <= node["threshold"]
                else node["right"]
            )
        return float(node["value"])

    def _raw_predict(self, X: NDArray[Any]) -> NDArray[np.floating]:
        checked = check_predict_input(self, X, "tree_")
        return np.array([self._route(row, self.tree_) for row in checked])

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Predict values or class labels."""
        raw = self._raw_predict(X)
        if self.task == "classification":
            return self.classes_[(raw > 0.5).astype(int)]
        return raw

    def _is_classifier(self) -> bool:
        """Return whether classification-only methods should be exposed."""
        return self.task == "classification"

    @available_if(_is_classifier)
    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.floating]:
        """Return binary class probabilities from leaf proportions."""
        raw = np.clip(self._raw_predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - raw, raw])

    def score(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> float:
        """Return accuracy for classification or R-squared for regression."""
        predictions = self.predict(X)
        if self.task == "classification":
            return float(accuracy_score(y, predictions, sample_weight=sample_weight))
        return float(r2_score(y, predictions, sample_weight=sample_weight))

    def get_n_leaves(self) -> int:
        """Return the number of terminal nodes."""
        check_is_fitted(self, "tree_")

        def count(node: dict[str, Any]) -> int:
            return (
                1
                if node["type"] == "leaf"
                else count(node["left"]) + count(node["right"])
            )

        return count(self.tree_)

    def split_supports(self) -> list[float]:
        """Return the winning vote share for each internal node, root first."""
        check_is_fitted(self, "tree_")

        def walk(node: dict[str, Any]) -> list[float]:
            if node["type"] == "leaf":
                return []
            return [node["support"], *walk(node["left"]), *walk(node["right"])]

        return walk(self.tree_)

    def _leaf_values_and_rows(self, X: NDArray[Any], y: NDArray[Any]):
        checked = check_predict_input(self, X, "tree_")
        numeric_y = np.asarray(y, dtype=float)
        output = []

        def walk(node: dict[str, Any], mask: NDArray[np.bool_]) -> None:
            if node["type"] == "leaf":
                output.append((node["value"], numeric_y[mask]))
                return
            left = mask & (checked[:, node["feature"]] <= node["threshold"])
            walk(node["left"], left)
            walk(node["right"], mask & ~left)

        walk(self.tree_, np.ones(len(checked), dtype=bool))
        return output

    def __sklearn_tags__(self):
        """Expose the task-dependent estimator type to scikit-learn."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        if self.task == "classification":
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags(multi_class=False)
            tags.regressor_tags = None
        else:
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
            tags.classifier_tags = None
        return tags
