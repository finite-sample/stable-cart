"""Exhaustive split scoring for the experimental CART variants."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_is_fitted

Task = Literal["regression", "classification"]


def check_predict_input(
    estimator: Any, X: NDArray[Any], fitted_attribute: str
) -> NDArray[Any]:
    """
    Check that an estimator is fitted and that X is the width it was fitted on.

    Parameters
    ----------
    estimator
        The estimator being asked to predict.
    X
        Feature matrix supplied at prediction time.
    fitted_attribute
        Attribute that only exists once the estimator has been fitted.

    Returns
    -------
    NDArray[Any]
        The validated feature matrix.

    Raises
    ------
    ValueError
        If X does not have ``n_features_in_`` columns. Predicting from a matrix
        of the wrong width is the failure mode that returns plausible numbers
        computed from the wrong columns, so it is refused rather than
        broadcast.
    """
    check_is_fitted(estimator, fitted_attribute)
    X = check_array(X, accept_sparse=False)
    expected = getattr(estimator, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise ValueError(
            f"X has {X.shape[1]} features, but this {type(estimator).__name__} "
            f"was fitted with {expected} features."
        )
    return X


@dataclass(slots=True)
class SplitCandidate:
    """A scored axis-aligned split and its child observation indices."""

    feature_idx: int
    threshold: float
    gain: float
    left_indices: NDArray[np.int_]
    right_indices: NDArray[np.int_]


def _find_candidate_splits(
    X: np.ndarray,
    y: np.ndarray,
    task: Task,
    max_candidates: int = 20,
    min_samples_leaf: int = 1,
) -> list[SplitCandidate]:
    """
    Find basic axis-aligned split candidates.

    Parameters
    ----------
    X
        Feature matrix for split finding.
    y
        Target values for split evaluation.
    task
        Regression uses variance reduction; classification uses Gini reduction.
    max_candidates
        Maximum number of candidates to return.
    min_samples_leaf
        Minimum rows a split must leave on each side. Enforced while the
        candidates are generated, not afterwards: a caller that vetoes an
        inadmissible candidate later has no second choice and stops growing the
        tree at that node.

    Returns
    -------
    list[SplitCandidate]
        List of split candidates, every one of them admissible.
    """
    candidates = []
    n_features = X.shape[1]

    # Per-feature budget, so no single feature can crowd the others out of the pool.
    splits_per_feature = max(1, max_candidates // n_features)

    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        thresholds, gains = _all_split_gains(feature_values, y, task, min_samples_leaf)

        if thresholds.size == 0:
            continue

        # Keep this feature's *best* thresholds. Scanning every midpoint first is
        # what makes that possible: taking a prefix of the sorted unique values
        # only ever sees the bottom of the feature's range.
        keep = np.argsort(gains)[::-1][:splits_per_feature]

        for i in np.sort(keep):
            threshold = float(thresholds[i])
            left_mask = feature_values <= threshold
            candidates.append(
                SplitCandidate(
                    feature_idx=feature_idx,
                    threshold=threshold,
                    gain=float(gains[i]),
                    left_indices=np.where(left_mask)[0],
                    right_indices=np.where(~left_mask)[0],
                )
            )

    # Return top candidates
    candidates.sort(key=lambda c: c.gain, reverse=True)
    return candidates[:max_candidates]


def _all_split_gains(
    feature_values: np.ndarray, y: np.ndarray, task: Task, min_samples_leaf: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score every admissible threshold of one feature in a single vectorized pass.

    Equivalent to calling :func:`_evaluate_split_gain` on each midpoint between
    consecutive distinct feature values, but O(n log n) rather than O(n) per
    threshold, which is what makes an exhaustive scan affordable.

    Parameters
    ----------
    feature_values
        One column of the feature matrix.
    y
        Target values.
    task
        Regression uses variance reduction; classification uses Gini reduction.
    min_samples_leaf
        Minimum rows on each side of the split.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Candidate thresholds and their gains, in ascending threshold order.
    """
    n = len(y)
    empty = (np.empty(0), np.empty(0))
    if n < 2:
        return empty

    order = np.argsort(feature_values, kind="mergesort")
    xs = feature_values[order]
    ys = y[order]

    # A split is admissible only between two distinct feature values and only if
    # it leaves enough rows on both sides.
    admissible = xs[:-1] < xs[1:]
    if min_samples_leaf > 1:
        left_count = np.arange(1, n)
        admissible &= (left_count >= min_samples_leaf) & (
            n - left_count >= min_samples_leaf
        )
    if not np.any(admissible):
        return empty

    n_left = np.arange(1, n, dtype=float)
    n_right = n - n_left

    if task == "regression":
        ys = ys.astype(float)
        ys = ys - ys[0]
        csum = np.cumsum(ys)[:-1]
        csum_sq = np.cumsum(ys**2)[:-1]
        total_sum, total_sq = float(ys.sum()), float((ys**2).sum())

        mean_left = csum / n_left
        mean_right = (total_sum - csum) / n_right
        var_left = np.maximum(csum_sq / n_left - mean_left**2, 0.0)
        var_right = np.maximum((total_sq - csum_sq) / n_right - mean_right**2, 0.0)

        total_impurity = float(np.var(ys)) if n > 1 else 0.0
        weighted = (n_left * var_left + n_right * var_right) / n
    else:
        _, codes = np.unique(ys, return_inverse=True)
        onehot = np.zeros((n, codes.max() + 1))
        onehot[np.arange(n), codes] = 1.0
        counts_left = np.cumsum(onehot, axis=0)[:-1]
        counts_right = onehot.sum(axis=0) - counts_left

        gini_left = 1.0 - np.sum((counts_left / n_left[:, None]) ** 2, axis=1)
        gini_right = 1.0 - np.sum((counts_right / n_right[:, None]) ** 2, axis=1)

        total_impurity = _gini_impurity(ys)
        weighted = (n_left * gini_left + n_right * gini_right) / n

    thresholds = (xs[:-1] + xs[1:]) / 2.0
    gains = total_impurity - weighted
    return thresholds[admissible], gains[admissible]


def _evaluate_split_gain(y: np.ndarray, left_mask: np.ndarray, task: Task) -> float:
    """
    Evaluate information gain from a split.

    Parameters
    ----------
    y
        Target values array.
    left_mask
        Boolean mask for left split.
    task
        Regression uses variance reduction; classification uses Gini reduction.

    Returns
    -------
    float
        Information gain value.
    """
    if len(y) == 0 or np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
        return 0.0

    # Determine if this looks like regression or classification
    if task == "regression":
        # Regression: use variance reduction
        total_var = np.var(y) if len(y) > 1 else 0
        left_var = np.var(y[left_mask]) if np.sum(left_mask) > 1 else 0
        right_var = np.var(y[~left_mask]) if np.sum(~left_mask) > 1 else 0

        n_left = np.sum(left_mask)
        n_right = np.sum(~left_mask)
        n_total = len(y)

        weighted_var = (n_left * left_var + n_right * right_var) / n_total
        return total_var - weighted_var
    else:
        # Classification: use Gini reduction
        total_gini = _gini_impurity(y)
        left_gini = _gini_impurity(y[left_mask])
        right_gini = _gini_impurity(y[~left_mask])

        n_left = np.sum(left_mask)
        n_right = np.sum(~left_mask)
        n_total = len(y)

        weighted_gini = (n_left * left_gini + n_right * right_gini) / n_total
        return total_gini - weighted_gini


def _gini_impurity(y: np.ndarray) -> float:
    """
    Calculate Gini impurity.

    Parameters
    ----------
    y
        Class labels array.

    Returns
    -------
    float
        Gini impurity value.
    """
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities**2)
