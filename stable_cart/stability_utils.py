"""
Core stability utility functions implementing the 7 stability primitives.

These are the fundamental "atoms" of tree stability that can be composed
across different methods.
"""

import numpy as np
from typing import Tuple, List, Optional, Literal, Union
from dataclasses import dataclass
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split


@dataclass
class SplitCandidate:
    """Represents a potential split with all relevant information."""

    feature_idx: int
    threshold: float
    gain: float
    left_indices: np.ndarray
    right_indices: np.ndarray
    is_oblique: bool = False
    oblique_weights: Optional[np.ndarray] = None
    validation_score: Optional[float] = None
    variance_estimate: Optional[float] = None
    consensus_support: Optional[float] = None


@dataclass
class StabilityMetrics:
    """Container for stability diagnostic information."""

    prefix_consensus_scores: List[float]
    validation_consistency: float
    leaf_variance_estimates: List[float]
    split_margins: List[float]
    bootstrap_variance: Optional[float] = None


# ============================================================================
# 1. PREFIX STABILITY
# ============================================================================


def bootstrap_consensus_split(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 12,
    max_candidates: int = 20,
    threshold: float = 0.5,
    enable_quantile_binning: bool = True,
    max_bins: int = 24,
    random_state: Optional[int] = None,
) -> Tuple[Optional[SplitCandidate], List[SplitCandidate]]:
    """
    Find consensus split using bootstrap voting with quantile-binned thresholds.

    Returns
    -------
    best_split : SplitCandidate or None
        Consensus split if one achieves threshold support
    all_candidates : List[SplitCandidate]
        All evaluated candidates with their consensus scores
    """
    if len(X) < 10:  # Too few samples for meaningful consensus
        return None, []

    rng = np.random.RandomState(random_state)
    n_samples_bootstrap = max(len(X) // 2, 10)

    # Collect votes from bootstrap samples
    candidate_votes = {}  # (feature, binned_threshold) -> count

    for _ in range(n_samples):
        # Bootstrap sample
        bootstrap_idx = rng.choice(len(X), size=n_samples_bootstrap, replace=True)
        X_boot, y_boot = X[bootstrap_idx], y[bootstrap_idx]

        # Find best splits in this sample
        candidates = _find_candidate_splits(X_boot, y_boot, max_candidates)

        for candidate in candidates:
            # Bin the threshold if enabled
            if enable_quantile_binning:
                feature_values = X[:, candidate.feature_idx]
                binned_threshold = _bin_threshold(candidate.threshold, feature_values, max_bins)
            else:
                binned_threshold = candidate.threshold

            key = (candidate.feature_idx, binned_threshold)
            candidate_votes[key] = candidate_votes.get(key, 0) + 1

    if not candidate_votes:
        return None, []

    # Convert votes to candidates with consensus scores
    consensus_candidates = []
    for (feature_idx, threshold), votes in candidate_votes.items():
        consensus_score = votes / n_samples

        if consensus_score >= threshold:
            # Evaluate this consensus candidate on full data
            left_mask = X[:, feature_idx] <= threshold
            if np.sum(left_mask) > 0 and np.sum(~left_mask) > 0:
                gain = _evaluate_split_gain(y, left_mask)

                candidate = SplitCandidate(
                    feature_idx=feature_idx,
                    threshold=threshold,
                    gain=gain,
                    left_indices=np.where(left_mask)[0],
                    right_indices=np.where(~left_mask)[0],
                    consensus_support=consensus_score,
                )
                consensus_candidates.append(candidate)

    if not consensus_candidates:
        return None, []

    # Return best consensus candidate
    best_candidate = max(consensus_candidates, key=lambda c: c.gain)
    return best_candidate, consensus_candidates


def _bin_threshold(threshold: float, feature_values: np.ndarray, max_bins: int) -> float:
    """Bin threshold to quantile grid to reduce micro-jitter."""
    if len(np.unique(feature_values)) <= max_bins:
        return threshold

    quantiles = np.linspace(0, 1, max_bins + 1)
    bins = np.quantile(feature_values, quantiles)
    bins = np.unique(bins)  # Remove duplicates

    # Find closest bin
    closest_idx = np.argmin(np.abs(bins - threshold))
    return bins[closest_idx]


def enable_deterministic_tiebreaking(candidates: List[SplitCandidate]) -> List[SplitCandidate]:
    """Sort candidates deterministically to break ties consistently."""
    return sorted(
        candidates,
        key=lambda c: (
            -c.gain,  # Best gain first
            c.feature_idx,  # Consistent feature ordering
            c.threshold,  # Consistent threshold ordering
        ),
    )


def apply_margin_veto(
    candidates: List[SplitCandidate], margin_threshold: float = 0.03
) -> List[SplitCandidate]:
    """Veto splits where the margin between best candidates is too small."""
    if len(candidates) < 2:
        return candidates

    # Sort by gain
    sorted_candidates = sorted(candidates, key=lambda c: c.gain, reverse=True)
    best_gain = sorted_candidates[0].gain
    second_best_gain = sorted_candidates[1].gain if len(sorted_candidates) > 1 else 0

    margin = best_gain - second_best_gain
    relative_margin = margin / (best_gain + 1e-10)

    if relative_margin < margin_threshold:
        # Margin too small - return empty to trigger more careful evaluation
        return []

    return [sorted_candidates[0]]  # Return only clear winner


# ============================================================================
# 2. VALIDATION-CHECKED SPLIT SELECTION
# ============================================================================


def validation_checked_split_selection(
    X_split: np.ndarray,
    y_split: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    candidates: List[SplitCandidate],
    metric: Literal["median", "one_se", "variance_penalized"] = "variance_penalized",
    consistency_weight: float = 1.0,
    task: str = "regression",
) -> Optional[SplitCandidate]:
    """
    Evaluate split candidates on validation data and select based on consistency.
    """
    if not candidates:
        return None

    scored_candidates = []

    for candidate in candidates:
        # Apply split to validation data
        if candidate.is_oblique and candidate.oblique_weights is not None:
            val_score = X_val @ candidate.oblique_weights
            left_mask_val = val_score <= candidate.threshold
        else:
            left_mask_val = X_val[:, candidate.feature_idx] <= candidate.threshold

        # Evaluate on validation set
        val_score = _evaluate_split_performance(y_val, left_mask_val, task)
        candidate.validation_score = val_score
        scored_candidates.append(candidate)

    # Select based on metric
    if metric == "median":
        return _select_by_median_score(scored_candidates)
    elif metric == "one_se":
        return _select_by_one_se_rule(scored_candidates)
    elif metric == "variance_penalized":
        return _select_by_variance_penalty(scored_candidates, consistency_weight)
    else:
        raise ValueError(f"Unknown validation metric: {metric}")


def _evaluate_split_performance(y: np.ndarray, left_mask: np.ndarray, task: str) -> float:
    """Evaluate split performance on validation data."""
    if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
        return 0.0

    if task == "regression":
        # Use reduction in MSE
        total_var = np.var(y) if len(y) > 1 else 0
        left_var = np.var(y[left_mask]) if np.sum(left_mask) > 1 else 0
        right_var = np.var(y[~left_mask]) if np.sum(~left_mask) > 1 else 0

        weighted_var = (np.sum(left_mask) * left_var + np.sum(~left_mask) * right_var) / len(y)
        return total_var - weighted_var
    else:
        # Use reduction in Gini impurity
        total_gini = _gini_impurity(y)
        left_gini = _gini_impurity(y[left_mask]) if np.sum(left_mask) > 0 else 0
        right_gini = _gini_impurity(y[~left_mask]) if np.sum(~left_mask) > 0 else 0

        weighted_gini = (np.sum(left_mask) * left_gini + np.sum(~left_mask) * right_gini) / len(y)
        return total_gini - weighted_gini


def _select_by_variance_penalty(
    candidates: List[SplitCandidate], penalty_weight: float
) -> Optional[SplitCandidate]:
    """Select split using validation score minus variance penalty."""
    if not candidates:
        return None

    best_candidate = None
    best_score = -np.inf

    for candidate in candidates:
        if candidate.validation_score is None:
            continue

        # Estimate variance (placeholder - would use bootstrap in practice)
        variance_penalty = penalty_weight * 0.1  # Simplified
        penalized_score = candidate.validation_score - variance_penalty

        if penalized_score > best_score:
            best_score = penalized_score
            best_candidate = candidate

    return best_candidate


# ============================================================================
# 3. HONESTY (SPLIT vs EST)
# ============================================================================


def honest_data_partition(
    X: np.ndarray,
    y: np.ndarray,
    split_frac: float = 0.6,
    val_frac: float = 0.2,
    est_frac: float = 0.2,
    enable_stratification: bool = True,
    task: str = "regression",
    random_state: Optional[int] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    """
    Partition data into SPLIT/VAL/EST subsets with optional stratification.

    Returns
    -------
    (X_split, y_split), (X_val, y_val), (X_est, y_est)
    """
    assert abs(split_frac + val_frac + est_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    if enable_stratification and task == "regression":
        # Stratify by target quantiles for regression
        y_binned = _create_target_bins(y, n_bins=5)
        stratify = y_binned
    elif enable_stratification and task == "classification":
        stratify = y
    else:
        stratify = None

    # First split: SPLIT vs (VAL + EST)
    test_size = val_frac + est_frac
    X_split, X_temp, y_split, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )

    # Second split: VAL vs EST
    est_size_relative = est_frac / (val_frac + est_frac)

    if enable_stratification and task == "regression":
        temp_stratify = _create_target_bins(y_temp, n_bins=5)
    elif enable_stratification and task == "classification":
        temp_stratify = y_temp
    else:
        temp_stratify = None

    X_val, X_est, y_val, y_est = train_test_split(
        X_temp,
        y_temp,
        test_size=est_size_relative,
        stratify=temp_stratify,
        random_state=random_state,
    )

    return (X_split, y_split), (X_val, y_val), (X_est, y_est)


def _create_target_bins(y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Create stratification bins for regression targets using quantiles."""
    if len(np.unique(y)) <= n_bins:
        return y.astype(int)

    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y, quantiles)
    return np.digitize(y, bins[1:-1])  # Exclude first/last bins to avoid edge effects


# ============================================================================
# 4. LEAF STABILIZATION
# ============================================================================


def stabilize_leaf_estimate(
    y_est: np.ndarray,
    y_parent: np.ndarray,
    strategy: Literal["m_estimate", "shrink_to_parent", "beta_smoothing"] = "m_estimate",
    smoothing: float = 1.0,
    task: str = "regression",
    min_samples: int = 5,
) -> Union[float, np.ndarray]:
    """
    Stabilize leaf estimates using various smoothing strategies.
    """
    if len(y_est) == 0:
        # Fall back to parent estimate
        if task == "regression":
            return float(np.mean(y_parent)) if len(y_parent) > 0 else 0.0
        else:
            # Return uniform probabilities
            n_classes = len(np.unique(y_parent)) if len(y_parent) > 0 else 2
            return np.ones(n_classes) / n_classes

    if len(y_est) < min_samples and strategy != "shrink_to_parent":
        # Force shrinkage for very small leaves
        strategy = "shrink_to_parent"

    if task == "regression":
        return _stabilize_regression_leaf(y_est, y_parent, strategy, smoothing)
    else:
        return _stabilize_classification_leaf(y_est, y_parent, strategy, smoothing)


def _stabilize_regression_leaf(
    y_est: np.ndarray, y_parent: np.ndarray, strategy: str, smoothing: float
) -> float:
    """Stabilize regression leaf estimate."""
    leaf_mean = np.mean(y_est)
    parent_mean = np.mean(y_parent) if len(y_parent) > 0 else leaf_mean

    if strategy == "m_estimate":
        # M-estimate: weighted average with parent
        n = len(y_est)
        return (n * leaf_mean + smoothing * parent_mean) / (n + smoothing)
    elif strategy == "shrink_to_parent":
        # James-Stein style shrinkage
        shrinkage_factor = smoothing / (1 + smoothing)
        return (1 - shrinkage_factor) * leaf_mean + shrinkage_factor * parent_mean
    else:  # beta_smoothing - simplified for regression
        return _stabilize_regression_leaf(y_est, y_parent, "m_estimate", smoothing)


def _stabilize_classification_leaf(
    y_est: np.ndarray, y_parent: np.ndarray, strategy: str, smoothing: float
) -> np.ndarray:
    """Stabilize classification leaf probabilities."""
    unique_classes = (
        np.unique(np.concatenate([y_est, y_parent])) if len(y_parent) > 0 else np.unique(y_est)
    )
    n_classes = len(unique_classes)

    # Leaf counts
    leaf_counts = np.bincount(y_est.astype(int), minlength=n_classes)
    parent_counts = (
        np.bincount(y_parent.astype(int), minlength=n_classes) if len(y_parent) > 0 else leaf_counts
    )

    if strategy == "m_estimate":
        # M-estimate smoothing
        prior = (
            parent_counts / np.sum(parent_counts)
            if np.sum(parent_counts) > 0
            else np.ones(n_classes) / n_classes
        )
        smoothed_counts = leaf_counts + smoothing * prior * np.sum(leaf_counts)
        return smoothed_counts / np.sum(smoothed_counts)
    elif strategy == "beta_smoothing":
        # Beta-Binomial smoothing
        alpha = smoothing
        beta = smoothing
        return (leaf_counts + alpha) / (np.sum(leaf_counts) + alpha + beta)
    else:  # shrink_to_parent
        parent_probs = (
            parent_counts / np.sum(parent_counts)
            if np.sum(parent_counts) > 0
            else np.ones(n_classes) / n_classes
        )
        leaf_probs = leaf_counts / np.sum(leaf_counts) if np.sum(leaf_counts) > 0 else parent_probs
        shrinkage_factor = smoothing / (1 + smoothing)
        return (1 - shrinkage_factor) * leaf_probs + shrinkage_factor * parent_probs


# ============================================================================
# 5. DATA REGULARIZATION
# ============================================================================


def winsorize_features(
    X: np.ndarray,
    quantiles: Tuple[float, float] = (0.01, 0.99),
    fitted_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Winsorize features to reduce outlier influence.

    Returns
    -------
    X_winsorized : np.ndarray
        Winsorized features
    bounds : Tuple[np.ndarray, np.ndarray]
        (lower_bounds, upper_bounds) for future application
    """
    if fitted_bounds is not None:
        lower_bounds, upper_bounds = fitted_bounds
    else:
        lower_bounds = np.quantile(X, quantiles[0], axis=0)
        upper_bounds = np.quantile(X, quantiles[1], axis=0)

    X_winsorized = np.clip(X, lower_bounds, upper_bounds)
    return X_winsorized, (lower_bounds, upper_bounds)


# ============================================================================
# 6. CANDIDATE DIVERSITY WITH DETERMINISTIC RESOLUTION
# ============================================================================


def generate_oblique_candidates(
    X: np.ndarray,
    y: np.ndarray,
    strategy: Literal["lasso", "ridge", "elastic_net"] = "lasso",
    enable_correlation_gating: bool = True,
    min_correlation: float = 0.3,
    task: str = "regression",
    random_state: Optional[int] = None,
) -> List[SplitCandidate]:
    """
    Generate oblique split candidates using linear projections.
    """
    if X.shape[1] < 2:
        return []  # Need at least 2 features for oblique splits

    if enable_correlation_gating:
        # Check if features are correlated enough to justify oblique splits
        corr_matrix = np.corrcoef(X.T)
        max_corr = np.max(np.abs(corr_matrix - np.eye(X.shape[1])))
        if max_corr < min_correlation:
            return []  # Features not correlated enough

    try:
        if task == "regression":
            if strategy == "lasso":
                model = LassoCV(cv=3, random_state=random_state)
            elif strategy == "ridge":
                model = RidgeCV(cv=3)
            else:  # elastic_net
                model = ElasticNetCV(cv=3, random_state=random_state)

            model.fit(X, y)
            weights = model.coef_
        else:  # classification
            model = LogisticRegressionCV(
                cv=3, random_state=random_state, penalty="l1", solver="liblinear"
            )
            model.fit(X, y)
            weights = model.coef_[0] if model.coef_.ndim > 1 else model.coef_

        # Only proceed if we got non-trivial weights
        if np.sum(np.abs(weights) > 1e-6) < 2:
            return []

        # Create oblique split candidate
        projections = X @ weights

        # Try different threshold percentiles
        candidates = []
        for percentile in [25, 50, 75]:
            threshold = np.percentile(projections, percentile)
            left_mask = projections <= threshold

            if np.sum(left_mask) > 0 and np.sum(~left_mask) > 0:
                gain = _evaluate_split_gain(y, left_mask)

                candidate = SplitCandidate(
                    feature_idx=-1,  # Special marker for oblique
                    threshold=threshold,
                    gain=gain,
                    left_indices=np.where(left_mask)[0],
                    right_indices=np.where(~left_mask)[0],
                    is_oblique=True,
                    oblique_weights=weights,
                )
                candidates.append(candidate)

        return candidates

    except Exception:
        # Fallback gracefully if oblique fitting fails
        return []


def beam_search_splits(
    X: np.ndarray,
    y: np.ndarray,
    depth: int = 2,
    beam_width: int = 12,
    enable_ambiguity_gating: bool = True,
    ambiguity_threshold: float = 0.05,
    task: str = "regression",
) -> List[SplitCandidate]:
    """
    Use beam search to find splits with lookahead.
    """
    if len(X) < 20:  # Too small for meaningful beam search
        return _find_candidate_splits(X, y, max_candidates=beam_width)

    # Get initial candidates
    candidates = _find_candidate_splits(X, y, max_candidates=beam_width * 2)

    if enable_ambiguity_gating and len(candidates) >= 2:
        # Check if top candidates are ambiguous enough to justify beam search
        sorted_candidates = sorted(candidates, key=lambda c: c.gain, reverse=True)
        top_gain = sorted_candidates[0].gain
        second_gain = sorted_candidates[1].gain if len(sorted_candidates) > 1 else 0

        if top_gain > 0:
            relative_gap = (top_gain - second_gain) / top_gain
            if relative_gap > ambiguity_threshold:
                # Clear winner - no need for expensive beam search
                return [sorted_candidates[0]]

    # Perform beam search if we reach here
    return _perform_beam_search(X, y, candidates[:beam_width], depth, task)


def _perform_beam_search(
    X: np.ndarray, y: np.ndarray, initial_candidates: List[SplitCandidate], depth: int, task: str
) -> List[SplitCandidate]:
    """Simplified beam search implementation."""
    if depth <= 1:
        return initial_candidates

    # For now, return initial candidates with improved evaluation
    # Full beam search would recursively evaluate subsequent splits
    for candidate in initial_candidates:
        # Add lookahead score (simplified)
        lookahead_bonus = 0.1 * candidate.gain  # Placeholder for actual lookahead
        candidate.gain += lookahead_bonus

    return sorted(initial_candidates, key=lambda c: c.gain, reverse=True)


# ============================================================================
# 7. VARIANCE-AWARE STOPPING
# ============================================================================


def should_stop_splitting(
    current_gain: float,
    variance_estimate: float,
    variance_weight: float = 1.0,
    strategy: Literal["one_se", "variance_penalty", "both"] = "variance_penalty",
) -> bool:
    """
    Determine if splitting should stop based on variance-aware criteria.
    """
    if strategy == "variance_penalty":
        return current_gain < variance_weight * variance_estimate
    elif strategy == "one_se":
        # Simplified 1-SE rule
        return current_gain < variance_estimate  # Would use SE in practice
    else:  # both
        penalty_stop = current_gain < variance_weight * variance_estimate
        se_stop = current_gain < variance_estimate
        return penalty_stop or se_stop


def estimate_split_variance(
    X: np.ndarray,
    y: np.ndarray,
    split_candidate: SplitCandidate,
    n_bootstrap: int = 10,
    task: str = "regression",
    random_state: Optional[int] = None,
) -> float:
    """
    Estimate variance that would be introduced by this split.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[bootstrap_idx]
        y_boot = y[bootstrap_idx]

        # Apply the split to bootstrap sample
        if split_candidate.is_oblique and split_candidate.oblique_weights is not None:
            projections = X_boot @ split_candidate.oblique_weights
            left_mask = projections <= split_candidate.threshold
        else:
            left_mask = X_boot[:, split_candidate.feature_idx] <= split_candidate.threshold

        # Evaluate split on this bootstrap sample
        if np.sum(left_mask) > 0 and np.sum(~left_mask) > 0:
            score = _evaluate_split_gain(y_boot, left_mask)
            bootstrap_scores.append(score)

    if len(bootstrap_scores) < 2:
        return 0.0

    return float(np.var(bootstrap_scores, ddof=1))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _find_candidate_splits(
    X: np.ndarray, y: np.ndarray, max_candidates: int = 20
) -> List[SplitCandidate]:
    """Find basic axis-aligned split candidates."""
    candidates = []
    n_features = X.shape[1]

    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)

        if len(unique_values) < 2:
            continue

        # Try thresholds between unique values
        for i in range(min(len(unique_values) - 1, max_candidates // n_features)):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            left_mask = feature_values <= threshold

            if np.sum(left_mask) > 0 and np.sum(~left_mask) > 0:
                gain = _evaluate_split_gain(y, left_mask)

                candidate = SplitCandidate(
                    feature_idx=feature_idx,
                    threshold=threshold,
                    gain=gain,
                    left_indices=np.where(left_mask)[0],
                    right_indices=np.where(~left_mask)[0],
                )
                candidates.append(candidate)

    # Return top candidates
    candidates.sort(key=lambda c: c.gain, reverse=True)
    return candidates[:max_candidates]


def _evaluate_split_gain(y: np.ndarray, left_mask: np.ndarray) -> float:
    """Evaluate information gain from a split."""
    if len(y) == 0 or np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
        return 0.0

    # Determine if this looks like regression or classification
    if len(np.unique(y)) > 10 or y.dtype in [np.float32, np.float64]:
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
    """Calculate Gini impurity."""
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities**2)


def _select_by_median_score(candidates: List[SplitCandidate]) -> Optional[SplitCandidate]:
    """Select candidate with best median validation score."""
    if not candidates:
        return None

    scored = [c for c in candidates if c.validation_score is not None]
    if not scored:
        return None

    return max(scored, key=lambda c: c.validation_score)


def _select_by_one_se_rule(candidates: List[SplitCandidate]) -> Optional[SplitCandidate]:
    """Select using one-standard-error rule."""
    if not candidates:
        return None

    scored = [c for c in candidates if c.validation_score is not None]
    if not scored:
        return None

    scores = [c.validation_score for c in scored]
    best_score = max(scores)
    score_std = np.std(scores) if len(scores) > 1 else 0

    # Find simplest model within one SE of best
    threshold = best_score - score_std
    viable_candidates = [c for c in scored if c.validation_score >= threshold]

    # Return "simplest" (axis-aligned over oblique, lower feature index)
    return min(viable_candidates, key=lambda c: (c.is_oblique, c.feature_idx))
