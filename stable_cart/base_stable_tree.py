"""
BaseStableTree: Unified implementation of all stability primitives.

This base class implements the 7 core stability "atoms" that can be composed
across different tree methods. Each method can inherit from this and configure
different defaults to maintain their distinct personalities.
"""

import time
from typing import Optional, Tuple, Literal
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils.validation import check_X_y, check_array

from .stability_utils import (
    honest_data_partition,
    winsorize_features,
    stabilize_leaf_estimate,
)
from .split_strategies import HybridStrategy, create_split_strategy


class BaseStableTree(BaseEstimator):
    """
    Unified base class implementing all 7 stability primitives.

    The 7 stability primitives are:
    1. Prefix stability (robust consensus on early splits)
    2. Validation-checked split selection
    3. Honesty (separate data for structure vs estimation)
    4. Leaf stabilization (shrinkage/smoothing)
    5. Data regularization (winsorization, etc.)
    6. Candidate diversity with deterministic resolution
    7. Variance-aware stopping

    All tree methods inherit from this and configure different defaults
    to maintain their distinct personalities while sharing the unified
    stability infrastructure.
    """

    def __init__(
        self,
        # === TASK AND CORE PARAMETERS ===
        task: Literal["regression", "classification"] = "regression",
        max_depth: int = 5,
        min_samples_split: int = 40,
        min_samples_leaf: int = 20,
        # === 3. HONESTY - Data Partitioning ===
        enable_honest_estimation: bool = True,
        split_frac: float = 0.6,
        val_frac: float = 0.2,
        est_frac: float = 0.2,
        enable_stratified_sampling: bool = True,
        # === 2. VALIDATION-CHECKED SPLIT SELECTION ===
        enable_validation_checking: bool = True,
        validation_metric: Literal["median", "one_se", "variance_penalized"] = "variance_penalized",
        validation_consistency_weight: float = 1.0,
        # === 1. PREFIX STABILITY ===
        enable_prefix_consensus: bool = False,
        prefix_levels: int = 2,
        consensus_samples: int = 12,
        consensus_threshold: float = 0.5,
        enable_quantile_grid_thresholds: bool = False,
        max_threshold_bins: int = 24,
        # === 4. LEAF STABILIZATION ===
        leaf_smoothing: float = 0.0,
        leaf_smoothing_strategy: Literal[
            "m_estimate", "shrink_to_parent", "beta_smoothing"
        ] = "m_estimate",
        enable_calibrated_smoothing: bool = False,
        min_leaf_samples_for_stability: int = 5,
        # === 5. DATA REGULARIZATION ===
        enable_winsorization: bool = False,
        winsor_quantiles: Tuple[float, float] = (0.01, 0.99),
        enable_feature_standardization: bool = False,
        # === 6. CANDIDATE DIVERSITY ===
        enable_oblique_splits: bool = False,
        oblique_strategy: Literal["root_only", "all_levels", "adaptive"] = "root_only",
        oblique_regularization: Literal["lasso", "ridge", "elastic_net"] = "lasso",
        enable_correlation_gating: bool = True,
        min_correlation_threshold: float = 0.3,
        enable_lookahead: bool = False,
        lookahead_depth: int = 1,
        beam_width: int = 8,
        enable_ambiguity_gating: bool = True,
        ambiguity_threshold: float = 0.05,
        min_samples_for_lookahead: int = 100,
        enable_deterministic_preprocessing: bool = False,
        enable_deterministic_tiebreaks: bool = True,
        enable_margin_vetoes: bool = False,
        margin_threshold: float = 0.03,
        # === 7. VARIANCE-AWARE STOPPING ===
        enable_variance_aware_stopping: bool = False,
        variance_stopping_weight: float = 1.0,
        variance_stopping_strategy: Literal[
            "one_se", "variance_penalty", "both"
        ] = "variance_penalty",
        enable_bootstrap_variance_tracking: bool = False,
        variance_tracking_samples: int = 10,
        enable_explicit_variance_penalty: bool = False,
        variance_penalty_weight: float = 0.1,
        # === ADVANCED CONFIGURATION ===
        split_strategy: Optional[str] = None,
        algorithm_focus: Literal["speed", "accuracy", "stability"] = "stability",
        # === CLASSIFICATION ===
        classification_criterion: Literal["gini", "entropy"] = "gini",
        # === OTHER ===
        random_state: Optional[int] = None,
        # === ADDITIONAL PARAMETERS FOR CROSS-METHOD LEARNING ===
        enable_threshold_binning: bool = False,
        enable_gain_margin_logic: bool = False,
        enable_beam_search_for_consensus: bool = False,
        enable_robust_consensus_for_ambiguous: bool = False,
    ):
        # Validate fractions sum to 1
        if abs(split_frac + val_frac + est_frac - 1.0) > 1e-6:
            raise ValueError("split_frac + val_frac + est_frac must sum to 1.0")

        # === CORE PARAMETERS ===
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        # === 3. HONESTY ===
        self.enable_honest_estimation = enable_honest_estimation
        self.split_frac = split_frac
        self.val_frac = val_frac
        self.est_frac = est_frac
        self.enable_stratified_sampling = enable_stratified_sampling

        # === 2. VALIDATION ===
        self.enable_validation_checking = enable_validation_checking
        self.validation_metric = validation_metric
        self.validation_consistency_weight = validation_consistency_weight

        # === 1. PREFIX STABILITY ===
        self.enable_prefix_consensus = enable_prefix_consensus
        self.prefix_levels = prefix_levels
        self.consensus_samples = consensus_samples
        self.consensus_threshold = consensus_threshold
        self.enable_quantile_grid_thresholds = enable_quantile_grid_thresholds
        self.max_threshold_bins = max_threshold_bins

        # === 4. LEAF STABILIZATION ===
        self.leaf_smoothing = leaf_smoothing
        self.leaf_smoothing_strategy = leaf_smoothing_strategy
        self.enable_calibrated_smoothing = enable_calibrated_smoothing
        self.min_leaf_samples_for_stability = min_leaf_samples_for_stability

        # === 5. DATA REGULARIZATION ===
        self.enable_winsorization = enable_winsorization
        self.winsor_quantiles = winsor_quantiles
        self.enable_feature_standardization = enable_feature_standardization

        # === 6. CANDIDATE DIVERSITY ===
        self.enable_oblique_splits = enable_oblique_splits
        self.oblique_strategy = oblique_strategy
        self.oblique_regularization = oblique_regularization
        self.enable_correlation_gating = enable_correlation_gating
        self.min_correlation_threshold = min_correlation_threshold

        self.enable_lookahead = enable_lookahead
        self.lookahead_depth = lookahead_depth
        self.beam_width = beam_width
        self.enable_ambiguity_gating = enable_ambiguity_gating
        self.ambiguity_threshold = ambiguity_threshold
        self.min_samples_for_lookahead = min_samples_for_lookahead

        self.enable_deterministic_preprocessing = enable_deterministic_preprocessing
        self.enable_deterministic_tiebreaks = enable_deterministic_tiebreaks
        self.enable_margin_vetoes = enable_margin_vetoes
        self.margin_threshold = margin_threshold

        # === 7. VARIANCE-AWARE STOPPING ===
        self.enable_variance_aware_stopping = enable_variance_aware_stopping
        self.variance_stopping_weight = variance_stopping_weight
        self.variance_stopping_strategy = variance_stopping_strategy
        self.enable_bootstrap_variance_tracking = enable_bootstrap_variance_tracking
        self.variance_tracking_samples = variance_tracking_samples
        self.enable_explicit_variance_penalty = enable_explicit_variance_penalty
        self.variance_penalty_weight = variance_penalty_weight

        # === ADVANCED ===
        self.split_strategy = split_strategy
        self.algorithm_focus = algorithm_focus

        # === CLASSIFICATION ===
        self.classification_criterion = classification_criterion

        # === OTHER ===
        self.random_state = random_state

        # === CROSS-METHOD LEARNING ===
        self.enable_threshold_binning = enable_threshold_binning
        self.enable_gain_margin_logic = enable_gain_margin_logic
        self.enable_beam_search_for_consensus = enable_beam_search_for_consensus
        self.enable_robust_consensus_for_ambiguous = enable_robust_consensus_for_ambiguous

        # Initialize fitted attributes
        self.tree_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.fit_time_sec_ = None
        self._split_strategy_ = None
        self._winsor_bounds_ = None
        self._global_prior_ = None

    def fit(self, X, y):
        """Fit the stable tree to the training data."""
        start_time = time.time()

        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False)

        # === 1. TASK SETUP ===
        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            if self.n_classes_ > 2:
                raise ValueError(
                    "Multi-class classification not yet supported. "
                    "Use binary classification or regression."
                )

            # Convert to 0/1 for binary classification
            y = (y == self.classes_[1]).astype(int)
            self._global_prior_ = np.mean(y)
        else:
            self.classes_ = None
            self.n_classes_ = None
            self._global_prior_ = np.mean(y) if len(y) > 0 else 0.0

        # === 5. DATA REGULARIZATION ===
        X_processed = self._preprocess_features(X)

        # === 3. HONESTY - Data Partitioning ===
        data_splits = self._partition_data(X_processed, y)
        (X_split, y_split), (X_val, y_val), (X_est, y_est) = data_splits

        # === Configure Split Strategy ===
        self._split_strategy_ = self._create_split_strategy()

        # === Build Tree Structure ===
        self.tree_ = self._build_tree(X_split, y_split, X_val, y_val, X_est, y_est, depth=0)

        # Record timing and diagnostics
        self.fit_time_sec_ = time.time() - start_time

        return self

    def predict(self, X):
        """Predict targets for samples in X."""
        check_array(X, accept_sparse=False)

        if self.tree_ is None:
            raise ValueError("Tree not fitted yet")

        # Apply same preprocessing as training
        X_processed = self._preprocess_features(X, fitted=True)

        predictions = np.array([self._predict_sample(x, self.tree_) for x in X_processed])

        if self.task == "classification":
            # Convert back to original class labels
            return np.where(predictions > 0.5, self.classes_[1], self.classes_[0])
        else:
            return predictions

    def predict_proba(self, X):
        """Predict class probabilities for classification tasks."""
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification tasks")

        check_array(X, accept_sparse=False)

        if self.tree_ is None:
            raise ValueError("Tree not fitted yet")

        # Apply same preprocessing as training
        X_processed = self._preprocess_features(X, fitted=True)

        # Get probability of positive class
        proba_positive = np.array([self._predict_sample(x, self.tree_) for x in X_processed])

        # Return as [P(class=0), P(class=1)]
        proba_negative = 1 - proba_positive
        return np.column_stack([proba_negative, proba_positive])

    def score(self, X, y):
        """Return the mean accuracy (classification) or R² (regression)."""
        y_pred = self.predict(X)

        if self.task == "regression":
            return r2_score(y, y_pred)
        else:
            return accuracy_score(y, y_pred)

    def count_leaves(self):
        """Count the number of leaf nodes in the tree."""
        if self.tree_ is None:
            return 0
        return self._count_leaves_recursive(self.tree_)

    def _count_leaves_recursive(self, node):
        """Recursively count leaves."""
        if node["type"] == "leaf":
            return 1
        else:
            left_count = self._count_leaves_recursive(node["left"]) if "left" in node else 0
            right_count = self._count_leaves_recursive(node["right"]) if "right" in node else 0
            return left_count + right_count

    # ========================================================================
    # INTERNAL METHODS - STABILITY PRIMITIVES
    # ========================================================================

    def _preprocess_features(self, X, fitted=False):
        """Apply data regularization preprocessing."""
        X_processed = X.copy()

        # === 5. DATA REGULARIZATION ===
        if self.enable_winsorization:
            if fitted and self._winsor_bounds_ is not None:
                X_processed, _ = winsorize_features(X_processed, fitted_bounds=self._winsor_bounds_)
            else:
                X_processed, self._winsor_bounds_ = winsorize_features(
                    X_processed, self.winsor_quantiles
                )

        # Feature standardization (rarely needed for trees)
        if self.enable_feature_standardization:
            # Would implement standardization here
            pass

        return X_processed

    def _partition_data(self, X, y):
        """Partition data using honest splitting."""
        if not self.enable_honest_estimation:
            # Use all data for both structure and estimation
            return (X, y), (X, y), (X, y)

        return honest_data_partition(
            X,
            y,
            split_frac=self.split_frac,
            val_frac=self.val_frac,
            est_frac=self.est_frac,
            enable_stratification=self.enable_stratified_sampling,
            task=self.task,
            random_state=self.random_state,
        )

    def _create_split_strategy(self):
        """Create the split strategy based on enabled features."""
        if self.split_strategy is not None:
            # Explicit strategy specified
            return create_split_strategy(
                self.split_strategy,
                task=self.task,
                random_state=self.random_state,
                # Pass relevant parameters
                oblique_regularization=self.oblique_regularization,
                enable_correlation_gating=self.enable_correlation_gating,
                min_correlation=self.min_correlation_threshold,
                consensus_samples=self.consensus_samples,
                consensus_threshold=self.consensus_threshold,
                lookahead_depth=self.lookahead_depth,
                beam_width=self.beam_width,
                variance_penalty_weight=self.variance_penalty_weight,
            )
        else:
            # Auto-select based on enabled features and algorithm focus
            return HybridStrategy(
                focus=self.algorithm_focus, task=self.task, random_state=self.random_state
            )

    def _build_tree(self, X_split, y_split, X_val, y_val, X_est, y_est, depth=0):
        """Recursively build the tree structure."""
        n_samples = len(X_split)

        # Base stopping conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y_split)) <= 1
        ):
            return self._make_leaf(y_est, y_split, depth)

        # Find best split using configured strategy
        best_split = self._split_strategy_.find_best_split(
            X_split,
            y_split,
            X_val if self.enable_validation_checking else None,
            y_val if self.enable_validation_checking else None,
            depth=depth,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )

        if best_split is None:
            return self._make_leaf(y_est, y_split, depth)

        # === 7. VARIANCE-AWARE STOPPING ===
        if self.enable_variance_aware_stopping and best_split.variance_estimate is not None:
            should_stop = self._split_strategy_.should_stop(
                X_split,
                y_split,
                best_split.gain,
                depth,
                variance_estimate=best_split.variance_estimate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            if should_stop:
                return self._make_leaf(y_est, y_split, depth)

        # Apply split to all data partitions
        left_indices_split, right_indices_split = self._apply_split_to_data(X_split, best_split)
        left_indices_val, right_indices_val = self._apply_split_to_data(X_val, best_split)
        left_indices_est, right_indices_est = self._apply_split_to_data(X_est, best_split)

        # Check minimum leaf size
        if (
            len(left_indices_split) < self.min_samples_leaf
            or len(right_indices_split) < self.min_samples_leaf
        ):
            return self._make_leaf(y_est, y_split, depth)

        # Recursively build children
        left_child = self._build_tree(
            X_split[left_indices_split],
            y_split[left_indices_split],
            X_val[left_indices_val],
            y_val[left_indices_val],
            X_est[left_indices_est],
            y_est[left_indices_est],
            depth + 1,
        )

        right_child = self._build_tree(
            X_split[right_indices_split],
            y_split[right_indices_split],
            X_val[right_indices_val],
            y_val[right_indices_val],
            X_est[right_indices_est],
            y_est[right_indices_est],
            depth + 1,
        )

        # Create internal node
        return {
            "type": "split_oblique" if best_split.is_oblique else "split",
            "feature_idx": best_split.feature_idx,
            "threshold": best_split.threshold,
            "gain": best_split.gain,
            "depth": depth,
            "n_samples_split": len(X_split),
            "n_samples_val": len(X_val),
            "n_samples_est": len(X_est),
            "oblique_weights": best_split.oblique_weights if best_split.is_oblique else None,
            "consensus_support": getattr(best_split, "consensus_support", None),
            "variance_estimate": getattr(best_split, "variance_estimate", None),
            "left": left_child,
            "right": right_child,
        }

    def _apply_split_to_data(self, X, split_candidate):
        """Apply a split to data and return left/right indices."""
        if split_candidate.is_oblique and split_candidate.oblique_weights is not None:
            projections = X @ split_candidate.oblique_weights
            left_mask = projections <= split_candidate.threshold
        else:
            left_mask = X[:, split_candidate.feature_idx] <= split_candidate.threshold

        left_indices = np.where(left_mask)[0]
        right_indices = np.where(~left_mask)[0]
        return left_indices, right_indices

    def _make_leaf(self, y_est, y_split, depth):
        """Create a leaf node with stabilized estimates."""
        # === 4. LEAF STABILIZATION ===
        if len(y_est) == 0:
            y_est = y_split  # Fallback to split data

        # Get parent data for shrinkage (use split data as proxy)
        stabilized_value = stabilize_leaf_estimate(
            y_est,
            y_split,
            strategy=self.leaf_smoothing_strategy,
            smoothing=self.leaf_smoothing,
            task=self.task,
            min_samples=self.min_leaf_samples_for_stability,
        )

        if self.task == "regression":
            return {
                "type": "leaf",
                "value": stabilized_value,
                "depth": depth,
                "n_samples_split": len(y_split),
                "n_samples_est": len(y_est),
            }
        else:
            # For classification, stabilized_value is probability array or scalar
            if isinstance(stabilized_value, (float, int)):
                prob = stabilized_value
            else:
                # stabilized_value is an array of class probabilities
                if len(stabilized_value) >= 2:
                    prob = stabilized_value[1]  # P(class=1) for binary classification
                else:
                    # Only one class present, assume class 0
                    prob = 0.0
            return {
                "type": "leaf",
                "proba": float(prob),
                "depth": depth,
                "n_samples_split": len(y_split),
                "n_samples_est": len(y_est),
            }

    def _predict_sample(self, x, node):
        """Predict a single sample by traversing the tree."""
        if node["type"] == "leaf":
            if self.task == "regression":
                return node["value"]
            else:
                return node["proba"]

        # Apply split
        if node["type"] == "split_oblique" and node["oblique_weights"] is not None:
            projection = x @ node["oblique_weights"]
            go_left = projection <= node["threshold"]
        else:
            go_left = x[node["feature_idx"]] <= node["threshold"]

        # Recurse
        if go_left:
            return self._predict_sample(x, node["left"])
        else:
            return self._predict_sample(x, node["right"])
