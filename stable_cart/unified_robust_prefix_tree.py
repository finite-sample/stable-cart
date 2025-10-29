"""
RobustPrefixHonestTree: Enhanced with cross-method learning.

Now inherits from BaseStableTree and incorporates lessons from:
- LessGreedyHybridTree: Oblique splits, lookahead with beam search, ambiguity gating, correlation gating
- BootstrapVariancePenalizedTree: Explicit variance tracking
"""

import numpy as np
from typing import Literal, Optional, Tuple
from .base_stable_tree import BaseStableTree


class RobustPrefixHonestTree(BaseStableTree):
    """
    Robust prefix honest tree with unified stability primitives.

    Enhanced with cross-method learning:
    - Oblique splits (from LessGreedy): Add Lasso-based oblique splits to the locked prefix
    - Lookahead with beam search (from LessGreedy): Replace depth-1 stumps with k-step lookahead
    - Ambiguity gating (from LessGreedy): Only apply expensive consensus when splits are actually ambiguous
    - Correlation gating (from LessGreedy): Check if features are correlated before attempting oblique splits
    - Explicit variance tracking (from Bootstrap): Monitor prediction variance as diagnostic

    Core Features:
    - Robust consensus-based prefix splits with honest leaf estimation
    - Winsorization for outlier robustness
    - Stratified honest data partitioning
    - Advanced consensus mechanisms with threshold binning
    """

    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        # === CORE TREE PARAMETERS ===
        max_depth: int = 6,
        min_samples_leaf: int = 2,  # More aggressive for RobustPrefix style
        # === ROBUST PREFIX CONSENSUS ===
        top_levels: int = 2,  # Signature feature: how many levels to make robust
        consensus_samples: int = 12,  # Signature feature: bootstrap samples for consensus
        consensus_threshold: float = 0.5,
        consensus_subsample_frac: float = 0.8,
        # === HONEST PARTITIONING ===
        val_frac: float = 0.2,
        est_frac: float = 0.4,  # Larger estimation set for robust leaves
        enable_stratified_sampling: bool = True,  # Signature feature
        # === OUTLIER ROBUSTNESS ===
        enable_winsorization: bool = True,  # Signature feature
        winsor_quantiles: Tuple[float, float] = (0.01, 0.99),
        # === THRESHOLD DISCRETIZATION ===
        enable_threshold_binning: bool = True,  # Signature feature: reduce micro-jitter
        max_threshold_bins: int = 24,
        # === ENHANCED: OBLIQUE SPLITS (from LessGreedy) ===
        enable_oblique_splits: bool = True,  # NEW: add to locked prefix
        oblique_strategy: Literal[
            "root_only", "all_levels", "adaptive"
        ] = "root_only",  # Conservative for robust method
        oblique_regularization: Literal["lasso", "ridge", "elastic_net"] = "lasso",
        enable_correlation_gating: bool = True,  # NEW: from LessGreedy
        min_correlation_threshold: float = 0.3,
        # === ENHANCED: LOOKAHEAD WITH BEAM SEARCH (from LessGreedy) ===
        enable_lookahead: bool = True,  # NEW: replace depth-1 stumps in consensus
        lookahead_depth: int = 2,  # More informed prefix decisions
        beam_width: int = 12,
        enable_beam_search_for_consensus: bool = True,  # NEW: enhanced consensus
        # === ENHANCED: AMBIGUITY GATING (from LessGreedy) ===
        enable_ambiguity_gating: bool = True,  # NEW: only apply expensive consensus when needed
        ambiguity_threshold: float = 0.05,  # Similar to gain-margin logic
        enable_gain_margin_logic: bool = True,
        margin_threshold: float = 0.03,
        # === ENHANCED: VARIANCE TRACKING (from Bootstrap) ===
        enable_bootstrap_variance_tracking: bool = True,  # NEW: diagnostic
        variance_tracking_samples: int = 10,
        enable_explicit_variance_penalty: bool = False,  # Optional enhancement
        variance_penalty_weight: float = 0.1,
        # === LEAF STABILIZATION ===
        smoothing: float = 1.0,  # m-estimate smoothing
        leaf_smoothing_strategy: Literal["m_estimate", "shrink_to_parent"] = "m_estimate",
        # === CLASSIFICATION ===
        classification_criterion: Literal["gini", "entropy"] = "gini",
        random_state: Optional[int] = None,
    ):
        # Compute split_frac from val_frac and est_frac
        split_frac = 1.0 - val_frac - est_frac

        # Configure defaults that reflect RobustPrefix's personality
        super().__init__(
            task=task,
            max_depth=max_depth,
            min_samples_split=min_samples_leaf * 2,  # Derive from min_samples_leaf
            min_samples_leaf=min_samples_leaf,
            # Honest partitioning - core feature
            enable_honest_estimation=True,
            split_frac=split_frac,
            val_frac=val_frac,
            est_frac=est_frac,
            enable_stratified_sampling=enable_stratified_sampling,
            # Validation checking - always enabled
            enable_validation_checking=True,
            validation_metric="median",  # Robust approach
            # Robust prefix consensus - signature feature
            enable_prefix_consensus=True,
            prefix_levels=top_levels,
            consensus_samples=consensus_samples,
            consensus_threshold=consensus_threshold,
            # Outlier robustness - signature feature
            enable_winsorization=enable_winsorization,
            winsor_quantiles=winsor_quantiles,
            # Threshold discretization - signature feature
            enable_threshold_binning=enable_threshold_binning,
            enable_quantile_grid_thresholds=enable_threshold_binning,
            max_threshold_bins=max_threshold_bins,
            # ENHANCED: Oblique splits (from LessGreedy)
            enable_oblique_splits=enable_oblique_splits,
            oblique_strategy=oblique_strategy,
            oblique_regularization=oblique_regularization,
            enable_correlation_gating=enable_correlation_gating,
            min_correlation_threshold=min_correlation_threshold,
            # ENHANCED: Lookahead (from LessGreedy)
            enable_lookahead=enable_lookahead,
            lookahead_depth=lookahead_depth,
            beam_width=beam_width,
            enable_beam_search_for_consensus=enable_beam_search_for_consensus,
            # ENHANCED: Ambiguity gating (from LessGreedy)
            enable_ambiguity_gating=enable_ambiguity_gating,
            ambiguity_threshold=ambiguity_threshold,
            enable_gain_margin_logic=enable_gain_margin_logic,
            enable_margin_vetoes=enable_gain_margin_logic,
            margin_threshold=margin_threshold,
            # ENHANCED: Variance tracking (from Bootstrap)
            enable_bootstrap_variance_tracking=enable_bootstrap_variance_tracking,
            variance_tracking_samples=variance_tracking_samples,
            enable_explicit_variance_penalty=enable_explicit_variance_penalty,
            variance_penalty_weight=variance_penalty_weight,
            # Deterministic processing - signature feature
            enable_deterministic_preprocessing=True,
            enable_deterministic_tiebreaks=True,
            # Leaf stabilization - signature feature
            leaf_smoothing=smoothing,
            leaf_smoothing_strategy=leaf_smoothing_strategy,
            enable_calibrated_smoothing=True,
            # Classification
            classification_criterion=classification_criterion,
            # Focus on maximum stability
            algorithm_focus="stability",
            random_state=random_state,
        )

        # Store RobustPrefix-specific parameters for backwards compatibility
        self.top_levels = top_levels
        self.smoothing = smoothing
        self.consensus_B = consensus_samples
        self.consensus_subsample_frac = consensus_subsample_frac
        self.consensus_max_bins = max_threshold_bins

        # Cross-method enhancement flags
        self.enable_beam_search_for_consensus = enable_beam_search_for_consensus
        self.enable_bootstrap_variance_tracking = enable_bootstrap_variance_tracking
        self.enable_explicit_variance_penalty = enable_explicit_variance_penalty

    def fit(self, X, y):
        """Fit with robust prefix consensus."""
        # Validate for binary classification only
        if self.task == "classification":
            unique_classes = np.unique(y)
            if len(unique_classes) > 2:
                raise ValueError(
                    "Multi-class classification not yet supported. "
                    "RobustPrefixHonestTree currently supports binary classification only."
                )

        return super().fit(X, y)

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return super().get_params(deep=deep)

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        return super().set_params(**params)


# Create the backwards-compatible aliases
RobustPrefixHonestRegressor = RobustPrefixHonestTree  # Will need task='regression'
RobustPrefixHonestClassifier = RobustPrefixHonestTree  # Will need task='classification'
