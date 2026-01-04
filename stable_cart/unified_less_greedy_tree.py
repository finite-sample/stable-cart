"""
LessGreedyHybridTree: Enhanced with cross-method learning.

Now inherits from BaseStableTree and incorporates lessons from:
- RobustPrefixHonestTree: Winsorization, consensus for ambiguous splits, stratified sampling
- BootstrapVariancePenalizedTree: Explicit variance tracking
"""

from typing import Any, Literal

from .base_stable_tree import BaseStableTree


class LessGreedyHybridTree(BaseStableTree):
    """
    LessGreedyHybridTree with unified stability primitives.

    Enhanced with cross-method learning:
    - Winsorization (from RobustPrefix)
    - Bootstrap consensus for ambiguous splits (from RobustPrefix)
    - Stratified sampling (from RobustPrefix)
    - Explicit variance tracking (from Bootstrap)

    Core Features:
    - Honest data partitioning with lookahead beam search
    - Optional oblique root splits using regularized linear models
    - Leaf smoothing (shrinkage for regression, m-estimate for classification)
    - Advanced split selection with multiple strategies

    Parameters
    ----------
    task
        Prediction task type.
    max_depth
        Maximum tree depth.
    min_samples_split
        Minimum samples to split a node.
    min_samples_leaf
        Minimum samples per leaf.
    split_frac
        Fraction of data for structure building.
    val_frac
        Fraction of data for validation.
    est_frac
        Fraction of data for estimation.
    enable_stratified_sampling
        Enable stratified sampling in data partitioning.
    enable_oblique_splits
        Enable oblique split capability.
    oblique_strategy
        Strategy for oblique splits.
    oblique_regularization
        Regularization type for oblique splits.
    enable_correlation_gating
        Enable correlation-based feature gating.
    min_correlation_threshold
        Minimum correlation for feature selection.
    enable_lookahead
        Enable lookahead search.
    lookahead_depth
        Depth for lookahead search.
    beam_width
        Width of beam search.
    enable_ambiguity_gating
        Enable ambiguity-based gating.
    ambiguity_threshold
        Threshold for ambiguity detection.
    min_samples_for_lookahead
        Minimum samples required for lookahead.
    enable_robust_consensus_for_ambiguous
        Enable robust consensus for ambiguous splits.
    consensus_samples
        Number of samples for consensus.
    consensus_threshold
        Threshold for consensus decisions.
    enable_threshold_binning
        Enable threshold binning to reduce micro-jitter.
    max_threshold_bins
        Maximum number of threshold bins.
    enable_winsorization
        Enable feature winsorization.
    winsor_quantiles
        Quantile bounds for winsorization.
    enable_bootstrap_variance_tracking
        Enable bootstrap variance tracking.
    variance_tracking_samples
        Number of samples for variance tracking.
    enable_explicit_variance_penalty
        Enable explicit variance penalty.
    variance_penalty_weight
        Weight for variance penalty.
    leaf_smoothing
        Smoothing parameter for leaf estimates.
    leaf_smoothing_strategy
        Strategy for leaf smoothing.
    enable_gain_margin_logic
        Enable gain margin logic.
    margin_threshold
        Threshold for margin-based decisions.
    classification_criterion
        Criterion for classification splits.
    random_state
        Random state for reproducibility.
    """

    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        # === CORE TREE PARAMETERS ===
        max_depth: int = 5,
        min_samples_split: int = 40,
        min_samples_leaf: int = 20,
        # === HONEST PARTITIONING ===
        split_frac: float = 0.6,
        val_frac: float = 0.2,
        est_frac: float = 0.2,
        enable_stratified_sampling: bool = True,  # ENHANCED: from RobustPrefix
        # === OBLIQUE SPLITS ===
        enable_oblique_splits: bool = True,  # Signature feature
        oblique_strategy: Literal["root_only", "all_levels", "adaptive"] = "root_only",
        oblique_regularization: Literal["lasso", "ridge", "elastic_net"] = "lasso",
        enable_correlation_gating: bool = True,
        min_correlation_threshold: float = 0.3,
        # === LOOKAHEAD WITH BEAM SEARCH ===
        enable_lookahead: bool = True,  # Signature feature
        lookahead_depth: int = 2,
        beam_width: int = 12,
        enable_ambiguity_gating: bool = True,
        ambiguity_threshold: float = 0.05,
        min_samples_for_lookahead: int = 600,
        # === ENHANCED: CONSENSUS FOR AMBIGUOUS SPLITS (from RobustPrefix) ===
        enable_robust_consensus_for_ambiguous: bool = True,  # NEW
        consensus_samples: int = 12,
        consensus_threshold: float = 0.5,
        enable_threshold_binning: bool = True,  # NEW: reduce micro-jitter
        max_threshold_bins: int = 24,
        # === ENHANCED: OUTLIER ROBUSTNESS (from RobustPrefix) ===
        enable_winsorization: bool = True,  # NEW: robust preprocessing
        winsor_quantiles: tuple = (0.01, 0.99),
        # === ENHANCED: VARIANCE TRACKING (from Bootstrap) ===
        enable_bootstrap_variance_tracking: bool = True,  # NEW: diagnostic
        variance_tracking_samples: int = 10,
        enable_explicit_variance_penalty: bool = False,  # Optional enhancement
        variance_penalty_weight: float = 0.1,
        # === LEAF STABILIZATION ===
        leaf_smoothing: float = 0.0,  # Conservative default for LessGreedy
        leaf_smoothing_strategy: Literal[
            "m_estimate", "shrink_to_parent"
        ] = "shrink_to_parent",
        # === MARGIN-BASED LOGIC ===
        enable_gain_margin_logic: bool = True,  # Signature feature
        margin_threshold: float = 0.03,
        # === CLASSIFICATION ===
        classification_criterion: Literal["gini", "entropy"] = "gini",
        random_state: int | None = None,
    ):
        # Configure defaults that reflect LessGreedy's personality
        super().__init__(
            task=task,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            # Honest partitioning - core feature
            enable_honest_estimation=True,
            split_frac=split_frac,
            val_frac=val_frac,
            est_frac=est_frac,
            enable_stratified_sampling=enable_stratified_sampling,
            # Validation checking - always enabled
            enable_validation_checking=True,
            validation_metric="variance_penalized",
            # ENHANCED: Outlier robustness (from RobustPrefix)
            enable_winsorization=enable_winsorization,
            winsor_quantiles=winsor_quantiles,
            # Oblique splits - signature feature
            enable_oblique_splits=enable_oblique_splits,
            oblique_strategy=oblique_strategy,
            oblique_regularization=oblique_regularization,
            enable_correlation_gating=enable_correlation_gating,
            min_correlation_threshold=min_correlation_threshold,
            # Lookahead - signature feature
            enable_lookahead=enable_lookahead,
            lookahead_depth=lookahead_depth,
            beam_width=beam_width,
            enable_ambiguity_gating=enable_ambiguity_gating,
            ambiguity_threshold=ambiguity_threshold,
            min_samples_for_lookahead=min_samples_for_lookahead,
            # ENHANCED: Consensus for ambiguous splits (from RobustPrefix)
            enable_robust_consensus_for_ambiguous=enable_robust_consensus_for_ambiguous,
            enable_prefix_consensus=enable_robust_consensus_for_ambiguous,  # For ambiguous splits
            consensus_samples=consensus_samples,
            consensus_threshold=consensus_threshold,
            enable_threshold_binning=enable_threshold_binning,
            max_threshold_bins=max_threshold_bins,
            # Margin logic - signature feature
            enable_margin_vetoes=enable_gain_margin_logic,
            margin_threshold=margin_threshold,
            # ENHANCED: Variance tracking (from Bootstrap)
            enable_bootstrap_variance_tracking=enable_bootstrap_variance_tracking,
            variance_tracking_samples=variance_tracking_samples,
            enable_explicit_variance_penalty=enable_explicit_variance_penalty,
            variance_penalty_weight=variance_penalty_weight,
            # Leaf stabilization - conservative for accuracy
            leaf_smoothing=leaf_smoothing,
            leaf_smoothing_strategy=leaf_smoothing_strategy,
            # Classification
            classification_criterion=classification_criterion,
            # Focus on balanced accuracy + stability
            algorithm_focus="accuracy",
            random_state=random_state,
        )

        # Store LessGreedy-specific parameters
        self.inner_k = 1  # Simplified for unified version
        self.oblique_cv = 5  # Fixed for unified version

        # Cross-method enhancement flags
        self.enable_robust_consensus_for_ambiguous = (
            enable_robust_consensus_for_ambiguous
        )
        self.enable_bootstrap_variance_tracking = enable_bootstrap_variance_tracking
        self.enable_explicit_variance_penalty = enable_explicit_variance_penalty

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for sklearn compatibility.

        Parameters
        ----------
        deep
            Whether to return deep parameter copy.

        Returns
        -------
        dict[str, Any]
            Parameter dictionary.
        """
        # Use the parent method which gets constructor parameters
        return super().get_params(deep=deep)

    def set_params(self, **params: Any) -> "LessGreedyHybridTree":
        """
        Set parameters for sklearn compatibility.

        Parameters
        ----------
        **params
            Parameter values to set.

        Returns
        -------
        LessGreedyHybridTree
            Self with updated parameters.
        """
        return super().set_params(**params)
