"""
BootstrapVariancePenalizedTree: Enhanced with cross-method learning.

Now inherits from BaseStableTree and incorporates lessons from:
- RobustPrefixHonestTree: Stratified bootstraps, winsorization, threshold binning, robust consensus
- LessGreedyHybridTree: Oblique splits, lookahead, beam search
"""

from typing import Any, ClassVar, Literal

import numpy as np

from .base_stable_tree import BaseStableTree


class BootstrapVariancePenalizedTree(BaseStableTree):
    """
    Bootstrap variance penalized tree with unified stability primitives.

    Enhanced with cross-method learning:
    - Stratified bootstraps (from RobustPrefix)
    - Winsorization (from RobustPrefix)
    - Threshold binning/bucketing (from RobustPrefix)
    - Robust consensus mechanism (from RobustPrefix)
    - Oblique splits (from LessGreedy)
    - Lookahead (from LessGreedy)
    - Beam search (from LessGreedy)

    Core Features:
    - Explicit bootstrap variance penalty during split selection
    - Honest data partitioning for unbiased estimation
    - Advanced split strategies with variance awareness

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
    variance_penalty
        Weight for bootstrap variance penalty.
    n_bootstrap
        Number of bootstrap samples for variance estimation.
    enable_variance_aware_stopping
        Enable variance-aware stopping criteria.
    split_frac
        Fraction of data for structure building.
    val_frac
        Fraction of data for validation.
    est_frac
        Fraction of data for estimation.
    enable_stratified_sampling
        Enable stratified sampling in data partitioning.
    enable_winsorization
        Enable feature winsorization before bootstrap sampling.
    winsor_quantiles
        Quantile bounds for winsorization.
    enable_threshold_binning
        Enable threshold binning to reduce micro-jitter.
    max_threshold_bins
        Maximum number of threshold bins.
    enable_robust_consensus
        Enable robust consensus mechanism.
    consensus_samples
        Number of samples for consensus.
    consensus_threshold
        Threshold for consensus decisions.
    enable_oblique_splits
        Enable oblique split capability.
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
    leaf_smoothing
        Smoothing parameter for leaf estimates.
    leaf_smoothing_strategy
        Strategy for leaf smoothing.
    enable_gain_margin_logic
        Enable gain margin logic.
    margin_threshold
        Threshold for margin-based decisions.
    random_state
        Random state for reproducibility.
    """

    _PARAM_ALIASES: ClassVar[dict[str, str | tuple[str, ...]]] = {
        "enable_gain_margin_logic": "enable_margin_vetoes",
        "enable_robust_consensus": "enable_prefix_consensus",
        "enable_threshold_binning": "enable_quantile_grid_thresholds",
        "variance_penalty": ("variance_penalty_weight", "variance_stopping_weight"),
        "n_bootstrap": "variance_tracking_samples",
    }

    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        # === CORE TREE PARAMETERS ===
        max_depth: int = 5,
        min_samples_split: int = 40,
        min_samples_leaf: int = 20,
        # === BOOTSTRAP VARIANCE PENALTY ===
        variance_penalty: float = 1.0,  # Signature feature
        n_bootstrap: int = 10,
        enable_variance_aware_stopping: bool = True,  # Signature feature
        # === HONEST PARTITIONING ===
        split_frac: float = 0.6,
        val_frac: float = 0.2,
        est_frac: float = 0.2,
        enable_stratified_sampling: bool = True,  # ENHANCED: from RobustPrefix
        # === ENHANCED: STRATIFIED BOOTSTRAPS (from RobustPrefix) ===
        # === ENHANCED: WINSORIZATION (from RobustPrefix) ===
        enable_winsorization: bool = True,  # NEW: apply before bootstrap sampling
        winsor_quantiles: tuple = (0.01, 0.99),
        # === ENHANCED: THRESHOLD BINNING (from RobustPrefix) ===
        enable_threshold_binning: bool = True,  # NEW: bin thresholds to reduce micro-jitter
        max_threshold_bins: int = 24,
        # === ENHANCED: ROBUST CONSENSUS (from RobustPrefix) ===
        enable_robust_consensus: bool = True,  # NEW: replace SimpleTree with consensus
        consensus_samples: int = 12,
        consensus_threshold: float = 0.5,
        # === ENHANCED: OBLIQUE SPLITS (from LessGreedy) ===
        enable_oblique_splits: bool = True,  # NEW: can significantly reduce bootstrap variance
        oblique_regularization: Literal["lasso", "ridge", "elastic_net"] = "lasso",
        enable_correlation_gating: bool = True,
        min_correlation_threshold: float = 0.3,
        # === ENHANCED: LOOKAHEAD (from LessGreedy) ===
        enable_lookahead: bool = True,  # NEW: combine with variance penalty
        lookahead_depth: int = 1,  # Conservative for variance method
        beam_width: int = 8,  # Smaller beam for efficiency
        enable_ambiguity_gating: bool = True,  # Use lookahead when penalty alone is ambiguous
        ambiguity_threshold: float = 0.1,  # More conservative threshold
        min_samples_for_lookahead: int = 100,
        # === LEAF STABILIZATION ===
        leaf_smoothing: float = 0.0,  # Conservative default
        leaf_smoothing_strategy: Literal[
            "m_estimate", "shrink_to_parent"
        ] = "m_estimate",
        # === MARGIN-BASED LOGIC ===
        enable_gain_margin_logic: bool = True,
        margin_threshold: float = 0.03,
        # === CLASSIFICATION ===
        random_state: int | None = None,
    ):
        # Configure defaults that reflect Bootstrap method's personality
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
            # ENHANCED: Winsorization (from RobustPrefix)
            enable_winsorization=enable_winsorization,
            winsor_quantiles=winsor_quantiles,
            # ENHANCED: Threshold binning (from RobustPrefix)
            enable_threshold_binning=enable_threshold_binning,
            max_threshold_bins=max_threshold_bins,
            # ENHANCED: Robust consensus (from RobustPrefix)
            enable_prefix_consensus=enable_robust_consensus,
            consensus_samples=consensus_samples,
            consensus_threshold=consensus_threshold,
            enable_quantile_grid_thresholds=enable_threshold_binning,
            # ENHANCED: Oblique splits (from LessGreedy)
            enable_oblique_splits=enable_oblique_splits,
            oblique_regularization=oblique_regularization,
            enable_correlation_gating=enable_correlation_gating,
            min_correlation_threshold=min_correlation_threshold,
            # ENHANCED: Lookahead (from LessGreedy)
            enable_lookahead=enable_lookahead,
            lookahead_depth=lookahead_depth,
            beam_width=beam_width,
            enable_ambiguity_gating=enable_ambiguity_gating,
            ambiguity_threshold=ambiguity_threshold,
            min_samples_for_lookahead=min_samples_for_lookahead,
            # Variance awareness - signature feature
            enable_variance_aware_stopping=enable_variance_aware_stopping,
            variance_tracking_samples=n_bootstrap,
            enable_explicit_variance_penalty=True,  # Core feature
            variance_penalty_weight=variance_penalty,
            # Margin logic
            enable_margin_vetoes=enable_gain_margin_logic,
            margin_threshold=margin_threshold,
            # Leaf stabilization
            leaf_smoothing=leaf_smoothing,
            leaf_smoothing_strategy=leaf_smoothing_strategy,
            # Classification
            # Focus on maximum stability
            algorithm_focus="stability",
            random_state=random_state,
        )

        # Store Bootstrap-specific parameters for sklearn compatibility
        self.variance_penalty = variance_penalty
        self.n_bootstrap = n_bootstrap
        self.enable_variance_aware_stopping = enable_variance_aware_stopping

        # Cross-method enhancement flags
        self.enable_robust_consensus = enable_robust_consensus
        # The base class also has a parameter of this name and would otherwise
        # leave its own default here, so ``get_params`` would report the
        # opposite of what the caller asked for.
        self.enable_gain_margin_logic = enable_gain_margin_logic

        # Initialize fitted attributes

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrapVariancePenalizedTree":
        """
        Fit with bootstrap variance tracking.

        Parameters
        ----------
        X
            Training features.
        y
            Training targets.

        Returns
        -------
        BootstrapVariancePenalizedTree
            Fitted estimator.
        """
        # Call parent fit method
        super().fit(X, y)

        return self

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
        return super().get_params(deep=deep)

    def set_params(self, **params: Any) -> "BootstrapVariancePenalizedTree":
        """
        Set parameters for sklearn compatibility.

        Parameters
        ----------
        **params
            Parameter values to set.

        Returns
        -------
        BootstrapVariancePenalizedTree
            Self with updated parameters.
        """
        return super().set_params(**params)
