"""
BootstrapVariancePenalizedTree: Enhanced with cross-method learning.

Now inherits from BaseStableTree and incorporates lessons from:
- RobustPrefixHonestTree: Stratified bootstraps, winsorization, threshold binning, robust consensus
- LessGreedyHybridTree: Oblique splits, lookahead, beam search
"""

from typing import Literal

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
    """

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
        bootstrap_max_depth: int = 2,  # Depth for variance estimation trees
        enable_variance_aware_stopping: bool = True,  # Signature feature
        # === HONEST PARTITIONING ===
        split_frac: float = 0.6,
        val_frac: float = 0.2,
        est_frac: float = 0.2,
        enable_stratified_sampling: bool = True,  # ENHANCED: from RobustPrefix
        # === ENHANCED: STRATIFIED BOOTSTRAPS (from RobustPrefix) ===
        enable_stratified_bootstraps: bool = True,  # NEW: target-stratified sampling
        bootstrap_stratification_bins: int = 5,  # For regression quantile bins
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
        oblique_strategy: Literal["root_only", "all_levels", "adaptive"] = "adaptive",
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
        classification_criterion: Literal["gini", "entropy"] = "gini",
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
            validation_metric="variance_penalized",  # Signature approach
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
            oblique_strategy=oblique_strategy,
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
            variance_stopping_weight=variance_penalty,
            enable_bootstrap_variance_tracking=True,
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
            classification_criterion=classification_criterion,
            # Focus on maximum stability
            algorithm_focus="stability",
            random_state=random_state,
        )

        # Store Bootstrap-specific parameters for backwards compatibility
        self.variance_penalty = variance_penalty
        self.n_bootstrap = n_bootstrap
        self.bootstrap_max_depth = bootstrap_max_depth

        # Cross-method enhancement flags
        self.enable_stratified_bootstraps = enable_stratified_bootstraps
        self.bootstrap_stratification_bins = bootstrap_stratification_bins
        self.enable_robust_consensus = enable_robust_consensus

        # Initialize fitted attributes
        self.bootstrap_evaluations_ = 0

    def fit(self, X, y):
        """Fit with bootstrap variance tracking."""
        # Call parent fit method
        result = super().fit(X, y)

        # Set bootstrap evaluations for backwards compatibility
        if self.enable_explicit_variance_penalty:
            # Estimate number of bootstrap evaluations based on tree structure
            self.bootstrap_evaluations_ = self._estimate_bootstrap_evaluations()
        else:
            self.bootstrap_evaluations_ = 0

        return result

    def _estimate_bootstrap_evaluations(self):
        """Estimate total bootstrap evaluations performed during training."""
        if self.tree_ is None:
            return 0

        # Rough estimate: internal nodes * n_bootstrap * candidate evaluations
        internal_nodes = self._count_internal_nodes(self.tree_)
        candidates_per_node = 10  # Rough estimate

        return internal_nodes * self.n_bootstrap * candidates_per_node

    def _count_internal_nodes(self, node):
        """Count internal (non-leaf) nodes recursively."""
        if node["type"] == "leaf":
            return 0

        count = 1  # This node
        if "left" in node:
            count += self._count_internal_nodes(node["left"])
        if "right" in node:
            count += self._count_internal_nodes(node["right"])

        return count

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return super().get_params(deep=deep)

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        return super().set_params(**params)


# Create the backwards-compatible aliases
BootstrapVariancePenalizedRegressor = (
    BootstrapVariancePenalizedTree  # Will need task='regression'
)
BootstrapVariancePenalizedClassifier = (
    BootstrapVariancePenalizedTree  # Will need task='classification'
)
