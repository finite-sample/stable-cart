"""
Unified split finding strategies that implement different approaches to
split selection while maintaining consistent interfaces.

This allows different tree methods to compose split strategies flexibly.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from .stability_utils import (
    SplitCandidate,
    _find_candidate_splits,
    apply_margin_veto,
    beam_search_splits,
    bootstrap_consensus_split,
    enable_deterministic_tiebreaking,
    estimate_split_variance,
    generate_oblique_candidates,
    should_stop_splitting,
    validation_checked_split_selection,
)


class SplitStrategy(ABC):
    """Abstract base class for split finding strategies."""

    @abstractmethod
    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        depth: int = 0,
        **kwargs,
    ) -> SplitCandidate | None:
        """
        Find the best split for the given data.

        Parameters
        ----------
        X, y : np.ndarray
            Training data for structure learning
        X_val, y_val : np.ndarray, optional
            Validation data for split evaluation
        depth : int
            Current depth in the tree
        **kwargs
            Strategy-specific parameters

        Returns
        -------
        best_split : SplitCandidate or None
            Best split found, or None if no good split exists
        """
        pass

    @abstractmethod
    def should_stop(
        self, X: np.ndarray, y: np.ndarray, current_gain: float, depth: int, **kwargs
    ) -> bool:
        """
        Determine if splitting should stop at this node.
        """
        pass


class AxisAlignedStrategy(SplitStrategy):
    """Traditional axis-aligned splits with optional enhancements."""

    def __init__(
        self,
        max_candidates: int = 20,
        enable_deterministic_tiebreaking: bool = True,
        enable_margin_veto: bool = False,
        margin_threshold: float = 0.03,
        task: str = "regression",
    ):
        self.max_candidates = max_candidates
        self.enable_deterministic_tiebreaking = enable_deterministic_tiebreaking
        self.enable_margin_veto = enable_margin_veto
        self.margin_threshold = margin_threshold
        self.task = task

    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        depth: int = 0,
        **kwargs,
    ) -> SplitCandidate | None:
        """Find best axis-aligned split."""
        candidates = _find_candidate_splits(X, y, self.max_candidates)

        if not candidates:
            return None

        if self.enable_margin_veto:
            candidates = apply_margin_veto(candidates, self.margin_threshold)
            if not candidates:
                return None

        if self.enable_deterministic_tiebreaking:
            candidates = enable_deterministic_tiebreaking(candidates)

        # Use validation if available
        if X_val is not None and y_val is not None:
            return validation_checked_split_selection(
                X, y, X_val, y_val, candidates, task=self.task
            )

        return candidates[0] if candidates else None

    def should_stop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        current_gain: float,
        depth: int,
        max_depth: int = 10,
        min_samples_split: int = 2,
        **kwargs,
    ) -> bool:
        """Basic stopping criteria."""
        if depth >= max_depth:
            return True
        if len(X) < min_samples_split:
            return True
        if current_gain <= 0:
            return True
        return False


class ConsensusStrategy(SplitStrategy):
    """Bootstrap consensus-based split selection."""

    def __init__(
        self,
        consensus_samples: int = 12,
        consensus_threshold: float = 0.5,
        enable_quantile_binning: bool = True,
        max_bins: int = 24,
        fallback_strategy: SplitStrategy | None = None,
        task: str = "regression",
        random_state: int | None = None,
    ):
        self.consensus_samples = consensus_samples
        self.consensus_threshold = consensus_threshold
        self.enable_quantile_binning = enable_quantile_binning
        self.max_bins = max_bins
        self.fallback_strategy = fallback_strategy or AxisAlignedStrategy(task=task)
        self.task = task
        self.random_state = random_state

    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        depth: int = 0,
        **kwargs,
    ) -> SplitCandidate | None:
        """Find consensus split using bootstrap voting."""
        best_split, all_candidates = bootstrap_consensus_split(
            X,
            y,
            n_samples=self.consensus_samples,
            threshold=self.consensus_threshold,
            enable_quantile_binning=self.enable_quantile_binning,
            max_bins=self.max_bins,
            random_state=self.random_state,
        )

        if best_split is not None:
            # Use validation to refine if available
            if X_val is not None and y_val is not None and all_candidates:
                validated_split = validation_checked_split_selection(
                    X, y, X_val, y_val, all_candidates, task=self.task
                )
                return validated_split or best_split
            return best_split

        # Fall back to simpler strategy if consensus fails
        return self.fallback_strategy.find_best_split(X, y, X_val, y_val, depth, **kwargs)

    def should_stop(
        self, X: np.ndarray, y: np.ndarray, current_gain: float, depth: int, **kwargs
    ) -> bool:
        """Use fallback strategy for stopping criteria."""
        return self.fallback_strategy.should_stop(X, y, current_gain, depth, **kwargs)


class ObliqueStrategy(SplitStrategy):
    """Oblique splits using linear projections."""

    def __init__(
        self,
        oblique_regularization: Literal["lasso", "ridge", "elastic_net"] = "lasso",
        enable_correlation_gating: bool = True,
        min_correlation: float = 0.3,
        fallback_strategy: SplitStrategy | None = None,
        task: str = "regression",
        random_state: int | None = None,
    ):
        self.oblique_regularization = oblique_regularization
        self.enable_correlation_gating = enable_correlation_gating
        self.min_correlation = min_correlation
        self.fallback_strategy = fallback_strategy or AxisAlignedStrategy(task=task)
        self.task = task
        self.random_state = random_state

    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        depth: int = 0,
        **kwargs,
    ) -> SplitCandidate | None:
        """Find best oblique split."""
        oblique_candidates = generate_oblique_candidates(
            X,
            y,
            strategy=self.oblique_regularization,
            enable_correlation_gating=self.enable_correlation_gating,
            min_correlation=self.min_correlation,
            task=self.task,
            random_state=self.random_state,
        )

        # Also get axis-aligned candidates for comparison
        axis_candidates = _find_candidate_splits(X, y, max_candidates=10)

        all_candidates = oblique_candidates + axis_candidates

        if not all_candidates:
            return None

        # Use validation to select best
        if X_val is not None and y_val is not None:
            return validation_checked_split_selection(
                X, y, X_val, y_val, all_candidates, task=self.task
            )

        # Otherwise return best by training gain
        return max(all_candidates, key=lambda c: c.gain)

    def should_stop(
        self, X: np.ndarray, y: np.ndarray, current_gain: float, depth: int, **kwargs
    ) -> bool:
        """Use fallback strategy for stopping criteria."""
        return self.fallback_strategy.should_stop(X, y, current_gain, depth, **kwargs)


class LookaheadStrategy(SplitStrategy):
    """Lookahead with beam search."""

    def __init__(
        self,
        lookahead_depth: int = 2,
        beam_width: int = 12,
        enable_ambiguity_gating: bool = True,
        ambiguity_threshold: float = 0.05,
        min_samples_for_lookahead: int = 100,
        fallback_strategy: SplitStrategy | None = None,
        task: str = "regression",
    ):
        self.lookahead_depth = lookahead_depth
        self.beam_width = beam_width
        self.enable_ambiguity_gating = enable_ambiguity_gating
        self.ambiguity_threshold = ambiguity_threshold
        self.min_samples_for_lookahead = min_samples_for_lookahead
        self.fallback_strategy = fallback_strategy or AxisAlignedStrategy(task=task)
        self.task = task

    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        depth: int = 0,
        **kwargs,
    ) -> SplitCandidate | None:
        """Find split using lookahead beam search."""
        if len(X) < self.min_samples_for_lookahead:
            # Fall back for small datasets
            return self.fallback_strategy.find_best_split(X, y, X_val, y_val, depth, **kwargs)

        candidates = beam_search_splits(
            X,
            y,
            depth=self.lookahead_depth,
            beam_width=self.beam_width,
            enable_ambiguity_gating=self.enable_ambiguity_gating,
            ambiguity_threshold=self.ambiguity_threshold,
            task=self.task,
        )

        if not candidates:
            return None

        # Use validation to refine
        if X_val is not None and y_val is not None:
            return validation_checked_split_selection(
                X, y, X_val, y_val, candidates, task=self.task
            )

        return candidates[0]

    def should_stop(
        self, X: np.ndarray, y: np.ndarray, current_gain: float, depth: int, **kwargs
    ) -> bool:
        """Use fallback strategy for stopping criteria."""
        return self.fallback_strategy.should_stop(X, y, current_gain, depth, **kwargs)


class VariancePenalizedStrategy(SplitStrategy):
    """Variance-aware split selection with explicit penalties."""

    def __init__(
        self,
        variance_penalty_weight: float = 1.0,
        variance_estimation_samples: int = 10,
        stopping_strategy: Literal["one_se", "variance_penalty", "both"] = "variance_penalty",
        base_strategy: SplitStrategy | None = None,
        task: str = "regression",
        random_state: int | None = None,
    ):
        self.variance_penalty_weight = variance_penalty_weight
        self.variance_estimation_samples = variance_estimation_samples
        self.stopping_strategy = stopping_strategy
        self.base_strategy = base_strategy or AxisAlignedStrategy(task=task)
        self.task = task
        self.random_state = random_state

    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        depth: int = 0,
        **kwargs,
    ) -> SplitCandidate | None:
        """Find split with explicit variance penalty."""
        # Get candidates from base strategy
        base_split = self.base_strategy.find_best_split(X, y, X_val, y_val, depth, **kwargs)

        if base_split is None:
            return None

        # Estimate variance of this split
        variance_estimate = estimate_split_variance(
            X,
            y,
            base_split,
            n_bootstrap=self.variance_estimation_samples,
            task=self.task,
            random_state=self.random_state,
        )

        base_split.variance_estimate = variance_estimate

        # Apply variance penalty to gain
        penalized_gain = base_split.gain - self.variance_penalty_weight * variance_estimate

        if penalized_gain <= 0:
            return None  # Split not worth the variance cost

        # Update gain with penalty
        base_split.gain = penalized_gain
        return base_split

    def should_stop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        current_gain: float,
        depth: int,
        variance_estimate: float = 0.0,
        **kwargs,
    ) -> bool:
        """Variance-aware stopping criteria."""
        # Base stopping criteria
        if self.base_strategy.should_stop(X, y, current_gain, depth, **kwargs):
            return True

        # Variance-aware stopping
        return should_stop_splitting(
            current_gain, variance_estimate, self.variance_penalty_weight, self.stopping_strategy
        )


class CompositeStrategy(SplitStrategy):
    """Composite strategy that tries multiple approaches and selects the best."""

    def __init__(
        self,
        strategies: list[SplitStrategy],
        selection_metric: Literal["gain", "validation", "variance_penalized"] = "validation",
        task: str = "regression",
    ):
        self.strategies = strategies
        self.selection_metric = selection_metric
        self.task = task

        if not strategies:
            raise ValueError("Must provide at least one strategy")

    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        depth: int = 0,
        **kwargs,
    ) -> SplitCandidate | None:
        """Try all strategies and select the best split."""
        candidates = []

        for strategy in self.strategies:
            try:
                split = strategy.find_best_split(X, y, X_val, y_val, depth, **kwargs)
                if split is not None:
                    candidates.append(split)
            except Exception:
                # Continue if one strategy fails
                continue

        if not candidates:
            return None

        # Select best based on metric
        if self.selection_metric == "gain":
            return max(candidates, key=lambda c: c.gain)
        elif self.selection_metric == "validation" and X_val is not None and y_val is not None:
            return validation_checked_split_selection(
                X, y, X_val, y_val, candidates, task=self.task
            )
        elif self.selection_metric == "variance_penalized":
            # Prefer candidates with lower variance estimates
            valid_candidates = [c for c in candidates if c.variance_estimate is not None]
            if valid_candidates:
                return min(
                    valid_candidates, key=lambda c: c.variance_estimate - c.gain
                )  # Lower variance, higher gain
            else:
                return max(candidates, key=lambda c: c.gain)
        else:
            return max(candidates, key=lambda c: c.gain)

    def should_stop(
        self, X: np.ndarray, y: np.ndarray, current_gain: float, depth: int, **kwargs
    ) -> bool:
        """Stop if any strategy says to stop."""
        return any(
            strategy.should_stop(X, y, current_gain, depth, **kwargs)
            for strategy in self.strategies
        )


class HybridStrategy(SplitStrategy):
    """
    Hybrid strategy that adapts behavior based on data characteristics.

    This implements the "algorithm focus" concept where we can emphasize
    speed, stability, or accuracy based on the situation.
    """

    def __init__(
        self,
        focus: Literal["speed", "stability", "accuracy"] = "stability",
        task: str = "regression",
        random_state: int | None = None,
    ):
        self.focus = focus
        self.task = task
        self.random_state = random_state

        # Build appropriate strategy based on focus
        if focus == "speed":
            self.strategy = AxisAlignedStrategy(
                max_candidates=10, enable_deterministic_tiebreaking=True, task=task
            )
        elif focus == "accuracy":
            # Composite of oblique + lookahead for best accuracy
            self.strategy = CompositeStrategy(
                [
                    ObliqueStrategy(task=task, random_state=random_state),
                    LookaheadStrategy(task=task),
                    AxisAlignedStrategy(task=task),
                ],
                selection_metric="validation",
                task=task,
            )
        else:  # stability
            # Consensus + variance penalty for maximum stability
            self.strategy = CompositeStrategy(
                [
                    VariancePenalizedStrategy(
                        base_strategy=ConsensusStrategy(task=task, random_state=random_state),
                        task=task,
                        random_state=random_state,
                    ),
                    ConsensusStrategy(task=task, random_state=random_state),
                ],
                selection_metric="variance_penalized",
                task=task,
            )

    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        depth: int = 0,
        **kwargs,
    ) -> SplitCandidate | None:
        """Delegate to the configured strategy."""
        return self.strategy.find_best_split(X, y, X_val, y_val, depth, **kwargs)

    def should_stop(
        self, X: np.ndarray, y: np.ndarray, current_gain: float, depth: int, **kwargs
    ) -> bool:
        """Delegate to the configured strategy."""
        return self.strategy.should_stop(X, y, current_gain, depth, **kwargs)


# ============================================================================
# STRATEGY FACTORY
# ============================================================================


def create_split_strategy(strategy_type: str, task: str = "regression", **kwargs) -> SplitStrategy:
    """
    Factory function to create split strategies by name.

    Parameters
    ----------
    strategy_type : str
        Type of strategy: 'axis_aligned', 'consensus', 'oblique',
        'lookahead', 'variance_penalized', 'composite', 'hybrid'
    task : str
        'regression' or 'classification'
    **kwargs
        Strategy-specific parameters

    Returns
    -------
    strategy : SplitStrategy
        Configured split strategy
    """
    if strategy_type == "axis_aligned":
        return AxisAlignedStrategy(task=task, **kwargs)
    elif strategy_type == "consensus":
        return ConsensusStrategy(task=task, **kwargs)
    elif strategy_type == "oblique":
        return ObliqueStrategy(task=task, **kwargs)
    elif strategy_type == "lookahead":
        return LookaheadStrategy(task=task, **kwargs)
    elif strategy_type == "variance_penalized":
        return VariancePenalizedStrategy(task=task, **kwargs)
    elif strategy_type == "hybrid":
        return HybridStrategy(task=task, **kwargs)
    elif strategy_type == "composite":
        # Default composite with common strategies
        strategies = [
            AxisAlignedStrategy(task=task),
            ConsensusStrategy(task=task, **kwargs),
        ]
        if kwargs.get("enable_oblique", False):
            strategies.append(ObliqueStrategy(task=task, **kwargs))
        if kwargs.get("enable_lookahead", False):
            strategies.append(LookaheadStrategy(task=task, **kwargs))

        return CompositeStrategy(strategies, task=task)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
