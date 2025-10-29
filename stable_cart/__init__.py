"""Public package exports for stable_cart."""

from .evaluation import prediction_stability, evaluate_models

# Unified tree classes with all stability primitives
from .unified_less_greedy_tree import LessGreedyHybridTree
from .unified_bootstrap_variance_tree import BootstrapVariancePenalizedTree
from .unified_robust_prefix_tree import RobustPrefixHonestTree

# Base class for advanced users
from .base_stable_tree import BaseStableTree

# Stability utilities for researchers
from .stability_utils import SplitCandidate, StabilityMetrics
from .split_strategies import SplitStrategy, create_split_strategy

__all__ = [
    # Evaluation utilities
    "prediction_stability",
    "evaluate_models",
    # Main tree classes
    "LessGreedyHybridTree",
    "BootstrapVariancePenalizedTree",
    "RobustPrefixHonestTree",
    # Advanced/research APIs
    "BaseStableTree",
    "SplitCandidate",
    "StabilityMetrics",
    "SplitStrategy",
    "create_split_strategy",
]

__version__ = "0.3.0"  # Major enhancement with unified stability primitives
