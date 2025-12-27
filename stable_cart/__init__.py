"""Public package exports for stable_cart."""

# Base class for advanced users
from .base_stable_tree import BaseStableTree
from .evaluation import evaluate_models, prediction_stability
from .split_strategies import SplitStrategy, create_split_strategy

# Stability utilities for researchers
from .stability_utils import SplitCandidate, StabilityMetrics
from .unified_bootstrap_variance_tree import BootstrapVariancePenalizedTree

# Unified tree classes with all stability primitives
from .unified_less_greedy_tree import LessGreedyHybridTree
from .unified_robust_prefix_tree import RobustPrefixHonestTree

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

__version__ = "0.3.1"  # Enhanced type safety and documentation quality
