"""Public package exports for stable_cart."""

from importlib.metadata import PackageNotFoundError, version

# Base class for advanced users
from .base_stable_tree import BaseStableTree

# Centroid tree: select tree closest to ensemble mean
from .centroid_tree import CentroidTree
from .evaluation import (
    bootstrap_instability,
    bootstrap_predictions,
    evaluate_models,
    prediction_stability,
)
from .frontier import pareto_front, stability_frontier
from .split_strategies import SplitStrategy, create_split_strategy

# Plots: matplotlib is imported lazily inside these, so the extra is only
# needed by callers who actually draw something.
from .stability_plots import (
    plot_mape_by_prediction,
    plot_prediction_instability,
    plot_stability_frontier,
)

# Stability utilities for researchers
from .stability_utils import SplitCandidate, StabilityMetrics
from .stable_tree import StableTree
from .unified_bootstrap_variance_tree import BootstrapVariancePenalizedTree

# Unified tree classes with all stability primitives
from .unified_less_greedy_tree import LessGreedyHybridTree
from .unified_robust_prefix_tree import RobustPrefixHonestTree

__all__ = [
    # Evaluation utilities
    "prediction_stability",
    "bootstrap_instability",
    "bootstrap_predictions",
    "stability_frontier",
    "pareto_front",
    "evaluate_models",
    # Plots (need the optional matplotlib extra)
    "plot_prediction_instability",
    "plot_mape_by_prediction",
    "plot_stability_frontier",
    # Main tree classes
    "StableTree",
    "LessGreedyHybridTree",
    "BootstrapVariancePenalizedTree",
    "RobustPrefixHonestTree",
    "CentroidTree",
    # Advanced/research APIs
    "BaseStableTree",
    "SplitCandidate",
    "StabilityMetrics",
    "SplitStrategy",
    "create_split_strategy",
]

try:
    __version__ = version("stable-cart")
except PackageNotFoundError:
    # Package is not installed, likely in development mode
    __version__ = "0.0.0.dev"
