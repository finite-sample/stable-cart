"""Public package exports for stable_cart."""

from importlib.metadata import PackageNotFoundError, version

from .evaluation import (
    bootstrap_instability,
    bootstrap_predictions,
)
from .explanation_stability import (
    explanation_instability,
    path_agreement,
    root_agreement,
    split_feature_paths,
    split_features,
)
from .frontier import pareto_front, stability_frontier
from .linear import linear_frontier, linear_instability, shrinkage_coefficients
from .representative_estimator import RepresentativeEstimator

# Plots: matplotlib is imported lazily inside these, so the extra is only
# needed by callers who actually draw something.
from .stability_plots import (
    plot_mape_by_prediction,
    plot_prediction_instability,
    plot_stability_frontier,
)

__all__ = [
    # Evaluation utilities
    "bootstrap_instability",
    "bootstrap_predictions",
    "stability_frontier",
    "pareto_front",
    # Supported estimator and fixed-design calibration
    "RepresentativeEstimator",
    "linear_instability",
    "linear_frontier",
    "shrinkage_coefficients",
    # Tree-structure diagnostics
    "split_features",
    "split_feature_paths",
    "explanation_instability",
    "root_agreement",
    "path_agreement",
    # Plots (need the optional matplotlib extra)
    "plot_prediction_instability",
    "plot_mape_by_prediction",
    "plot_stability_frontier",
]

try:
    __version__ = version("stable-cart")
except PackageNotFoundError:
    # Package is not installed, likely in development mode
    __version__ = "0.0.0.dev"
